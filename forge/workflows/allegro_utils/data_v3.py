"""V3 implementation that creates indices before dataset distribution."""

from typing import Dict, Any, Optional, List
import logging
import torch
from torch.utils.data import DataLoader, Sampler
from hydra.utils import instantiate
from nequip.data.datamodule import ASEDataModule

logger = logging.getLogger(__name__)


class PrecomputedIndicesSampler(Sampler):
    """Simple sampler that uses precomputed indices."""
    def __init__(self, indices: List[int]):
        self.indices = indices
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class CustomSamplingASEDataModuleV3(ASEDataModule):
    """V3 implementation that precomputes sampling indices before dataset creation.
    
    This approach:
    1. Loads the dataset file to count structures and compute rare indices
    2. Precomputes all sampling indices with replication
    3. Creates a simple indices sampler that works with distributed training
    """
    
    def __init__(self, sampler_config: Optional[Dict[str, Any]] = None, **kwargs):
        # Store config
        self._sampler_config = sampler_config
        self._precomputed_indices = None
        
        # If sampler config is provided, precompute indices
        if sampler_config is not None:
            self._precompute_indices(kwargs.get('train_file_path'), sampler_config)
        
        # Call parent
        super().__init__(**kwargs)
    
    def _precompute_indices(self, train_file_path: str, sampler_config: Dict[str, Any]):
        """Precompute sampling indices based on the training file."""
        try:
            # Count structures in the file
            import ase.io
            structures = ase.io.read(train_file_path, index=':')
            num_structures = len(structures)
            logger.info(f"Precomputing indices for {num_structures} structures from {train_file_path}")
            
            # Get sampler parameters
            replica = sampler_config.get('replica', 1)
            rare_idx = set(sampler_config.get('rare_idx', []))
            alpha = sampler_config.get('alpha')
            
            # Build indices with replication
            indices = []
            weights = []
            rare_count = 0
            
            for i in range(num_structures):
                num_replicas = replica if i in rare_idx else 1
                if i in rare_idx:
                    rare_count += 1
                indices.extend([i] * num_replicas)
                
                # For simplicity, just use uniform weights in V3
                # (force weighting would require loading all data)
                weight = 1.0
                weights.extend([weight] * num_replicas)
            
            logger.info(f"Precomputed {len(indices)} total indices with {rare_count} rare structures "
                       f"replicated {replica}x")
            
            # Shuffle indices for better distribution
            import random
            random.Random(42).shuffle(indices)  # Use fixed seed for reproducibility
            
            self._precomputed_indices = indices
            
        except Exception as e:
            logger.error(f"Failed to precompute indices: {e}", exc_info=True)
            self._precomputed_indices = None
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with precomputed indices."""
        # Get base dataloader
        dataloader = super().train_dataloader()
        
        # If we have precomputed indices, replace the sampler
        if self._precomputed_indices is not None:
            logger.info(f"Creating dataloader with precomputed {len(self._precomputed_indices)} indices")
            
            # Check if distributed
            is_distributed = hasattr(dataloader.sampler, '__class__') and \
                           'DistributedSampler' in dataloader.sampler.__class__.__name__
            
            if is_distributed:
                # Use PyTorch's DistributedSampler on our indices
                from torch.utils.data.distributed import DistributedSampler
                
                # Create a dummy dataset with our indices length
                class IndicesDataset:
                    def __init__(self, indices):
                        self.indices = indices
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        return self.indices[idx]
                
                indices_dataset = IndicesDataset(self._precomputed_indices)
                
                # Get distributed params from original sampler
                orig = dataloader.sampler
                dist_sampler = DistributedSampler(
                    indices_dataset,
                    num_replicas=getattr(orig, 'num_replicas', None),
                    rank=getattr(orig, 'rank', None),
                    shuffle=False,  # We already shuffled
                    seed=getattr(orig, 'seed', 0)
                )
                
                # Create a sampler that maps distributed indices back to dataset indices
                class MappedDistributedSampler(Sampler):
                    def __init__(self, dist_sampler, indices_dataset):
                        self.dist_sampler = dist_sampler
                        self.indices_dataset = indices_dataset
                    
                    def __iter__(self):
                        for idx in self.dist_sampler:
                            yield self.indices_dataset[idx]
                    
                    def __len__(self):
                        return len(self.dist_sampler)
                
                final_sampler = MappedDistributedSampler(dist_sampler, indices_dataset)
                logger.info(f"Created distributed sampler with {len(final_sampler)} samples per rank")
            else:
                # Use simple sampler with our precomputed indices
                final_sampler = PrecomputedIndicesSampler(self._precomputed_indices)
            
            # Recreate dataloader
            dl_params = {
                'dataset': dataloader.dataset,
                'batch_size': dataloader.batch_size,
                'sampler': final_sampler,
                'num_workers': dataloader.num_workers,
                'collate_fn': dataloader.collate_fn,
                'pin_memory': dataloader.pin_memory,
                'drop_last': dataloader.drop_last,
                'shuffle': False,
            }
            dataloader = DataLoader(**dl_params)
            logger.info(f"Created dataloader with {len(dataloader)} batches")
        
        return dataloader 