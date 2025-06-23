"""Alternative data module implementation that works with NequIP's dataloader creation flow."""

from typing import Dict, Any, Optional
import logging

from nequip.data.datamodule import ASEDataModule
from torch.utils.data import DataLoader, Sampler
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class CustomSamplingASEDataModuleV2(ASEDataModule):
    """ASEDataModule with custom sampler support that works with NequIP's flow.
    
    Instead of overriding train_dataloader, this version hooks into the
    setup phase to modify the dataloader configuration before it's used.
    """
    
    def __init__(self, sampler_config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize with optional sampler configuration.
        
        Args:
            sampler_config: Configuration dict for the sampler (with _target_ key)
            **kwargs: All other arguments passed to ASEDataModule
        """
        # Store sampler config before calling parent
        self._sampler_config = sampler_config
        self._custom_sampler = None
        
        # Call parent constructor with remaining kwargs
        super().__init__(**kwargs)
        
        logger.info(f"Initialized {self.__class__.__name__} with sampler_config: {sampler_config}")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets and potentially modify dataloader configs.
        
        This is called by Lightning before requesting dataloaders.
        """
        # Store the sampler config but don't create sampler yet
        # We'll create it in train_dataloader() when we have access to the full dataset
        if stage == "fit" and self._sampler_config is not None:
            logger.info(f"Will create custom sampler with config: {self._sampler_config}")
            # Modify dataloader config to disable shuffle when using sampler
            if hasattr(self, 'train_dataloader') and isinstance(self.train_dataloader, (dict, DictConfig)):
                self._original_shuffle = self.train_dataloader.get('shuffle', None)
                self.train_dataloader['shuffle'] = False
                self.train_dataloader.pop('batch_sampler', None)
        
        # Call parent setup to create datasets
        super().setup(stage=stage)

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader, potentially with custom sampler."""
        # First, check if we need to create a custom sampler
        if self._sampler_config is not None and self._custom_sampler is None:
            # Get the full dataset before it's distributed
            if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                logger.info(f"Creating custom sampler with dataset of size: {len(self.train_dataset)}")
                try:
                    self._custom_sampler = instantiate(
                        self._sampler_config,
                        data_source=self.train_dataset
                    )
                    logger.info(f"Successfully created sampler: {type(self._custom_sampler)} with {len(self._custom_sampler)} indices")
                except Exception as e:
                    logger.error(f"Failed to create custom sampler: {e}", exc_info=True)
                    self._custom_sampler = None
        
        # Get the base dataloader from parent
        dataloader = super().train_dataloader()
        
        # If we have a custom sampler, we need to recreate the dataloader with it
        if self._custom_sampler is not None:
            logger.info("Recreating train dataloader with custom sampler")
            
            # Check if we're in distributed training
            if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, '__class__'):
                sampler_class_name = dataloader.sampler.__class__.__name__
                is_distributed = 'DistributedSampler' in sampler_class_name
            else:
                is_distributed = False
            
            # If distributed, we need to wrap our custom sampler
            final_sampler = self._custom_sampler
            if is_distributed:
                logger.info("Detected distributed training - wrapping custom sampler with DistributedSampler")
                from torch.utils.data.distributed import DistributedSampler
                
                # Create a wrapper that combines DistributedSampler with our custom sampler
                class DistributedCustomSampler(DistributedSampler):
                    """Wrapper that applies distributed sampling to our custom sampler's indices."""
                    def __init__(self, custom_sampler, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
                        # Get the indices from the custom sampler
                        self.custom_sampler = custom_sampler
                        self.indices = list(custom_sampler.indices)  # Get pre-computed indices
                        
                        # Initialize distributed sampler with a dummy dataset of the right length
                        class DummyDataset:
                            def __len__(self):
                                return len(self.indices)
                        
                        super().__init__(
                            dataset=DummyDataset(),
                            num_replicas=num_replicas,
                            rank=rank,
                            shuffle=shuffle,
                            seed=seed,
                            drop_last=drop_last
                        )
                    
                    def __iter__(self):
                        # Get distributed indices from parent
                        distributed_indices = list(super().__iter__())
                        # Map back to the custom sampler's indices
                        for idx in distributed_indices:
                            yield self.indices[idx]
                    
                    def __len__(self):
                        return super().__len__()
                
                # Get distributed parameters from original sampler if available
                orig_sampler = dataloader.sampler
                num_replicas = getattr(orig_sampler, 'num_replicas', None)
                rank = getattr(orig_sampler, 'rank', None)
                shuffle = getattr(orig_sampler, 'shuffle', True)
                seed = getattr(orig_sampler, 'seed', 0)
                drop_last = getattr(orig_sampler, 'drop_last', False)
                
                final_sampler = DistributedCustomSampler(
                    custom_sampler=self._custom_sampler,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=shuffle,
                    seed=seed,
                    drop_last=drop_last
                )
                logger.info(f"Created distributed sampler with {len(final_sampler)} samples for this rank")
            
            # Extract key parameters from the existing dataloader
            dl_params = {
                'dataset': dataloader.dataset,
                'batch_size': dataloader.batch_size,
                'sampler': final_sampler,
                'num_workers': dataloader.num_workers,
                'collate_fn': dataloader.collate_fn,
                'pin_memory': dataloader.pin_memory,
                'drop_last': dataloader.drop_last,
                'timeout': dataloader.timeout,
                'worker_init_fn': dataloader.worker_init_fn,
                'prefetch_factor': getattr(dataloader, 'prefetch_factor', None),
                'persistent_workers': getattr(dataloader, 'persistent_workers', False),
                'shuffle': False,  # Must be False when using a sampler
            }
            
            # Remove None values
            dl_params = {k: v for k, v in dl_params.items() if v is not None}
            
            # Create new dataloader with custom sampler
            dataloader = DataLoader(**dl_params)
            logger.info(f"Successfully created dataloader with custom sampler - {len(dataloader)} batches")
        
        return dataloader 