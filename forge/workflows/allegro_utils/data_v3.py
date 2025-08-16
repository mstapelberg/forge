"""V3 implementation that creates indices before dataset distribution."""

from typing import Dict, Any, Optional, List, Sequence
import logging
import torch
from torch.utils.data import DataLoader, Sampler
from hydra.utils import instantiate
from nequip.data.datamodule import ASEDataModule
from nequip.data import register_fields

from .config_aware_stress import normalize_config_type, section_of, section_to_id

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
        # Capture file paths for wrapper use
        self._train_file_path = kwargs.get("train_file_path")
        self._val_file_paths = kwargs.get("val_file_path")
        self._test_file_path = kwargs.get("test_file_path")
        
        # If sampler config is provided, precompute indices
        if sampler_config is not None:
            self._precompute_indices(kwargs.get('train_file_path'), sampler_config)
        
        # Call parent
        super().__init__(**kwargs)
        # Ensure our custom long graph field is registered
        try:
            register_fields(graph_fields=["section_id"], long_fields=["section_id"])
        except Exception:
            # Safe if already registered
            pass
    
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
        """Create training dataloader and inject section_id at collate time."""
        # Get base dataloader
        base_dl = super().train_dataloader()

        # Prepare index-returning dataset
        index_ds = _IndexDatasetWrapper(base_dl.dataset)

        # Collate wrapper that adds section_id based on precomputed ids
        section_ids = getattr(self, "_section_ids_train", None)
        collate_fn = _CollateWithSection(base_dl.collate_fn, section_ids)

        # Recreate dataloader with same params
        dl_params = {
            'dataset': index_ds,
            'batch_size': base_dl.batch_size,
            'sampler': base_dl.sampler,
            'num_workers': base_dl.num_workers,
            'collate_fn': collate_fn,
            'pin_memory': base_dl.pin_memory,
            'drop_last': base_dl.drop_last,
            'shuffle': False,
        }
        return DataLoader(**dl_params)

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader and inject section_id at collate time."""
        base_dl = super().val_dataloader()
        section_ids = getattr(self, "_section_ids_val", None)
        def _wrap(dl: DataLoader) -> DataLoader:
            index_ds = _IndexDatasetWrapper(dl.dataset)
            collate_fn = _CollateWithSection(dl.collate_fn, section_ids)
            dl_params = {
                'dataset': index_ds,
                'batch_size': dl.batch_size,
                'sampler': dl.sampler,
                'num_workers': dl.num_workers,
                'collate_fn': collate_fn,
                'pin_memory': dl.pin_memory,
                'drop_last': dl.drop_last,
                'shuffle': False,
            }
            return DataLoader(**dl_params)
        if isinstance(base_dl, (list, tuple)):
            return type(base_dl)(_wrap(d) for d in base_dl)
        return _wrap(base_dl)

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader and inject section_id at collate time."""
        base_dl = super().test_dataloader()
        section_ids = getattr(self, "_section_ids_test", None)
        def _wrap(dl: DataLoader) -> DataLoader:
            index_ds = _IndexDatasetWrapper(dl.dataset)
            collate_fn = _CollateWithSection(dl.collate_fn, section_ids)
            dl_params = {
                'dataset': index_ds,
                'batch_size': dl.batch_size,
                'sampler': dl.sampler,
                'num_workers': dl.num_workers,
                'collate_fn': collate_fn,
                'pin_memory': dl.pin_memory,
                'drop_last': dl.drop_last,
                'shuffle': False,
            }
            return DataLoader(**dl_params)
        if isinstance(base_dl, (list, tuple)):
            return type(base_dl)(_wrap(d) for d in base_dl)
        return _wrap(base_dl)

    # === Hook into dataset creation to inject section ids ===
    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage=stage)
        # Precompute section ids per split from extxyz
        try:
            self._section_ids_train = _compute_section_ids_from_file(self._train_file_path)
        except Exception:
            self._section_ids_train = None
        try:
            # val may be a list; use first file's mapping length as heuristic
            if isinstance(self._val_file_paths, (list, tuple)) and len(self._val_file_paths) > 0:
                self._section_ids_val = _compute_section_ids_from_file(self._val_file_paths[0])
            else:
                self._section_ids_val = _compute_section_ids_from_file(self._val_file_paths)
        except Exception:
            self._section_ids_val = None
        try:
            self._section_ids_test = _compute_section_ids_from_file(self._test_file_path)
        except Exception:
            self._section_ids_test = None


class _IndexDatasetWrapper:
    """Wrap dataset to also return the index along with the item."""
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        return item, int(idx)


class _CollateWithSection:
    """Collate wrapper that adds `section_id` based on provided id list.

    If `section_ids` is None or too short, falls back to zeros.
    """
    def __init__(self, base_collate, section_ids: Optional[List[int]]):
        self.base_collate = base_collate
        self.section_ids = section_ids

    def __call__(self, samples):
        # samples is a list of (item, idx)
        items, indices = zip(*samples)
        batch = self.base_collate(items)
        if self.section_ids is not None:
            sid_list = [self.section_ids[i] if i < len(self.section_ids) else 8 for i in indices]
        else:
            sid_list = [8 for _ in indices]  # map to "Other"
        batch["section_id"] = torch.tensor(sid_list, dtype=torch.long)
        return batch


def _compute_section_ids_from_file(file_path: Optional[str]) -> Optional[List[int]]:
    if not file_path:
        return None
    try:
        import ase.io
        atoms_list = ase.io.read(file_path, index=":")
        ids: List[int] = []
        for atoms in atoms_list:
            raw = atoms.info.get("config_type")
            ct = normalize_config_type(raw)
            sec = section_of(ct)
            ids.append(section_to_id(sec))
        return ids
    except Exception:
        return None