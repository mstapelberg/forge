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
        # Call parent setup first to ensure datasets are created
        super().setup(stage=stage)
        
        # Now that datasets are created, we can instantiate the sampler if needed
        if stage == "fit" and self._sampler_config is not None:
            if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                logger.info("Creating custom sampler for training dataset")
                try:
                    # Instantiate the sampler with the actual dataset
                    self._custom_sampler = instantiate(
                        self._sampler_config, 
                        data_source=self.train_dataset
                    )
                    logger.info(f"Successfully created sampler: {type(self._custom_sampler)}")
                    
                    # Modify the train_dataloader configuration to use our sampler
                    # This assumes train_dataloader is a config dict
                    if hasattr(self, 'train_dataloader') and isinstance(self.train_dataloader, (dict, DictConfig)):
                        # Store original shuffle setting
                        self._original_shuffle = self.train_dataloader.get('shuffle', None)
                        
                        # When using a sampler, shuffle must be False
                        self.train_dataloader['shuffle'] = False
                        # Remove any batch_sampler if present
                        self.train_dataloader.pop('batch_sampler', None)
                        
                        logger.info("Modified train_dataloader config to prepare for custom sampler")
                    
                except Exception as e:
                    logger.error(f"Failed to create custom sampler: {e}", exc_info=True)
                    self._custom_sampler = None
            else:
                logger.warning("train_dataset not available in setup, cannot create sampler")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader, potentially with custom sampler."""
        # Get the base dataloader from parent
        dataloader = super().train_dataloader()
        
        # If we have a custom sampler, we need to recreate the dataloader with it
        if self._custom_sampler is not None:
            logger.info("Recreating train dataloader with custom sampler")
            
            # Extract key parameters from the existing dataloader
            dl_params = {
                'dataset': dataloader.dataset,
                'batch_size': dataloader.batch_size,
                'sampler': self._custom_sampler,
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
            logger.info("Successfully created dataloader with custom sampler")
        
        return dataloader 