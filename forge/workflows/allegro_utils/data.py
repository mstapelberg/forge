from typing import Dict, Any
import logging

from nequip.data.datamodule import ASEDataModule
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from .samplers import RareWeightedSampler

logger = logging.getLogger(__name__)


class CustomSamplingASEDataModule(ASEDataModule):
    """An ASEDataModule that supports a custom training set sampler.

    This class extends the standard NequIP ASEDataModule to allow for the
    injection of a custom sampler for the training DataLoader. It works by
    overriding the `train_dataloader` method to manually instantiate the
    sampler with the prepared training dataset before creating the DataLoader.
    
    It is careful to pop its custom arguments from kwargs before calling the
    parent constructor, to avoid breaking the parent's initialization.
    """
    def __init__(self, **kwargs):
        # Pop our custom key before passing the rest to the parent.
        sampler_config = kwargs.pop("sampler_config", None)
        
        # Now, kwargs contains only arguments that ASEDataModule expects.
        super().__init__(**kwargs)
        
        self.sampler_config = sampler_config

    def train_dataloader(self) -> DataLoader:
        """Builds the training DataLoader with the custom sampler if provided."""
        # First check if datasets have been set up
        train_dataset = getattr(self, 'train_dataset', None)
        
        # Debug logging
        logger.debug(f"train_dataset type: {type(train_dataset)}")
        logger.debug(f"train_dataset value: {train_dataset}")
        
        # If train_dataset is a config object, we need to call setup first
        if isinstance(train_dataset, (DictConfig, ListConfig)) or train_dataset is None:
            logger.warning("train_dataset is not yet instantiated. Calling parent's train_dataloader().")
            # Let the parent handle dataset setup and dataloader creation
            return super().train_dataloader()
        
        if self.sampler_config is None:
            # If no sampler is configured, use the default behavior
            return super().train_dataloader()

        # Instantiate the custom sampler, providing the dataset it needs
        sampler = instantiate(self.sampler_config, data_source=train_dataset)
        
        # Get the dataloader configuration
        # In NequIP, this is typically stored as train_dataloader attribute
        train_dl_config = getattr(self, 'train_dataloader', None)
        if train_dl_config is None:
            train_dl_config = getattr(self, 'train_dataloader_config', None)
            if train_dl_config is None:
                logger.error("Could not find train dataloader configuration.")
                # Fall back to parent implementation
                return super().train_dataloader()
        
        # Get collate_fn from the dataset if it exists
        collate_fn = getattr(train_dataset, 'collate_fn', None)
        if collate_fn is None:
            # Try to get it from the datamodule itself
            collate_fn = getattr(self, 'collate_fn', None)
        
        logger.debug(f"Using collate_fn: {collate_fn}")
        
        # Build kwargs for DataLoader instantiation
        dl_kwargs = {
            "dataset": train_dataset,
            "sampler": sampler,
            "batch_sampler": None,  # Sampler and batch_sampler are mutually exclusive
            "shuffle": False,  # Shuffle is mutually exclusive with a sampler
        }
        
        # Only add collate_fn if it exists
        if collate_fn is not None:
            dl_kwargs["collate_fn"] = collate_fn
        
        # Instantiate the DataLoader
        return instantiate(train_dl_config, **dl_kwargs) 