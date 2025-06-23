from typing import Dict, Any

from nequip.data.datamodule import ASEDataModule
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from .samplers import RareWeightedSampler


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
        if self.train_dataset is None:
            raise RuntimeError("The training dataset has not been prepared. Call `setup()` first.")

        if self.sampler_config is None:
            # If no sampler is configured, use the default behavior
            return super().train_dataloader()

        # Instantiate the custom sampler, providing the dataset it needs
        sampler = instantiate(self.sampler_config, data_source=self.train_dataset)
        
        # Instantiate the DataLoader, providing the dataset AND our custom sampler
        # and crucially, the collate_fn from the dataset
        return instantiate(
            self.train_dataloader_config,
            dataset=self.train_dataset,
            sampler=sampler,
            batch_sampler=None, # Sampler and batch_sampler are mutually exclusive
            shuffle=False, # Shuffle is mutually exclusive with a sampler
            collate_fn=self.train_dataset.collate_fn, # The collate_fn lives on the *dataset*
        ) 