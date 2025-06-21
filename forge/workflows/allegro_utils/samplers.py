# mlip_utils/samplers.py
from typing import List, Optional, Iterator

import torch, random
from torch.utils.data import Sampler, WeightedRandomSampler, Dataset


class RareWeightedSampler(Sampler[int]):
    """A Torch Sampler that oversamples rare configurations and can apply force-based weighting.

    This sampler modifies the dataset indices to achieve two goals:
    1.  **Replica Oversampling**: Specified "rare" configurations are duplicated in the
        sampling pool, increasing their likelihood of being selected in a batch.
    2.  **Force-based Weighting**: Samples can be weighted based on the norm of the
        forces they contain. This prioritizes configurations with high forces,
        which are often more informative for training.

    It is designed to be compatible with NequIP's `ASEDataModule`.

    Args:
        dataset (Dataset): The dataset from which to sample. It is expected that
            each item has a `.metadata` attribute that may contain a "force_norm" key
            if force-weighting is used.
        rare_idx (List[int]): A list of integer indices corresponding to the
            "rare" configurations in the dataset that should be oversampled.
        replica (int): The number of times to duplicate the rare configurations.
            Defaults to 1 (no oversampling).
        alpha (Optional[float]): The scaling factor for force-based weighting.
            If None, weighting is disabled. The weight is calculated as
            `1.0 + alpha * force_norm`. Defaults to None.
    """
    def __init__(self, dataset: Dataset, rare_idx: List[int], replica: int = 1, alpha: Optional[float] = None):
        self.dataset = dataset
        self.rare_idx = set(rare_idx)
        self.replica = replica
        self.alpha = alpha
        
        self.indices: List[int] = []
        self.weights: List[float] = []

        self._build_sampler()

    def _build_sampler(self):
        """Constructs the indices and weights for the sampler."""
        self.indices = []
        self.weights = []
        for i in range(len(self.dataset)):
            num_replicas = self.replica if i in self.rare_idx else 1
            self.indices.extend([i] * num_replicas)
            
            weight = 1.0
            if self.alpha is not None and self.alpha > 0:
                # Assuming dataset[i] returns a Data object with a metadata dict
                force_norm = self.dataset[i].metadata.get("force_norm", 0.0)
                weight += self.alpha * force_norm
            
            self.weights.extend([weight] * num_replicas)

    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over the dataset indices."""
        if self.alpha is not None and self.alpha > 0:
            # Use weighted random sampling if alpha is set
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(self.weights, dtype=torch.double),
                num_samples=len(self.indices),
                replacement=True,
            )
            yield from sampler
        else:
            # Otherwise, just shuffle the (potentially replicated) indices
            indices = self.indices.copy()
            torch.randperm(len(indices), generator=None).tolist()
            yield from (indices[i] for i in torch.randperm(len(indices)))

    def __len__(self) -> int:
        """Returns the total number of samples in the iterator."""
        return len(self.indices) 