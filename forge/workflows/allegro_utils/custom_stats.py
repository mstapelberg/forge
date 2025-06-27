from typing import List, Dict, Union, Callable, Iterable, Optional
import os
import torch
from torchmetrics import Metric
from nequip.data import AtomicDataDict
from nequip.data.modifier import BaseModifier, PerAtomModifier, NumNeighbors
from nequip.data.stats import Mean, RootMeanSquare, StandardDeviation, _MeanX
from nequip.data.stats_manager import DataStatisticsManager
from nequip.train.metrics import StratifiedHuberForceLoss
from nequip.utils.logger import RankedLogger
from tqdm.auto import tqdm
import sys
import random

random.seed(42)
torch.manual_seed(42)

logger = RankedLogger(__name__, rank_zero_only=True)


class Quantile(Metric):
    """Computes the q-th quantile of a stream of data.

    This metric accumulates all data points and computes the exact quantile
    upon `compute()`.

    To avoid performance degradation from concatenating a large number of
    tensors, this implementation uses a buffering strategy. Tensors are
    first collected in a temporary list (`_buffer`) and then concatenated
    into a larger tensor once the buffer size exceeds a threshold. This
    keeps the main `data` list short.

    Args:
        q (float): The quantile to compute (must be between 0.0 and 1.0).
        buffer_size (int): The number of tensors to buffer before concatenating.
        **kwargs: Additional keyword arguments for the torchmetrics.Metric class.
    """
    full_state_update: bool = False

    def __init__(self, q: float, buffer_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile `q` must be between 0 and 1, but got {q}")
        self.q = q
        self.buffer_size = buffer_size
        # "sum" for lists is concatenation. This is the correct way for this state.
        self.add_state("data", default=[], dist_reduce_fx="sum")
        self._buffer: List[torch.Tensor] = []

    def update(self, data: torch.Tensor) -> None:
        """Append data to the internal buffer and consolidate if full."""
        if data.numel() > 0:
            self._buffer.append(data.flatten().cpu())
            if len(self._buffer) >= self.buffer_size:
                self._consolidate_buffer()

    def _consolidate_buffer(self):
        if not self._buffer:
            return
        # Move buffered tensors to the main data state
        self.data.append(torch.cat(self._buffer))
        self._buffer.clear()

    def compute(self) -> torch.Tensor:
        """Compute the quantile of all collected data."""
        # Ensure any remaining tensors in the buffer are included
        self._consolidate_buffer()
        
        if not self.data:
            return torch.tensor(float('nan'))

        data_cat = torch.cat(self.data)

        if data_cat.numel() == 0:
            return torch.tensor(float('nan'))

        return torch.quantile(data_cat.to(torch.float32), self.q)

    def __str__(self) -> str:
        return f"q_{self.q}"


class ForceMagnitude(BaseModifier):
    """A modifier to compute the magnitude of force vectors."""
    def __init__(self, field: str = AtomicDataDict.FORCE_KEY):
        super().__init__(field=field)

    def forward(self, forces: torch.Tensor) -> torch.Tensor:
        """Operates on the `forces` tensor directly.
        
        The `forces` argument is `data[self.field]`, which is automatically
        extracted by the `BaseModifier.__call__` method since `self.field`
        is set in `__init__`.
        """
        # --- Final Debugging Step ---
        if torch.any(torch.isnan(forces)):
            print(f"!!! DEBUG ForceMagnitude: NaN values found in input `forces` tensor!", file=sys.stderr)
        if torch.any(torch.isinf(forces)):
            print(f"!!! DEBUG ForceMagnitude: Inf values found in input `forces` tensor!", file=sys.stderr)
        # --- End Final Debugging Step ---
        
        magnitudes = torch.linalg.norm(forces, dim=-1)
        if torch.any(magnitudes < 0):
            print(f"!!! DEBUG ForceMagnitude: Negative values found in input `forces` tensor!", file=sys.stderr)
        # Replace any potential NaNs from zero-vectors with 0.0
        magnitudes = torch.nan_to_num(magnitudes, nan=0.0)
        return magnitudes

    def _func(self, data: AtomicDataDict.Type) -> torch.Tensor:
        return self.forward(data[self.field])

    def __str__(self) -> str:
        return "force_magnitude"


def ExtendedDataStatisticsManager(
    dataloader_kwargs: Dict = {},
    type_names: List[str] = None,
):
    """A DataStatisticsManager that includes extended statistics for custom losses.
    
    This manager computes:
    - Standard statistics from CommonDataStatisticsManager.
    - Quantiles of force magnitudes (q10, q50, q90, q95) for Huber/Focal loss params.
    - Derived values for `focal_beta` and `stratified_huber_deltas`.
    """
    # --- Performance optimizations for statistics calculation ---
    if 'num_workers' not in dataloader_kwargs:
        try:
            # Default to half the available CPU cores, capped at 8.
            num_cpus = os.cpu_count()
            num_workers = min(num_cpus // 2 if num_cpus else 0, 8)
            if num_workers > 0:
                dataloader_kwargs['num_workers'] = num_workers
                logger.info(f"Automatically setting num_workers for statistics to {num_workers} for faster data loading.")
        except NotImplementedError:
            logger.warning("Could not determine the number of CPUs. Statistics calculation might be slow.")

    if 'pin_memory' not in dataloader_kwargs and torch.cuda.is_available():
        dataloader_kwargs['pin_memory'] = True
        logger.info("Setting pin_memory=True for statistics calculation to speed up CPU-GPU data transfer.")
    # --- End of performance optimizations ---
    
    metrics = [
        # Common stats
        {"name": "num_neighbors_mean", "field": NumNeighbors(), "metric": Mean()},
        {"name": "per_atom_energy_mean", "field": PerAtomModifier(field=AtomicDataDict.TOTAL_ENERGY_KEY), "metric": Mean()},
        {"name": "forces_rms", "field": AtomicDataDict.FORCE_KEY, "metric": RootMeanSquare()},
        {"name": "per_type_forces_rms", "field": AtomicDataDict.FORCE_KEY, "metric": RootMeanSquare(), "per_type": True},
        # Extended stats for loss parameters
        {"name": "force_magnitude_q10", "field": ForceMagnitude(), "metric": Quantile(0.10, buffer_size=1000)},
        {"name": "force_magnitude_q50", "field": ForceMagnitude(), "metric": Quantile(0.50, buffer_size=1000)},
        {"name": "force_magnitude_q90", "field": ForceMagnitude(), "metric": Quantile(0.90, buffer_size=1000)},
        {"name": "force_magnitude_q95", "field": ForceMagnitude(), "metric": Quantile(0.95, buffer_size=1000)},
    ]
    
    base_manager = DataStatisticsManager(metrics, dataloader_kwargs, type_names)

    # Monkey-patch the compute method to add derived statistics
    original_compute = base_manager.compute
    
    def extended_compute(self):
        # Call original compute to get base stats
        stats = original_compute()
        logger.info("Computing extended statistics for loss functions...")

        # --- Compute FocalMSELoss beta ---
        if 'forces_rms' in stats:
            forces_rms = stats['forces_rms']
            if forces_rms > 1e-6:
                focal_beta = 1.0 / forces_rms
                stats['focal_beta'] = focal_beta
                logger.info(f"focal_beta: {focal_beta:.4f} (from 1/forces_rms)")
            else:
                logger.warning("forces_rms is very small, skipping focal_beta calculation.")

        # --- Compute Huber delta (from q95) ---
        if 'force_magnitude_q95' in stats:
            huber_delta = stats['force_magnitude_q95']
            stats['huber_delta'] = huber_delta
            logger.info(f"huber_delta: {huber_delta:.4f} (from force_magnitude_q95)")

        # --- Compute Stratified Huber deltas ---
        q10 = stats.get('force_magnitude_q10')
        q50 = stats.get('force_magnitude_q50')
        q90 = stats.get('force_magnitude_q90')
        
        if all(k is not None for k in [q10, q50, q90]):
            # Boundaries for stratified loss
            boundaries = [q10, q50, q90]
            stats['stratified_huber_boundaries'] = boundaries
            logger.info(f"stratified_huber_boundaries: {[f'{b:.4f}' for b in boundaries]}")
            
            # Deltas for each bin
            # For the first bin (0, p10), the lower bound is 0, which would give a
            # delta of 0. This is ill-defined for Huber loss. We'll use 0.1 * p10
            # for the first bin as a sensible default.
            delta1 = 0.1 * q10
            delta2 = 0.1 * q10
            delta3 = 0.1 * q50
            delta4 = 0.1 * q90
            deltas = [delta1, delta2, delta3, delta4]
            stats['stratified_huber_deltas'] = deltas
            logger.info(f"stratified_huber_deltas: {[f'{d:.4f}' for d in deltas]}")
        else:
            logger.warning("Could not compute stratified huber deltas because one of the required quantiles (q10, q50, q90) is missing.")

        return stats

    base_manager.compute = extended_compute.__get__(base_manager, DataStatisticsManager)

    # Monkey-patch the get_statistics method to add tqdm progress bar
    def extended_get_statistics(self, data_source: Iterable[AtomicDataDict.Type]):
        """A get_statistics method with a tqdm progress bar."""
        if isinstance(data_source, dict):
            raise TypeError(
                f"The data source passed to get_statistics was a dictionary, not a DataLoader or other iterable. "
                f"It seems you passed dataloader keyword arguments instead of an instantiated dataloader. "
                f"Received dict with keys: {list(data_source.keys())}"
            )
            
        pbar = tqdm(
            data_source,
            desc="Calculating statistics",
            total=len(data_source) if hasattr(data_source, '__len__') else None,
            disable=getattr(self, 'rank', 0) != 0 # Show progress bar only on rank 0
        )
        # The caller of get_statistics is responsible for calling .reset()
        for data in pbar:
            self(data) # This calls the forward method
        return self.compute()

    base_manager.get_statistics = extended_get_statistics.__get__(base_manager, DataStatisticsManager)

    return base_manager 