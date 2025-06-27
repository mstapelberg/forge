from typing import List, Dict, Union, Callable, Iterable, Optional
import torch
from torchmetrics import Metric
from nequip.data import AtomicDataDict
from nequip.data.modifier import BaseModifier, PerAtomModifier, NumNeighbors
from nequip.data.stats import Mean, RootMeanSquare, StandardDeviation, _MeanX
from nequip.data.stats_manager import DataStatisticsManager
from nequip.utils.logger import RankedLogger


logger = RankedLogger(__name__, rank_zero_only=True)


class Quantile(Metric):
    """Computes the q-th quantile of a stream of data.

    This metric accumulates all data points and computes the exact quantile
    upon `compute()`.

    Args:
        q (float): The quantile to compute (must be between 0.0 and 1.0).
        **kwargs: Additional keyword arguments for the torchmetrics.Metric class.
    """
    full_state_update: bool = False

    def __init__(self, q: float, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile `q` must be between 0 and 1, but got {q}")
        self.q = q
        self.add_state("data", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, data: torch.Tensor) -> None:
        """Append data to the state tensor."""
        if data.numel() > 0:
            # Ensure data is on the same device as the state tensor before concatenating
            self.data = torch.cat([self.data, data.flatten().to(self.data.device)])

    def compute(self) -> torch.Tensor:
        """Compute the quantile of all collected data."""
        if self.data.numel() == 0:
            return torch.tensor(float('nan'))
            
        return torch.quantile(self.data.to(torch.float32), self.q)

    def __str__(self) -> str:
        return f"q_{self.q}"


class ForceMagnitude(BaseModifier):
    """A modifier to compute the magnitude of force vectors."""
    def __init__(self, field: str = AtomicDataDict.FORCE_KEY):
        super().__init__(field=field)

    def forward(self, data: AtomicDataDict.Type) -> torch.Tensor:
        forces = super().forward(data)
        return torch.linalg.norm(forces, dim=-1)

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
    metrics = [
        # Common stats
        {"name": "num_neighbors_mean", "field": NumNeighbors(), "metric": Mean()},
        {"name": "per_atom_energy_mean", "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY), "metric": Mean()},
        {"name": "forces_rms", "field": AtomicDataDict.FORCE_KEY, "metric": RootMeanSquare()},
        {"name": "per_type_forces_rms", "field": AtomicDataDict.FORCE_KEY, "metric": RootMeanSquare(), "per_type": True},
        # Extended stats for loss parameters
        {"name": "force_magnitude_q10", "field": ForceMagnitude(), "metric": Quantile(0.10)},
        {"name": "force_magnitude_q50", "field": ForceMagnitude(), "metric": Quantile(0.50)},
        {"name": "force_magnitude_q90", "field": ForceMagnitude(), "metric": Quantile(0.90)},
        {"name": "force_magnitude_q95", "field": ForceMagnitude(), "metric": Quantile(0.95)},
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

    return base_manager 