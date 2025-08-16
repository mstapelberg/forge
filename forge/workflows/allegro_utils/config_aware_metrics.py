"""
Config-type aware stress metrics / losses for Allegro/NequIP.

This module provides training/validation metrics that apply section-aware
weighting to stress terms, using per-structure configuration labels
(``config_type``) mapped to broad sections (bulk, liquids, surfaces, NEB, ...).

Design choices for minimal integration footprint:
- Accept either ``section`` or raw ``config_type`` as extra input.
- Accept stresses in either eV/Å^3 (default) or GPa via a ``units`` parameter.
- No changes required to the data pipeline beyond passing the extra input name.
"""

from __future__ import annotations
from typing import Sequence, Optional

import torch
import torch.nn.functional as F
import logging

from nequip.data.stats import _MeanX

logger = logging.getLogger(__name__)

from .config_aware_stress import (
    eVa3_to_GPa,
    GPa_to_eVa3,
    voigt6_to_mat,
    decompose_stress,
    von_mises,
    normalize_config_type,
    section_of,
    build_section_masks,
    id_to_section,
)


def _ensure_mat33(x: torch.Tensor) -> torch.Tensor:
    """Ensure a stress tensor has shape ``(N, 3, 3)``.

    Supports inputs shaped ``(N, 6)`` in Voigt ordering or already ``(N, 3, 3)``.

    Args:
        x: Input tensor.

    Returns:
        torch.Tensor: Tensor of shape ``(N, 3, 3)``.

    Raises:
        ValueError: If the input cannot be interpreted as a batch of stresses.
    """
    if x.dim() == 2 and x.shape[-1] == 6:
        return voigt6_to_mat(x)
    if x.shape[-2:] == (3, 3):
        return x
    if x.numel() % 9 == 0 and x.shape[-1] == 9:
        return x.reshape(-1, 3, 3)
    raise ValueError(f"Stress must be (N,3,3) or (N,6); got {tuple(x.shape)}")


class ConfigAwareStressHuber(_MeanX):
    """Huber loss on stress with section-aware selection of terms.

    Section policy (fixed defaults):
    - Bulk crystals & Elastic: pressure AND deviatoric terms
    - Liquids & explore: pressure ONLY
    - Surfaces & γ, NEB, Intermetallics: excluded entirely
    - Point defects: pressure ONLY if ``allow_pressure_for_defects=True``

    Args:
        delta_p: Huber delta for pressure residuals (in GPa domain).
        delta_s: Huber delta for deviatoric residuals (in GPa domain).
        weight_p: Weight for pressure term in combined per-sample loss.
        weight_s: Weight for deviatoric term in combined per-sample loss.
        allow_pressure_for_defects: If True, include pressure-only for defects.
        include_intermetallics_in_full: If True, treat intermetallics like bulk.
            Note: the default section policy excludes intermetallics entirely;
            this flag only has effect if the exclusion policy is changed.
        ignore_nan: If True, mask samples with NaNs in the target stress.
        units: Input stress units for ``pred`` and ``target``. Either 'eVa3'
            (default) or 'GPa'. Computation is performed in GPa.
        loss_return_units: Control the units of the returned per-sample loss.
            One of ``"match_inputs"`` (default), ``"eVa3"``, or ``"GPa"``.
            This allows keeping legacy stress coefficients comparable while
            still computing in GPa internally for numerical stability.
        debug_masks: If True, logs a short summary of routing masks and
            normalization statistics per batch for debugging/ablations.
    """

    def __init__(
        self,
        delta_p: float = 1.0,
        delta_s: float = 2.0,
        weight_p: float = 0.30,
        weight_s: float = 0.70,
        allow_pressure_for_defects: bool = True,
        include_intermetallics_in_full: bool = False,
        ignore_nan: bool = True,
        units: str = "eVa3",
        loss_return_units: str = "match_inputs",  # {"match_inputs","eVa3","GPa"}
        debug_masks: bool = False,
        **kwargs,
    ):
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        self.delta_p = float(delta_p)
        self.delta_s = float(delta_s)
        self.weight_p = float(weight_p)
        self.weight_s = float(weight_s)
        self.allow_pressure_for_defects = bool(allow_pressure_for_defects)
        self.include_intermetallics_in_full = bool(include_intermetallics_in_full)
        self.ignore_nan = bool(ignore_nan)
        self.units = units
        self.loss_return_units = str(loss_return_units).lower()
        self.debug_masks = bool(debug_masks)
        # Differentiable per-batch values for training loss
        self.last_batch_value: Optional[torch.Tensor] = None

    def _maybe_to_gpa(self, S: torch.Tensor) -> torch.Tensor:
        return eVa3_to_GPa(S) if self.units.lower() in ("eva3", "eV/A^3".lower()) else S

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        section: Optional[Sequence[str]] = None,
        config_type: Optional[Sequence[str]] = None,
        section_id: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate section-aware Huber losses for a batch.

        Args:
            pred: Predicted stress, shape ``(N,3,3)`` or ``(N,6)``.
            target: Reference stress, same shape as ``pred``.
            section: Optional list of section labels per sample.
            config_type: Optional list of raw ``config_type`` strings per sample.
                Used only if ``section`` is not provided.
        """
        if pred.numel() == 0:
            # Differentiable zero with correct batch shape
            S_pred = _ensure_mat33(pred if pred.numel() > 0 else target)
            zero = (S_pred - S_pred).sum(dim=(-2, -1))  # (N,)
            self.last_batch_value = zero
            super().update(self.last_batch_value)
            return

        S_pred = _ensure_mat33(pred)
        S_ref = _ensure_mat33(target)

        # Convert units if necessary (compute in GPa domain)
        S_pred = self._maybe_to_gpa(S_pred)
        S_ref = self._maybe_to_gpa(S_ref)

        if self.ignore_nan:
            nan_mask = torch.isnan(S_ref).view(S_ref.shape[0], -1).any(dim=-1)
        else:
            nan_mask = torch.zeros(S_ref.shape[0], dtype=torch.bool, device=S_ref.device)

        p_pred, dev_pred = decompose_stress(S_pred)
        p_ref, dev_ref = decompose_stress(S_ref)

        p_loss = F.huber_loss(p_pred, p_ref, delta=self.delta_p, reduction="none")
        dev_loss_comp = F.huber_loss(dev_pred, dev_ref, delta=self.delta_s, reduction="none")
        dev_loss = dev_loss_comp.mean(dim=(-2, -1))

        # Resolve sections (graceful fallback if none provided)
        # If neither `section` nor `config_type` are provided (e.g., when used via
        # NequIP's MetricsManager that does not pass extra inputs), we fall back to
        # treating all samples as "full_sigma" (i.e., include both pressure and
        # deviatoric terms) so training can proceed without special routing.
        if section is None:
            if section_id is not None:
                # section_id may be shape (N,) LongTensor; map to labels
                section = [id_to_section(int(sid)) for sid in section_id.view(-1).tolist()]
            elif config_type is not None:
                section = [section_of(normalize_config_type(ct)) for ct in config_type]
            else:
                # Fallback: assign all to a section that maps to full-sigma handling
                section = ["Bulk crystals"] * S_ref.shape[0]

        masks = build_section_masks(
            section,
            allow_pressure_for_defects=self.allow_pressure_for_defects,
            include_intermetallics_in_full=self.include_intermetallics_in_full,
        )
        m_full = masks["full_sigma"].to(p_loss.device)
        m_press = masks["pressure"].to(p_loss.device)
        m_excl = masks["exclude"].to(p_loss.device)

        drop = m_excl | nan_mask
        keep = ~drop

        use_p = (m_press & keep).float()
        use_s = (m_full & keep).float()

        combined = self.weight_p * (use_p * p_loss) + self.weight_s * (use_s * dev_loss)
        norm = (self.weight_p * use_p + self.weight_s * use_s).clamp_min(1e-8)
        per_sample_gpa = combined / norm

        # Optional debug: mask counts & norm summary
        if self.debug_masks:
            with torch.no_grad():
                n = int(S_pred.shape[0])
                logger.info(
                    "[ConfigAwareStressHuber] N=%d full=%d press=%d excl=%d nan=%d norm_mean=%.4g",
                    n,
                    int(m_full.sum()),
                    int(m_press.sum()),
                    int(m_excl.sum()),
                    int(nan_mask.sum()),
                    float(norm.mean()),
                )

        # Return domain control: keep GPa internally; convert for the returned scalar if requested
        dest = self.loss_return_units
        if dest == "match_inputs":
            dest = "eva3" if self.units.lower() in ("eva3", "ev/a^3") else "gpa"
        if dest == "eva3":
            per_sample = GPa_to_eVa3(per_sample_gpa)
        elif dest == "gpa":
            per_sample = per_sample_gpa
        else:
            raise ValueError(f"loss_return_units must be one of match_inputs|eVa3|GPa, got {self.loss_return_units}")

        # Cache differentiable per-batch values for the manager to build the loss
        self.last_batch_value = per_sample
        super().update(per_sample)


class PressureMAE(_MeanX):
    """Mean absolute error on hydrostatic pressure (in GPa domain by default)."""

    def __init__(self, ignore_nan: bool = True, units: str = "eVa3", expose_for_loss: bool = False, **kwargs):
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        self.ignore_nan = bool(ignore_nan)
        self.units = units
        self.expose_for_loss = bool(expose_for_loss)
        self.last_batch_value: Optional[torch.Tensor] = None

    def _maybe_to_gpa(self, S: torch.Tensor) -> torch.Tensor:
        return eVa3_to_GPa(S) if self.units.lower() in ("eva3", "eV/A^3".lower()) else S

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        S_pred = _ensure_mat33(pred)
        S_ref = _ensure_mat33(target)
        S_pred = self._maybe_to_gpa(S_pred)
        S_ref = self._maybe_to_gpa(S_ref)
        p_pred, _ = decompose_stress(S_pred)
        p_ref, _ = decompose_stress(S_ref)
        err = torch.abs(p_pred - p_ref)
        self.last_batch_value = err  # expose differentiable values if included in loss
        if self.ignore_nan:
            mask = ~torch.isnan(err)
            if mask.any():
                super().update(err[mask])
            return
        super().update(err)


class VonMisesMAE(_MeanX):
    """Mean absolute error on von Mises equivalent stress (in GPa domain by default)."""

    def __init__(self, ignore_nan: bool = True, units: str = "eVa3", expose_for_loss: bool = False, **kwargs):
        super().__init__(modifier=torch.nn.Identity(), **kwargs)
        self.ignore_nan = bool(ignore_nan)
        self.units = units
        self.expose_for_loss = bool(expose_for_loss)
        self.last_batch_value: Optional[torch.Tensor] = None

    def _maybe_to_gpa(self, S: torch.Tensor) -> torch.Tensor:
        return eVa3_to_GPa(S) if self.units.lower() in ("eva3", "eV/A^3".lower()) else S

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        S_pred = _ensure_mat33(pred)
        S_ref = _ensure_mat33(target)
        S_pred = self._maybe_to_gpa(S_pred)
        S_ref = self._maybe_to_gpa(S_ref)
        vm_pred = von_mises(decompose_stress(S_pred)[1])
        vm_ref = von_mises(decompose_stress(S_ref)[1])
        err = torch.abs(vm_pred - vm_ref)
        self.last_batch_value = err  # expose differentiable values if included in loss
        if self.ignore_nan:
            mask = ~torch.isnan(err)
            if mask.any():
                super().update(err[mask])
            return
        super().update(err)


