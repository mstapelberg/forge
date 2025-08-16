"""
Utilities for config-type-aware stress handling.

This module provides:
- Conversion between Voigt and 3x3 tensors
- eV/Å^3 ↔ GPa conversions
- Decomposition into pressure (hydrostatic) + deviatoric components
- Von Mises equivalent stress
- Section mapping derived from your `config_type` field

All functions use PyTorch tensors for seamless GPU support.
"""

from __future__ import annotations
import re
from typing import Dict, Iterable, List, Tuple

import torch

GPA_PER_EVA3 = 160.21766208


def voigt6_to_mat(v6: torch.Tensor) -> torch.Tensor:
    """Convert Voigt-6 stress to 3x3 tensor.

    Args:
        v6: Tensor shaped ``(N, 6)`` or broadcastable to it, ordered as
            ``(xx, yy, zz, yz, xz, xy)``.

    Returns:
        torch.Tensor: Tensor of shape ``(N, 3, 3)`` (symmetric).
    """
    v6 = v6.reshape(-1, 6)
    xx, yy, zz, yz, xz, xy = [v6[:, i] for i in range(6)]
    S = torch.stack([
        torch.stack([xx, xy, xz], dim=-1),
        torch.stack([xy, yy, yz], dim=-1),
        torch.stack([xz, yz, zz], dim=-1)
    ], dim=-2)  # (N,3,3)
    return S


def mat_to_voigt6(S: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 stress tensor to Voigt-6 ordering.

    Args:
        S: Tensor with trailing shape ``(3, 3)``.

    Returns:
        torch.Tensor: Tensor with trailing shape ``(6,)`` in ordering
        ``(xx, yy, zz, yz, xz, xy)``.
    """
    return torch.stack([S[..., 0, 0], S[..., 1, 1], S[..., 2, 2],
                        S[..., 1, 2], S[..., 0, 2], S[..., 0, 1]], dim=-1)


def eVa3_to_GPa(S: torch.Tensor) -> torch.Tensor:
    """Convert stress from eV/Å^3 to GPa.

    Args:
        S: Stress tensor in eV/Å^3.

    Returns:
        torch.Tensor: Stress in GPa.
    """
    return S * GPA_PER_EVA3


def GPa_to_eVa3(S: torch.Tensor) -> torch.Tensor:
    """Convert stress from GPa to eV/Å^3.

    Args:
        S: Stress tensor in GPa.

    Returns:
        torch.Tensor: Stress in eV/Å^3.
    """
    return S / GPA_PER_EVA3


def decompose_stress(S_GPa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose stress into pressure and deviatoric part (both in GPa).

    Args:
        S_GPa: Stress tensor shaped ``(..., 3, 3)`` in GPa.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(pressure, deviatoric_tensor)`` where
        ``pressure`` is shape ``(...)`` (positive in compression) and ``deviatoric_tensor`` is shape ``(..., 3, 3)``.
    """
    I = torch.eye(3, device=S_GPa.device, dtype=S_GPa.dtype)
    tr = S_GPa[..., 0, 0] + S_GPa[..., 1, 1] + S_GPa[..., 2, 2]
    p = -tr / 3.0
    dev = S_GPa + p[..., None, None] * I
    return p, dev


def von_mises(dev_GPa: torch.Tensor) -> torch.Tensor:
    """Compute Von Mises stress from deviatoric tensor (GPa).

    Args:
        dev_GPa: Deviatoric stress tensor shaped ``(..., 3, 3)`` in GPa.

    Returns:
        torch.Tensor: Von Mises equivalent stress (GPa), shape ``(...)``.
    """
    return torch.sqrt(1.5 * (dev_GPa * dev_GPa).sum(dim=(-2, -1)))


def normalize_config_type(raw: str) -> str:
    """Normalize raw ``config_type`` strings to canonical tokens.

    Args:
        raw: Raw config type string (e.g., from ``atoms.info['config_type']``).

    Returns:
        str: Normalized token.
    """
    if raw is None:
        return "unknown"
    ct = str(raw).strip().replace(" ", "_")
    ct = re.sub(r"_aa(-mc)?$", "", ct)
    ct = ct.replace("vac_", "vacancy_")
    if ct in {"vac", "vacancy-alloy"}:
        ct = "vacancy"
    return ct


def section_of(ct: str) -> str:
    """Map a canonical config type to a broad section label.

    Args:
        ct: Canonical config type string.

    Returns:
        str: Section label among known categories.
    """
    if ct.startswith("surface_") or ct.startswith("gamma_surface"):
        return "Surfaces & γ"
    if (ct.startswith("vac") or ct in {"di-vacancy", "tri-vacancy", "sia", "di-sia"}):
        return "Point defects"
    if ct in {"A15", "C15"}:
        return "Intermetallics"
    if ct in {"bcc_distorted", "fcc", "hcp", "dia"}:
        return "Bulk crystals"
    if (ct.startswith("liquid") or ct.startswith("comp-explore") or ct.startswith("surf_liquid")):
        return "Liquids & explore"
    if ct.startswith("phonon"):
        return "Phonon"
    if ct.startswith("elastic"):
        return "Elastic"
    if ct.startswith("neb"):
        return "NEB"
    return "Other"


SECTIONS_FULL_SIGMA = {"Bulk crystals", "Elastic"}
SECTIONS_PRESSURE_ONLY = {"Liquids & explore"}
# Exclude Surfaces/γ, NEB, and Intermetallics entirely by default
SECTIONS_EXCLUDE = {"Surfaces & γ", "NEB", "Intermetallics"}


# Stable section id mapping to allow integer labels in data batches
SECTION_LABEL_TO_ID: Dict[str, int] = {
    "Bulk crystals": 0,
    "Elastic": 1,
    "Liquids & explore": 2,
    "Surfaces & γ": 3,
    "NEB": 4,
    "Intermetallics": 5,
    "Point defects": 6,
    "Phonon": 7,
    "Other": 8,
}
ID_TO_SECTION_LABEL: Dict[int, str] = {v: k for k, v in SECTION_LABEL_TO_ID.items()}


def section_to_id(section_label: str) -> int:
    """Convert a section label to its stable integer id.

    Args:
        section_label: One of the known section labels.

    Returns:
        int: Integer id for the section. Unknown sections map to "Other".
    """
    return SECTION_LABEL_TO_ID.get(section_label, SECTION_LABEL_TO_ID["Other"])


def id_to_section(section_id: int) -> str:
    """Convert a section id to its label.

    Args:
        section_id: Integer id.

    Returns:
        str: Section label.
    """
    return ID_TO_SECTION_LABEL.get(int(section_id), "Other")


def build_section_masks(
    sections: Iterable[str],
    allow_pressure_for_defects: bool = False,
    include_intermetallics_in_full: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build boolean masks for section-based stress treatment.

    Args:
        sections: Iterable of section labels per sample.
        allow_pressure_for_defects: If True, apply pressure-only term to "Point defects".
        include_intermetallics_in_full: If True, include "Intermetallics" in full-sigma set.

    Returns:
        Dict[str, torch.Tensor]: Keys ``full_sigma``, ``pressure``, ``exclude`` mapping to ``(N,)`` bool tensors.
    """
    sec_list = list(sections)
    full = set(SECTIONS_FULL_SIGMA)
    if include_intermetallics_in_full:
        full |= {"Intermetallics"}

    mask_full = torch.tensor([(s in full) for s in sec_list])
    mask_press = torch.tensor([
        (s in SECTIONS_PRESSURE_ONLY)
        or (allow_pressure_for_defects and s == "Point defects")
        or (s in full)
        for s in sec_list
    ])
    mask_excl = torch.tensor([(s in SECTIONS_EXCLUDE) for s in sec_list])
    return {"full_sigma": mask_full, "pressure": mask_press, "exclude": mask_excl}


