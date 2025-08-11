#!/usr/bin/env python3
"""
Self-contained NEB endpoint debug script.

This script:
1) Builds a small BCC alloy structure (V with small fractions of Ti, Cr, Zr, W).
2) Relaxes the perfect structure using your MLIP.
3) Uses VacancyDiffusion to construct relaxed start/end vacancy endpoints (from the relaxed perfect).
4) Saves a single multi-frame extxyz with three frames:
   - atoms.info['config_type'] = 'perf' | 'start' | 'end'
5) Saves a JSON summary with indices, site positions, natoms, simple diagnostics, and energies.

Assumptions:
- Backend and model path are set near the top.
- Species map is fixed: {'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}.
- Units: energy in eV, lengths in Å.

Adjust MODEL_PATH and OUTPUT_DIR before running.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import json
import logging
import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.io import write

from forge.calculators.factory import create_ensemble_calculator
from forge.workflows.relax import relax
from forge.workflows.neb import VacancyDiffusion

# =========================
# User-configurable section
# =========================
MODEL_PATH: Path = Path("/Users/myless/Packages/forge/scratch/data/potentials/allegro/fast_rmax5.50_lmax1_layers1_mlp256_seed42.nequip.zip")  # set me
BACKEND: Optional[str] = "allegro"  # "allegro", "mace", or None for auto
OUTPUT_DIR: Path = Path("../../data/neb_debug_outputs").resolve()
OUTPUT_XYZ: Path = OUTPUT_DIR / "neb_endpoints_debug.xyz"
OUTPUT_JSON: Path = OUTPUT_DIR / "neb_endpoints_debug.json"
SEED: int = 42

# Species map as requested
SPECIES_TO_TYPE_NAME: Dict[str, int] = {"Ti": 0, "V": 1, "Cr": 2, "Zr": 3, "W": 4}

# Relax params
RELAX_FMAX: float = 0.01
RELAX_STEPS: int = 500

# NEB endpoint params
NN_CUTOFF: float = 2.8
NNN_CUTOFF: float = 3.2
VERBOSE: int = 1  # 0=silent, 1=basic, 2=more


@dataclass
class EndpointDiagnostics:
    """Container for simple endpoint diagnostics."""
    min_dist_to_vac_site_start: float
    min_dist_to_target_site_end: float


def device_str() -> str:
    """Return device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def attach_calc(atoms: Atoms, calc) -> None:
    """Attach calculator returned by create_ensemble_calculator to Atoms.

    Handles both unified wrappers and direct ASE calculators.

    Args:
        atoms: ASE Atoms to modify.
        calc: Calculator or wrapper from create_ensemble_calculator.
    """
    if hasattr(calc, "calculator"):
        atoms.calc = calc.calculator
    else:
        atoms.calc = calc


def build_initial_structure(a: float = 3.01, reps: Tuple[int, int, int] = (3, 3, 3)) -> Atoms:
    """Build a small multi-component BCC structure.

    Starts with BCC V and substitutes a few atoms to Ti/Cr/Zr/W for heterogeneity.

    Args:
        a: Lattice constant in Å.
        reps: Supercell repetition.

    Returns:
        ASE Atoms with mixed species.
    """
    atoms = bulk("V", "bcc", a=a).repeat(reps)  # 54 atoms for (3,3,3)
    rng = np.random.default_rng(SEED)
    idxs = np.arange(len(atoms))
    rng.shuffle(idxs)

    # Substitute a handful for variety (safe counts for 54 atoms)
    for sym, count in [("Ti", 3), ("Cr", 3), ("Zr", 2), ("W", 2)]:
        take = idxs[:count]
        idxs = idxs[count:]
        for i in take:
            atoms[i].symbol = sym

    return atoms


def relax_perfect_structure(perfect: Atoms, model_paths: List[str], backend: Optional[str]) -> Atoms:
    """Relax the perfect structure with the ensemble calculator.

    Args:
        perfect: Initial perfect structure.
        model_paths: Paths to one or more model files.
        backend: 'allegro', 'mace', or None.

    Returns:
        Relaxed perfect structure.
    """
    calc = create_ensemble_calculator(
        model_paths=model_paths,
        backend=backend,
        device=device_str(),
        species_to_type_name=SPECIES_TO_TYPE_NAME,
    )
    relaxed = relax(
        atoms=perfect,
        calculator=calc,
        relax_cell=True,
        fmax=RELAX_FMAX,
        steps=RELAX_STEPS,
        optimizer="FIRE",
        logfile='-' if VERBOSE > 0 else None,
        verbose=VERBOSE,
    )
    return relaxed


def choose_indices(relaxed: Atoms, vd: VacancyDiffusion) -> Tuple[int, int]:
    """Choose a vacancy site and a nearest neighbor target index.

    Picks the atom closest to the cell center as vacancy_index, then selects
    its nearest neighbor as target_index.

    Args:
        relaxed: Relaxed perfect structure.
        vd: VacancyDiffusion (constructed with relaxed).

    Returns:
        Tuple (vacancy_index, target_index).
    """
    cell = relaxed.get_cell()
    center = np.dot([0.5, 0.5, 0.5], cell)
    dists = np.linalg.norm(relaxed.positions - center[None, :], axis=1)
    vac_idx = int(np.argmin(dists))

    neighbors = vd.get_neighbors(vac_idx)
    if len(neighbors.nn_indices) == 0:
        # fallback: if no nn found with the given cutoff, pick the closest atom
        all_idx = np.arange(len(relaxed))
        all_idx = all_idx[all_idx != vac_idx]
        tgt_idx = int(all_idx[np.argmin(np.linalg.norm(relaxed.positions[all_idx] - relaxed.positions[vac_idx], axis=1))])
    else:
        tgt_idx = int(neighbors.nn_indices[0])

    return vac_idx, tgt_idx


def min_distance_to_point(atoms: Atoms, point: np.ndarray) -> float:
    """Compute min distance from any atom to a reference point."""
    pos = atoms.get_positions()
    return float(np.min(np.linalg.norm(pos - point[None, :], axis=1)))


def build_frames_with_metadata(
    perfect: Atoms,
    start_atoms: Atoms,
    end_atoms: Atoms,
    vacancy_index: int,
    target_index: int,
) -> Tuple[List[Atoms], EndpointDiagnostics]:
    """Create frames with info annotations and basic diagnostics.

    Args:
        perfect: Relaxed perfect structure.
        start_atoms: Relaxed start endpoint (vacancy at vacancy_index).
        end_atoms: Relaxed end endpoint (vacancy at original target site).
        vacancy_index: Index in the perfect structure.
        target_index: Index in the perfect structure.

    Returns:
        (frames, diagnostics) where frames = [perf, start, end].
    """
    vac_pos = perfect.positions[vacancy_index].copy()
    tgt_pos = perfect.positions[target_index].copy()

    perf = perfect.copy()
    perf.info["config_type"] = "perf"
    perf.info["vacancy_index"] = int(vacancy_index)
    perf.info["target_index"] = int(target_index)
    perf.info["vacancy_pos"] = vac_pos.tolist()
    perf.info["target_pos"] = tgt_pos.tolist()
    perf.info["natoms"] = len(perf)

    start = start_atoms.copy()
    start.info["config_type"] = "start"
    start.info["vacancy_index"] = int(vacancy_index)
    start.info["target_index"] = int(target_index)
    start.info["vacancy_pos"] = vac_pos.tolist()
    start.info["target_pos"] = tgt_pos.tolist()
    start.info["natoms"] = len(start)

    end = end_atoms.copy()
    end.info["config_type"] = "end"
    end.info["vacancy_index"] = int(vacancy_index)
    end.info["target_index"] = int(target_index)
    end.info["vacancy_pos"] = vac_pos.tolist()
    end.info["target_pos"] = tgt_pos.tolist()
    end.info["natoms"] = len(end)

    diags = EndpointDiagnostics(
        min_dist_to_vac_site_start=min_distance_to_point(start, vac_pos),
        min_dist_to_target_site_end=min_distance_to_point(end, tgt_pos),
    )

    start.info["diagnostic_min_dist_to_vac_site"] = diags.min_dist_to_vac_site_start
    end.info["diagnostic_min_dist_to_target_site"] = diags.min_dist_to_target_site_end

    return [perf, start, end], diags


def compute_energies(frames: List[Atoms], model_paths: List[str], backend: Optional[str]) -> List[float]:
    """Compute single-point energies for each frame with a fresh calculator."""
    calc = create_ensemble_calculator(
        model_paths=model_paths,
        backend=backend,
        device=device_str(),
        species_to_type_name=SPECIES_TO_TYPE_NAME,
    )
    energies: List[float] = []
    for fr in frames:
        fr = fr.copy()
        attach_calc(fr, calc)
        energies.append(float(fr.get_potential_energy()))
    return energies


def save_outputs(frames: List[Atoms], energies: List[float], json_path: Path, xyz_path: Path) -> None:
    """Save the multi-frame extxyz and a JSON summary."""
    # Put energies into frame info for convenience
    for fr, e in zip(frames, energies):
        fr.info["energy"] = e

    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(xyz_path), frames, format="extxyz")

    summary = {
        "frames": [
            {
                "config_type": fr.info.get("config_type"),
                "natoms": int(fr.info.get("natoms", len(fr))),
                "energy_eV": float(fr.info.get("energy")),
                "vacancy_index": fr.info.get("vacancy_index"),
                "target_index": fr.info.get("target_index"),
                "vacancy_pos": fr.info.get("vacancy_pos"),
                "target_pos": fr.info.get("target_pos"),
                "diagnostic_min_dist_to_vac_site": fr.info.get("diagnostic_min_dist_to_vac_site"),
                "diagnostic_min_dist_to_target_site": fr.info.get("diagnostic_min_dist_to_target_site"),
            }
            for fr in frames
        ]
    }

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Wrote XYZ: %s", xyz_path)
    logging.info("Wrote JSON: %s", json_path)


def main() -> None:
    """Run the self-contained pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Please set MODEL_PATH to a valid file: {MODEL_PATH}")

    # 1) Build initial structure
    base = build_initial_structure()

    # 2) Relax perfect
    relaxed = relax_perfect_structure(base, [str(MODEL_PATH)], BACKEND)

    # 3) Create endpoints from the relaxed perfect
    vd = VacancyDiffusion(
        atoms=relaxed,
        model_path=[str(MODEL_PATH)],
        nn_cutoff=NN_CUTOFF,
        nnn_cutoff=NNN_CUTOFF,
        seed=SEED,
        backend=BACKEND,
        species_to_type_name=SPECIES_TO_TYPE_NAME,
    )

    vac_idx, tgt_idx = choose_indices(relaxed, vd)
    logging.info("Chosen indices: vacancy=%d, target=%d", vac_idx, tgt_idx)

    start_atoms, end_atoms, _meta = vd.create_endpoints(
        vacancy_index=vac_idx,
        target_index=tgt_idx,
        relax_fmax=RELAX_FMAX,
        relax_steps=RELAX_STEPS,
        verbose=VERBOSE,
    )

    # 4) Build frames and diagnostics
    frames, diags = build_frames_with_metadata(relaxed, start_atoms, end_atoms, vac_idx, tgt_idx)
    assert len(frames) == 3, "Expected exactly three frames"
    assert len(frames[1]) == len(frames[0]) - 1, "Start natoms should be perfect-1"
    assert len(frames[2]) == len(frames[0]) - 1, "End natoms should be perfect-1"

    # 5) Energies and save
    energies = compute_energies(frames, [str(MODEL_PATH)], BACKEND)
    save_outputs(frames, energies, OUTPUT_JSON, OUTPUT_XYZ)

    logging.info("Diagnostics: start min-dist to vac site = %.4f Å, end min-dist to target site = %.4f Å",
                 diags.min_dist_to_vac_site_start, diags.min_dist_to_target_site_end)


if __name__ == "__main__":
    main()