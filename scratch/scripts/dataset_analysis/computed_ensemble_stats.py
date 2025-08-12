"""Ensemble stats and bad datapoint identification for XYZ datasets.

This script computes ensemble statistics and training metrics for frames in
EXTXYZ files using an ensemble calculator. It can be invoked as a module or via
CLI. When used from CLI, inputs such as model paths, data paths, device, and
output file paths can be provided at runtime.

Overview:
1) Loads `.xyz` frames from provided `data_paths` (files or directories)
2) Builds a calculator via `forge.calculators.factory.create_ensemble_calculator`
3) Computes ensemble mean and variance for energy, forces, stress
4) Uses training module metrics to score frames and flag likely-bad datapoints
5) Writes a CSV summary and optional JSONL with per-frame details
6) Optionally annotates each frame with dataset split (`train`/`val`/`test`)
   using a provided `structure_splits.json`.

Defaults are provided for convenience but can be overridden via CLI.
"""

from __future__ import annotations

import csv
import json
import logging
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Set

import numpy as np
from ase import Atoms
from ase.io import iread

from forge.calculators.factory import create_ensemble_calculator
from forge.analysis.training.metrics.statistical import (
    ForceErrorStats,
    EnergyErrorStats,
    StressErrorStats,
)
from forge.analysis.training.difficulty.ensemble import calculate_ensemble_variance
from tqdm.auto import tqdm


# ------------------------------- Configuration ------------------------------ #

# Species mapping for Allegro/NequIP: symbol -> type index
species_to_type_name: Dict[str, int] = {
    "V": 1,
    "Cr": 2,
    "Ti": 0,
    "W": 4,
    "Zr": 3,
}

# Hardcoded model and data paths (already provided/adjust as needed)
import glob as _glob

DEFAULT_MODEL_GLOB = (
    "/home/myless/Packages/forge/scratch/data/potentials/allegro/gen-8-aa/*.nequip.zip"
)

model_paths: List[str] = sorted(
    _glob.glob(
        DEFAULT_MODEL_GLOB
    )
)

DEFAULT_DATA_GLOB = (
    "/home/myless/Packages/forge/scratch/data/allegro_experiments/allegro-exploit-gen-10/"
    "exploit_rmax5.50_lmax1_layers2_mlp256_seed42/data/*.xyz"
)

data_paths: List[str] = sorted(_glob.glob(DEFAULT_DATA_GLOB))

device: str = "cuda"

out_csv: Path = Path("./generation_10_data_stats.csv").resolve()
out_jsonl: Optional[Path] = Path("./generation_10_data_stats.jsonl").resolve()

# Optional: path to structure_splits.json to annotate rows with split labels
splits_json: Optional[Path] = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("computed_ensemble_stats")


# ----------------------------- Data Structures ------------------------------ #


@dataclass
class FrameSummary:
    """Per-frame summary suitable for CSV output.

    Args:
        source_path: Source `.xyz` file path.
        frame_index: Index within file.
        num_atoms: Number of atoms in the frame.
        ref_energy: Reference energy if available.
        mean_energy: Ensemble mean energy.
        var_energy: Ensemble variance of total energy.
        force_rmse_metric: Force RMSE from training metrics (vs REF_force).
        force_mae_metric: Force MAE from training metrics.
        force_max_metric: Force max error.
        energy_rmse_per_atom_metric: Energy RMSE per atom (if REF_energy present).
        energy_mae_per_atom_metric: Energy MAE per atom (if REF_energy present).
        stress_frobenius_error_metric: Frobenius norm of stress error (if REF_stress present).
        force_variance_metric: Ensemble force variance metric (training difficulty).
        energy_variance_per_atom_metric: Ensemble energy var per atom.
        stress_variance_metric: Ensemble stress variance metric.
        split: Dataset split label (train/val/test/unknown) if available.
        is_bad: Flag set after dataset-level thresholding.
        bad_reasons: Comma-separated reasons exceeding thresholds.
    """

    source_path: str
    structure_id: str
    frame_index: int
    num_atoms: int

    ref_energy: Optional[float]
    mean_energy: Optional[float]
    var_energy: Optional[float]

    force_rmse_metric: Optional[float]
    force_mae_metric: Optional[float]
    force_max_metric: Optional[float]

    energy_rmse_per_atom_metric: Optional[float]
    energy_mae_per_atom_metric: Optional[float]

    stress_frobenius_error_metric: Optional[float]

    force_variance_metric: Optional[float]
    energy_variance_per_atom_metric: Optional[float]
    stress_variance_metric: Optional[float]

    split: str = "unknown"
    is_bad: bool = False
    bad_reasons: str = ""


@dataclass
class FrameDetails:
    """Per-frame detailed arrays for JSONL output."""

    source_path: str
    structure_id: str
    frame_index: int
    num_atoms: int

    ref_energy: Optional[float]
    ref_stress: Optional[List[float]]
    ref_force: Optional[List[List[float]]]

    mean_energy: Optional[float]
    var_energy: Optional[float]

    mean_force: Optional[List[List[float]]]
    var_force: Optional[List[List[float]]]

    mean_stress: Optional[List[float]]
    var_stress: Optional[List[float]]


# --------------------------------- Helpers --------------------------------- #


def gather_xyz_files(paths: Sequence[str]) -> List[Path]:
    """Collect `.xyz` files from provided paths.

    Args:
        paths: Files and/or directories.

    Returns:
        List of absolute paths to `.xyz` files.

    Raises:
        FileNotFoundError: If none found.
    """
    xyz_files: List[Path] = []
    for p in paths:
        pth = Path(p).expanduser().resolve()
        if pth.is_file() and pth.suffix.lower() == ".xyz":
            xyz_files.append(pth)
        elif pth.is_dir():
            xyz_files.extend(sorted(pth.rglob("*.xyz")))
        else:
            logger.warning("Not an .xyz file or directory: %s", pth)

    if not xyz_files:
        raise FileNotFoundError("No .xyz files found")
    return xyz_files


def iter_frames(xyz_path: Path) -> Iterator[Tuple[int, Atoms]]:
    """Yield `(frame_index, Atoms)` for an `.xyz` file."""
    for idx, atoms in enumerate(iread(str(xyz_path), format="extxyz", index=":")):
        yield idx, atoms


def extract_refs(atoms: Atoms) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract REF labels from `Atoms`.

    Returns:
        (ref_energy, ref_stress_voigt6, ref_force)
    """
    e = None
    s = None
    f = None
    try:
        if "REF_energy" in atoms.info:
            e = float(atoms.info["REF_energy"])  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        if "REF_stress" in atoms.info:
            s = np.asarray(atoms.info["REF_stress"], dtype=float).reshape(-1)
            if s.shape[0] != 6:
                s = None
    except Exception:
        pass
    try:
        if "REF_force" in atoms.arrays:
            f = np.asarray(atoms.arrays["REF_force"], dtype=float)
            if f.ndim != 2 or f.shape[1] != 3:
                f = None
    except Exception:
        pass
    return e, s, f


def ensemble_evaluate(atoms: Atoms, calculator):
    """Evaluate ensemble predictions for a frame.

    Returns:
        dict with keys energies_all, forces_all, stresses_all if available.
    """
    results = {}
    # Temporarily detach to avoid from_ase interference in Allegro backend
    original_calc = atoms.calc
    atoms.calc = None
    try:
        if hasattr(calculator, "energies_all"):
            results["energies_all"] = calculator.energies_all(atoms)
        if hasattr(calculator, "forces_all"):
            results["forces_all"] = calculator.forces_all(atoms)
        if hasattr(calculator, "stresses_all"):
            results["stresses_all"] = calculator.stresses_all(atoms)
    finally:
        atoms.calc = original_calc
    return results


def voigt6_to_mat3x3(voigt: np.ndarray) -> Optional[np.ndarray]:
    """Convert Voigt-6 stress to 3x3 matrix (ASE Voigt order: xx, yy, zz, yz, zx, xy)."""
    if voigt is None or voigt.size != 6:
        return None
    # ASE uses [xx, yy, zz, yz, xz, xy] per docs; map accordingly
    xx, yy, zz, yz, xz, xy = voigt.tolist()
    return np.array(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float
    )


def compute_frame_metrics(
    atoms: Atoms,
    ens: Dict[str, np.ndarray],
    ref_energy: Optional[float],
    ref_stress_v6: Optional[np.ndarray],
    ref_force: Optional[np.ndarray],
) -> Tuple[FrameSummary, FrameDetails]:
    """Compute ensemble stats and training metrics for a single frame.

    Args:
        atoms: Frame atoms.
        ens: Dict with ensemble arrays.
        ref_energy: Reference energy.
        ref_stress_v6: Reference stress in Voigt-6.
        ref_force: Reference forces.

    Returns:
        (FrameSummary, FrameDetails)
    """
    n_atoms = len(atoms)

    e_all = ens.get("energies_all")
    f_all = ens.get("forces_all")
    s_all = ens.get("stresses_all")

    # Means and variances
    mean_energy = float(np.mean(e_all)) if e_all is not None else None
    var_energy = float(np.var(e_all)) if e_all is not None else None

    mean_force = np.mean(f_all, axis=0) if f_all is not None else None
    var_force = np.var(f_all, axis=0) if f_all is not None else None

    mean_stress = np.mean(s_all, axis=0) if s_all is not None else None
    var_stress = np.var(s_all, axis=0) if s_all is not None else None

    # Training metrics
    force_stats = None
    energy_stats = None
    stress_stats = None

    if mean_force is not None and ref_force is not None:
        force_stats = ForceErrorStats().calculate(mean_force, ref_force)

    if mean_energy is not None and ref_energy is not None:
        energy_stats = EnergyErrorStats().calculate(np.array([mean_energy]), np.array([ref_energy]), n_atoms=n_atoms)

    if mean_stress is not None and ref_stress_v6 is not None:
        pred_s_mat = voigt6_to_mat3x3(np.asarray(mean_stress))
        ref_s_mat = voigt6_to_mat3x3(np.asarray(ref_stress_v6))
        if pred_s_mat is not None and ref_s_mat is not None:
            stress_stats = StressErrorStats().calculate(pred_s_mat, ref_s_mat)

    # Ensemble variance metrics via training.difficulty
    ensemble_predictions = []
    if e_all is not None or f_all is not None or s_all is not None:
        n_models = 0
        if e_all is not None:
            n_models = max(n_models, int(e_all.shape[0]))
        if f_all is not None:
            n_models = max(n_models, int(f_all.shape[0]))
        if s_all is not None:
            n_models = max(n_models, int(s_all.shape[0]))
        for i in range(n_models):
            pred: Dict[str, np.ndarray] = {}
            if e_all is not None:
                pred["energy"] = float(e_all[i])
            if f_all is not None:
                pred["forces"] = f_all[i]
            if s_all is not None:
                pred["stress"] = np.asarray(s_all[i])
            ensemble_predictions.append(pred)

    var_metrics = calculate_ensemble_variance(ensemble_predictions, n_atoms=n_atoms)

    summary = FrameSummary(
        source_path=str(atoms.info.get("source_path", "")),
        structure_id=str(atoms.info.get("structure_id", "")),
        frame_index=int(atoms.info.get("frame_index", -1)),
        num_atoms=n_atoms,
        ref_energy=float(ref_energy) if ref_energy is not None else None,
        mean_energy=mean_energy,
        var_energy=var_energy,
        force_rmse_metric=force_stats.get("force_rmse_metric") if force_stats else None,
        force_mae_metric=force_stats.get("force_mae_metric") if force_stats else None,
        force_max_metric=force_stats.get("force_max_metric") if force_stats else None,
        energy_rmse_per_atom_metric=energy_stats.get("energy_rmse_per_atom_metric") if energy_stats else None,
        energy_mae_per_atom_metric=energy_stats.get("energy_mae_per_atom_metric") if energy_stats else None,
        stress_frobenius_error_metric=stress_stats.get("stress_frobenius_error_metric") if stress_stats else None,
        force_variance_metric=var_metrics.get("force_variance_metric"),
        energy_variance_per_atom_metric=var_metrics.get("energy_variance_per_atom_metric"),
        stress_variance_metric=var_metrics.get("stress_variance_metric"),
    )

    details = FrameDetails(
        source_path=str(atoms.info.get("source_path", "")),
        structure_id=str(atoms.info.get("structure_id", "")),
        frame_index=int(atoms.info.get("frame_index", -1)),
        num_atoms=n_atoms,
        ref_energy=float(ref_energy) if ref_energy is not None else None,
        ref_stress=ref_stress_v6.tolist() if ref_stress_v6 is not None else None,
        ref_force=ref_force.tolist() if ref_force is not None else None,
        mean_energy=mean_energy,
        var_energy=var_energy,
        mean_force=mean_force.tolist() if mean_force is not None else None,
        var_force=var_force.tolist() if var_force is not None else None,
        mean_stress=mean_stress.tolist() if mean_stress is not None else None,
        var_stress=var_stress.tolist() if var_stress is not None else None,
    )

    return summary, details


def flag_bad_datapoints(rows: List[FrameSummary]) -> None:
    """Flag rows exceeding dataset-level thresholds.

    Thresholds are 95th percentile for:
    - force_rmse_metric
    - force_variance_metric
    - energy_rmse_per_atom_metric (if present)
    - stress_frobenius_error_metric (if present)
    """
    # Collect series with None filtered out
    def values(attr: str) -> np.ndarray:
        vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        return np.array(vals, dtype=float) if vals else np.array([])

    thresholds: Dict[str, float] = {}
    for key in (
        "force_rmse_metric",
        "force_variance_metric",
        "energy_rmse_per_atom_metric",
        "stress_frobenius_error_metric",
    ):
        arr = values(key)
        thresholds[key] = float(np.quantile(arr, 0.95)) if arr.size > 0 else float("inf")

    # Flagging
    for r in rows:
        reasons = []
        if r.force_rmse_metric is not None and r.force_rmse_metric >= thresholds["force_rmse_metric"]:
            reasons.append("high_force_rmse")
        if r.force_variance_metric is not None and r.force_variance_metric >= thresholds["force_variance_metric"]:
            reasons.append("high_ensemble_force_var")
        if (
            r.energy_rmse_per_atom_metric is not None
            and r.energy_rmse_per_atom_metric >= thresholds["energy_rmse_per_atom_metric"]
        ):
            reasons.append("high_energy_rmse_per_atom")
        if (
            r.stress_frobenius_error_metric is not None
            and r.stress_frobenius_error_metric >= thresholds["stress_frobenius_error_metric"]
        ):
            reasons.append("high_stress_error")

        r.is_bad = len(reasons) > 0
        r.bad_reasons = ",".join(reasons)


# --------------------------------- Runner ---------------------------------- #


def _load_structure_splits(path: Optional[Path]) -> Dict[str, Set[str]]:
    """Load mapping of structure IDs to dataset split labels.

    Args:
        path: Path to `structure_splits.json`.

    Returns:
        Mapping from split label to set of structure IDs. Returns empty mapping
        if `path` is None or file does not exist.

    Examples:
        >>> mapping = _load_structure_splits(Path('structure_splits.json'))
        >>> isinstance(mapping, dict)
        True
    """
    if path is None:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        result: Dict[str, Set[str]] = {}
        for split_name, id_list in data.items():
            result[split_name] = set(str(x) for x in id_list)
        return result
    except FileNotFoundError:
        logger.warning("structure_splits.json not found at %s", path)
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read structure_splits.json at %s: %s", path, exc)
        return {}


def _infer_split_label(structure_id: str, splits: Dict[str, Set[str]]) -> str:
    """Infer dataset split label for a given structure ID.

    Args:
        structure_id: Structure identifier from `Atoms.info['structure_id']`.
        splits: Mapping from split name to set of IDs.

    Returns:
        Split label ("train", "val", "test") if found, otherwise "unknown".
    """
    if not structure_id or not splits:
        return "unknown"
    for split_name, ids in splits.items():
        if structure_id in ids:
            return split_name
    return "unknown"


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    """Run analysis over configured datasets and write outputs.

    Args:
        cli_args: Optional sequence of CLI arguments for programmatic invocation.
            If None, arguments are taken from `sys.argv`.
    """
    parser = argparse.ArgumentParser(description="Compute ensemble stats and metrics over .xyz datasets")
    parser.add_argument(
        "--model-glob",
        type=str,
        default=DEFAULT_MODEL_GLOB,
        help="Glob for model files (e.g., *.nequip.zip)",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="*",
        default=None,
        help="Explicit list of model files (overrides --model-glob if provided)",
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        default=None,
        help=".xyz files or directories to scan recursively for .xyz",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=device,
        choices=["cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=out_csv,
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=out_jsonl,
        help="Output JSONL path (omit to disable)",
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=None,
        help="Path to structure_splits.json to annotate dataset split labels",
    )

    args = parser.parse_args(list(cli_args) if cli_args is not None else None)

    # Resolve model paths
    models: List[str]
    if args.model_paths:
        models = [str(Path(p).expanduser().resolve()) for p in args.model_paths]
    else:
        models = sorted(_glob.glob(args.model_glob))

    logger.info("Found %d model(s)", len(models))
    if not models:
        raise FileNotFoundError("No model files found for analysis")

    # Resolve data files
    if args.data_paths is not None:
        paths_in: List[str] = []
        for p in args.data_paths:
            paths_in.append(str(Path(p).expanduser().resolve()))
        xyz_files = gather_xyz_files(paths_in)
    else:
        xyz_files = gather_xyz_files(data_paths)
    logger.info("Found %d .xyz file(s)", len(xyz_files))

    # Load structure splits if provided
    splits_map = _load_structure_splits(args.splits_json)

    # Build calculator (auto-detect backend if desired; here we use Allegro)
    calculator = create_ensemble_calculator(
        model_paths=models,
        backend="allegro",
        device=args.device,
        species_to_type_name=species_to_type_name,
    )

    summaries: List[FrameSummary] = []
    jsonl_file = None
    jsonl_path: Optional[Path] = args.out_jsonl
    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path).expanduser().resolve()
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")

    try:
        for xyz_path in tqdm(xyz_files, desc="XYZ files", unit="file"):
            frames_pbar = tqdm(desc=f"{xyz_path.name}", unit="frame", leave=False)
            for idx, atoms in iter_frames(xyz_path):
                # Annotate for provenance in outputs
                atoms.info["source_path"] = str(xyz_path)
                atoms.info["frame_index"] = idx

                ref_e, ref_s_v6, ref_f = extract_refs(atoms)
                # Use unified predict method if available for speed
                if hasattr(calculator, 'predict_all'):
                    preds = calculator.predict_all(atoms)
                    # Map to expected keys for downstream code
                    ens = {}
                    if 'energies' in preds:
                        ens['energies_all'] = preds['energies']
                    if 'forces' in preds:
                        ens['forces_all'] = preds['forces']
                    if 'stresses' in preds:
                        ens['stresses_all'] = preds['stresses']
                else:
                    ens = ensemble_evaluate(atoms, calculator)

                summary, details = compute_frame_metrics(atoms, ens, ref_e, ref_s_v6, ref_f)

                # Annotate dataset split if mapping provided
                structure_id = str(atoms.info.get("structure_id", ""))
                summary.split = _infer_split_label(structure_id, splits_map)

                summaries.append(summary)

                if jsonl_file is not None:
                    jsonl_file.write(json.dumps(asdict(details)) + "\n")

                frames_pbar.update(1)
            frames_pbar.close()

    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    # Identify bad datapoints using training metrics
    flag_bad_datapoints(summaries)

    # Write CSV
    out_csv_path = Path(args.out_csv).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_path",
                "structure_id",
                "frame_index",
                "num_atoms",
                "ref_energy",
                "mean_energy",
                "var_energy",
                "force_rmse_metric",
                "force_mae_metric",
                "force_max_metric",
                "energy_rmse_per_atom_metric",
                "energy_mae_per_atom_metric",
                "stress_frobenius_error_metric",
                "force_variance_metric",
                "energy_variance_per_atom_metric",
                "stress_variance_metric",
                "split",
                "is_bad",
                "bad_reasons",
            ]
        )
        for r in tqdm(summaries, desc="Writing CSV", unit="frame"):
            writer.writerow(
                [
                    r.source_path,
                    r.structure_id,
                    r.frame_index,
                    r.num_atoms,
                    r.ref_energy if r.ref_energy is not None else "",
                    r.mean_energy if r.mean_energy is not None else "",
                    r.var_energy if r.var_energy is not None else "",
                    r.force_rmse_metric if r.force_rmse_metric is not None else "",
                    r.force_mae_metric if r.force_mae_metric is not None else "",
                    r.force_max_metric if r.force_max_metric is not None else "",
                    r.energy_rmse_per_atom_metric if r.energy_rmse_per_atom_metric is not None else "",
                    r.energy_mae_per_atom_metric if r.energy_mae_per_atom_metric is not None else "",
                    r.stress_frobenius_error_metric if r.stress_frobenius_error_metric is not None else "",
                    r.force_variance_metric if r.force_variance_metric is not None else "",
                    r.energy_variance_per_atom_metric if r.energy_variance_per_atom_metric is not None else "",
                    r.stress_variance_metric if r.stress_variance_metric is not None else "",
                    r.split,
                    int(r.is_bad),
                    r.bad_reasons,
                ]
            )

    logger.info("Wrote CSV summary: %s", out_csv_path)
    if jsonl_path is not None:
        logger.info("Wrote JSONL details: %s", jsonl_path)


if __name__ == "__main__":
    main()


