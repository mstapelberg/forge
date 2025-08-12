"""Compute ensemble statistics over XYZ datasets using Forge calculators.

This script iterates over one or more `.xyz` files, reads reference labels
stored in `Atoms.info['REF_energy']`, `Atoms.info['REF_stress']`, and
`Atoms.arrays['REF_force']`, evaluates an ensemble (or single) calculator
created via `forge.calculators.factory.create_ensemble_calculator`, and
computes per-frame mean and variance for energy, forces, and stress.

Results are summarized to a CSV file by default, with optional JSONL output
containing per-frame detailed arrays.

Example:
    python -m dataset_analysis.compute_ensemble_stats \
        --models /path/to/model1.nequip.zip /path/to/model2.nequip.zip \
        --backend allegro --device cuda \
        data/train1.xyz data/train2.xyz

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import iread

from forge.calculators.factory import create_ensemble_calculator


# ----------------------------- Data Structures ----------------------------- #


@dataclass
class FrameStats:
    """Container for per-frame statistics.

    Attributes capture both reference values (if provided in the file) and the
    ensemble mean and variance for energy, forces, and stress. Array-valued
    fields are summarized to scalar aggregates for CSV output, with the full
    arrays optionally written to a JSONL file when requested.

    Args:
        source_path: Absolute path to the source `.xyz` file.
        frame_index: Zero-based frame index within the source file.
        num_atoms: Number of atoms in the frame.
        ref_energy: Reference energy if present, otherwise None.
        ref_stress: Reference stress Voigt vector (6,) if present, otherwise None.
        ref_force: Reference forces array (n_atoms, 3) if present, otherwise None.
        mean_energy: Ensemble mean energy (scalar).
        var_energy: Ensemble variance of energy (scalar).
        mean_force: Ensemble mean forces (n_atoms, 3) or None if unavailable.
        var_force: Ensemble variance of forces across models (n_atoms, 3) or None.
        mean_stress: Ensemble mean stress Voigt vector (6,) or None if unavailable.
        var_stress: Ensemble variance of stress across models (6,) or None.
    """

    source_path: str
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

    # Scalar summaries for CSV convenience
    var_force_mean_abs: Optional[float] = None
    var_force_max_abs: Optional[float] = None
    var_stress_mean_abs: Optional[float] = None
    var_stress_max_abs: Optional[float] = None


# --------------------------------- Utils ---------------------------------- #


def _gather_xyz_files(paths: Sequence[Union[str, Path]]) -> List[Path]:
    """Collect `.xyz` files from provided paths.

    Args:
        paths: Files and/or directories to search.

    Returns:
        List of absolute `Path` objects to `.xyz` files.

    Raises:
        FileNotFoundError: If no `.xyz` files are found.
    """
    xyz_files: List[Path] = []
    for p in paths:
        pth = Path(p).expanduser().resolve()
        if pth.is_file() and pth.suffix.lower() == ".xyz":
            xyz_files.append(pth)
        elif pth.is_dir():
            xyz_files.extend(sorted(pth.rglob("*.xyz")))
        else:
            logging.warning("Path is not an .xyz file or directory: %s", pth)

    if not xyz_files:
        raise FileNotFoundError("No .xyz files found in the provided paths")
    return xyz_files


def _extract_references(atoms: Atoms) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract reference labels from an `Atoms` object.

    Args:
        atoms: ASE `Atoms` containing reference fields.

    Returns:
        Tuple of (ref_energy, ref_stress, ref_force) where values may be None if
        not present. Stress is returned as shape (6,), force as shape (n_atoms, 3).
    """
    ref_energy: Optional[float] = None
    ref_stress: Optional[np.ndarray] = None
    ref_force: Optional[np.ndarray] = None

    try:
        if "REF_energy" in atoms.info:
            ref_energy = float(atoms.info["REF_energy"])  # type: ignore[arg-type]
    except Exception as exc:
        logging.debug("Failed to parse REF_energy: %s", exc)

    try:
        if "REF_stress" in atoms.info:
            stress_val = atoms.info["REF_stress"]
            ref_stress = np.asarray(stress_val, dtype=float).reshape(-1)
            if ref_stress.shape[0] != 6:
                logging.debug("REF_stress has unexpected shape: %s", ref_stress.shape)
    except Exception as exc:
        logging.debug("Failed to parse REF_stress: %s", exc)

    try:
        if "REF_force" in atoms.arrays:
            ref_force = np.asarray(atoms.arrays["REF_force"], dtype=float)
            if ref_force.ndim != 2 or ref_force.shape[1] != 3:
                logging.debug("REF_force has unexpected shape: %s", ref_force.shape)
    except Exception as exc:
        logging.debug("Failed to parse REF_force: %s", exc)

    return ref_energy, ref_stress, ref_force


def _compute_mean_var_from_ensemble(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance across the model axis.

    Expects input shaped as `(n_models, ...)` and returns `(mean, var)` with the
    model axis reduced.

    Args:
        array: Array of predictions with leading model dimension.

    Returns:
        A tuple `(mean, var)` with arrays shaped as `array.shape[1:]`.
    """
    mean_arr = array.mean(axis=0)
    var_arr = array.var(axis=0)
    return mean_arr, var_arr


def _summarize_abs(var_arr: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
    """Summarize an array via mean and max absolute value.

    Args:
        var_arr: Array to summarize, or None.

    Returns:
        Tuple `(mean_abs, max_abs)` or `(None, None)` if input is None.
    """
    if var_arr is None:
        return None, None
    abs_arr = np.abs(var_arr)
    return float(abs_arr.mean()), float(abs_arr.max())


def _maybe_to_list(arr: Optional[np.ndarray]) -> Optional[List]:
    """Convert numpy array to nested Python list if not None."""
    if arr is None:
        return None
    return arr.tolist()


# --------------------------------- Runner ---------------------------------- #


def iter_frames(xyz_path: Path) -> Iterator[Tuple[int, Atoms]]:
    """Yield `(frame_index, Atoms)` from an `.xyz` file.

    Args:
        xyz_path: Path to an `.xyz` file.

    Yields:
        Tuples of `(index, atoms)` for each frame.
    """
    for idx, atoms in enumerate(iread(str(xyz_path), format="extxyz", index=":")):
        yield idx, atoms


def evaluate_frame(
    atoms: Atoms,
    calculator,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate energy, forces, and stress using the provided calculator.

    Supports both ensemble calculators exposing `energies_all`/`forces_all`/
    `stresses_all` and native ASE calculators exposing `get_*` methods.

    Args:
        atoms: Frame to evaluate.
        calculator: Calculator created by `create_ensemble_calculator`.

    Returns:
        Tuple of `(mean_energy, mean_force, mean_stress, var_energy, var_force, var_stress)`
        where array-valued entries are numpy arrays with shapes matching their
        physical quantities, or None if the quantity cannot be computed.
    """
    # Prefer ensemble methods if available
    has_ensemble = all(
        hasattr(calculator, name) for name in ("energies_all", "forces_all", "stresses_all")
    )

    if has_ensemble:
        # Energies
        mean_energy: Optional[float]
        var_energy: Optional[float]
        try:
            e_all = calculator.energies_all(atoms)  # shape: (n_models,)
            mean_energy = float(np.mean(e_all))
            var_energy = float(np.var(e_all))
        except Exception as exc:  # energy should generally be available
            logging.warning("Energy evaluation failed: %s", exc)
            mean_energy = None
            var_energy = None

        # Forces
        mean_force: Optional[np.ndarray]
        var_force: Optional[np.ndarray]
        try:
            f_all = calculator.forces_all(atoms)  # shape: (n_models, n_atoms, 3)
            mean_force, var_force = _compute_mean_var_from_ensemble(f_all)
        except Exception as exc:
            logging.info("Forces unavailable for this model: %s", exc)
            mean_force, var_force = None, None

        # Stress
        mean_stress: Optional[np.ndarray]
        var_stress: Optional[np.ndarray]
        try:
            s_all = calculator.stresses_all(atoms)  # shape: (n_models, 6)
            mean_stress, var_stress = _compute_mean_var_from_ensemble(s_all)
        except Exception as exc:
            logging.info("Stress unavailable for this model: %s", exc)
            mean_stress, var_stress = None, None

        return mean_energy, mean_force, mean_stress, var_energy, var_force, var_stress

    # Fallback: native ASE calculator
    atoms_copy = atoms.copy()
    atoms_copy.calc = calculator

    # Energy
    mean_energy = None
    var_energy = None
    try:
        mean_energy = float(atoms_copy.get_potential_energy())
        var_energy = 0.0
    except Exception as exc:
        logging.warning("Energy evaluation failed with native calculator: %s", exc)

    # Forces
    mean_force_arr: Optional[np.ndarray] = None
    var_force_arr: Optional[np.ndarray] = None
    try:
        mean_force_arr = np.asarray(atoms_copy.get_forces(), dtype=float)
        var_force_arr = np.zeros_like(mean_force_arr)
    except Exception as exc:
        logging.info("Forces unavailable with native calculator: %s", exc)

    # Stress
    mean_stress_arr: Optional[np.ndarray] = None
    var_stress_arr: Optional[np.ndarray] = None
    try:
        stress_val = atoms_copy.get_stress()
        mean_stress_arr = np.asarray(stress_val, dtype=float).reshape(-1)
        if mean_stress_arr.shape[0] != 6:
            logging.debug("Native stress has unexpected shape: %s", mean_stress_arr.shape)
        var_stress_arr = np.zeros_like(mean_stress_arr)
    except Exception as exc:
        logging.info("Stress unavailable with native calculator: %s", exc)

    return mean_energy, mean_force_arr, mean_stress_arr, var_energy, var_force_arr, var_stress_arr


def run(
    xyz_inputs: Sequence[Union[str, Path]],
    model_paths: Sequence[str],
    backend: str,
    device: str,
    species_to_type_name: Optional[Dict[str, int]] = None,
    out_csv: Optional[Union[str, Path]] = None,
    out_jsonl: Optional[Union[str, Path]] = None,
) -> List[FrameStats]:
    """Main execution: compute statistics for all frames across inputs.

    Args:
        xyz_inputs: Paths (files or directories) containing `.xyz` files.
        model_paths: Paths to one or more model files for the calculator.
        backend: Backend name (`mace`, `allegro`, or `auto`).
        device: Compute device: `cpu` or `cuda`.
        species_to_type_name: Optional mapping for Allegro/NequIP models.
        out_csv: Optional CSV path for scalar summaries.
        out_jsonl: Optional JSONL path for detailed array outputs.

    Returns:
        List of `FrameStats` objects for all processed frames.

    Raises:
        FileNotFoundError: If no `.xyz` files are found.
        ImportError: If the requested backend or detected backend is unavailable.
        ValueError: For unsupported configurations.
    """
    xyz_files = _gather_xyz_files(xyz_inputs)

    # Build calculator once
    calculator_kwargs: Dict[str, object] = {}
    if species_to_type_name is not None:
        calculator_kwargs["species_to_type_name"] = species_to_type_name

    calculator = create_ensemble_calculator(
        model_paths=list(model_paths), backend=backend, device=device, **calculator_kwargs
    )

    results: List[FrameStats] = []
    jsonl_file = None
    if out_jsonl is not None:
        jsonl_path = Path(out_jsonl).expanduser().resolve()
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")

    try:
        for xyz_path in xyz_files:
            for idx, atoms in iter_frames(xyz_path):
                ref_energy, ref_stress, ref_force = _extract_references(atoms)

                mean_energy, mean_force, mean_stress, var_energy, var_force, var_stress = evaluate_frame(
                    atoms, calculator
                )

                # Aggregate scalar summaries
                var_force_mean_abs, var_force_max_abs = _summarize_abs(var_force)
                var_stress_mean_abs, var_stress_max_abs = _summarize_abs(var_stress)

                frame_stats = FrameStats(
                    source_path=str(xyz_path),
                    frame_index=idx,
                    num_atoms=len(atoms),
                    ref_energy=ref_energy,
                    ref_stress=_maybe_to_list(ref_stress),
                    ref_force=_maybe_to_list(ref_force),
                    mean_energy=mean_energy,
                    var_energy=var_energy,
                    mean_force=_maybe_to_list(mean_force),
                    var_force=_maybe_to_list(var_force),
                    mean_stress=_maybe_to_list(mean_stress),
                    var_stress=_maybe_to_list(var_stress),
                    var_force_mean_abs=var_force_mean_abs,
                    var_force_max_abs=var_force_max_abs,
                    var_stress_mean_abs=var_stress_mean_abs,
                    var_stress_max_abs=var_stress_max_abs,
                )

                results.append(frame_stats)

                # Stream JSONL if requested
                if jsonl_file is not None:
                    jsonl_file.write(json.dumps(asdict(frame_stats)) + "\n")

    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    # Write CSV summary if requested
    if out_csv is not None:
        csv_path = Path(out_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "source_path",
                    "frame_index",
                    "num_atoms",
                    "ref_energy",
                    "mean_energy",
                    "var_energy",
                    "var_force_mean_abs",
                    "var_force_max_abs",
                    "var_stress_mean_abs",
                    "var_stress_max_abs",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.source_path,
                        r.frame_index,
                        r.num_atoms,
                        r.ref_energy if r.ref_energy is not None else "",
                        r.mean_energy if r.mean_energy is not None else "",
                        r.var_energy if r.var_energy is not None else "",
                        r.var_force_mean_abs if r.var_force_mean_abs is not None else "",
                        r.var_force_max_abs if r.var_force_max_abs is not None else "",
                        r.var_stress_mean_abs if r.var_stress_mean_abs is not None else "",
                        r.var_stress_max_abs if r.var_stress_max_abs is not None else "",
                    ]
                )

    return results


def _parse_species_map(spec: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse species mapping from a JSON string or file path.

    The expected format is a mapping from chemical symbol to type index, e.g.:
    `{ "V": 0, "Cr": 1, "Ti": 2, "W": 3, "Zr": 4 }`.

    Args:
        spec: JSON content or a path to a JSON file. If None, returns None.

    Returns:
        Parsed mapping or None.

    Raises:
        ValueError: If parsing fails.
    """
    if spec is None:
        return None

    path = Path(spec)
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Fall back to attempting to parse the string as JSON directly
        pass

    try:
        return json.loads(spec)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Failed to parse species map. Provide a JSON string or a path to a JSON file."
        ) from exc


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help=".xyz files and/or directories to scan recursively for .xyz",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Path(s) to model files for the calculator (e.g., .model, .pt2, .zip)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "mace", "allegro", "nequip"],
        help="Calculator backend (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for evaluation (default: cpu)",
    )
    parser.add_argument(
        "--species-map",
        default=None,
        help=(
            "JSON mapping or path for Allegro/NequIP species_to_type_name, e.g. "
            "'{\"V\":0,\"Cr\":1,...}' or /path/to/map.json"
        ),
    )
    parser.add_argument(
        "--out-csv",
        default=str(Path(__file__).resolve().parent / "ensemble_stats.csv"),
        help="Output CSV path for scalar summaries (default: dataset_analysis/ensemble_stats.csv)",
    )
    parser.add_argument(
        "--out-jsonl",
        default=None,
        help="Optional JSONL output path for detailed per-frame arrays",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    species_map = _parse_species_map(args.species_map)

    run(
        xyz_inputs=args.inputs,
        model_paths=args.models,
        backend=args.backend,
        device=args.device,
        species_to_type_name=species_map,
        out_csv=args.out_csv,
        out_jsonl=args.out_jsonl,
    )


if __name__ == "__main__":
    main()


