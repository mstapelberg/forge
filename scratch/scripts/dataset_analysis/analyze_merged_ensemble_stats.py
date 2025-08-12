"""Analyze merged ensemble statistics to flag bad structures and reasons.

This tool consumes the merged CSV (and optionally JSONL) produced by the
ensemble stats array jobs and identifies structures that should be flagged as
"bad" based on dataset-level thresholds.

Two modes are supported:
- Recompute global thresholds across the merged dataset (default). Thresholds
  are computed as quantiles (e.g., 95th percentile) for selected metrics.
- Use provided per-frame flags (`is_bad`, `bad_reasons`) from the CSV and
  aggregate to structure level.

Outputs:
- `bad_structure_ids.json`: list of unique structure IDs flagged as bad.
- `bad_structures_detailed.json`: mapping of structure ID to summary including
  reasons, split, counts, and metric maxima.
- `bad_structures_summary.csv`: per-structure CSV summary.

Examples:
    # Recompute global thresholds at 95th percentile
    python analyze_merged_ensemble_stats.py \
        --merged-csv /abs/run/merged/ensemble_stats_merged.csv \
        --out-dir /abs/run/analysis --quantile 0.95

    # Use existing per-frame flags instead of recomputing thresholds
    python analyze_merged_ensemble_stats.py \
        --merged-csv /abs/run/merged/ensemble_stats_merged.csv \
        --out-dir /abs/run/analysis --use-provided-bad
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger("analyze_merged_ensemble_stats")


# ------------------------------- Data classes ------------------------------- #


@dataclass
class FrameRecord:
    """Single-frame record parsed from the merged CSV.

    Attributes mirror the CSV headers expected from `computed_ensemble_stats.py`.
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
    split: str
    is_bad: bool
    bad_reasons: str


@dataclass
class StructureSummary:
    """Aggregated metrics and labels for a structure across frames.

    Args:
        structure_id: Unique structure identifier.
        split: Dataset split label if available.
        num_frames: Number of frames encountered for this structure.
        any_frame_bad: True if any frame had `is_bad=True`.
        frame_bad_count: Number of frames flagged bad.
        max_force_rmse_metric: Max of force RMSE across frames.
        max_force_variance_metric: Max of ensemble force variance across frames.
        max_energy_rmse_per_atom_metric: Max of energy RMSE/atom across frames.
        max_stress_frobenius_error_metric: Max of stress frob error across frames.
        reasons: Comma-separated reasons for flagging (populated after evaluation).
        source_paths: Sorted unique source `.xyz` files contributing frames.
    """

    structure_id: str
    split: str
    num_frames: int
    any_frame_bad: bool
    frame_bad_count: int

    max_force_rmse_metric: Optional[float]
    max_force_variance_metric: Optional[float]
    max_energy_rmse_per_atom_metric: Optional[float]
    max_stress_frobenius_error_metric: Optional[float]

    reasons: str
    source_paths: List[str]


# ------------------------------- Core helpers ------------------------------- #


CSV_EXPECTED_HEADERS: Tuple[str, ...] = (
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
)


METRIC_KEYS_TO_REASON: Dict[str, str] = {
    "force_rmse_metric": "high_force_rmse",
    "force_variance_metric": "high_ensemble_force_var",
    "energy_rmse_per_atom_metric": "high_energy_rmse_per_atom",
    "stress_frobenius_error_metric": "high_stress_error",
}


def _safe_float(value: str) -> Optional[float]:
    """Parse optional float from CSV.

    Args:
        value: String value, may be empty.

    Returns:
        Float or None if empty/invalid.
    """
    v = value.strip()
    if v == "":
        return None
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def _parse_bool(value: str) -> bool:
    """Parse CSV boolean stored as int or string.

    Accepts "1"/"0", "true"/"false" (case-insensitive).
    """
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes"}:
        return True
    if v in {"0", "false", "f", "no"}:
        return False
    # Fallback: non-empty -> True
    try:
        return bool(int(v))
    except Exception:  # noqa: BLE001
        return v != ""


def load_frames_from_csv(path: Path) -> List[FrameRecord]:
    """Load per-frame records from the merged CSV.

    Args:
        path: Path to the merged CSV file.

    Returns:
        List of `FrameRecord` entries.
    """
    path = path.expanduser().resolve()
    frames: List[FrameRecord] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = tuple(reader.fieldnames or [])
        missing = [h for h in CSV_EXPECTED_HEADERS if h not in headers]
        if missing:
            logger.warning("CSV missing expected headers: %s", ", ".join(missing))
        for row in reader:
            frames.append(
                FrameRecord(
                    source_path=row.get("source_path", ""),
                    structure_id=row.get("structure_id", ""),
                    frame_index=int(row.get("frame_index", 0) or 0),
                    num_atoms=int(row.get("num_atoms", 0) or 0),
                    ref_energy=_safe_float(row.get("ref_energy", "") or ""),
                    mean_energy=_safe_float(row.get("mean_energy", "") or ""),
                    var_energy=_safe_float(row.get("var_energy", "") or ""),
                    force_rmse_metric=_safe_float(row.get("force_rmse_metric", "") or ""),
                    force_mae_metric=_safe_float(row.get("force_mae_metric", "") or ""),
                    force_max_metric=_safe_float(row.get("force_max_metric", "") or ""),
                    energy_rmse_per_atom_metric=_safe_float(row.get("energy_rmse_per_atom_metric", "") or ""),
                    energy_mae_per_atom_metric=_safe_float(row.get("energy_mae_per_atom_metric", "") or ""),
                    stress_frobenius_error_metric=_safe_float(row.get("stress_frobenius_error_metric", "") or ""),
                    force_variance_metric=_safe_float(row.get("force_variance_metric", "") or ""),
                    energy_variance_per_atom_metric=_safe_float(row.get("energy_variance_per_atom_metric", "") or ""),
                    stress_variance_metric=_safe_float(row.get("stress_variance_metric", "") or ""),
                    split=row.get("split", "unknown"),
                    is_bad=_parse_bool(row.get("is_bad", "0")),
                    bad_reasons=row.get("bad_reasons", ""),
                )
            )
    return frames


def compute_thresholds(frames: Iterable[FrameRecord], quantile: float) -> Dict[str, float]:
    """Compute dataset-level thresholds for metrics using a quantile.

    Args:
        frames: Iterable of per-frame records.
        quantile: Quantile in (0, 1], e.g., 0.95.

    Returns:
        Mapping metric key -> threshold value (float). If no data for a metric,
        returns `inf` for that metric.
    """
    thresholds: Dict[str, float] = {}
    for key in METRIC_KEYS_TO_REASON.keys():
        values = [getattr(fr, key) for fr in frames if getattr(fr, key) is not None]
        if values:
            arr = np.asarray(values, dtype=float)
            thresholds[key] = float(np.quantile(arr, quantile))
        else:
            thresholds[key] = float("inf")
    return thresholds


def aggregate_by_structure(
    frames: Iterable[FrameRecord],
) -> Dict[str, StructureSummary]:
    """Aggregate frame-level records into per-structure summaries.

    Args:
        frames: Iterable of per-frame records.

    Returns:
        Mapping from structure ID to `StructureSummary`.
    """
    per_struct: Dict[str, StructureSummary] = {}
    for fr in frames:
        sid = fr.structure_id
        if sid not in per_struct:
            per_struct[sid] = StructureSummary(
                structure_id=sid,
                split=fr.split or "unknown",
                num_frames=0,
                any_frame_bad=False,
                frame_bad_count=0,
                max_force_rmse_metric=None,
                max_force_variance_metric=None,
                max_energy_rmse_per_atom_metric=None,
                max_stress_frobenius_error_metric=None,
                reasons="",
                source_paths=[],
            )
        ss = per_struct[sid]
        ss.num_frames += 1
        ss.any_frame_bad = ss.any_frame_bad or fr.is_bad
        ss.frame_bad_count += int(fr.is_bad)
        # Aggregate maxima for the key metrics
        for key in (
            "force_rmse_metric",
            "force_variance_metric",
            "energy_rmse_per_atom_metric",
            "stress_frobenius_error_metric",
        ):
            val = getattr(fr, key)
            if val is None:
                continue
            max_key = f"max_{key}"
            current = getattr(ss, max_key)
            if current is None or val > current:
                setattr(ss, max_key, float(val))
        # Track contributing sources
        if fr.source_path and fr.source_path not in ss.source_paths:
            ss.source_paths.append(fr.source_path)
    # Sort source paths for determinism
    for ss in per_struct.values():
        ss.source_paths.sort()
    return per_struct


def label_structures_with_thresholds(
    summaries: Dict[str, StructureSummary], thresholds: Dict[str, float]
) -> None:
    """Populate `reasons` in structure summaries using thresholds on max metrics.

    Args:
        summaries: Mapping of structure ID to summary (modified in place).
        thresholds: Mapping from metric key to threshold values.
    """
    for ss in summaries.values():
        reasons: List[str] = []
        for key, reason in METRIC_KEYS_TO_REASON.items():
            max_key = f"max_{key}"
            val = getattr(ss, max_key)
            thr = thresholds.get(key, float("inf"))
            if val is not None and val >= thr:
                reasons.append(reason)
        ss.reasons = ",".join(sorted(set(reasons)))


def label_structures_from_frames(summaries: Dict[str, StructureSummary], frames: Iterable[FrameRecord]) -> None:
    """Populate `reasons` from frame-level `bad_reasons` aggregated per structure.

    Args:
        summaries: Mapping of structure ID to summary (modified in place).
        frames: Iterable of per-frame records.
    """
    per_sid_reasons: Dict[str, List[str]] = {}
    for fr in frames:
        if not fr.bad_reasons:
            continue
        sid = fr.structure_id
        per_sid_reasons.setdefault(sid, []).extend([r for r in fr.bad_reasons.split(",") if r])
    for sid, reasons in per_sid_reasons.items():
        if sid in summaries:
            summaries[sid].reasons = ",".join(sorted(set(reasons)))


def write_outputs(
    summaries: Dict[str, StructureSummary], out_dir: Path
) -> Tuple[Path, Path, Path]:
    """Write JSON and CSV outputs for bad structures.

    Args:
        summaries: Per-structure summaries with `reasons` populated.
        out_dir: Output directory.

    Returns:
        Tuple of paths: (bad_structure_ids.json, bad_structures_detailed.json, bad_structures_summary.csv)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to those with non-empty reasons
    bad_items = {sid: ss for sid, ss in summaries.items() if ss.reasons}
    bad_ids = sorted(bad_items.keys())

    ids_path = out_dir / "bad_structure_ids.json"
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(bad_ids, f, indent=2)

    detailed_path = out_dir / "bad_structures_detailed.json"
    with detailed_path.open("w", encoding="utf-8") as f:
        json.dump({sid: asdict(ss) for sid, ss in bad_items.items()}, f, indent=2)

    csv_path = out_dir / "bad_structures_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "structure_id",
                "split",
                "num_frames",
                "any_frame_bad",
                "frame_bad_count",
                "max_force_rmse_metric",
                "max_force_variance_metric",
                "max_energy_rmse_per_atom_metric",
                "max_stress_frobenius_error_metric",
                "reasons",
                "source_paths",
            ]
        )
        for sid in bad_ids:
            ss = bad_items[sid]
            writer.writerow(
                [
                    ss.structure_id,
                    ss.split,
                    ss.num_frames,
                    int(ss.any_frame_bad),
                    ss.frame_bad_count,
                    ss.max_force_rmse_metric if ss.max_force_rmse_metric is not None else "",
                    ss.max_force_variance_metric if ss.max_force_variance_metric is not None else "",
                    ss.max_energy_rmse_per_atom_metric if ss.max_energy_rmse_per_atom_metric is not None else "",
                    ss.max_stress_frobenius_error_metric if ss.max_stress_frobenius_error_metric is not None else "",
                    ss.reasons,
                    ";".join(ss.source_paths),
                ]
            )

    return ids_path, detailed_path, csv_path


# ----------------------------------- CLI ----------------------------------- #


def sweep_quantiles(
    merged_csv: Path,
    out_dir: Path,
    quantiles: Sequence[float],
    use_provided_bad: bool = False,
) -> Dict[float, Dict[str, int]]:
    """Sweep over quantiles, write outputs per-quantile, and summarize counts.

    For each quantile `q` provided, this function:
      1) Loads frames from `merged_csv`
      2) Aggregates by structure
      3) Either recomputes thresholds at `q` and labels reasons from maxima, or
         aggregates provided per-frame flags if `use_provided_bad=True`
      4) Writes outputs into `<out_dir>/q-<q*100>` via `write_outputs`
      5) Writes a small JSON summary including thresholds, total flagged count,
         and per-reason counts

    Args:
        merged_csv: Path to the merged CSV file from ensemble stats.
        out_dir: Base directory to contain one subdirectory per quantile.
        quantiles: Iterable of quantiles in (0, 1], e.g., [0.95, 0.99].
        use_provided_bad: If True, do not recompute thresholds; aggregate
            provided per-frame flags instead.

    Returns:
        Mapping from quantile to a dict of summary counts, e.g.:
            {
              0.95: {"flagged_structures": 123, "high_force_rmse": 80, ...},
              0.99: {...}
            }
    """
    merged_csv = merged_csv.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    frames = load_frames_from_csv(merged_csv)
    logger.info("Loaded %d frame records from %s", len(frames), merged_csv)

    summary_by_q: Dict[float, Dict[str, int]] = {}

    def _reason_counts(summaries: Dict[str, StructureSummary]) -> Dict[str, int]:
        counts: Dict[str, int] = {r: 0 for r in METRIC_KEYS_TO_REASON.values()}
        for ss in summaries.values():
            if not ss.reasons:
                continue
            for r in ss.reasons.split(","):
                if not r:
                    continue
                counts[r] = counts.get(r, 0) + 1
        return counts

    for q in quantiles:
        # Defensive checks
        try:
            qf = float(q)
        except Exception:
            logger.warning("Skipping non-float quantile: %r", q)
            continue
        if not (0.0 < qf <= 1.0):
            logger.warning("Skipping out-of-range quantile: %s", qf)
            continue

        # Aggregate by structure fresh each loop to ensure clean state
        summaries = aggregate_by_structure(frames)

        thresholds: Dict[str, float] = {}
        if use_provided_bad:
            # Label from per-frame provided flags
            label_structures_from_frames(summaries, frames)
        else:
            thresholds = compute_thresholds(frames, quantile=qf)
            label_structures_with_thresholds(summaries, thresholds)

        # Prepare per-quantile directory
        q_dir_name = f"q-{qf*100:.3f}".rstrip("0").rstrip(".")  # e.g., q-95, q-99.5
        q_out_dir = out_dir / q_dir_name
        q_out_dir.mkdir(parents=True, exist_ok=True)

        # Write standard outputs
        ids_path, detailed_path, csv_path = write_outputs(summaries, q_out_dir)
        logger.info("[%s] Wrote outputs: ids=%s, detailed=%s, csv=%s", q_dir_name, ids_path, detailed_path, csv_path)

        # Build and write summary JSON
        flagged_count = sum(1 for s in summaries.values() if s.reasons)
        reason_cts = _reason_counts(summaries)
        summary = {
            "quantile": qf,
            "use_provided_bad": bool(use_provided_bad),
            "thresholds": thresholds,
            "flagged_structures": flagged_count,
            "reason_counts": reason_cts,
        }
        (q_out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        summary_by_q[qf] = {"flagged_structures": flagged_count, **reason_cts}

    return summary_by_q


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional sequence of arguments, otherwise uses `sys.argv`.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description="Analyze merged ensemble stats to flag bad structures")
    p.add_argument("--merged-csv", type=Path, required=True, help="Path to merged CSV file")
    p.add_argument("--merged-jsonl", type=Path, default=None, help="Path to merged JSONL (optional)")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory to write outputs")
    p.add_argument("--quantile", type=float, default=0.95, help="Quantile for thresholds (0-1]")
    p.add_argument("--use-provided-bad", action="store_true", help="Use per-frame is_bad/bad_reasons instead of recomputing thresholds")
    p.add_argument("--sweep-quantiles", type=str, default=None, help="Comma-separated list of quantiles to sweep (e.g., '0.95,0.975,0.99')")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point to analyze merged ensemble stats and flag bad structures.

    This function loads the merged CSV, aggregates metrics per structure, then
    either recomputes dataset-wide thresholds to assign reasons or uses
    `is_bad`/`bad_reasons` provided per frame to aggregate labels by structure.
    Outputs are written to the specified directory.
    """
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # If sweep mode requested, run sweep and exit
    if args.sweep_quantiles is not None:
        try:
            q_list = [float(x.strip()) for x in args.sweep_quantiles.split(",") if x.strip()]
        except Exception as exc:
            raise ValueError("Failed to parse --sweep-quantiles; provide comma-separated floats") from exc
        summary = sweep_quantiles(
            merged_csv=args.merged_csv,
            out_dir=args.out_dir,
            quantiles=q_list,
            use_provided_bad=bool(args.use_provided_bad),
        )
        for qf, counts in sorted(summary.items()):
            logger.info("q=%.6g -> %s", qf, counts)
        return

    frames = load_frames_from_csv(args.merged_csv)
    logger.info("Loaded %d frame records from %s", len(frames), args.merged_csv)

    summaries = aggregate_by_structure(frames)
    logger.info("Aggregated into %d structures", len(summaries))

    if args.use_provided_bad:
        label_structures_from_frames(summaries, frames)
    else:
        thresholds = compute_thresholds(frames, quantile=args.quantile)
        logger.info(
            "Thresholds: force_rmse=%.6g, force_var=%.6g, energy_rmse_pa=%.6g, stress_frob=%.6g",
            thresholds["force_rmse_metric"],
            thresholds["force_variance_metric"],
            thresholds["energy_rmse_per_atom_metric"],
            thresholds["stress_frobenius_error_metric"],
        )
        label_structures_with_thresholds(summaries, thresholds)

    ids_path, detailed_path, csv_path = write_outputs(summaries, args.out_dir.expanduser().resolve())
    logger.info("Wrote bad_structure_ids: %s", ids_path)
    logger.info("Wrote detailed JSON: %s", detailed_path)
    logger.info("Wrote summary CSV: %s", csv_path)


if __name__ == "__main__":
    main()


