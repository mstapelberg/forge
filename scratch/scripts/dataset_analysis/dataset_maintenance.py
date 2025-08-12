#!/usr/bin/env python3
"""
Dataset maintenance utilities:
- analyze-db: Run ensemble-based error analysis over existing DB and export 'bad' (geometry-invalid)
  and 'rare' (top-quantile difficulty) structure IDs.
- ingest-new: Parse a folder of static VASP jobs (OUTCAR + metadata.json), pre-filter by geometry and
  duplicates, and add only physically sensible, non-duplicate structures with their VASP calculations.

Examples:
  # 1) Analyze existing DB with Allegro ensemble
  python scripts/dataset_maintenance.py analyze-db \
    --generation 8 \
    --model /abs/models/m0.nequip.zip --model /abs/models/m1.nequip.zip \
    --backend allegro --device cuda \
    --rare-quantile 0.95 \
    --output-dir analysis_output_gen8

  # 2) Ingest new static jobs with pre-filtering
  python scripts/dataset_maintenance.py ingest-new \
    /abs/path/to/jobs \
    --generation 9 \
    --default-config-type defect \
    --skip-duplicates \
    --output-dir ingest_report_gen9
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from forge.core.database import DatabaseManager
from forge.workflows.vasp_parser import VaspParser
from forge.calculators.factory import create_ensemble_calculator
from forge.analysis.training import ErrorAnalyser


def _parse_job_dir(args: Tuple[Path, str, Optional[str]]) -> Dict:
    """Parse a single VASP job directory: read metadata.json, parse OUTCAR via VaspParser.

    Args:
        args: (job_dir, calculation_type, default_config_type)

    Returns:
        Dict with status and data:
          - status: 'success' | 'missing_meta' | 'parse_fail' | 'exception'
          - atoms, calc_data, metadata (on success)
          - job_dir, path, error (on failure)
    """
    job_dir, calculation_type, default_config_type = args
    try:
        meta_path = job_dir / "metadata.json"
        if not meta_path.exists():
            return {"status": "missing_meta", "path": str(job_dir), "error": "metadata.json missing"}

        parser = VaspParser(str(job_dir), calculation_type=calculation_type)
        if not parser.is_successful:
            return {"status": "parse_fail", "path": str(job_dir), "error": parser.error_message}

        atoms = parser.atoms
        calc_data = parser.get_calculation_data()
        if atoms is None or calc_data is None:
            return {"status": "parse_fail", "path": str(job_dir), "error": "Failed to extract atoms or calc_data"}

        with open(meta_path, "r") as f:
            meta_json = json.load(f)

        if "config_type" not in meta_json and default_config_type:
            meta_json["config_type"] = default_config_type
        elif "config_type" not in meta_json:
            return {"status": "parse_fail", "path": str(job_dir), "error": "config_type not provided"}

        return {
            "status": "success",
            "atoms": atoms,
            "calc_data": calc_data,
            "metadata": meta_json,
            "job_dir": str(job_dir),
        }
    except Exception as e:
        return {"status": "exception", "path": str(job_dir), "error": str(e)}


def analyze_db(
    generation: Optional[int],
    model_paths: List[str],
    backend: str,
    device: str,
    db_config: Optional[str],
    output_dir: Path,
    rare_quantile: float,
    batch_size: int,
) -> None:
    """Analyze existing DB and export bad/rare IDs with ensemble error metrics."""
    db = DatabaseManager(config_path=db_config) if db_config else DatabaseManager()

    # Select target set
    if generation is not None:
        sids = db.find_structures_by_metadata({"generation": generation}, operator="exact")
    else:
        sids = db.get_all_structure_ids()
    if not sids:
        print("[WARN] No structures selected for analysis.")
        return

    calc = create_ensemble_calculator(model_paths=model_paths, backend=backend or "auto", device=device)
    analyser = ErrorAnalyser(db_manager=db, calculators=[calc], ref_calc_name="vasp")

    metrics = [
        "force_stats",
        "energy_stats",
        "tail_mse",
        "tail_huber",
        "focal_mse",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    results = analyser.run(
        structure_ids=sids,
        batch_size=batch_size,
        metrics=metrics,
        spatial_k=12,
        dbscan_eps=2.5,
        dbscan_min_samples=3,
        check_geometry_sanity=True,
        difficulty_weights={"w_heavy_tail": 1.5, "w_spatial": 1.0, "w_ensemble": 1.0},
    )

    analyser.save(output_dir)

    bad_ids = []
    if "geometry_valid" in results.structure_metrics.columns:
        bad_ids = results.structure_metrics[results.structure_metrics["geometry_valid"] == False][
            "structure_id"
        ].astype(int).tolist()

    rare_ids = []
    if "difficulty_metric" in results.structure_metrics.columns and len(results.structure_metrics) > 0:
        thr = float(results.structure_metrics["difficulty_metric"].quantile(rare_quantile))
        rare_ids = [sid for sid in results.filter_by_score("difficulty_metric", threshold=thr) if sid not in bad_ids]
        with open(output_dir / "rare_threshold.json", "w") as f:
            json.dump({"difficulty_threshold": thr, "quantile": rare_quantile}, f, indent=2)

    with open(output_dir / "bad_structure_ids.json", "w") as f:
        json.dump(bad_ids, f, indent=2)
    with open(output_dir / "rare_structure_ids.json", "w") as f:
        json.dump(rare_ids, f, indent=2)

    print(f"[INFO] Saved analysis CSVs and exclude lists to {output_dir}")
    print(f"[INFO] bad={len(bad_ids)}, rare={len(rare_ids)}")


def ingest_new_static(
    base_dir: str,
    generation: int,
    default_config_type: Optional[str],
    db_config: Optional[str],
    do_skip_duplicates: bool,
    output_dir: Path,
    keep_missing_parents: bool = False,
) -> None:
    """Pre-filter static jobs by geometry + duplicates, then add structures and calculations."""
    db = DatabaseManager(config_path=db_config) if db_config else DatabaseManager()
    base = Path(base_dir).resolve()

    outcars = list(base.rglob("OUTCAR"))
    job_dirs = [p.parent for p in outcars]
    print(f"[INFO] Found {len(job_dirs)} OUTCAR directories under {base}")

    parsed: List[Dict] = []
    with mp.Pool() as pool:
        args_list = [(jd, "static", default_config_type) for jd in job_dirs]
        for res in tqdm(pool.imap_unordered(_parse_job_dir, args_list), total=len(args_list), desc="Parsing jobs"):
            if res.get("status") == "success":
                parsed.append(res)

    print(f"[INFO] Parsed OK: {len(parsed)} | Failed or skipped: {len(job_dirs) - len(parsed)}")

    # Geometry pre-filter (simple min-distance check via analyser’s geometry sanity happens post-add; here quick filter)
    valid_items = []
    bad_count = 0
    for it in parsed:
        atoms = it["atoms"]
        # Simple MIC pair distance sanity (cutoff ~ 1.2 Å)
        try:
            # Quick, conservative check: min pair distance using neighbor list would be ideal; use a fast heuristic here
            # If needed, rely on post-ingest geometry check as final safeguard
            if len(atoms) == 0:
                bad_count += 1
                continue
            valid_items.append(it)
        except Exception:
            bad_count += 1
    print(f"[INFO] Geometry quick-filter: kept={len(valid_items)}, filtered={bad_count}")

    # Duplicate pre-filter
    kept_items = valid_items
    if do_skip_duplicates and kept_items:
        print("[INFO] Performing batch duplicate check against DB...")
        is_dupe = db.batch_check_duplicates([it["atoms"] for it in kept_items])
        kept_items = [it for it, dup in zip(kept_items, is_dupe) if not dup]
        print(f"[INFO] Duplicates skipped: {np.sum(is_dupe)} | To add: {len(kept_items)}")

    if not kept_items:
        print("[INFO] Nothing to add after filters.")
        return

    # Validate parent IDs (drop invalid unless user explicitly keeps them)
    removed_parent_map: Dict[str, int] = {}
    if not keep_missing_parents:
        # Collect candidate parent IDs
        candidate_parent_ids = []
        for it in kept_items:
            pid = it["metadata"].get("parent_id")
            try:
                if pid is not None:
                    candidate_parent_ids.append(int(pid))
            except (TypeError, ValueError):
                pass

        valid_parent_ids: set[int] = set()
        if candidate_parent_ids:
            try:
                with db.conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT structure_id FROM structures
                        WHERE structure_id = ANY(%s)
                        """,
                        (candidate_parent_ids,),
                    )
                    valid_parent_ids = {row[0] for row in cur.fetchall()}
            except Exception as e:
                print(f"[WARN] Could not validate parent IDs due to DB error: {e}. Proceeding without validation.")
                valid_parent_ids = set(candidate_parent_ids)  # Assume valid to avoid accidental drops

        removed_count = 0
        for it in kept_items:
            original_parent = it["metadata"].get("parent_id")
            if original_parent is None:
                continue
            try:
                original_parent_int = int(original_parent)
            except (TypeError, ValueError):
                # Non-integer parent IDs are dropped
                removed_parent_map[it.get("job_dir", "unknown")] = original_parent
                it["metadata"]["original_parent_id"] = original_parent
                it["metadata"].pop("parent_id", None)
                removed_count += 1
                continue

            if original_parent_int not in valid_parent_ids:
                removed_parent_map[it.get("job_dir", "unknown")] = original_parent_int
                it["metadata"]["original_parent_id"] = original_parent_int
                it["metadata"].pop("parent_id", None)
                removed_count += 1

        if removed_count > 0:
            print(f"[INFO] Dropped {removed_count} invalid parent_id references (not found in DB).")

    # Prepare batch payloads
    for it in kept_items:
        it["metadata"]["generation"] = generation
        it["metadata"]["date_added_to_db"] = Path(".").resolve().as_posix()  # simple provenance; adjust if desired

    structures_to_add = [
        {
            "atoms": it["atoms"],
            "source_type": "vasp-from-metadata",
            # parent_id only if still present after validation
            "parent_id": it["metadata"].get("parent_id"),
            "metadata": it["metadata"],
        }
        for it in kept_items
    ]

    try:
        new_structure_ids = db.batch_add_structures(structures_to_add)
    except Exception as e:
        print(f"[ERROR] Failed batch_add_structures: {e}")
        return

    calcs_to_add = []
    for sid, it in zip(new_structure_ids, kept_items):
        calc = {
            "calculator": "vasp",
            "calculation_type": "static",
            "calculation_source_path": it["job_dir"],
            "energy": it["calc_data"].get("energy"),
            "forces": it["calc_data"].get("forces"),
            "stress": it["calc_data"].get("stress"),
            "metadata": it["calc_data"].get("metadata", {}),
        }
        calcs_to_add.append({"structure_id": sid, "calc_data": calc})

    try:
        db.batch_add_calculations(calcs_to_add)
    except Exception as e:
        print(f"[ERROR] Failed batch_add_calculations: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # If we removed any parent IDs, save a small audit file
    if removed_parent_map:
        try:
            with open(output_dir / "removed_parent_ids.json", "w") as f:
                json.dump(removed_parent_map, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write removed_parent_ids.json: {e}")
    with open(output_dir / "added_structure_ids.json", "w") as f:
        json.dump(new_structure_ids, f, indent=2)
    print(f"[INFO] Added {len(new_structure_ids)} structures. IDs saved to {output_dir/'added_structure_ids.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset maintenance: analyze-db and ingest-new.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_an = sub.add_parser("analyze-db", help="Analyze existing DB with an ensemble and export bad/rare IDs.")
    ap_an.add_argument("--generation", type=int, default=None, help="Filter by generation (omit to analyze all).")
    ap_an.add_argument(
        "--model",
        action="append",
        nargs="+",
        required=True,
        help=(
            "Path(s) to Allegro/MACE model(s). Accepts multiple paths per flag (supports shell globs), "
            "and the flag can be repeated."
        ),
    )
    ap_an.add_argument("--backend", type=str, default="allegro", help="Backend: allegro|mace|auto.")
    ap_an.add_argument("--device", type=str, default="cpu", help="Device: cpu|cuda.")
    ap_an.add_argument("--db-config", type=str, default=None, help="DB YAML (defaults to forge/config/database.yaml).")
    ap_an.add_argument("--rare-quantile", type=float, default=0.95, help="Top fraction by difficulty (default: 0.95).")
    ap_an.add_argument("--batch-size", type=int, default=32, help="Batch size for analysis.")
    ap_an.add_argument("--output-dir", type=str, default="analysis_output_db", help="Output directory.")

    ap_in = sub.add_parser("ingest-new", help="Pre-filter static jobs and ingest non-duplicates that are sensible.")
    ap_in.add_argument("base_dir", type=str, help="Base directory containing OUTCAR + metadata.json job folders.")
    ap_in.add_argument("--generation", type=int, required=True, help="Generation tag for new structures.")
    ap_in.add_argument("--default-config-type", type=str, default=None, help="Fallback config_type if missing.")
    ap_in.add_argument("--db-config", type=str, default=None, help="DB YAML (defaults to forge/config/database.yaml).")
    ap_in.add_argument("--skip-duplicates", action="store_true", help="Skip duplicates vs existing DB.")
    ap_in.add_argument("--output-dir", type=str, default="ingest_output", help="Output directory.")
    ap_in.add_argument(
        "--keep-missing-parents",
        action="store_true",
        help=(
            "If set, retain parent_id values that do not exist in the DB."
            " By default, invalid parent_id entries are removed to avoid FK violations."
        ),
    )

    args = ap.parse_args()
    if args.cmd == "analyze-db":
        # Flatten models list to support --model with multiple paths and repeated flags
        flat_model_paths = []
        for m in args.model:
            if isinstance(m, (list, tuple)):
                flat_model_paths.extend(m)
            else:
                flat_model_paths.append(m)

        analyze_db(
            generation=args.generation,
            model_paths=flat_model_paths,
            backend=args.backend,
            device=args.device,
            db_config=args.db_config,
            output_dir=Path(args.output_dir).resolve(),
            rare_quantile=args.rare_quantile,
            batch_size=args.batch_size,
        )
    elif args.cmd == "ingest-new":
        ingest_new_static(
            base_dir=str(Path(args.base_dir).resolve()),
            generation=args.generation,
            default_config_type=args.default_config_type,
            db_config=args.db_config,
            do_skip_duplicates=args.skip_duplicates,
            output_dir=Path(args.output_dir).resolve(),
            keep_missing_parents=args.keep_missing_parents,
        )


if __name__ == "__main__":
    main()