#!/usr/bin/env python3
"""
Build a composite test set:
  1) From generation 10, sample k_per_bucket structures per (composition, config_type) bucket.
  2) From all other generations, sample a global fraction (default 10%).

Excludes bad structure IDs loaded from multiple known files. Writes the final
list of test IDs to a JSON file (default: ./test_structure_ids_gen10.json).

Usage:
  python scratch/scripts/dataset_analysis/build_test_set_gen10_plus_global.py \
    --k-per-bucket 3 --global-fraction 0.10 --output ./test_structure_ids_gen10.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

from forge.core.database import DatabaseManager


def load_bad_ids_and_metafilter_ids(db: DatabaseManager) -> Set[int]:
    """Load bad structure IDs from known files and apply additional meta-filters.

    This function aggregates bad structure IDs from multiple sources and
    further excludes structures matching known problematic config_types
    (e.g., dimers, short_range, etc.) as in the main experiment filtering.

    Args:
        db (DatabaseManager): An active database manager instance.

    Returns:
        Set[int]: Set of all structure IDs to exclude.

    Raises:
        None

    Examples:
        >>> with DatabaseManager() as db:
        ...     bad_ids = load_bad_ids_and_metafilter_ids(db)
        ...     print(len(bad_ids))
    """
    bad_id_files = [
        Path("./analysis_output_full/bad_structure_ids.json"),
        Path("./bad_structure_ids_gen8.json"),
        Path("./bad_structure_ids_gen9.json"),
        Path("./bad_structure_ids_gen10.json"),
        Path("./scratch/data/dataset_analysis_output_gen_8/bad_structure_ids.json"),
        Path("./scratch/data/dataset_analysis_output_gen_9/bad_structure_ids.json"),
        Path("./scratch/data/dataset_analysis_output_gen_10/bad_structure_ids.json"),
    ]
    bad_ids: Set[int] = set()
    for p in bad_id_files:
        try:
            if p.exists():
                ids = json.load(p.open())
                bad_ids.update(int(x) for x in ids)
        except Exception:
            pass

    # Additional meta-filters as in run_allegro_exploit_gen10.py
    dimer_ids = db.find_structures_by_metadata({'config_type': 'dimer'})
    sr_dimer_ids = db.find_structures_by_metadata({'config_type': 'short_range_dimer'})
    sr_ids = db.find_structures_by_metadata({'config_type': 'short_range'})
    sr_aa_ids = db.find_structures_by_metadata({'config_type': 'short_range_aa'})
    dimer_aa_ids = db.find_structures_by_metadata({'config_type': 'dimer_aa'})

    bad_ids.update(dimer_ids)
    bad_ids.update(sr_dimer_ids)
    bad_ids.update(sr_ids)
    bad_ids.update(sr_aa_ids)
    bad_ids.update(dimer_aa_ids)

    return bad_ids


def sample_gen10_by_bucket(db: DatabaseManager, k_per_bucket: int, seed: int) -> List[int]:
    random.seed(seed)
    gen10_ids = db.find_structures_by_metadata({"generation": 10}, operator="exact")
    if not gen10_ids:
        return []
    meta_map = db.get_structure_metadata_batch(gen10_ids)

    # Build buckets by (composition, config_type)
    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for sid, meta in meta_map.items():
        comp = meta.get("composition_str") or meta.get("formula_string") or meta.get("formula") or "UNKNOWN"
        cfg = meta.get("config_type") or meta.get("structure_type") or "UNKNOWN"
        buckets[(comp, cfg)].append(sid)

    sampled: List[int] = []
    for _, sids in buckets.items():
        if len(sids) <= k_per_bucket:
            sampled.extend(sids)
        else:
            sampled.extend(random.sample(sids, k_per_bucket))
    return sampled


def sample_global_excluding_gen(db: DatabaseManager, exclude_generation: int, fraction: float, seed: int) -> List[int]:
    random.seed(seed)
    all_ids = db.find_structures_by_metadata({"generation": 0}, operator=">=")
    if not all_ids:
        return []
    meta_map = db.get_structure_metadata_batch(all_ids)
    pool = [sid for sid, meta in meta_map.items() if int(meta.get("generation", -1)) != exclude_generation]
    n = int(round(len(pool) * fraction))
    if n <= 0:
        return []
    return random.sample(pool, min(n, len(pool)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build test set with gen10 bucket sampling plus global fraction.")
    ap.add_argument("--k-per-bucket", type=int, default=3, help="Per (composition, config_type) samples in gen 10.")
    ap.add_argument("--global-fraction", type=float, default=0.10, help="Fraction sampled from all non-gen10 structures.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--output", type=str, default="./test_structure_ids_gen10.json", help="Output JSON for test IDs.")
    args = ap.parse_args()

    db = DatabaseManager()
    bad_ids = load_bad_ids_and_metafilter_ids(db)

    gen10_sample = sample_gen10_by_bucket(db, k_per_bucket=args.k_per_bucket, seed=args.seed)
    gen10_sample = [sid for sid in gen10_sample if sid not in bad_ids]

    global_sample = sample_global_excluding_gen(db, exclude_generation=10, fraction=args.global_fraction, seed=args.seed)
    global_sample = [sid for sid in global_sample if sid not in bad_ids and sid not in gen10_sample]

    final_ids = gen10_sample + global_sample

    print(f"[INFO] Buckets (gen10): selected {len(gen10_sample)} IDs")
    print(f"[INFO] Global (non-gen10): selected {len(global_sample)} IDs")
    print(f"[INFO] Total test set size: {len(final_ids)}")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(final_ids, indent=2))
    print(f"[INFO] Wrote test structure IDs to {out_path.resolve()}")


if __name__ == "__main__":
    main()


