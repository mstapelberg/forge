#!/usr/bin/env python3
"""
Summarize composition and configuration-type counts for a set of structures.

This script can summarize either:
- All structures matching a specific generation (via database metadata), or
- An explicit list of structure IDs provided in a JSON file.

Outputs:
- Total number of structures considered
- Unique compositions and counts
- Unique config_types and counts
- Composition x config_type matrix counts

Examples:
  Summarize by generation:
    python scratch/scripts/dataset_analysis/config_summary.py --generation 10

  Summarize by an explicit list of IDs from JSON:
    python scratch/scripts/dataset_analysis/config_summary.py --ids scratch/scripts/dataset_analysis/test_structure_ids_gen10.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from forge.core.database import DatabaseManager


def load_ids_from_json(json_path: str | Path) -> List[int]:
    """Load structure IDs from a JSON file.

    The file must contain a JSON array of integers (or strings convertible to integers).

    Args:
        json_path: Path to the JSON file containing structure IDs.

    Returns:
        A list of structure IDs as integers.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If the JSON file contents are not a list or contain non-integer values.

    Examples:
        >>> ids = load_ids_from_json("/path/to/ids.json")
        >>> isinstance(ids, list)
        True
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"IDs JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("IDs JSON must be a list of integers.")

    ids: List[int] = []
    for item in data:
        try:
            ids.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid ID value in JSON list: {item!r} (must be integer-convertible)"
            ) from exc

    return ids


def summarize_structure_ids(
    db: DatabaseManager,
    structure_ids: Iterable[int],
    context_label: str,
    per_generation: bool = False,
    to_json: Optional[str] = None,
) -> None:
    """Print summary of compositions and config_types for provided structure IDs.

    Args:
        db: Connected `DatabaseManager` instance.
        structure_ids: Iterable of structure IDs to summarize.
        context_label: Short descriptor of the selection (e.g., "generation 10" or
            "IDs file path"). Used only for informational printing.
        per_generation: If True, also print a per-generation breakdown of
            config_type x composition counts using `atoms.info['generation']`.
        to_json: Optional path to write all summaries as a single JSON file.

    Returns:
        None. Results are printed to stdout.

    Raises:
        RuntimeError: If metadata retrieval fails unexpectedly.

    Examples:
        >>> db = DatabaseManager()
        >>> summarize_structure_ids(db, [1, 2, 3], "example")
        # Prints summary to stdout
    """
    structure_id_list: List[int] = list(structure_ids)
    print(
        f"[INFO] Found {len(structure_id_list)} structures for {context_label}."
    )
    if not structure_id_list:
        return

    # Retrieve atoms in batch for efficiency
    atoms_list = db.get_batch_atoms_with_calculation(structure_id_list)

    compositions: List[str] = []
    config_types: List[str] = []
    matrix: Dict[Tuple[str, str], int] = defaultdict(int)

    for atoms in atoms_list:
        comp = atoms.get_chemical_formula() or "UNKNOWN"
        cfg_raw = atoms.info.get("config_type")
        cfg = str(cfg_raw) if cfg_raw is not None else "UNKNOWN"

        compositions.append(comp)
        config_types.append(cfg)
        matrix[(comp, cfg)] += 1

    comp_counts = Counter(compositions)
    cfg_counts = Counter(config_types)

    print("\n[INFO] Unique compositions and counts:")
    for comp, cnt in comp_counts.most_common():
        print(f"  - {comp}: {cnt}")

    print("\n[INFO] Unique config_types and counts:")
    for cfg, cnt in cfg_counts.most_common():
        print(f"  - {cfg}: {cnt}")

    print("\n[INFO] Composition x config_type counts:")
    for (comp, cfg), cnt in sorted(matrix.items()):
        print(f"  - {comp} | {cfg}: {cnt}")

    if per_generation:
        per_gen_counts: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
        for atoms in atoms_list:
            # Determine generation label
            gen_raw = atoms.info.get("generation")
            generation = str(gen_raw) if gen_raw is not None else "UNKNOWN"
            # Keys as (config_type, composition) to match requested ordering
            comp = atoms.get_chemical_formula() or "UNKNOWN"
            cfg = str(atoms.info.get("config_type")) if atoms.info.get("config_type") is not None else "UNKNOWN"
            per_gen_counts[generation][(cfg, comp)] += 1

        print("\n[INFO] Per-generation config_type x composition counts:")
        for generation in sorted(per_gen_counts.keys(), key=lambda g: ("zzz" if g == "UNKNOWN" else g)):
            print(f"  Generation {generation}:")
            inner = per_gen_counts[generation]
            for (cfg, comp), cnt in sorted(inner.items()):
                print(f"    - {cfg} | {comp}: {cnt}")

    # Optionally write JSON output for downstream analysis
    if to_json:
        # Convert matrix (comp, cfg) -> count into nested mapping comp -> cfg -> count
        matrix_nested: Dict[str, Dict[str, int]] = defaultdict(dict)
        for (c, k), v in matrix.items():
            matrix_nested.setdefault(c, {})[k] = v

        json_payload: Dict[str, Any] = {
            "context": context_label,
            "num_structures": len(atoms_list),
            "composition_counts": dict(comp_counts),
            "config_type_counts": dict(cfg_counts),
            "composition_x_config_type": matrix_nested,
        }

        if per_generation:
            # Build generation -> config_type -> composition -> count
            per_gen_nested: Dict[str, Dict[str, Dict[str, int]]] = {}
            for gen, inner in per_gen_counts.items():
                cfg_to_comp: Dict[str, Dict[str, int]] = {}
                for (cfg, comp), cnt in inner.items():
                    cfg_to_comp.setdefault(cfg, {})[comp] = cnt
                per_gen_nested[gen] = cfg_to_comp
            json_payload["per_generation_config_type_x_composition"] = per_gen_nested

        out_path = Path(to_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(json_payload, fh, indent=2, sort_keys=True)
        print(f"\n[INFO] Wrote JSON summary to: {out_path}")


def main() -> None:
    """CLI entrypoint for configuration summary.

    Provides mutually exclusive options to summarize by a generation filter or by
    an explicit list of structure IDs in a JSON file.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser(
        description=(
            "Summarize composition and config_type counts for a generation or a list of IDs."
        )
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generation",
        type=int,
        help="Target generation number (e.g., 10).",
    )
    group.add_argument(
        "--ids",
        type=str,
        help=(
            "Path to JSON file containing a list of structure IDs to summarize."
        ),
    )
    ap.add_argument(
        "--per-generation",
        action="store_true",
        help="Also print a per-generation breakdown of config_type x composition.",
    )
    ap.add_argument(
        "--to-json",
        type=str,
        default=None,
        help="Write the full summary to a JSON file for downstream analysis.",
    )

    args = ap.parse_args()

    db = DatabaseManager()

    if args.generation is not None:
        structure_ids = db.find_structures_by_metadata(
            {"generation": args.generation}, operator="exact"
        )
        summarize_structure_ids(
            db,
            structure_ids,
            f"generation {args.generation}",
            per_generation=args.per_generation,
            to_json=args.to_json,
        )
        return

    if args.ids is not None:
        ids = load_ids_from_json(args.ids)
        summarize_structure_ids(
            db,
            ids,
            f"IDs file {args.ids}",
            per_generation=args.per_generation,
            to_json=args.to_json,
        )
        return


if __name__ == "__main__":
    main()


