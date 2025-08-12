#!/usr/bin/env python3
"""
Analyze bad structure IDs from ensemble stats analysis.

This script loads bad structure IDs from a JSON file, queries the database
for actual ASE atoms objects, and shows breakdowns by generation, config_type, 
composition, and dataset split.

Usage:
  python analyze_bad_structures.py --bad-ids bad_structure_ids.json --splits structure_splits.json --detailed detailed.json --output analysis.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Optional

from forge.core.database import DatabaseManager


def load_bad_structure_ids(bad_ids_path: Path) -> List[int]:
    """Load bad structure IDs and convert to integers.

    Args:
        bad_ids_path: Path to JSON file containing a list of structure IDs (as strings or ints).

    Returns:
        List of structure IDs as integers.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    with open(bad_ids_path, 'r') as f:
        string_ids = json.load(f)
    return [int(sid) for sid in string_ids]


def load_structure_splits(splits_path: Path) -> Dict[str, Set[str]]:
    """Load structure splits mapping.

    Args:
        splits_path: Path to JSON file mapping split names to arrays of IDs.

    Returns:
        Dictionary mapping split name to a set of string IDs.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    with open(splits_path, 'r') as f:
        data = json.load(f)
    return {split: set(str(x) for x in ids) for split, ids in data.items()}


def load_detailed_bad_structures(detailed_path: Path) -> Dict[str, Dict]:
    """Load detailed bad structures data to get reasons.

    Args:
        detailed_path: Path to JSON file mapping structure ID (as string) to detail dicts.

    Returns:
        Dictionary of detailed records keyed by structure ID string.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    with open(detailed_path, 'r') as f:
        return json.load(f)


def _fetch_db_totals(db: DatabaseManager) -> Dict[str, Dict]:
    """Compute global totals across the entire database for key categories.

    Categories returned:
    - generation: Count of structures per generation (int) from `metadata->>'generation'`.
    - config_type: Count of structures per config_type (str) from `metadata->>'config_type'`.
                    Missing/null treated as 'UNKNOWN'.
    - formula: Count of structures per composition string from `formula` column.
    - natoms: Count of structures per number of atoms from `jsonb_array_length(positions)`.

    Args:
        db: Initialized `DatabaseManager` with an active connection.

    Returns:
        Dictionary with keys 'generation', 'config_type', 'formula', 'natoms', each
        mapping to a dictionary of totals.
    """
    totals: Dict[str, Dict] = {"generation": {}, "config_type": {}, "formula": {}, "natoms": {}}

    # Generation totals (guard against non-integer values)
    with db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT (metadata->>'generation')::int AS gen, COUNT(*)
            FROM structures
            WHERE (metadata->>'generation') ~ '^\\d+$'
            GROUP BY gen
            """
        )
        for gen, cnt in cur.fetchall():
            totals["generation"][int(gen)] = int(cnt)

    # Config type totals; missing/null -> 'UNKNOWN'
    with db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(metadata->>'config_type', 'UNKNOWN') AS cfg, COUNT(*)
            FROM structures
            GROUP BY cfg
            """
        )
        for cfg, cnt in cur.fetchall():
            totals["config_type"][str(cfg)] = int(cnt)

    # Composition (formula) totals
    with db.conn.cursor() as cur:
        cur.execute("SELECT formula, COUNT(*) FROM structures GROUP BY formula")
        for formula, cnt in cur.fetchall():
            totals["formula"][str(formula)] = int(cnt)

    # Number of atoms totals
    with db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT jsonb_array_length(positions) AS natoms, COUNT(*)
            FROM structures
            GROUP BY natoms
            """
        )
        for natoms, cnt in cur.fetchall():
            totals["natoms"][int(natoms)] = int(cnt)

    return totals


def _compute_split_totals(db: DatabaseManager, splits_map: Dict[str, Set[str]]) -> Dict[str, int]:
    """Compute total counts per dataset split, plus an 'unknown' bucket.

    Args:
        db: Initialized `DatabaseManager`.
        splits_map: Mapping of split name to a set of string structure IDs.

    Returns:
        Dictionary mapping each split name (and 'unknown') to total counts across the DB.
        For splits present in `splits_map`, totals are simply the length of that set.
        The 'unknown' total is computed as (all DB structures) - (union of all split IDs).
    """
    # Known split totals directly from provided mapping
    split_totals: Dict[str, int] = {name: len(ids) for name, ids in splits_map.items()}

    # Compute 'unknown' as DB_total - union_of_known
    try:
        all_ids = set(db.get_all_structure_ids())
    except Exception:
        # Fallback: if fetching all IDs fails, set unknown to 0 rather than crashing
        all_ids = set()
    known_ids: Set[int] = set()
    for ids in splits_map.values():
        for sid in ids:
            try:
                known_ids.add(int(sid))
            except Exception:
                # Ignore non-integer IDs in mapping
                continue
    unknown_total = max(0, len(all_ids - known_ids)) if all_ids else 0
    split_totals["unknown"] = unknown_total
    return split_totals


def _format_ratio(count: int, total: Optional[int]) -> str:
    """Format count/total with percentage.

    Args:
        count: Count in the subset (e.g., bad structures).
        total: Corresponding total in the full dataset; can be None.

    Returns:
        Formatted string like "count / total (X%)". If total is falsy, returns
        "count / 0 (N/A)".
    """
    if total and total > 0:
        pct = 100.0 * float(count) / float(total)
        return f"{count} / {total} ({pct:.1f}%)"
    return f"{count} / 0 (N/A)"


def analyze_bad_structures(bad_ids: List[int], splits_map: Dict[str, Set[str]], detailed_data: Dict[str, Dict], output_file: Path) -> None:
    """Analyze bad structures and write detailed breakdowns to file.

    Enriches each reported count with its total across the entire dataset and
    the corresponding percentage (count/total).

    Args:
        bad_ids: Structure IDs flagged as bad.
        splits_map: Mapping of dataset splits to structure ID sets (strings).
        detailed_data: Mapping of bad structure IDs (as strings) to detail dicts, including 'reasons'.
        output_file: Path to write the analysis report.
    """
    db = DatabaseManager()
    
    print(f"[INFO] Analyzing {len(bad_ids)} bad structures")
    
    # Query database for actual ASE atoms objects
    atoms_map = db.get_structures_batch(bad_ids)
    print(f"[INFO] Found {len(atoms_map)} structures with ASE atoms")

    # Fetch global totals from DB and totals per split
    db_totals = _fetch_db_totals(db)
    split_totals = _compute_split_totals(db, splits_map)
    
    # Track counts
    generations: List[int] = []
    config_types: List[str] = []
    compositions: List[str] = []
    num_atoms_list: List[int] = []
    split_counts: Dict[str, int] = defaultdict(int)
    reason_counts: Dict[str, int] = defaultdict(int)
    
    # Per-generation tracking
    gen_compositions: Dict[int, List[str]] = defaultdict(list)
    gen_config_types: Dict[int, List[str]] = defaultdict(list)
    
    # Analyze each structure
    for sid in bad_ids:
        sid_str = str(sid)
        atoms = atoms_map.get(sid)
        detailed = detailed_data.get(sid_str, {})
        
        if atoms is None:
            print(f"[WARNING] No atoms found for structure {sid}")
            continue
        
        # Get generation from atoms.info
        gen = atoms.info.get("generation")
        if gen is not None:
            gen = int(gen)
            generations.append(gen)
            
            # Track per-generation data
            comp = atoms.get_chemical_formula()
            gen_compositions[gen].append(comp)
            
            cfg = atoms.info.get("config_type", "UNKNOWN")
            gen_config_types[gen].append(cfg)
        
        # Get config_type from atoms.info
        cfg = atoms.info.get("config_type", "UNKNOWN")
        config_types.append(cfg)
        
        # Get composition from ASE atoms
        comp = atoms.get_chemical_formula()
        compositions.append(comp)
        
        # Get number of atoms from ASE atoms
        num_atoms_list.append(len(atoms))
        
        # Get dataset split
        split_found = False
        for split_name, ids in splits_map.items():
            if sid_str in ids:
                split_counts[split_name] += 1
                split_found = True
                break
        if not split_found:
            split_counts["unknown"] += 1
        
        # Get reasons
        reasons = detailed.get("reasons", "")
        if reasons:
            for reason in reasons.split(","):
                reason = reason.strip()
                if reason:
                    reason_counts[reason] += 1
    
    # Write detailed analysis to file
    with open(output_file, 'w') as f:
        f.write("BAD STRUCTURE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total bad structures: {len(bad_ids)}\n")
        f.write(f"Structures with ASE atoms: {len(atoms_map)}\n\n")
        
        # Generation breakdown
        if generations:
            gen_counts = Counter(generations)
            f.write(f"[GENERATION] Breakdown ({len(generations)} structures):\n")
            for gen, cnt in sorted(gen_counts.items()):
                total_gen = db_totals["generation"].get(gen, 0)
                f.write(f"  - Generation {gen}: {_format_ratio(cnt, total_gen)}\n")
            f.write("\n")
            
            # Per-generation detailed breakdowns
            f.write("[PER-GENERATION BREAKDOWNS]\n")
            f.write("-" * 30 + "\n\n")
            
            for gen in sorted(gen_compositions.keys()):
                f.write(f"Generation {gen} ({len(gen_compositions[gen])} structures):\n")
                
                # Composition breakdown for this generation
                comp_counts = Counter(gen_compositions[gen])
                f.write(f"  Compositions:\n")
                for comp, cnt in comp_counts.most_common():
                    total_comp = db_totals["formula"].get(comp, 0)
                    f.write(f"    - {comp}: {_format_ratio(cnt, total_comp)}\n")
                
                # Config type breakdown for this generation
                cfg_counts = Counter(gen_config_types[gen])
                f.write(f"  Config Types:\n")
                for cfg, cnt in cfg_counts.most_common():
                    total_cfg = db_totals["config_type"].get(cfg, 0)
                    f.write(f"    - {cfg}: {_format_ratio(cnt, total_cfg)}\n")
                f.write("\n")
        
        # Overall config type breakdown
        cfg_counts = Counter(config_types)
        f.write(f"[OVERALL CONFIG_TYPE] Breakdown ({len(config_types)} structures):\n")
        for cfg, cnt in cfg_counts.most_common():
            total_cfg = db_totals["config_type"].get(cfg, 0)
            f.write(f"  - {cfg}: {_format_ratio(cnt, total_cfg)}\n")
        f.write("\n")
        
        # Overall composition breakdown
        comp_counts = Counter(compositions)
        f.write(f"[OVERALL COMPOSITION] Breakdown ({len(compositions)} structures):\n")
        for comp, cnt in comp_counts.most_common():
            total_comp = db_totals["formula"].get(comp, 0)
            f.write(f"  - {comp}: {_format_ratio(cnt, total_comp)}\n")
        f.write("\n")
        
        # Dataset split breakdown
        f.write(f"[DATASET_SPLIT] Breakdown ({len(bad_ids)} structures):\n")
        for split, cnt in sorted(split_counts.items()):
            total_split = split_totals.get(split, 0)
            f.write(f"  - {split}: {_format_ratio(cnt, total_split)}\n")
        f.write("\n")
        
        # Reasons breakdown
        f.write(f"[REASONS] Breakdown ({len(bad_ids)} structures):\n")
        reason_counter = Counter(reason_counts)
        for reason, cnt in reason_counter.most_common():
            # No reliable DB-wide baseline for 'reasons' (sourced from detailed JSON of bad structures)
            f.write(f"  - {reason}: {cnt}\n")
        f.write("\n")
        
        # Number of atoms summary
        if num_atoms_list:
            f.write(f"[NUM_ATOMS] Summary ({len(num_atoms_list)} structures):\n")
            f.write(f"  - Min: {min(num_atoms_list)}\n")
            f.write(f"  - Max: {max(num_atoms_list)}\n")
            f.write(f"  - Mean: {sum(num_atoms_list) / len(num_atoms_list):.1f}\n")
            
            # Show distribution
            natom_counts = Counter(num_atoms_list)
            f.write(f"  - Most common:\n")
            for natoms, cnt in natom_counts.most_common(10):
                total_natoms = db_totals["natoms"].get(natoms, 0)
                f.write(f"    {natoms} atoms: {_format_ratio(cnt, total_natoms)}\n")
    
    # Also print summary to console
    print(f"\n[INFO] Analysis written to: {output_file}")
    print(f"[INFO] Found {len(atoms_map)} structures with ASE atoms")
    print(f"[INFO] Split distribution: {dict(split_counts)}")
    reason_counter = Counter(reason_counts)
    print(f"[INFO] Top reasons: {dict(reason_counter.most_common(5))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bad structure IDs from ensemble stats")
    parser.add_argument("--bad-ids", type=Path, required=True, help="Path to bad_structure_ids.json")
    parser.add_argument("--splits", type=Path, required=True, help="Path to structure_splits.json")
    parser.add_argument("--detailed", type=Path, required=True, help="Path to bad_structures_detailed.json")
    parser.add_argument("--output", type=Path, default=Path("bad_structures_analysis.txt"), help="Output file path")
    
    args = parser.parse_args()
    
    # Load data
    bad_ids = load_bad_structure_ids(args.bad_ids)
    splits_map = load_structure_splits(args.splits)
    detailed_data = load_detailed_bad_structures(args.detailed)
    
    # Analyze
    analyze_bad_structures(bad_ids, splits_map, detailed_data, args.output)


if __name__ == "__main__":
    main()