import json
import glob
import tqdm
from collections import Counter
import numpy as np
from itertools import combinations
import re

from ase.io import iread
from ase.geometry import get_duplicate_atoms
from ase.neighborlist import NeighborList
from forge.core.database import DatabaseManager

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

CUTOFF_GLOBAL = 1.2  # Å

def check_geometry(atoms_iterator, desc=""):
    """
    Runs geometry sanity checks on an iterator of ASE Atoms objects.
    Returns a dictionary of failure reasons to lists of structure info dicts.
    """
    bad_structures = {
        "duplicates": [],
        "too_close": [],
    }
    for at in tqdm.tqdm(atoms_iterator, desc=desc):
        # (a) Check for duplicate atoms
        duplicates = get_duplicate_atoms(at, cutoff=CUTOFF_GLOBAL, delete=False)
        if duplicates.size > 0:
            bad_structures["duplicates"].append(at.info)
            continue
        # (b) Check for atoms too close
        nlist = NeighborList([CUTOFF_GLOBAL / 2] * len(at), self_interaction=False, bothways=True)
        nlist.update(at)
        if any(len(nlist.get_neighbors(i)[0]) for i in range(len(at))):
            bad_structures["too_close"].append(at.info)
            continue
    return bad_structures

def save_bad_structures_to_json(bad_ids_by_reason, suffix=""):
    """Saves the results of the geometry check to JSON files."""
    for reason, infos in bad_ids_by_reason.items():
        if infos:
            filename = f"bad_structures_{reason}{suffix}.json"
            with open(filename, 'w') as f:
                json.dump(infos, f, indent=2, cls=NumpyEncoder)
            print(f"Saved {len(infos)} bad structures for reason '{reason}' to {filename}")

def get_category_totals(atoms_list):
    """Calculates total counts for different categories from a list of atoms."""
    return {
        "generation": Counter(at.info.get('generation', 'N/A') for at in atoms_list),
        "config_type": Counter(at.info.get('config_type', 'N/A') for at in atoms_list),
        "composition": Counter(at.info.get('composition', 'N/A') for at in atoms_list),
    }

def analyze_bad_ids(bad_ids_by_reason, totals, title):
    """Prints a summary analysis of the bad structures."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    generation_totals = totals.get("generation", Counter())
    config_type_totals = totals.get("config_type", Counter())
    composition_totals = totals.get("composition", Counter())

    for reason, infos in bad_ids_by_reason.items():
        if not infos:
            print(f"\n--- Reason: {reason} (0 found) ---")
            continue
        print(f"\n--- Reason: {reason} ({len(infos)} found) ---")
        
        breakdowns = {
            "Generation": (Counter(info.get('generation', 'N/A') for info in infos), generation_totals),
            "Config Type": (Counter(info.get('config_type', 'N/A') for info in infos), config_type_totals),
            "Composition": (Counter(info.get('composition', 'N/A') for info in infos), composition_totals)
        }
        
        for name, (counts, totals_counts) in breakdowns.items():
            print(f"\n  By {name}:")
            for key, count in sorted(counts.items(), key=lambda item: str(item[0])):
                total_count = totals_counts.get(key, 0)
                percentage = (count / total_count * 100) if total_count > 0 else 0
                print(f"    - {key}: {count} / {total_count} ({percentage:.2f}%)")
    print("\n" + "="*80)

def analyze_id_overlap(sids_by_metric):
    """Analyzes and prints the overlap of structure IDs between different metrics."""
    metrics = list(sids_by_metric.keys())
    if len(metrics) < 2:
        return

    print("\n" + "="*80)
    print("Hotspot ID Overlap Analysis")
    print("="*80)
    print("Total unique IDs per metric:")
    for metric in metrics:
        print(f"- {metric}: {len(sids_by_metric[metric])}")

    print("\nIntersections:")
    for i in range(2, len(metrics) + 1):
        for combo in combinations(metrics, i):
            intersecting_set = set.intersection(*(sids_by_metric[m] for m in combo))
            label = " & ".join(combo)
            print(f"- {label}: {len(intersecting_set)} IDs")

    print("\nUnique to each metric:")
    for metric in metrics:
        others = set(metrics) - {metric}
        unique_set = sids_by_metric[metric].difference(*(sids_by_metric[o] for o in others))
        print(f"- Unique to {metric}: {len(unique_set)} IDs")
    print("\n" + "="*80)

def main():
    # --- Stage 1A: Analyze structures from individual local extxyz files ---
    print("--- Stage 1: Analyzing local hotspot extxyz files ---")
    extxyz_files = sorted(glob.glob('hotspot_structures_*.extxyz'))
    
    all_hotspot_sids = {}
    all_bad_ids_from_hotspots = set()

    if not extxyz_files:
        print("No 'hotspot_structures_*.extxyz' files found. Skipping Stage 1.")
    else:
        for f in extxyz_files:
            match = re.search(r'hotspot_structures_(\w+)\.extxyz', f)
            metric_name = match.group(1) if match else f
            print(f"\n--- Analyzing hotspot file for metric: {metric_name} ---")

            atoms_list = list(iread(f, ':'))
            current_sids = set()
            for at in atoms_list:
                at.info['composition'] = at.get_chemical_formula()
                sid = at.info.get('structure_id')
                if sid is not None:
                    current_sids.add(int(sid))
            
            all_hotspot_sids[metric_name] = current_sids

            totals = get_category_totals(atoms_list)
            bad_structures = check_geometry(atoms_list, desc=f"Checking {metric_name}")
            save_bad_structures_to_json(bad_structures, suffix=f"_from_{metric_name}")
            analyze_bad_ids(bad_structures, totals, f"Analysis for {metric_name.capitalize()} Hotspots")

            for reason in bad_structures:
                for info in bad_structures[reason]:
                    all_bad_ids_from_hotspots.add(int(info['structure_id']))

    # --- Stage 1B: Analyze overlap between hotspot file IDs ---
    analyze_id_overlap(all_hotspot_sids)

    # --- Stage 2: Analyze structures from the database ---
    print("\n--- Stage 2: Analyzing structures from database ---")
    db = DatabaseManager()
    
    print(f"Found {len(all_bad_ids_from_hotspots)} unique bad structure IDs in Stage 1 to exclude.")

    print("Fetching structure IDs from database...")
    structure_ids = db.find_structures_by_metadata(metadata_filters={'generation': 0}, operator='>=')
    dimer_ids = set(db.find_structures_by_metadata(metadata_filters={'config_type': 'dimer'}))
    
    print(f"Total structures (gen >= 0): {len(structure_ids)}")
    print(f"Dimer structures to exclude: {len(dimer_ids)}")

    ids_to_check = [
        sid for sid in structure_ids 
        if sid not in dimer_ids and sid not in all_bad_ids_from_hotspots
    ]

    print(f"Structures to check in database after filtering: {len(ids_to_check)}")
    
    if not ids_to_check:
         print("No new structures to check in the database.")
    else:
        db_structures = db.get_batch_atoms_with_calculation(structure_ids=ids_to_check)
        for at in db_structures:
            at.info['composition'] = at.get_chemical_formula()
        
        stage2_totals = get_category_totals(db_structures)
        bad_from_db = check_geometry(db_structures, desc="Checking DB structures")
        save_bad_structures_to_json(bad_from_db, suffix="_from_db")
        analyze_bad_ids(bad_from_db, stage2_totals, "Analysis of Bad Structures from Remainder of Database")

if __name__ == "__main__":
    main()
