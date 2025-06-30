import json
import glob
from pathlib import Path
import argparse
import pandas as pd

def load_ids_from_old_json(glob_pattern: str) -> set:
    """Loads structure IDs from multiple old-format JSON files."""
    old_ids = set()
    file_paths = glob.glob(glob_pattern, recursive=True)
    
    if not file_paths:
        print(f"Warning: No files found for glob pattern: {glob_pattern}")
        return old_ids

    print(f"Found {len(file_paths)} files from old script to process.")
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Data is a list of dicts, each with 'structure_id'
                for item in data:
                    if 'structure_id' in item:
                        old_ids.add(int(item['structure_id']))
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")
            
    return old_ids

def load_ids_from_new_json(file_path: str) -> set:
    """Loads a simple list of IDs from a new-format JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return set(int(i) for i in data)
    except FileNotFoundError:
        print(f"Error: New IDs file not found at {file_path}")
        return set()
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error: Could not decode new IDs file {file_path}. Error: {e}")
        return set()

def main():
    parser = argparse.ArgumentParser(
        description="Compare bad structure IDs from the old and new analysis scripts."
    )
    parser.add_argument(
        '--new-ids-file', type=str, required=True,
        help='Path to the JSON file with bad IDs from the new script (e.g., analysis_output_full/bad_structure_ids.json).'
    )
    parser.add_argument(
        '--new-rare-ids-file', type=str, required=True,
        help='Path to the JSON file with rare IDs from the new script (e.g., analysis_output_full/rare_structure_ids.json).'
    )
    parser.add_argument(
        '--old-ids-glob', type=str, required=True,
        help='Glob pattern to find all old JSON files (e.g., "old_results/**/bad_structures_*.json").'
    )
    args = parser.parse_args()


    from forge.core.database import DatabaseManager
    db = DatabaseManager()

    # Load IDs
    print(f"Loading new IDs from: {args.new_ids_file}")
    new_ids = load_ids_from_new_json(args.new_ids_file)

    new_ids_atoms = db.get_batch_atoms_with_calculation(new_ids)

    new_ids_configs = [atoms.info['config_type'] for atoms in new_ids_atoms]
    new_ids_compositions = [atoms.get_chemical_formula() for atoms in new_ids_atoms]
    new_ids_generations = [atoms.info['generation'] for atoms in new_ids_atoms]

    # get the unique config types, compositions, and generations
    unique_configs = list(set(new_ids_configs))
    unique_compositions = list(set(new_ids_compositions))
    unique_generations = list(set(new_ids_generations))

    print(f"Unique config types: {unique_configs}")
    print(f"Unique compositions: {unique_compositions}")
    print(f"Unique generations: {unique_generations}")


    print(f"Loading new rare IDs from: {args.new_rare_ids_file}")
    new_rare_ids = load_ids_from_new_json(args.new_rare_ids_file)
    print(f"\nLoading old IDs with pattern: {args.old_ids_glob}")
    old_ids = load_ids_from_old_json(args.old_ids_glob)

    if not new_ids and not old_ids:
        print("\nNo IDs found from either source. Exiting.")
        return

    # Perform comparison
    intersection = new_ids.intersection(old_ids)
    intersection_rare = new_rare_ids.intersection(old_ids)
    only_in_new = new_ids.difference(old_ids)
    only_in_new_rare = new_rare_ids.difference(old_ids)
    only_in_old = old_ids.difference(new_ids)
    only_in_old_rare = old_ids.difference(new_rare_ids.union(new_ids))

    # Print report
    print("\n" + "="*80)
    print("Bad Structure ID Comparison Report")
    print("="*80)
    print(f"{'Total unique IDs from old script:':<40} {len(old_ids)}")
    print(f"{'Total IDs from new script:':<40} {len(new_ids)}")
    print("-" * 80)
    print(f"{'IDs found in BOTH old and new:':<40} {len(intersection)}")
    print(f"{'IDs found ONLY in new script:':<40} {len(only_in_new)}")
    print(f"{'IDs found ONLY in old script:':<40} {len(only_in_old)}")
    print(f"{'IDs found in BOTH old and new (rare):':<40} {len(intersection_rare)}")
    print(f"{'IDs found ONLY in new (rare):':<40} {len(only_in_new_rare)}")
    print(f"{'IDs found ONLY in old (rare):':<40} {len(only_in_old_rare)}")
    print("="*80)

    if only_in_old:
        print(f"\n--- {len(only_in_old)} IDs found only in the OLD script (potential misses by new script) ---")
        print(sorted(list(only_in_old)))
    
    if only_in_new:
        print(f"\n--- {len(only_in_new)} IDs found only in the NEW script (potential new finds) ---")
        print(sorted(list(only_in_new)))
    
    if only_in_new_rare:
        print(f"\n--- {len(only_in_new_rare)} IDs found only in the NEW script (rare) (potential new finds) ---")
        print(sorted(list(only_in_new_rare)))
    
    if only_in_old_rare:
        print(f"\n--- {len(only_in_old_rare)} IDs found only in the OLD script (but not in newrare) (potential misses by new script) ---")

if __name__ == "__main__":
    main() 