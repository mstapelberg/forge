# combine_variances.py
import json
import os
from pathlib import Path
import ase
from ase.atoms import Atoms
import numpy as np

def combine_variance_files(variance_dir: str, n_structures: int = 120, output_dir: str = None):
    """
    Combine all variance JSON files and select top N structures with highest variance
    
    Args:
        variance_dir: Directory containing variance JSON files
        n_structures: Number of top structures to select
        output_dir: Directory to save combined results and structure batches
    """
    # Load all variance files
    all_results = []
    for file in Path(variance_dir).glob('*_variances.json'):
        with open(file, 'r') as f:
            results = json.load(f)
            all_results.extend(results)
    
    # Sort by variance
    sorted_results = sorted(all_results, key=lambda x: x['variance'], reverse=True)
    
    # Select top N structures
    top_structures = sorted_results[:n_structures]
    
    # Create batches (24 batches of 5 structures each)
    n_batches = 24
    structures_per_batch = n_structures // n_batches
    batches = []
    
    for i in range(n_batches):
        start_idx = i * structures_per_batch
        end_idx = start_idx + structures_per_batch
        batch = top_structures[start_idx:end_idx]
        batches.append(batch)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined results
        combined_file = os.path.join(output_dir, 'combined_variances.json')
        with open(combined_file, 'w') as f:
            json.dump({
                'all_structures': len(all_results),
                'top_structures': top_structures
            }, f, indent=2)
        
        # Save batch assignments
        for i, batch in enumerate(batches):
            batch_file = os.path.join(output_dir, f'batch_{i}_structures.json')
            with open(batch_file, 'w') as f:
                json.dump(batch, f, indent=2)
    
    return batches

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine variance results and create batches')
    parser.add_argument('variance_dir', help='Directory containing variance JSON files')
    parser.add_argument('--n_structures', type=int, default=120,
                      help='Number of top structures to select')
    parser.add_argument('--output_dir', help='Directory to save combined results')
    
    args = parser.parse_args()
    
    combine_variance_files(
        variance_dir=args.variance_dir,
        n_structures=args.n_structures,
        output_dir=args.output_dir
    )
