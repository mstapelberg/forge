# combine_variances.py
import json
import os
from pathlib import Path
import ase
from ase.io import read, write
import numpy as np
from monty.json import MontyEncoder, MontyDecoder

def combine_variance_files(variance_dir: str, original_xyz: str, n_structures: int = 120, output_dir: str = None):
    """
    Combine all variance JSON files, select top N structures with highest variance,
    and create batch xyz files
    
    Args:
        variance_dir: Directory containing variance JSON files
        original_xyz: Path to original xyz file (to get the atoms objects)
        n_structures: Number of top structures to select
        output_dir: Directory to save combined results and structure batches
    """
    # Load all variance files
    all_results = []
    for file in Path(variance_dir).glob('*_variances.json'):
        with open(file, 'r') as f:
            results = json.load(f, cls=MontyDecoder)
            all_results.extend(results)
    
    # Sort by variance
    sorted_results = sorted(all_results, key=lambda x: x['variance'], reverse=True)
    
    # Select top N structures
    top_structures = sorted_results[:n_structures]
    
    # Load original xyz file
    all_atoms = read(original_xyz, ':')
    
    # Create mapping of structure names to atoms objects
    atoms_dict = {atoms.info['structure_name']: atoms for atoms in all_atoms}
    
    # Create batches (24 batches of 5 structures each)
    n_batches = 24
    structures_per_batch = n_structures // n_batches
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined results json for reference
        combined_file = os.path.join(output_dir, 'combined_variances.json')
        with open(combined_file, 'w') as f:
            summary_data = {
                'all_structures': len(all_results),
                'total_selected': len(top_structures),
                'structures_per_batch': structures_per_batch,
                'n_batches': n_batches,
                'top_structures': [{
                    'structure_name': s['structure_name'],
                    'variance': s['variance'],
                    'position_in_file': s['position_in_file']
                } for s in top_structures],
                'variance_statistics': {
                    'mean': float(np.mean([s['variance'] for s in all_results])),
                    'max': float(np.max([s['variance'] for s in all_results])),
                    'min': float(np.min([s['variance'] for s in all_results]))
                }
            }
            json.dump(summary_data, f, cls=MontyEncoder, indent=2)
        
        # Create batch xyz files
        for i in range(n_batches):
            start_idx = i * structures_per_batch
            end_idx = start_idx + structures_per_batch
            batch = top_structures[start_idx:end_idx]
            
            # Get atoms objects for this batch
            batch_atoms = [atoms_dict[s['structure_name']] for s in batch]
            
            # Write batch xyz file
            batch_file = os.path.join(output_dir, f'batch_{i}.xyz')
            write(batch_file, batch_atoms)
            
            # Save batch metadata
            batch_meta_file = os.path.join(output_dir, f'batch_{i}_meta.json')
            with open(batch_meta_file, 'w') as f:
                batch_data = {
                    'batch_id': i,
                    'structures': [{
                        'structure_name': s['structure_name'],
                        'variance': s['variance'],
                        'position_in_file': s['position_in_file']
                    } for s in batch]
                }
                json.dump(batch_data, f, cls=MontyEncoder, indent=2)
            
            print(f"Written batch {i} with {len(batch_atoms)} structures to {batch_file}")
            print(f"Batch metadata saved to {batch_meta_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine variance results and create batches')
    parser.add_argument('variance_dir', help='Directory containing variance JSON files')
    parser.add_argument('original_xyz', help='Path to original xyz file')
    parser.add_argument('--n_structures', type=int, default=120,
                      help='Number of top structures to select')
    parser.add_argument('--output_dir', help='Directory to save combined results')
    
    args = parser.parse_args()
    
    combine_variance_files(
        variance_dir=args.variance_dir,
        original_xyz=args.original_xyz,
        n_structures=args.n_structures,
        output_dir=args.output_dir
    )
