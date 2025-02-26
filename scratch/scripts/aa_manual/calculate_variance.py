# calculate_variance.py
import os
import numpy as np
import ase.io
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('../../Modules')
from al_functions import AdversarialCalculator
from monty.json import MontyEncoder, MontyDecoder

def calculate_and_save_variances(
    model_paths: list,
    xyz_file: str,
    output_dir: str,
    device: str = 'cuda'
):
    """
    Calculate force variances for structures and save results
    
    Args:
        model_paths: List of paths to model files
        xyz_file: Path to input xyz file
        output_dir: Directory to save results
        device: Device to run calculations on
    """
    # Initialize calculator
    calculator = AdversarialCalculator(
        model_paths=model_paths,
        device=device
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name of input file for naming output
    base_name = Path(xyz_file).stem
    
    # Load all frames
    print(f"Loading structures from {xyz_file}...")
    atoms_list = ase.io.read(xyz_file, ':')
    
    # Calculate variances and store results
    results = []
    print("Calculating force variances...")
    for i, atoms in enumerate(tqdm(atoms_list)):
        if 'structure_name' not in atoms.info:
            atoms.info['structure_name'] = f'structure_{i}'
            
        # Calculate forces and variance
        forces = calculator.calculate_forces(atoms)
        atom_variances = calculator.calculate_normalized_force_variance(forces)
        structure_variance = float(np.mean(atom_variances))
        
        # Store results 
        result = {
            'structure_name': atoms.info['structure_name'],
            'variance': structure_variance,
            'position_in_file': i
        }
        results.append(result)
    
    # Save results using monty
    import json
    output_file = os.path.join(output_dir, f'{base_name}_variances.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, cls=MontyEncoder, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Processed {len(atoms_list)} structures")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate force variances for structures')
    parser.add_argument('xyz_file', help='Input xyz file')
    parser.add_argument('output_dir', help='Directory to save results')
    parser.add_argument('--model_paths', nargs='+', required=True,
                      help='Paths to model files')
    parser.add_argument('--device', default='cuda',
                      help='Device to run calculations on')
    
    args = parser.parse_args()
    
    calculate_and_save_variances(
        model_paths=args.model_paths,
        xyz_file=args.xyz_file,
        output_dir=args.output_dir,
        device=args.device
    )
