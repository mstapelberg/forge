import os
import json
import numpy as np
import ase.io
from pathlib import Path
import argparse
from typing import List, Dict
from tqdm import tqdm

# Import your classes
#from adversarial_calculator import AdversarialCalculator
#from displacement_generator import DisplacementGenerator
#from adversarial_optimizer import AdversarialOptimizer
import sys
sys.path.append('../../Modules')
from al_functions import AdversarialCalculator, DisplacementGenerator, AdversarialOptimizer

def calculate_structure_variances(
    model_paths: List[str],
    xyz_file: str,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Calculate force variances for all structures in xyz file.
    
    Args:
        model_paths: List of paths to model files
        xyz_file: Path to input xyz file
        device: Device to run calculations on
        
    Returns:
        Dictionary mapping structure names to their force variances
    """
    # Initialize calculator
    calculator = AdversarialCalculator(
        model_paths=model_paths,
        device=device
    )
    
    # Load all frames
    print("Loading structures...")
    atoms_list = ase.io.read(xyz_file, ':')
    
    # Calculate variances
    variances = {}
    print("Calculating force variances...")
    for i, atoms in enumerate(tqdm(atoms_list)):
        # Ensure structure name exists
        if 'structure_name' not in atoms.info:
            atoms.info['structure_name'] = f'structure_{i}'
            
        # Calculate variance
        forces = calculator.calculate_forces(atoms)
        atom_variances = calculator.calculate_normalized_force_variance(forces)
        # Take mean over all atoms to get single structure variance
        structure_variance = float(np.mean(atom_variances))
        
        variances[atoms.info['structure_name']] = structure_variance
        
    return variances

def run_adversarial_attacks(
    model_paths: List[str],
    xyz_file: str,
    output_dir: str,
    n_structures: int,
    temperature: float = 1200,
    device: str = 'cpu'
):
    """
    Run complete adversarial attack workflow.
    
    Args:
        model_paths: List of paths to model files
        xyz_file: Path to input xyz file
        output_dir: Directory to save results
        n_structures: Number of highest-variance structures to select
        temperature: Temperature for adversarial optimization
        device: Device to run calculations on
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate variances for all structures
    variances = calculate_structure_variances(model_paths, xyz_file, device)
    
    # Save variances to json
    variance_file = output_dir / 'structure_variances.json'
    with open(variance_file, 'w') as f:
        json.dump(variances, f, indent=2)
        
    # Print variance summary
    print("\nVariance Summary:")
    print(f"Mean variance: {np.mean(list(variances.values())):.6f}")
    print(f"Max variance: {np.max(list(variances.values())):.6f}")
    print(f"Min variance: {np.min(list(variances.values())):.6f}")
    
    # Select N structures with highest variance
    sorted_structures = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    selected_structures = sorted_structures[:n_structures]
    
    print(f"\nSelected {n_structures} structures for adversarial attacks:")
    for name, variance in selected_structures:
        print(f"{name}: {variance:.6f}")
    
    # Initialize calculator and optimizer
    calculator = AdversarialCalculator(model_paths=model_paths, device=device)
    optimizer = AdversarialOptimizer(calculator)
    
    # Load all structures (we'll need to match names)
    atoms_list = ase.io.read(xyz_file, ':')
    atoms_dict = {atoms.info['structure_name']: atoms for atoms in atoms_list}
    
    # Run adversarial optimization on selected structures
    print("\nRunning adversarial attacks...")
    for struct_name, initial_variance in tqdm(selected_structures):
        # Get structure
        atoms = atoms_dict[struct_name]
        
        # Create output name
        output_name = f"{struct_name}_{temperature}K_all.xyz"
        output_path = output_dir / output_name
        
        print(f"\nOptimizing structure: {struct_name}")
        print(f"Initial variance: {initial_variance:.6f}")
        
        # Run optimization
        best_atoms, best_variance, accepted = optimizer.optimize(
            atoms=atoms,
            temperature=temperature,
            mode='all',
            output_dir=output_dir
        )

def main():
    parser = argparse.ArgumentParser(description='Run adversarial attacks on structures')
    parser.add_argument('xyz_file', help='Input xyz file')
    parser.add_argument('output_dir', help='Directory to save results')
    parser.add_argument('--model_paths', nargs='+', required=True,
                      help='Paths to model files')
    parser.add_argument('--n_structures', type=int, default=5,
                      help='Number of highest-variance structures to select')
    parser.add_argument('--temperature', type=float, default=1200,
                      help='Temperature for adversarial optimization')
    parser.add_argument('--device', default='cpu',
                      help='Device to run calculations on')
    
    args = parser.parse_args()
    
    run_adversarial_attacks(
        model_paths=args.model_paths,
        xyz_file=args.xyz_file,
        output_dir=args.output_dir,
        n_structures=args.n_structures,
        temperature=args.temperature,
        device=args.device
    )
    """
    model_paths = ['../Models/zr-w-v-ti-cr/gen_1_2024-11-09/gen_1_model_3-11-09_stagetwo_compiled.model',
                   '../Models/zr-w-v-ti-cr/gen_1_2024-11-09/gen_1_model_4-11-09_stagetwo_compiled.model',
                   '../Models/zr-w-v-ti-cr/gen_1_2024-11-09/gen_1_model_5-11-09_stagetwo_compiled.model']
    xyz_file = '../data/zr-w-v-ti-cr/gen_1_2024-11-09/gen_1_2024-11-09_cum_reduced.xyz'
    output_dir = '../data/zr-w-v-ti-cr/gen_1_2024-11-09/aa_out'
    # if the output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_structures = 15
    
    temperature = 1200

    device = 'cuda'

    run_adversarial_attacks(
        model_paths=model_paths,
        xyz_file=xyz_file,
        output_dir=output_dir,
        n_structures=n_structures,
        temperature=temperature,
        device=device
    )
    """

if __name__ == "__main__":
    main()
