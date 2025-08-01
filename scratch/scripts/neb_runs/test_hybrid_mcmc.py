"""
Test script for HybridMCMCSampler.

This script creates a simple alloy structure, runs the hybrid MCMC-MD optimization,
and compares before/after states.

Note: Uses EMT calculator for testing (supports Cu, Ag, Au, Ni, Pd, Pt, Al).
In production, use your MLIP calculator.

Confidence: 9/10
"""

import numpy as np
from ase.build import bulk
from ase.io import write
from nequip.ase import NequIPCalculator
from forge.workflows.hybrid_mcmc import HybridMCMCSampler
from typing import List, Tuple
from ase import Atoms

package_path = '/home/myless/Packages/forge/scratch/data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip'

def create_test_structure(
    base_element: str = 'V',
    alloy_elements: List[str] = ['Cr', 'Ti', 'W', 'Zr'],
    dimensions: Tuple[int, int, int] = (8, 8, 8),
    lattice_constant: float = 3.01,
    seed: int = 42
) -> Atoms:
    """
    Create a random alloy supercell for testing.

    Args:
        base_element: Base element for bulk structure
        alloy_elements: List of elements to randomly assign
        dimensions: Supercell repeat dimensions
        lattice_constant: Lattice parameter
        seed: Random seed

    Returns:
        ASE Atoms object with random alloy composition
    """
    rng = np.random.default_rng(seed)
    atoms = bulk(base_element, 'bcc', a=lattice_constant).repeat(dimensions)
    n_atoms = len(atoms)
    
    # Randomly assign symbols
    symbols = rng.choice(alloy_elements, size=n_atoms)
    for i in range(n_atoms):
        atoms[i].symbol = symbols[i]
    
    return atoms

def main():
    """Run hybrid MCMC-MD test."""
    # Create test structure
    atoms = create_test_structure()
    
    # Set up simple calculator (replace with your MLIP in production)
    calculator = NequIPCalculator._from_packaged_model(
        package_path=package_path,
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4},
        device='cuda'
    )
    
    # Print initial info
    atoms.calc = calculator
    initial_energy = atoms.get_potential_energy()
    print("Initial structure:")
    print(f"Formula: {atoms.get_chemical_formula()} - {len(atoms)} atoms")
    print(f"Energy: {initial_energy:.3f} eV")
    write('initial_structure.xyz', atoms)
    
    # Initialize sampler
    sampler = HybridMCMCSampler(
        atoms=atoms,
        calculator=calculator,
        temperature=1000.0,
        md_temperature=873.15,  # Slightly different for testing
        steps=10,  # Small number for quick test
        md_steps_per_cycle=min(20*len(atoms), 1000),
        mc_steps_per_cycle=min(50*len(atoms), 5000),
        final_cell_relax=True,  # Do final cell relaxation after convergence
        md_timestep=1.0,
        md_thermostat='langevin',
        friction=0.02,
        rng_seed=42
    )
    
    # Run optimization
    print("\nRunning hybrid MCMC-MD...")
    optimized_atoms = sampler.run_hybrid_mcmc(
        convergence_window=100,
        energy_threshold=0.002
    )
    
    # Print final info
    final_energy = optimized_atoms.get_potential_energy()
    print("\nFinal structure:")
    print(f"Formula: {optimized_atoms.get_chemical_formula()}")
    print(f"Energy: {final_energy:.3f} eV")
    print(f"Energy change: {final_energy - initial_energy:.3f} eV")
    write('optimized_structure.xyz', optimized_atoms)
    
    # If tracker was used (add tracker_settings to sampler init if desired)
    if sampler.tracker:
        sampler.tracker.plot_results(save_dir='test_results')

if __name__ == '__main__':
    main() 