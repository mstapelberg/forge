import numpy as np
from ase import Atoms 
from ase.io import read
from nequip.ase import NequIPCalculator
from mace.calculators.mace import MACECalculator 
import numpy as np
from tqdm import tqdm
import os
import ase.io
from pathlib import Path
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from scipy.spatial.distance import pdist, squareform
import copy

class AdversarialCalculator:
    def __init__(self, model_paths, calculator_type='mace', device='cpu', species=None, default_dtype='float32'):
        """
        Initialize the calculator with either single or ensemble of models.
        
        Args:
            model_paths (str or list): Path(s) to model file(s)
            calculator_type (str): Type of calculator ('mace' or 'allegro')
            device (str): Device to use ('cpu' or 'cuda')
            species (dict): Dictionary mapping species to type names (for Allegro)
            default_dtype (str): Default data type for calculations
        """
        self.calculator_type = calculator_type
        self.device = device
        self.default_dtype = default_dtype
        
        if isinstance(model_paths, str):
            self.is_ensemble = False
            self.model_paths = [model_paths]
        else:
            self.is_ensemble = True
            self.model_paths = model_paths
            
        self.calculators = self._initialize_calculators(species)
        
    def _initialize_calculators(self, species):
        """Initialize single calculator or ensemble of calculators."""
        if self.calculator_type == 'mace':
            return MACECalculator(model_paths=self.model_paths, 
                                device=self.device, 
                                default_dtype=self.default_dtype)
        elif self.calculator_type == 'allegro':
            return [NequIPCalculator.from_deployed_model(
                model_path=model_path,
                species_to_type_name=species,
                device=self.device
            ) for model_path in self.model_paths]
        else:
            raise ValueError(f"Unknown calculator type: {self.calculator_type}")
            
    def calculate_forces(self, atoms):
        """Calculate forces using single calculator or ensemble."""
        if self.calculator_type == 'mace':
            atoms.calc = self.calculators
            atoms.get_potential_energy()  # Initialize the calculator
            return atoms.calc.results['forces_comm']
        else:
            forces_comm = []
            for calc in self.calculators:
                atoms_copy = atoms.copy()
                atoms_copy.calc = calc
                forces = atoms_copy.get_forces()
                forces_comm.append(forces)
            return np.array(forces_comm)
            
    def calculate_normalized_force_variance(self, forces):
        """Calculate normalized force variance across ensemble predictions."""
        force_magnitudes = np.linalg.norm(forces, axis=2, keepdims=True)
        normalized_forces = np.where(force_magnitudes != 0, 
                                   forces / force_magnitudes, 
                                   0)
        atom_variances = np.var(normalized_forces, axis=0)
        total_atom_variances = np.sum(atom_variances, axis=1)
        return total_atom_variances
        
    def calculate_structure_disagreement(self, forces, atom_disagreement):
        """Calculate structure disagreement metrics."""
        force_magnitudes = np.mean(np.linalg.norm(forces, axis=2), axis=0)
        mean_disagreement = np.mean(atom_disagreement)
        max_disagreement = np.max(atom_disagreement)
        weighted_disagreement = np.sum(atom_disagreement * force_magnitudes) / np.sum(force_magnitudes)
        
        return {
            "mean_disagreement": mean_disagreement,
            "max_disagreement": max_disagreement,
            "weighted_disagreement": weighted_disagreement
        }
            
    def calculate_per_atom_variance(self, forces):
        """
        Calculate the normalized force variance for each atom across ensemble predictions.
        
        Args:
            forces (np.ndarray): Forces array of shape (n_models, n_atoms, 3)
                               from calculate_forces()
        
        Returns:
            np.ndarray: Array of shape (n_atoms,) containing the normalized variance
                       for each atom's forces across all models
        """
        # Calculate force magnitude for each atom in each model's prediction
        # Shape: (n_models, n_atoms, 1)
        force_magnitudes = np.linalg.norm(forces, axis=2, keepdims=True)
        
        # Normalize forces by their magnitudes
        # Shape: (n_models, n_atoms, 3)
        normalized_forces = np.where(force_magnitudes != 0,
                                   forces / force_magnitudes,
                                   0)
        
        # Calculate variance across models for each atom and direction
        # Shape: (n_atoms, 3)
        per_direction_variances = np.var(normalized_forces, axis=0)
        
        # Sum variances across directions to get single variance per atom
        # Shape: (n_atoms,)
        per_atom_variances = np.sum(per_direction_variances, axis=1)
        
        return per_atom_variances

class DisplacementGenerator:
    def __init__(self, min_distance=2.0):
        """
        Initialize the displacement generator.
        
        Args:
            min_distance (float): Minimum allowed distance between atoms in Angstroms
        """
        self.min_distance = min_distance
        
    def _check_overlaps(self, positions, cell, pbc):
        """
        Check if any atoms are closer than min_distance.
        
        Args:
            positions (np.ndarray): Atomic positions (N x 3)
            cell (np.ndarray): Unit cell (3 x 3)
            pbc (bool or list): Periodic boundary conditions
            
        Returns:
            bool: True if no overlaps, False if overlaps exist
        """
        # Get all pairwise distances
        distances = squareform(pdist(positions))
        # Set diagonal to infinity (distance to self)
        np.fill_diagonal(distances, np.inf)
        
        # Check if any distance is less than minimum
        return np.all(distances > self.min_distance)
    
    def _apply_mb_displacement(self, atoms, temperature, single_atom_idx=None, max_attempts=100):
        """
        Apply Maxwell-Boltzmann displacement with overlap checking.
        
        Args:
            atoms (Atoms): Input structure
            temperature (float): Temperature in Kelvin
            single_atom_idx (int, optional): Index of single atom to displace
            max_attempts (int): Maximum number of attempts to find valid configuration
            
        Returns:
            tuple: (displaced_positions, success)
        """
        # Create temporary atoms object for MB distribution
        temp_atoms = atoms.copy()
        
        for attempt in range(max_attempts):
            # Apply MB distribution to get velocities
            MaxwellBoltzmannDistribution(temp_atoms, temperature_K=temperature)
            
            # Convert velocities to displacements (arbitrary timestep of 1)
            displacements = temp_atoms.get_velocities()
            
            if single_atom_idx is not None:
                # Zero out all displacements except for chosen atom
                mask = np.zeros_like(displacements)
                mask[single_atom_idx] = 1
                displacements *= mask
            
            # Apply displacements
            new_positions = atoms.positions + displacements
            
            # Check for overlaps
            if self._check_overlaps(new_positions, atoms.cell, atoms.pbc):
                return new_positions, True
            
            # If overlaps found, reduce displacement magnitude and try again
            displacements *= 0.8
        
        return None, False
    
    def generate_displacement(self, atoms, temperature, single_atom_idx=None):
        """
        Generate thermally motivated displacements for atoms.
        
        Args:
            atoms (Atoms): Input atomic structure
            temperature (float): Temperature in Kelvin
            single_atom_idx (int, optional): If provided, only this atom will be displaced
            
        Returns:
            Atoms: New Atoms object with displaced positions, or None if valid 
                  configuration cannot be found
        """
        # Try to generate valid displacement
        new_positions, success = self._apply_mb_displacement(
            atoms, temperature, single_atom_idx
        )
        
        if not success:
            return None
            
        # Create new Atoms object with displaced positions
        new_atoms = atoms.copy()
        new_atoms.positions = new_positions
        
        # Store reference to parent structure
        if 'structure_name' in atoms.info:
            new_atoms.info['parent_structure'] = atoms.info['structure_name']
        
        return new_atoms
    
class AdversarialOptimizer:
    def __init__(self, adversarial_calc, displacement_gen=None, min_distance=2.0):
        """
        Initialize the optimizer with calculator and displacement generator.
        
        Args:
            adversarial_calc: Instance of AdversarialCalculator
            displacement_gen: Optional instance of DisplacementGenerator
            min_distance: Minimum atomic distance if creating new DisplacementGenerator
        """
        self.calculator = adversarial_calc
        self.displacement_gen = displacement_gen or DisplacementGenerator(min_distance=min_distance)
        
    def _get_output_filename(self, atoms):
        """Generate output filename based on parent structure name."""
        if 'structure_name' in atoms.info:
            base_name = atoms.info['structure_name']
        else:
            base_name = 'structure'
        return f"{base_name}_adversarial.xyz"
    
    def _calculate_variance(self, atoms):
        """Calculate force variance for structure."""
        forces = self.calculator.calculate_forces(atoms)
        atom_variances = self.calculator.calculate_normalized_force_variance(forces)
        structure_variance = float(np.mean(atom_variances))  # Convert to scalar
        return structure_variance, atom_variances, forces
    
    def _metropolis_acceptance(self, old_variance, new_variance, temperature):
        """Metropolis acceptance criterion."""
        # Ensure we're working with scalar values
        old_var = float(old_variance)
        new_var = float(new_variance)
        
        if new_var > old_var:
            return True
        else:
            delta = new_var - old_var
            probability = np.exp(delta * temperature)  # Temperature scales acceptance probability
            return float(np.random.random()) < probability
    
    def optimize(self, atoms, temperature, max_iterations=50, patience=25, 
                mode='all', output_dir='.'):
        """
        Run adversarial optimization to maximize force variance.
        
        Args:
            atoms: ASE Atoms object
            temperature: Temperature in Kelvin for both displacements and acceptance
            max_iterations: Maximum number of optimization steps
            patience: Stop if no improvement after this many steps
            mode: 'all' for all atoms or 'single' for highest variance atom
            output_dir: Directory to save trajectory
        
        Returns:
            best_atoms: Atoms object with highest achieved variance
            best_variance: Highest variance achieved
            accepted_moves: Number of accepted moves
        """
        # Setup output file
        output_file = os.path.join(output_dir, self._get_output_filename(atoms))
        
        # Calculate initial variance
        current_variance, atom_variances, _ = self._calculate_variance(atoms)
        best_variance = current_variance
        best_atoms = atoms.copy()
        
        # For single atom mode, identify atom with highest variance
        if mode == 'single':
            target_atom = np.argmax(atom_variances)
            print(f"Selected atom {target_atom} with initial variance {atom_variances[target_atom]}")
        
        # Initialize tracking variables
        steps_without_improvement = 0
        accepted_moves = 0
        current_atoms = atoms.copy()
        
        # Save initial structure
        if 'structure_name' in atoms.info:
            current_atoms.info['parent_structure'] = atoms.info['structure_name']
        ase.io.write(output_file, current_atoms, write_results=False)
        
        # Main optimization loop
        for step in tqdm(range(max_iterations), desc="Optimizing structure"):
            # Generate displacement
            if mode == 'all':
                new_atoms = self.displacement_gen.generate_displacement(
                    current_atoms, temperature)
            else:  # single atom mode
                new_atoms = self.displacement_gen.generate_displacement(
                    current_atoms, temperature, single_atom_idx=target_atom)
            
            if new_atoms is None:
                print("Warning: Could not generate valid displacement")
                continue
                
            # Calculate new variance
            new_variance, _, _ = self._calculate_variance(new_atoms)
            
            # Check acceptance
            if self._metropolis_acceptance(current_variance, new_variance, 1.0/temperature):
                current_atoms = new_atoms
                current_variance = new_variance
                accepted_moves += 1
                
                # Save accepted structure
                ase.io.write(output_file, current_atoms, append=True, write_results=False)
                
                # Print acceptance info
                print(f"\nStep {step}: Accepted move")
                print(f"New variance: {current_variance:.6f}")
                print(f"Acceptance rate: {accepted_moves/(step+1):.2%}")
                
                # Update best if improved
                if current_variance > best_variance:
                    best_variance = current_variance
                    best_atoms = current_atoms.copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
            else:
                steps_without_improvement += 1
            
            # Check patience
            if steps_without_improvement >= patience:
                print(f"\nStopping: No improvement for {patience} steps")
                break
        
        print(f"\nOptimization complete:")
        print(f"Best variance: {best_variance:.6f}")
        print(f"Accepted moves: {accepted_moves}/{step+1} ({accepted_moves/(step+1):.2%})")
        print(f"Trajectory saved to: {output_file}")
        
        return best_atoms, best_variance, accepted_moves