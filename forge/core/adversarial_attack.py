"""Core classes for adversarial attack workflow."""

import numpy as np
from ase import Atoms
from ase.io import read, write
from mace.calculators.mace import MACECalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from scipy.spatial.distance import pdist, squareform
import os
from pathlib import Path
from tqdm import tqdm
import json
import torch

class GradientAdversarialCalculator:
    """Calculator that enables gradient-based adversarial attacks using MACE models."""
    
    def __init__(self, model_paths, device='cuda', default_dtype='float32', temperature=0.86, 
                use_autograd=False, step_size=0.01):
        """Initialize with MACE models and setup for gradient calculation.
        
        Args:
            model_paths: List of paths to MACE models
            device: Device to run on ('cpu' or 'cuda')
            default_dtype: Data type for calculations
            temperature: Temperature in eV for probability weighting (kT)
            use_autograd: Whether to use Hessian for gradient calculation
            step_size: Step size for finite difference gradients if not using Hessian
        """
        self.device = device
        self.default_dtype = default_dtype
        self.temperature = temperature
        self.use_autograd = use_autograd
        self.step_size = step_size
        
        # Load MACE models
        self.models = []
        for model_path in model_paths:
            model = MACECalculator(
                model_paths=model_path,
                device=self.device,
                default_dtype=self.default_dtype
            )
            self.models.append(model)
    
    def calculate_loss(self, atoms, include_probability=False):
        """Calculate adversarial loss for a structure.
        
        Args:
            atoms: ASE Atoms object
            include_probability: Whether to include probability term
            
        Returns:
            float: Adversarial loss value
        """
        # Calculate forces from all models
        forces_list = []
        energies_list = []
        
        for model in self.models:
            atoms.calc = model
            # Force energy calculation to ensure forces are computed
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                forces_list.append(forces)
                energies_list.append(energy)
            except Exception as e:
                print(f"Warning: Model calculation failed: {e}")
                return 0.0
        
        # Convert to numpy arrays
        forces = np.array(forces_list)
        energies = np.array(energies_list)
        
        # Calculate force magnitudes
        force_magnitudes = np.linalg.norm(forces, axis=2, keepdims=True)
        # Avoid division by zero
        force_magnitudes = np.where(
            force_magnitudes < 1e-10, 
            np.ones_like(force_magnitudes), 
            force_magnitudes
        )
        
        # Normalize forces
        normalized_forces = forces / force_magnitudes
        
        # Calculate variance across models for each atom
        atom_variances = np.var(normalized_forces, axis=0)
        total_atom_variances = np.sum(atom_variances, axis=1)
        mean_variance = float(np.mean(total_atom_variances))
        
        # Calculate the adversarial loss
        if include_probability:
            # Calculate mean energy across models
            mean_energy = float(np.mean(energies))
            # Calculate probability term: exp(-E/kT)
            # Normalize by number of atoms to get per-atom energy
            n_atoms = len(atoms)
            energy_per_atom = mean_energy / n_atoms
            prob_term = np.exp(-energy_per_atom / self.temperature)
            # Final loss: p(X_δ) * σ_F^2(X_δ)
            loss = prob_term * mean_variance
        else:
            loss = mean_variance
        
        return loss
    
    def calculate_gradients(self, atoms, include_probability=False):
        """Calculate gradients of adversarial loss with respect to positions.
        
        Args:
            atoms: ASE Atoms object
            include_probability: Whether to include probability term
            
        Returns:
            np.ndarray: Gradients of loss with respect to positions (n_atoms, 3)
        """
        if self.use_autograd and all(hasattr(model, 'get_hessian') for model in self.models):
            try:
                # Use Hessian-based gradient calculation for efficiency
                return self._calculate_gradients_hessian(atoms, include_probability)
            except Exception as e:
                print(f"Warning: Hessian-based gradient calculation failed: {e}")
                print("Falling back to finite differences")
                return self._calculate_gradients_finite_diff(atoms, include_probability)
        else:
            # Use finite differences
            return self._calculate_gradients_finite_diff(atoms, include_probability)
            
    def _calculate_gradients_finite_diff(self, atoms, include_probability=False):
        """Calculate gradients using finite differences."""
        orig_positions = atoms.positions.copy()
        n_atoms = len(atoms)
        gradients = np.zeros((n_atoms, 3))
        
        # Calculate original loss
        orig_loss = self.calculate_loss(atoms, include_probability)
        
        # Calculate gradients for each atom and dimension
        for i in range(n_atoms):
            for j in range(3):
                # Forward difference
                atoms.positions[i, j] += self.step_size
                fwd_loss = self.calculate_loss(atoms, include_probability)
                
                # Backward difference for stability
                atoms.positions[i, j] -= 2 * self.step_size
                bwd_loss = self.calculate_loss(atoms, include_probability)
                
                # Central difference
                gradients[i, j] = (fwd_loss - bwd_loss) / (2 * self.step_size)
                
                # Restore position
                atoms.positions[i, j] = orig_positions[i, j]
        
        return gradients
        
    def _calculate_gradients_hessian(self, atoms, include_probability=False):
        """Calculate gradients using the Hessian from MACE models.
        
        This approach uses the Hessian to estimate force changes for small perturbations,
        which is more efficient than recomputing forces for each perturbation.
        """
        # Get the original positions
        orig_positions = atoms.positions.copy()
        n_atoms = len(atoms)
        
        # Compute forces, energies, and Hessians for all models
        all_forces = []
        all_hessians = []
        all_energies = []
        
        for model in self.models:
            atoms.calc = model
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces().copy()
            hessian = model.get_hessian(atoms)
            
            all_forces.append(forces)
            all_hessians.append(hessian)
            all_energies.append(energy)
        
        # Calculate original variance
        forces = np.array(all_forces)  # Shape: (n_models, n_atoms, 3)
        force_magnitudes = np.linalg.norm(forces, axis=2, keepdims=True)
        force_magnitudes = np.where(force_magnitudes < 1e-10, 1.0, force_magnitudes)
        normalized_forces = forces / force_magnitudes
        
        atom_variances = np.var(normalized_forces, axis=0)
        total_atom_variances = np.sum(atom_variances, axis=1)
        orig_variance = float(np.mean(total_atom_variances))
        
        # Compute gradients using Hessian to estimate force changes
        gradients = np.zeros((n_atoms, 3))
        eps = 1e-4  # Small perturbation
        
        for i in range(n_atoms):
            for j in range(3):
                # Perturb position
                dp = np.zeros_like(orig_positions)
                dp[i, j] = eps
                
                # Estimate new forces using Hessian for each model
                perturbed_forces = []
                for model_idx, (forces, hessian) in enumerate(zip(all_forces, all_hessians)):
                    # Reshape Hessian to match our needs
                    # The Hessian from MACE is (3N×3N) where each element is d²E/dx_i dx_j
                    # We use -Hessian to get dF/dx since F = -∇E
                    
                    # Initialize perturbed force
                    new_forces = forces.copy()
                    
                    # Apply force changes from Hessian
                    # The formula is: ΔF ≈ -H·Δx
                    # Since the Hessian is d²E/dx² = -dF/dx
                    for a in range(n_atoms):
                        for k in range(3):
                            # Row index in the Hessian
                            row = 3*a + k
                            # Apply changes from the perturbation
                            # We only perturbed position (i,j)
                            col = 3*i + j
                            # Negative Hessian gives force change
                            new_forces[a, k] -= hessian[row, col] * eps
                    
                    perturbed_forces.append(new_forces)
                
                # Calculate variance with perturbed forces
                p_forces = np.array(perturbed_forces)
                p_magnitudes = np.linalg.norm(p_forces, axis=2, keepdims=True)
                p_magnitudes = np.where(p_magnitudes < 1e-10, 1.0, p_magnitudes)
                p_norm_forces = p_forces / p_magnitudes
                
                p_variances = np.var(p_norm_forces, axis=0)
                p_total_variances = np.sum(p_variances, axis=1)
                perturbed_variance = float(np.mean(p_total_variances))
                
                # Compute gradient using forward difference
                gradients[i, j] = (perturbed_variance - orig_variance) / eps
        
        # Add probability term if needed
        if include_probability:
            # Calculate mean energy across models
            mean_energy = float(np.mean(all_energies))
            # Calculate probability term: exp(-E/kT)
            energy_per_atom = mean_energy / n_atoms
            prob_term = np.exp(-energy_per_atom / self.temperature)
            
            # Scale gradients by probability
            gradients *= prob_term
            
            # TODO: Add gradient of probability term if needed
            # This would require Hessian contribution to energy changes
        
        return gradients

class GradientAscentOptimizer:
    """Optimizer that uses gradient ascent to maximize adversarial loss."""
    
    def __init__(self, calculator, learning_rate=0.01, include_probability=False):
        """Initialize optimizer.
        
        Args:
            calculator: Instance of GradientAdversarialCalculator
            learning_rate: Learning rate for gradient ascent
            include_probability: Whether to include the probability term in the loss
        """
        self.calculator = calculator
        self.learning_rate = learning_rate
        self.include_probability = include_probability
    
    def optimize(self, atoms, n_iterations=60, min_distance=1.5, output_dir='.'):
        """Run gradient ascent to maximize adversarial loss.
        
        Args:
            atoms: ASE Atoms object with initial structure
            n_iterations: Number of optimization iterations
            min_distance: Minimum allowed distance between atoms
            output_dir: Directory to save trajectory
            
        Returns:
            tuple: (best_atoms, best_loss, loss_history)
        """
        from ase.io import write
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        import os
        import json
        
        # Setup output
        output_file = os.path.join(output_dir, f"{atoms.info.get('structure_name', 'structure')}_adversarial.xyz")
        
        # Initialization
        best_loss = -float('inf')
        best_positions = atoms.positions.copy()
        current_atoms = atoms.copy()
        current_positions = atoms.positions.copy()
        loss_history = []
        
        # Calculate initial loss
        current_loss = self.calculator.calculate_loss(
            current_atoms, 
            include_probability=self.include_probability
        )
        best_loss = current_loss
        
        # Store initial structure
        current_atoms.info['variance'] = current_loss
        write(output_file, current_atoms, write_results=False)
        loss_history.append(current_loss)
        
        # Optimization loop
        for step in range(n_iterations):
            # Calculate gradients
            gradients = self.calculator.calculate_gradients(
                current_atoms,
                include_probability=self.include_probability
            )
            
            # Update positions using gradient ascent
            new_positions = current_positions + self.learning_rate * gradients
            
            # Check for minimum distance constraint
            temp_atoms = current_atoms.copy()
            temp_atoms.positions = new_positions
            distances = squareform(pdist(new_positions))
            np.fill_diagonal(distances, np.inf)
            
            if np.all(distances > min_distance):
                # Accept new positions
                current_positions = new_positions
                current_atoms.positions = current_positions
                
                # Calculate new loss
                current_loss = self.calculator.calculate_loss(
                    current_atoms,
                    include_probability=self.include_probability
                )
                
                # Update best if improved
                if current_loss > best_loss:
                    best_loss = current_loss
                    best_positions = current_positions.copy()
                
                # Save current state
                current_atoms.info['variance'] = current_loss
                write(output_file, current_atoms, append=True, write_results=False)
                
                print(f"Step {step}: Loss = {current_loss:.6f}")
            else:
                print(f"Step {step}: Rejected due to close atoms")
                # Keep previous positions if constraint violated
                # No need to update current_atoms as we're keeping the same positions
            
            loss_history.append(current_loss)
        
        # Create best atoms object
        best_atoms = atoms.copy()
        best_atoms.positions = best_positions
        best_atoms.info['variance'] = best_loss
        
        # Save optimization summary
        summary_file = os.path.join(output_dir, 'optimization_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {'results': []}
        
        structure_name = atoms.info.get('structure_name', 'unknown')
        result = {
            'structure_name': structure_name,
            'initial_variance': loss_history[0],
            'final_variance': best_loss,
            'n_iterations': n_iterations,
            'learning_rate': self.learning_rate,
            'include_probability': self.include_probability,
            'loss_history': loss_history
        }
        summary['results'].append(result)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nOptimization complete:")
        print(f"Best variance: {best_loss:.6f}")
        print(f"Trajectory saved to: {output_file}")
        
        return best_atoms, best_loss, loss_history
    
class AdversarialCalculator:
    def __init__(self, model_paths, device='cpu', default_dtype='float32'):
        """Initialize calculator with MACE model ensemble.
        
        Args:
            model_paths (str or list): Path(s) to MACE model file(s)
            device (str): Device to use ('cpu' or 'cuda')
            default_dtype (str): Default data type for calculations
        """
        self.device = device
        self.default_dtype = default_dtype
        
        if isinstance(model_paths, str):
            self.is_ensemble = False
            self.model_paths = [model_paths]
        else:
            self.is_ensemble = True
            self.model_paths = model_paths
            
        # Initialize each model separately to ensure proper loading
        self.models = []
        for model_path in self.model_paths:
            model = MACECalculator(
                model_paths=model_path,
                device=self.device,
                default_dtype=self.default_dtype
            )
            self.models.append(model)
            
    def calculate_forces(self, atoms):
        """Calculate forces using MACE ensemble.
        
        Args:
            atoms (Atoms): ASE Atoms object
            
        Returns:
            np.ndarray: Forces array of shape (n_models, n_atoms, 3)
        """
        forces_list = []
        for model in self.models:
            atoms.calc = model
            try:
                # Force energy calculation to ensure forces are computed
                atoms.get_potential_energy()
                forces = atoms.get_forces()
                forces_list.append(forces)
            except Exception as e:
                print(f"Warning: Force calculation failed for model: {e}")
                return np.zeros((len(self.models), len(atoms), 3))
                
        return np.array(forces_list)
            
    def calculate_normalized_force_variance(self, forces):
        """Calculate normalized force variance across ensemble predictions.
        
        Args:
            forces (np.ndarray): Forces array from calculate_forces()
            
        Returns:
            np.ndarray: Array of shape (n_atoms,) with normalized variances
        """
        # Calculate force magnitudes, avoiding division by zero
        force_magnitudes = np.linalg.norm(forces, axis=2, keepdims=True)
        force_magnitudes = np.where(force_magnitudes < 1e-10, 1.0, force_magnitudes)
        
        # Normalize forces
        normalized_forces = forces / force_magnitudes
        
        # Calculate variance across models for each atom
        atom_variances = np.var(normalized_forces, axis=0)
        total_atom_variances = np.sum(atom_variances, axis=1)
        return total_atom_variances


class DisplacementGenerator:
    def __init__(self, min_distance=2.0):
        """Initialize displacement generator.
        
        Args:
            min_distance (float): Minimum allowed distance between atoms (Å)
        """
        self.min_distance = min_distance
        
    def _check_overlaps(self, positions, cell, pbc):
        """Check if any atoms are closer than min_distance."""
        distances = squareform(pdist(positions))
        np.fill_diagonal(distances, np.inf)
        return np.all(distances > self.min_distance)
    
    def _apply_mb_displacement(self, atoms, temperature, single_atom_idx=None, max_attempts=100):
        """Apply Maxwell-Boltzmann displacement with overlap checking."""
        temp_atoms = atoms.copy()
        
        for attempt in range(max_attempts):
            MaxwellBoltzmannDistribution(temp_atoms, temperature_K=temperature)
            displacements = temp_atoms.get_velocities()
            
            if single_atom_idx is not None:
                mask = np.zeros_like(displacements)
                mask[single_atom_idx] = 1
                displacements *= mask
            
            new_positions = atoms.positions + displacements
            
            if self._check_overlaps(new_positions, atoms.cell, atoms.pbc):
                return new_positions, True
            
            displacements *= 0.8
        
        return None, False
    
    def generate_displacement(self, atoms, temperature, single_atom_idx=None):
        """Generate thermally motivated displacements for atoms.
        
        Args:
            atoms (Atoms): Input atomic structure
            temperature (float): Temperature in Kelvin
            single_atom_idx (int, optional): If provided, only displace this atom
            
        Returns:
            Atoms: New Atoms object with displaced positions, or None if invalid
        """
        new_positions, success = self._apply_mb_displacement(
            atoms, temperature, single_atom_idx
        )
        
        if not success:
            return None
            
        new_atoms = atoms.copy()
        new_atoms.positions = new_positions
        
        if 'structure_name' in atoms.info:
            new_atoms.info['parent_structure'] = atoms.info['structure_name']
        
        return new_atoms


class AdversarialOptimizer:
    def __init__(self, adversarial_calc, displacement_gen=None, min_distance=2.0):
        """Initialize optimizer with calculator and displacement generator.
        
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
        structure_variance = float(np.mean(atom_variances))
        return structure_variance, atom_variances, forces
    
    def _metropolis_acceptance(self, old_variance, new_variance, temperature):
        """Metropolis acceptance criterion for variance maximization.
        
        Uses relative change in variance (delta_var/old_var) to make acceptance
        probability independent of the absolute scale of the variance.
        """
        old_var = float(old_variance)
        new_var = float(new_variance)
        
        if new_var > old_var:
            return True
        else:
            # Use relative change in variance
            relative_delta = (new_var - old_var) / (old_var + 1e-10)  # Add small epsilon to avoid division by zero
            probability = np.exp(relative_delta * 100 / temperature)  # Scale by 100 since relative changes are small
            return float(np.random.random()) < probability
    
    def optimize(self, atoms, temperature, max_iterations=50, patience=25,
                mode='all', output_dir='.'):
        """Run adversarial optimization to maximize force variance.
        
        Args:
            atoms: ASE Atoms object
            temperature: Temperature in Kelvin for displacements and acceptance
            max_iterations: Maximum optimization steps
            patience: Stop if no improvement after this many steps
            mode: 'all' for all atoms or 'single' for highest variance atom
            output_dir: Directory to save trajectory
            
        Returns:
            tuple: (best_atoms, best_variance, accepted_moves)
        """
        output_file = os.path.join(output_dir, self._get_output_filename(atoms))
        
        current_variance, atom_variances, _ = self._calculate_variance(atoms)
        best_variance = current_variance
        best_atoms = atoms.copy()
        
        if mode == 'single':
            target_atom = np.argmax(atom_variances)
            print(f"Selected atom {target_atom} with initial variance {atom_variances[target_atom]}")
        
        steps_without_improvement = 0
        accepted_moves = 0
        current_atoms = atoms.copy()
        step_variances = [current_variance]  # Track variances at each step
        
        if 'structure_name' in atoms.info:
            current_atoms.info['parent_structure'] = atoms.info['structure_name']
        current_atoms.info['variance'] = current_variance
        write(output_file, current_atoms, write_results=False)
        
        for step in tqdm(range(max_iterations), desc="Optimizing structure"):
            if mode == 'all':
                new_atoms = self.displacement_gen.generate_displacement(
                    current_atoms, temperature)
            else:
                new_atoms = self.displacement_gen.generate_displacement(
                    current_atoms, temperature, single_atom_idx=target_atom)
            
            if new_atoms is None:
                print("Warning: Could not generate valid displacement")
                continue
                
            new_variance, _, _ = self._calculate_variance(new_atoms)
            
            if self._metropolis_acceptance(current_variance, new_variance, temperature):
                current_atoms = new_atoms
                current_variance = new_variance
                accepted_moves += 1
                
                current_atoms.info['variance'] = current_variance
                write(output_file, current_atoms, append=True, write_results=False)
                step_variances.append(current_variance)
                
                print(f"\nStep {step}: Accepted move")
                print(f"New variance: {current_variance:.6f} (delta: {current_variance - best_variance:.6f})")
                print(f"Acceptance rate: {accepted_moves/(step+1):.2%}")
                
                if current_variance > best_variance:
                    best_variance = current_variance
                    best_atoms = current_atoms.copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
            else:
                steps_without_improvement += 1
                step_variances.append(current_variance)  # Keep previous variance for rejected moves
            
            if steps_without_improvement >= patience:
                print(f"\nStopping: No improvement for {patience} steps")
                break
        
        # Save optimization summary with step variances
        summary_file = os.path.join(output_dir, 'optimization_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {'results': []}
            
        structure_name = atoms.info.get('structure_name', 'unknown')
        result = {
            'structure_name': structure_name,
            'initial_variance': float(step_variances[0]),
            'final_variance': float(best_variance),
            'accepted_moves': accepted_moves,
            'total_steps': step + 1,
            'step_variances': [float(v) for v in step_variances]
        }
        summary['results'].append(result)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nOptimization complete:")
        print(f"Best variance: {best_variance:.6f}")
        print(f"Accepted moves: {accepted_moves}/{step+1} ({accepted_moves/(step+1):.2%})")
        print(f"Trajectory saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")
        
        return best_atoms, best_variance, accepted_moves 