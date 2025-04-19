#!/usr/bin/env python
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
import time
import matplotlib.pyplot as plt
import copy # Import the copy module

class Timer:
    """Simple timer for performance debugging."""

    def __init__(self, debug=False):
        """Initialize timer.

        Args:
            debug: Whether to print debug messages
        """
        self.debug = debug
        self.timers = {}
        self.starts = {}

    def start(self, name):
        """Start a named timer."""
        self.starts[name] = time.time()

    def stop(self, name):
        """Stop a named timer and record elapsed time."""
        if name in self.starts:
            elapsed = time.time() - self.starts[name]
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(elapsed)
            if self.debug:
                print(f"[DEBUG] {name}: {elapsed:.4f} seconds")
            return elapsed
        return 0

    def summary(self):
        """Print summary of all timers."""
        print("\n===== Performance Summary =====")
        for name, times in self.timers.items():
            total = sum(times)
            avg = total / len(times) if times else 0
            print(f"{name}:")
            print(f"  Total: {total:.4f} seconds")
            print(f"  Count: {len(times)}")
            print(f"  Average: {avg:.4f} seconds")
        print("==============================\n")

class GradientAdversarialOptimizer:
    """Optimizer that uses PyTorch autograd to maximize adversarial loss."""

    def __init__(self, model_paths, device='cuda', learning_rate=0.01,
                 temperature=0.86, include_probability=True, debug=False,
                 energy_list=None, use_energy_per_atom=False):
        """Initialize optimizer with model paths.

        Args:
            model_paths: List of paths to model files
            device: Device to run on ('cpu' or 'cuda')
            learning_rate: Learning rate for gradient ascent
            temperature: Temperature for probability weighting (eV)
            include_probability: Whether to include probability term in loss
            debug: Whether to print debug messages
            energy_list: List of energies (total or per atom) for normalization constant calculation
            use_energy_per_atom: If True, treat energy_list as energy/atom and use energy/atom for probability calc.
        """
        self.energy_list = energy_list
        self.model_paths = model_paths
        self.device = device
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.include_probability = include_probability
        self.debug = debug
        self.timer = Timer(debug=debug)
        self.dtype = torch.float32  # Explicitly set data type
        self.use_energy_per_atom = use_energy_per_atom

        # Initialize ASE calculator for force calculations
        self.timer.start("calculator_init")
        self.calculator = AdversarialCalculator(
            model_paths=model_paths,
            device=device
        )
        self.timer.stop("calculator_init")

        # Calculate normalization constant based on flag
        self.normalization_constant = 1.0
        if self.include_probability and self.energy_list is not None and len(self.energy_list) > 0:
            energies_for_Q = self.energy_list # Assume energy_list matches use_energy_per_atom flag
            try:
                 Q, _ = self._calculate_normalization_constant(energies_for_Q, self.temperature)
                 self.normalization_constant = Q
                 if self.debug:
                     print(f"[DEBUG] Normalization constant (Q): {self.normalization_constant:.6f} (based on {'energy/atom' if self.use_energy_per_atom else 'total energy'})")
            except Exception as e:
                 print(f"[WARN] Failed to calculate normalization constant: {e}. Using Q=1.0")
                 self.normalization_constant = 1.0

        elif self.include_probability:
             if self.debug:
                 print("[DEBUG] No energy list provided or empty, using default normalization constant Q=1.0")

    def _calculate_force_variance(self, atoms):
        """Calculate force variance and mean forces across ensemble models."""
        self.timer.start("force_calculation")
        forces = self.calculator.calculate_forces(atoms) # Shape (n_models, n_atoms, 3)
        self.timer.stop("force_calculation")

        # Calculate mean forces
        mean_forces = np.mean(forces, axis=0)

        self.timer.start("variance_calculation")
        atom_variances = self.calculator.calculate_normalized_force_variance(forces)
        variance = float(np.mean(atom_variances)) if atom_variances.size > 0 else 0.0
        self.timer.stop("variance_calculation")

        return variance, atom_variances, mean_forces # Return mean forces as well

    def _calculate_energy(self, atoms):
        """Calculate mean TOTAL energy across ensemble models."""
        self.timer.start("energy_calculation")
        energies = []
        for model in self.calculator.models:
            atoms.calc = model
            energy = atoms.get_potential_energy()
            energies.append(energy)
        mean_energy = float(np.mean(energies))
        self.timer.stop("energy_calculation")
        return mean_energy

    def _calculate_normalization_constant(self, energy_list, temperature):
        """Calculate 'normalization constant' using the shifted energies.
        Inputs:
            energy_list: List of energies
            temperature: Temperature (K)
        Outputs:
            Q: Normalization constant
            shifted_energies: Shifted energies
        """
        k_B = 8.617e-5  # eV/K
        e_min = np.min(energy_list)
        shifted_energies = energy_list - e_min
        exp_terms = np.exp(-shifted_energies / (k_B * temperature))
        Q = np.sum(exp_terms)
        return Q, shifted_energies

    def _calculate_probability(self, energy, temperature, normalization_constant=1.0):
        """Calculate Boltzmann probability for a structure.
        
        Args:
            energy: Energy per atom in eV
            temperature: Temperature in Kelvin
            normalization_constant: Partition function value (default: 1.0)
            
        Returns:
            float: Boltzmann probability (e^(-E/kT)/Q) for the structure
                  Returns 1.0 in case of invalid inputs or numerical errors
        """
        if temperature <= 0:
             print(f"[WARN] Temperature is non-positive ({temperature} K). Probability calculation is invalid. Returning 1.0.")
             return 1.0
        if normalization_constant <= 0:
             print(f"[WARN] Normalization constant Q is non-positive ({normalization_constant}). Probability calculation is invalid. Returning 1.0.")
             return 1.0

        k_B = 8.617e-5  # Boltzmann constant in eV/K
        exponent = -energy / (k_B * temperature)  # Convert temperature from K to energy units
        # Add safeguard against large positive exponent leading to overflow
        if exponent > 700: # Corresponds to exp(700), roughly float limit
             print(f"[WARN] Exponent {exponent:.2f} too large in probability calculation (Energy: {energy:.4f} eV, Temp: {temperature:.4f} K). Clamping probability.")
             probability = torch.finfo(self.dtype).max # Assign a large finite number instead of inf
        else:
            try:
                 probability = np.exp(exponent) / normalization_constant
            except FloatingPointError:
                 print(f"[WARN] Floating point error during probability calculation (Exponent: {exponent:.2f}). Returning 0.0.")
                 probability = 0.0

        # Check for NaN or Inf (should be less likely with checks above)
        if np.isnan(probability) or np.isinf(probability):
            print(f"[WARNING] Probability is nan or inf (E={energy:.4f} eV, T={temperature:.4f} K, Q={normalization_constant:.4f}). Setting probability to 1.0 (deterministic)." )
            return 1.0 # Fallback to deterministic if something unexpected happens
        return probability

    def optimize(self, atoms, generation: int, n_iterations=60, min_distance=1.5, output_dir='.',
                 patience: int = 20, shake_std: float = 0.05):
        """Run gradient-based adversarial attack optimization.

        Args:
            atoms: ASE Atoms object (must contain 'structure_id' and 'config_type' in info)
            generation: Generation identifier (integer) for the output structures.
            n_iterations: Maximum number of iterations
            min_distance: Minimum allowed distance between atoms
            output_dir: Directory to save plots
            patience: Number of steps without loss improvement before shaking (default: 10).
            shake_std: Standard deviation (in Angstrom) for random shake (default: 0.05).

        Returns:
            List[Atoms]: Trajectory of Atoms objects, each with detailed info and mean forces.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Extract initial info ---
        parent_id = atoms.info.get('structure_id')
        if parent_id is None:
            raise ValueError("Input 'atoms' object must have 'structure_id' in its info dictionary.")
        original_config_type = atoms.info.get('config_type', 'unknown')
        struct_name = atoms.info.get('structure_name', f'structure_{parent_id}')

        # Derive new config_type
        if '_aa' in original_config_type:
            new_config_type = original_config_type
        else:
            new_config_type = f"{original_config_type}_aa"

        # Initialize tracking variables
        original_positions = atoms.positions.copy()
        best_loss = -float('inf') # Still track best loss for potential future use/logging
        steps_without_improvement = 0 # Counter for patience
        trajectory = [] # Store Atoms objects for each step

        # Initialize displacement tensor with gradient tracking
        displacement = torch.zeros(
            (len(atoms), 3),
            requires_grad=True,
            device="cpu", # Keep displacement on CPU for numpy conversion
            dtype=self.dtype
        )

        # Create optimizer
        optimizer = torch.optim.Adam([displacement], lr=self.learning_rate)

        print(f"[INFO] Starting optimization for parent ID: {parent_id}, generation: {generation}, patience={patience}, shake_std={shake_std}")

        # --- Optimization loop ---
        for step in tqdm(range(n_iterations), desc=f"Optimizing {struct_name}"):
            self.timer.start(f"step_{step}")

            # --- Check for patience ---
            if steps_without_improvement >= patience:
                 print(f"\n[INFO] Step {step}: No improvement for {patience} steps. Applying random shake (std={shake_std} Å).")
                 noise = np.random.normal(0, shake_std, size=original_positions.shape)
                 # --- Apply shake to the *base* positions --- 
                 original_positions += noise
                 # --- Reset learned displacement and optimizer state --- 
                 displacement.data.zero_()
                 optimizer = torch.optim.Adam([displacement], lr=self.learning_rate) # Reinitialize optimizer
                 steps_without_improvement = 0 # Reset counter after shake
                 # NOTE: The next step will calculate loss based on the *shaken* structure

            # Zero gradients
            optimizer.zero_grad()

            # Apply displacements to get new positions
            # Ensure displacement is detached before numpy conversion
            new_positions = original_positions + displacement.detach().cpu().numpy()

            # Create current_atoms for this step's calculations
            current_atoms_step = atoms.copy()
            current_atoms_step.positions = new_positions

            # Check minimum distance constraint
            self.timer.start("distance_check")
            distances = current_atoms_step.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf) # Ignore self-distance
            min_dist = np.min(distances) if distances.size > 0 else np.inf
            self.timer.stop("distance_check")

            if min_dist < min_distance:
                if self.debug:
                    print(f"[DEBUG] Step {step}: Minimum distance constraint violated: {min_dist:.3f} Å. Scaling back.")
                # Scale back displacement to satisfy constraint
                scale_factor = 0.9
                with torch.no_grad():
                    displacement.data *= scale_factor
                self.timer.stop(f"step_{step}")
                # Skip gradient calculation and optimizer step for this iteration
                continue

            # Calculate variance and mean forces using the step's atoms
            self.timer.start("variance_forces_calc")
            variance, _, mean_forces = self._calculate_force_variance(current_atoms_step)
            self.timer.stop("variance_forces_calc")

            # Calculate energy using the step's atoms
            energy = self._calculate_energy(current_atoms_step)

            # Calculate probability if needed
            probability = 1.0 # Default if not included
            if self.include_probability:
                self.timer.start("probability_calculation")
                probability = self._calculate_probability(
                    energy, self.temperature, self.normalization_constant
                )
                self.timer.stop("probability_calculation")

            # Calculate loss
            loss = probability * variance if self.include_probability else variance

            # Update best loss tracking and patience counter
            if loss > best_loss:
                best_loss = loss
                steps_without_improvement = 0 # Reset patience counter
                if self.debug:
                     print(f"[DEBUG] Step {step}: New best loss: {loss:.6f}")
            else:
                steps_without_improvement += 1 # Increment patience counter

            # --- Store step results in trajectory ---
            self.timer.start("store_trajectory_step")
            step_atoms = current_atoms_step # Use the atoms object already created for this step

            # Clear previous calculator if attached by internal methods
            step_atoms.calc = None

            # Store results in info
            step_atoms.info['parent_id'] = parent_id
            step_atoms.info['generation'] = generation
            step_atoms.info['config_type'] = new_config_type
            step_atoms.info['step'] = step
            step_atoms.info['variance'] = variance
            step_atoms.info['energy'] = energy # Mean energy
            step_atoms.info['loss'] = loss
            if self.include_probability:
                step_atoms.info['probability'] = probability

            # Store mean forces in arrays
            step_atoms.arrays['forces'] = mean_forces # Store mean forces

            # --- Use deepcopy to ensure independence --- 
            trajectory.append(copy.deepcopy(step_atoms))
            self.timer.stop("store_trajectory_step")

            # --- Compute gradient using numerical approximation ---
            # (Calculation performed on current_atoms_step implicitly via forward diff)
            self.timer.start("gradient_calculation")
            epsilon = 1e-4
            grad = np.zeros_like(original_positions)

            if self.debug and step % 10 == 0: # Print less often
                print(f"[DEBUG] Grad calc step {step}: Starting gradient for {len(atoms)} atoms")

            # Re-use current_atoms_step for gradient calculation to avoid extra copies
            temp_atoms_for_grad = current_atoms_step.copy()

            for i in range(len(atoms)):
                for j in range(3):
                    # Forward difference positions
                    forward_positions = new_positions.copy()
                    forward_positions[i, j] += epsilon
                    temp_atoms_for_grad.positions = forward_positions # Modify positions of the temp object

                    # Calculate forward variance
                    forward_variance, _, _ = self._calculate_force_variance(temp_atoms_for_grad)

                    # Calculate forward loss
                    if self.include_probability:
                        forward_energy = self._calculate_energy(temp_atoms_for_grad)
                        forward_probability = self._calculate_probability(
                            forward_energy, self.temperature, self.normalization_constant
                        )
                        forward_loss = forward_probability * forward_variance
                    else:
                        forward_loss = forward_variance

                    # Estimate gradient component
                    grad[i, j] = (forward_loss - loss) / epsilon

            self.timer.stop("gradient_calculation")

            # Convert gradient to torch tensor
            grad_tensor = torch.tensor(grad, device=displacement.device, dtype=self.dtype)

            # Manually set the gradient on the displacement tensor
            # Gradient points towards increase, optimizer minimizes, so negate it.
            if displacement.grad is None:
                 displacement.grad = -grad_tensor
            else:
                 displacement.grad.copy_(-grad_tensor) # Use copy_ for in-place update if grad already exists

            # Step optimizer
            self.timer.start("optimizer_step")
            optimizer.step()
            self.timer.stop("optimizer_step")

            self.timer.stop(f"step_{step}")

            # Log progress
            if self.debug and (step % 5 == 0 or step == n_iterations - 1):
                 print(f"[DEBUG] Step {step}: Var={variance:.6f}, Prob={probability:.6f}, Loss={loss:.6f}, LR={optimizer.param_groups[0]['lr']:.1e}, Disp_norm={torch.norm(displacement.data):.4f}")
            elif not self.debug and (step % 10 == 0 or step == n_iterations - 1):
                 print(f"Step {step}: Variance={variance:.6f}, Probability={probability:.6f}, Loss={loss:.6f}")

        # --- Finalization ---
        print(f"[INFO] Optimization finished for parent ID: {parent_id}. Total steps: {len(trajectory)}")

        # Create plots using the generated trajectory
        if trajectory: # Ensure trajectory is not empty before plotting
             self._create_plots(output_path, trajectory)
        else:
             print("[WARN] No trajectory generated, skipping plot creation.")

        # Print performance summary
        self.timer.summary()

        # Return the full trajectory
        return trajectory

    def _create_plots(self, output_path, trajectory):
        """Create and save plots for optimization results using trajectory data."""
        if not trajectory:
            print("[WARN] Cannot create plots: Trajectory is empty.")
            return

        self.timer.start("create_plots")

        # Extract data from trajectory
        struct_name = trajectory[0].info.get('structure_name', f"structure_{trajectory[0].info.get('parent_id', 'unknown')}")
        losses = [atoms.info.get('loss', np.nan) for atoms in trajectory]
        variances = [atoms.info.get('variance', np.nan) for atoms in trajectory]
        energies = [atoms.info.get('energy', np.nan) for atoms in trajectory]
        # Check if probability was included during the run
        has_probability = 'probability' in trajectory[0].info
        probabilities = [atoms.info.get('probability', np.nan) for atoms in trajectory] if has_probability else None

        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(losses, marker='o', linestyle='-', label='Loss')
        plt.xlabel('Iteration Step')
        plt.ylabel('Loss Value')
        plt.title(f'Loss vs. Iteration - {struct_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path / f"{struct_name}_loss_plot.png")
        plt.close()

        # Plot variance history
        plt.figure(figsize=(10, 6))
        plt.plot(variances, marker='o', linestyle='-', color='green', label='Force Variance')
        plt.xlabel('Iteration Step')
        plt.ylabel('Force Variance')
        plt.title(f'Force Variance vs. Iteration - {struct_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path / f"{struct_name}_variance_plot.png")
        plt.close()

        # Plot energy history
        plt.figure(figsize=(10, 6))
        plt.plot(energies, marker='o', linestyle='-', color='red', label='Mean Energy (eV)')
        plt.xlabel('Iteration Step')
        plt.ylabel('Mean Energy (eV)')
        plt.title(f'Mean Energy vs. Iteration - {struct_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path / f"{struct_name}_energy_plot.png")
        plt.close()

        # Plot probability history if available
        if probabilities:
            plt.figure(figsize=(10, 6))
            plt.plot(probabilities, marker='o', linestyle='-', color='purple', label='Probability')
            plt.xlabel('Iteration Step')
            plt.ylabel('Boltzmann Probability')
            plt.title(f'Probability vs. Iteration - {struct_name}')
            plt.grid(True)
            plt.legend()
            plt.savefig(output_path / f"{struct_name}_probability_plot.png")
            plt.close()

        # Combined plot
        fig, ax1 = plt.subplots(figsize=(12, 7))

        ax1.set_xlabel('Iteration Step')
        ax1.set_ylabel('Loss / Variance', color='tab:blue')
        line1 = ax1.plot(losses, marker='o', linestyle='-', color='tab:blue', label='Loss')
        line2 = ax1.plot(variances, marker='s', linestyle=':', color='tab:cyan', label='Variance')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Mean Energy (eV)', color='tab:red')
        line3 = ax2.plot(energies, marker='^', linestyle='--', color='tab:red', label='Mean Energy')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')

        plt.title(f'Combined Metrics - {struct_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / f"{struct_name}_combined_plot.png")
        plt.close()

        self.timer.stop("create_plots")

    def _save_results(self, output_path, struct_name, initial_atoms, best_atoms,
                     initial_variance, best_variance, initial_energy, best_energy,
                     initial_probability, best_probability, initial_loss, best_loss,
                     loss_history, variance_history, energy_history, probability_history):
        # This method is no longer used by the modified optimize method.
        # It can be kept for other potential uses or removed.
        # For now, let's keep it but note it's disconnected from the main flow.
        print("[INFO] _save_results is no longer called by optimize method. Results are returned as trajectory.")
        pass # Keep the method signature but make it do nothing for now

        # self.timer.start("save_results")
        # import json
        # ... (rest of the original code) ...
        # self.timer.stop("save_results")

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