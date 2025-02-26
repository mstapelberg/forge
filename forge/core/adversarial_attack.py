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
                 temperature=0.86, include_probability=True, debug=False):
        """Initialize optimizer with model paths.

        Args:
            model_paths: List of paths to model files
            device: Device to run on ('cpu' or 'cuda')
            learning_rate: Learning rate for gradient ascent
            temperature: Temperature for probability weighting (eV)
            include_probability: Whether to include probability term in loss
            debug: Whether to print debug messages
        """
        self.model_paths = model_paths
        self.device = device
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.include_probability = include_probability
        self.debug = debug
        self.timer = Timer(debug=debug)
        self.dtype = torch.float32  # Explicitly set data type

        # Initialize ASE calculator for force calculations
        self.timer.start("calculator_init")
        self.calculator = AdversarialCalculator(
            model_paths=model_paths,
            device=device
        )
        self.timer.stop("calculator_init")

    def _calculate_force_variance(self, atoms):
        """Calculate force variance across ensemble models."""
        self.timer.start("force_calculation")
        forces = self.calculator.calculate_forces(atoms)
        self.timer.stop("force_calculation")

        self.timer.start("variance_calculation")
        atom_variances = self.calculator.calculate_normalized_force_variance(forces)
        variance = float(np.mean(atom_variances))
        self.timer.stop("variance_calculation")

        return variance, atom_variances, forces

    def _calculate_energy(self, atoms):
        """Calculate mean energy across ensemble models."""
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
        """Calculate Boltzmann probability for a structure."""
        k_B = 8.617e-5 #eV/K
        return np.exp(-energy / (k_B*temperature)) / normalization_constant

    def optimize(self, atoms, energy_list, n_iterations=60, min_distance=1.5, output_dir='.'):
        """Run gradient-based adversarial attack optimization.

        Args:
            atoms: ASE Atoms object
            energy_list: list of energies for normalization constant
            n_iterations: Maximum number of iterations
            min_distance: Minimum allowed distance between atoms
            output_dir: Directory to save results

        Returns:
            tuple: (best_atoms, best_loss, loss_history)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate normalization constant (approximated as 1.0 if working with single structure)
        # In a full dataset scenario, this would be calculated across all structures
        normalization_constant, shifted_energies = self._calculate_normalization_constant(energy_list, self.temperature)

        # Prepare output trajectory file
        struct_name = atoms.info.get('structure_name', 'structure')
        output_file = output_path / f"{struct_name}_adversarial.xyz"

        # Initialize tracking variables
        original_positions = atoms.positions.copy()
        best_loss = -float('inf')
        best_positions = original_positions.copy()
        loss_history = []
        variance_history = []
        probability_history = []
        energy_history = []

        # Save initial structure
        current_atoms = atoms.copy()

        # Calculate initial values
        if self.debug:
            print(f"[DEBUG] Calculating initial values for {len(atoms)} atoms")

        initial_variance, _, _ = self._calculate_force_variance(current_atoms)
        initial_energy = self._calculate_energy(current_atoms)
        initial_probability = self._calculate_probability(
            initial_energy, self.temperature, normalization_constant
        )

        # Calculate initial loss
        if self.include_probability:
            initial_loss = initial_probability * initial_variance
        else:
            initial_loss = initial_variance

        # Set initial values
        best_loss = initial_loss
        current_loss = initial_loss

        # Save initial state
        current_atoms.info['variance'] = initial_variance
        current_atoms.info['energy'] = initial_energy
        current_atoms.info['probability'] = initial_probability
        current_atoms.info['loss'] = initial_loss

        write(output_file, current_atoms, write_results=False)

        # Track history
        loss_history.append(initial_loss)
        variance_history.append(initial_variance)
        probability_history.append(initial_probability)
        energy_history.append(initial_energy)

        print(f"[INFO] Initial values:")
        print(f"  Variance: {initial_variance:.6f}")
        print(f"  Energy: {initial_energy:.6f} eV")
        print(f"  Probability: {initial_probability:.6f}")
        print(f"  Loss: {initial_loss:.6f}")

        # Initialize displacement tensor with gradient tracking - explicitly set dtype
        displacement = torch.zeros(
            (len(atoms), 3),
            requires_grad=True,
            device="cpu",
            dtype=self.dtype  # Explicitly set data type
        )

        # Create optimizer
        optimizer = torch.optim.Adam([displacement], lr=self.learning_rate)

        # Optimization loop
        for step in tqdm(range(n_iterations), desc="Optimizing structure"):
            self.timer.start(f"step_{step}")

            # Zero gradients
            optimizer.zero_grad()

            # Apply displacements to get new positions
            new_positions = original_positions + displacement.detach().cpu().numpy()

            # Check minimum distance constraint
            self.timer.start("distance_check")
            current_atoms.positions = new_positions
            distances = current_atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            self.timer.stop("distance_check")

            if min_dist < min_distance:
                if self.debug:
                    print(f"[DEBUG] Minimum distance constraint violated: {min_dist:.3f} Å")
                # Scale back displacement to satisfy constraint
                scale_factor = 0.9  # Scale back by 10%
                with torch.no_grad():
                    displacement.data *= scale_factor
                self.timer.stop(f"step_{step}")
                continue

            # Calculate variance
            variance, _, _ = self._calculate_force_variance(current_atoms)

            # Calculate energy and probability
            energy = self._calculate_energy(current_atoms)

            self.timer.start("probability_calculation")
            probability = self._calculate_probability(
                energy, self.temperature,
                normalization_constant
            )
            self.timer.stop("probability_calculation")

            # Calculate loss (negative because we're maximizing)
            if self.include_probability:
                loss = probability * variance
            else:
                loss = variance

            # Track values
            variance_history.append(variance)
            energy_history.append(energy)
            probability_history.append(probability)
            loss_history.append(loss)

            # Update best values if improved
            if loss > best_loss:
                best_loss = loss
                best_positions = new_positions.copy()
                print(f"[INFO] Step {step}: New best loss: {loss:.6f}")

            # Save current structure to trajectory
            self.timer.start("save_trajectory")
            current_atoms.info['variance'] = variance
            current_atoms.info['energy'] = energy
            current_atoms.info['probability'] = probability
            current_atoms.info['loss'] = loss
            current_atoms.info['step'] = step

            write(output_file, current_atoms, append=True, write_results=False)
            self.timer.stop("save_trajectory")

            # Compute gradient using numerical approximation
            self.timer.start("gradient_calculation")
            # This replaces the need for autograd/Hessian calculations
            # by directly estimating the gradient of the loss function
            epsilon = 1e-4  # Small perturbation
            grad = np.zeros_like(original_positions)

            if self.debug:
                print(f"[DEBUG] Starting gradient calculation for {len(atoms)} atoms")

            for i in range(len(atoms)):
                for j in range(3):
                    # Forward difference
                    forward_positions = new_positions.copy()
                    forward_positions[i, j] += epsilon

                    # Apply forward positions
                    current_atoms.positions = forward_positions

                    # Calculate forward variance
                    forward_variance, _, _ = self._calculate_force_variance(current_atoms)

                    # Calculate forward energy and probability if needed
                    if self.include_probability:
                        forward_energy = self._calculate_energy(current_atoms)
                        forward_probability = self._calculate_probability(
                            forward_energy, self.temperature, normalization_constant
                        )
                        forward_loss = forward_probability * forward_variance
                    else:
                        forward_loss = forward_variance

                    # Estimate gradient
                    grad[i, j] = (forward_loss - loss) / epsilon

            self.timer.stop("gradient_calculation")

            # Convert gradient to torch tensor with matching dtype
            grad_tensor = torch.tensor(grad, device="cpu", dtype=self.dtype)

            # Manually set the gradient
            displacement.grad = -grad_tensor  # Negative because optimizer minimizes

            # Step optimizer
            self.timer.start("optimizer_step")
            optimizer.step()
            self.timer.stop("optimizer_step")

            self.timer.stop(f"step_{step}")

            if self.debug:
                print(f"[DEBUG] Step {step} completed in {self.timer.timers[f'step_{step}'][-1]:.4f} seconds")
                print(f"[DEBUG] Current metrics - Variance: {variance:.6f}, Probability: {probability:.6f}, Loss: {loss:.6f}")
            else:
                print(f"Step {step}: Variance={variance:.6f}, Probability={probability:.6f}, Loss={loss:.6f}")

        # Create best atoms
        best_atoms = atoms.copy()
        best_atoms.positions = best_positions
        best_variance, _, _ = self._calculate_force_variance(best_atoms)
        best_energy = self._calculate_energy(best_atoms)
        best_probability = self._calculate_probability(
            best_energy, self.temperature, normalization_constant
        )

        # Save best structure
        best_atoms.info['variance'] = best_variance
        best_atoms.info['energy'] = best_energy
        best_atoms.info['probability'] = best_probability
        best_atoms.info['loss'] = best_loss
        best_atoms.info['final'] = True

        # Create plots
        self._create_plots(
            output_path, struct_name,
            loss_history, variance_history,
            energy_history, probability_history
        )

        # Save final results
        self._save_results(
            output_path, struct_name,
            atoms, best_atoms,
            initial_variance, best_variance,
            initial_energy, best_energy,
            initial_probability, best_probability,
            initial_loss, best_loss,
            loss_history, variance_history,
            energy_history, probability_history
        )

        # Print performance summary
        self.timer.summary()

        return best_atoms, best_loss, loss_history

    def _create_plots(self, output_path, struct_name, losses, variances, energies, probabilities):
        """Create and save plots for optimization results."""
        self.timer.start("create_plots")

        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(losses, marker='o', label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title(f'Loss vs. Iteration - {struct_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path / f"{struct_name}_loss_plot.png")
        plt.close()

        # Plot variance history
        plt.figure(figsize=(10, 6))
        plt.plot(variances, marker='o', color='green', label='Force Variance')
        plt.xlabel('Iteration')
        plt.ylabel('Force Variance')
        plt.title(f'Force Variance vs. Iteration - {struct_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path / f"{struct_name}_variance_plot.png")
        plt.close()

        # Plot energy history
        plt.figure(figsize=(10, 6))
        plt.plot(energies, marker='o', color='red', label='Energy (eV)')
        plt.xlabel('Iteration')
        plt.ylabel('Energy (eV)')
        plt.title(f'Energy vs. Iteration - {struct_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path / f"{struct_name}_energy_plot.png")
        plt.close()

        # Plot probability history if available
        if probabilities:
            plt.figure(figsize=(10, 6))
            plt.plot(probabilities, marker='o', color='purple', label='Probability')
            plt.xlabel('Iteration')
            plt.ylabel('Boltzmann Probability')
            plt.title(f'Probability vs. Iteration - {struct_name}')
            plt.grid(True)
            plt.legend()
            plt.savefig(output_path / f"{struct_name}_probability_plot.png")
            plt.close()

        # Combined plot
        fig, ax1 = plt.subplots(figsize=(12, 7))

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss / Variance', color='blue')
        ax1.plot(losses, marker='o', color='blue', label='Loss')
        ax1.plot(variances, marker='s', color='green', label='Variance')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Energy (eV)', color='red')
        ax2.plot(energies, marker='^', color='red', label='Energy')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')

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
        """Save detailed results to JSON file."""
        self.timer.start("save_results")

        import json

        results = {
            'structure_name': struct_name,
            'initial': {
                'variance': float(initial_variance),
                'energy': float(initial_energy),
                'probability': float(initial_probability),
                'loss': float(initial_loss)
            },
            'final': {
                'variance': float(best_variance),
                'energy': float(best_energy),
                'probability': float(best_probability),
                'loss': float(best_loss)
            },
            'improvement': {
                'variance_ratio': float(best_variance / initial_variance),
                'loss_ratio': float(best_loss / initial_loss)
            },
            'history': {
                'loss': [float(x) for x in loss_history],
                'variance': [float(x) for x in variance_history],
                'energy': [float(x) for x in energy_history],
                'probability': [float(x) for x in probability_history]
            },
            'parameters': {
                'learning_rate': self.learning_rate,
                'temperature': self.temperature,
                'include_probability': self.include_probability
            },
            'output_files': {
                'trajectory': f"{struct_name}_adversarial.xyz",
                'loss_plot': f"{struct_name}_loss_plot.png",
                'variance_plot': f"{struct_name}_variance_plot.png",
                'energy_plot': f"{struct_name}_energy_plot.png",
                'probability_plot': f"{struct_name}_probability_plot.png",
                'combined_plot': f"{struct_name}_combined_plot.png"
            },
            'performance': {
                'timers': {name: {'total': sum(times), 'average': sum(times)/len(times) if times else 0, 'count': len(times)}
                          for name, times in self.timer.timers.items()}
            }
        }

        with open(output_path / f"{struct_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.timer.stop("save_results")

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