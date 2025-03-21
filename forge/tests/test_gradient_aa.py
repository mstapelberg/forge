#!/usr/bin/env python
"""Test script for gradient-based adversarial attack using PyTorch autograd."""

import argparse
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import ase.io
from ase import Atoms
from pathlib import Path
from tqdm import tqdm

from forge.core.adversarial_attack import AdversarialCalculator


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
    
    def __init__(self, model_paths, energy_list, device='cuda', learning_rate=0.01, 
                 temperature=0.86, include_probability=True, debug=False):
        """Initialize optimizer with model paths.
        
        Args:
            model_paths: List of paths to model files
            energy_list: List of energies after running force variance calculations over entire dataset
            device: Device to run on ('cpu' or 'cuda')
            learning_rate: Learning rate for gradient ascent
            temperature: Temperature for probability weighting (eV)
            include_probability: Whether to include probability term in loss
            debug: Whether to print debug messages
        """
        self.model_paths = model_paths
        self.device = device
        self.learning_rate = learning_rate
        self.energy_list = energy_list
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
        return np.exp(-energy / (temperature)) / normalization_constant
    
    def optimize(self, atoms, n_iterations=60, min_distance=1.5, output_dir='.'):
        """Run gradient-based adversarial attack optimization.
        
        Args:
            atoms: ASE Atoms object
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
        normalization_constant = 1.0
        
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
        initial_Q, initial_shifted_energies = self._calculate_normalization_constant(
            self.energy_list, self.temperature
        )
        initial_probability = self._calculate_probability(
            initial_energy, self.temperature, initial_Q
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
        
        ase.io.write(output_file, current_atoms, write_results=False)
        
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
                initial_Q
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
            
            ase.io.write(output_file, current_atoms, append=True, write_results=False)
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
                            forward_energy, self.temperature, initial_Q
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
            best_energy, self.temperature, initial_Q 
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


def run_test(
    xyz_file,
    output_dir,
    model_paths,
    learning_rate=0.01,
    n_iterations=60,
    min_distance=1.5,
    include_probability=True,
    temperature=0.86,
    device="cuda",
    debug=False,
    profile_atom_indices=None,
    batch_size=None
):
    """Run gradient-based adversarial attack test."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load structures
    atoms_list = ase.io.read(xyz_file, ':')
    print(f"[INFO] Loaded {len(atoms_list)} structures from {xyz_file}")
    
    if debug:
        print(f"[DEBUG] Running with {len(model_paths)} models on device: {device}")
        for i, path in enumerate(model_paths):
            print(f"[DEBUG] Model {i+1}: {path}")
    
    # If profiling specific atoms, filter the list
    if profile_atom_indices is not None:
        indices = [int(idx) for idx in profile_atom_indices.split(',')]
        if debug:
            print(f"[DEBUG] Profiling only atoms at indices: {indices}")
        for i, atoms in enumerate(atoms_list):
            # Create a subset of atoms for profiling
            subset = atoms[indices].copy()
            subset.info = atoms.info.copy()
            atoms_list[i] = subset
            if debug:
                print(f"[DEBUG] Reduced structure {i} from {len(atoms)} to {len(subset)} atoms")
    
    # Initialize optimizer
    optimizer = GradientAdversarialOptimizer(
        model_paths=model_paths,
        device=device,
        learning_rate=learning_rate,
        temperature=temperature,
        include_probability=include_probability,
        debug=debug
    )
    
    # Process each structure
    results = []
    for i, atoms in enumerate(atoms_list):
        # Set structure name if not present
        if 'structure_name' not in atoms.info:
            atoms.info['structure_name'] = f"structure_{i}"
            
        struct_name = atoms.info['structure_name']
        print(f"\n[INFO] Processing structure: {struct_name} with {len(atoms)} atoms")
        
        # Create subdir for each structure
        struct_dir = output_path / struct_name
        struct_dir.mkdir(exist_ok=True)
        
        # Run optimization
        start_time = time.time()
        best_atoms, best_loss, loss_history = optimizer.optimize(
            atoms=atoms,
            n_iterations=n_iterations,
            min_distance=min_distance,
            output_dir=struct_dir
        )
        end_time = time.time()
        
        # Calculate improvement factor
        initial_loss = loss_history[0]
        improvement = best_loss / initial_loss if initial_loss != 0 else float('inf')
        
        # Add to results
        results.append({
            'structure_name': struct_name,
            'n_atoms': len(atoms),
            'initial_loss': initial_loss,
            'best_loss': best_loss,
            'improvement_factor': improvement,
            'runtime_seconds': end_time - start_time,
            'output_dir': str(struct_dir)
        })
        
        print(f"[INFO] Completed optimization for {struct_name}")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {best_loss:.6f}")
        print(f"  Improvement factor: {improvement:.2f}x")
        print(f"  Runtime: {end_time - start_time:.2f} seconds")
    
    # Save overall summary
    with open(output_path / "summary.json", 'w') as f:
        json.dump({
            'input_file': xyz_file,
            'number_of_structures': len(atoms_list),
            'parameters': {
                'learning_rate': learning_rate,
                'n_iterations': n_iterations,
                'min_distance': min_distance,
                'include_probability': include_probability,
                'temperature': temperature,
                'device': device,
                'debug': debug,
                'profile_atom_indices': profile_atom_indices,
                'batch_size': batch_size
            },
            'results': results
        }, f, indent=2)
    
    print("\n[INFO] Overall Results:")
    print(f"  Structures processed: {len(atoms_list)}")
    print(f"  Average improvement factor: {np.mean([r['improvement_factor'] for r in results]):.2f}x")
    print(f"  Average runtime per structure: {np.mean([r['runtime_seconds'] for r in results]):.2f} seconds")
    print(f"  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test gradient-based adversarial attack optimization."
    )
    parser.add_argument(
        "xyz_file",
        help="Input XYZ file containing structures"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save optimization results"
    )
    parser.add_argument(
        "--model_paths",
        nargs='+',
        required=True,
        help="Paths to model files"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient ascent"
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=60,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        default=1.5,
        help="Minimum allowed distance between atoms (Å)"
    )
    parser.add_argument(
        "--no_probability",
        action="store_true",
        help="Disable probability term in loss function (use variance only)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.86,
        help="Temperature for probability weighting (eV)"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda'],
        default='cuda',
        help="Device to run on"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output with detailed timing information"
    )
    parser.add_argument(
        "--profile_atoms",
        type=str,
        help="Comma-separated list of atom indices to profile (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of atoms to process in each batch (for large structures)"
    )
    
    args = parser.parse_args()
    
    run_test(
        xyz_file=args.xyz_file,
        output_dir=args.output_dir,
        model_paths=args.model_paths,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        min_distance=args.min_distance,
        include_probability=not args.no_probability,
        temperature=args.temperature,
        device=args.device,
        debug=args.debug,
        profile_atom_indices=args.profile_atoms,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 
