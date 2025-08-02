#!/usr/bin/env python
"""Core classes for adversarial attack workflow."""

import numpy as np
from ase import Atoms
from ase.io import read, write
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

# --- Conditional MACE imports for autograd functionality ---
try:
    from mace.calculators import MACECalculator
    from mace.data import AtomicData, config_from_atoms
    from mace.tools.torch_geometric import Batch
    MACE_AUTOGRAD_AVAILABLE = True
except ImportError:
    MACE_AUTOGRAD_AVAILABLE = False
    # Placeholder classes for when MACE is not available
    MACECalculator = None
    AtomicData = None
    config_from_atoms = None
    Batch = None

# --- Conditional NequIP imports for autograd functionality ---
try:
    from nequip.data import AtomicDataDict, from_ase
    NEQUIP_AUTOGRAD_AVAILABLE = True
except ImportError:
    NEQUIP_AUTOGRAD_AVAILABLE = False
    # Placeholder classes for when NequIP is not available  
    AtomicDataDict = None
    from_ase = None

# --- New Imports for Generic Calculator Interface ---
from forge.calculators import create_ensemble_calculator, BaseEnsembleCalculator


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
        if self.debug:
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
                 energy_list=None, use_energy_per_atom=False, backend='auto'):
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
            backend: Calculator backend to use ('mace', 'allegro', or 'auto' for auto-detection)
        """
        self.model_paths = model_paths
        self.device = device
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.include_probability = include_probability
        self.debug = debug
        self.timer = Timer(debug=debug)
        self.dtype = torch.float32
        self.use_energy_per_atom = use_energy_per_atom
        self.backend = backend
        
        # Initialize ensemble calculator using the generic interface
        self.timer.start("calculator_init")
        try:
            self.calculator = create_ensemble_calculator(
                model_paths=self.model_paths,
                backend=self.backend,
                device=self.device,
                default_dtype='float32'
            )
            if self.debug:
                print(f"[DEBUG] Initialized {type(self.calculator).__name__} with {len(self.calculator.models)} models")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ensemble calculator in optimizer: {e}")
            raise
        
        # Get the raw torch models - for Allegro, we need to access them differently
        if self.backend.lower() in ['allegro', 'nequip']:
            # For Allegro backend, the raw models are stored in calculator._models
            self.models = self.calculator._models
        else:
            # For other backends like MACE
            self.models = self.calculator.models
        
        # For gradient computation, we want parameters to allow gradient flow but not accumulate gradients
        # Don't modify requires_grad here - we'll handle it during optimization

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
                 patience: int = 20, shake_std: float = 0.05, shake: bool = False):
        """Run gradient-based adversarial attack optimization using PyTorch Autograd.

        Args:
            atoms: ASE Atoms object (must contain 'structure_id' and 'config_type' in info)
            generation: Generation identifier (integer) for the output structures.
            n_iterations: Maximum number of iterations
            min_distance: Minimum allowed distance between atoms
            output_dir: Directory to save plots
            patience: Number of steps without loss improvement before action (shake or stop).
            shake_std: Standard deviation (in Angstrom) for random shake (if shake=True).
            shake: If True, apply random shake when patience is reached. If False, stop optimization.

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

        # --- Setup for PyTorch Autograd ---
        self.timer.start("autograd_setup")
        
        # Displacement tensor will be optimized.
        displacement = torch.zeros(
            (len(atoms), 3),
            requires_grad=True,
            device=self.device,
            dtype=self.dtype
        )
        
        # Backend-specific setup
        if self.backend.lower() in ['mace']:
            if not MACE_AUTOGRAD_AVAILABLE:
                raise ImportError(
                    "MACE backend requested but MACE is not available for autograd optimization. "
                    "Please install MACE or use a different backend."
                )
            setup_result = self._setup_mace_autograd(atoms, displacement)
        elif self.backend.lower() in ['allegro', 'nequip']:
            setup_result = self._setup_allegro_autograd(atoms, displacement)
        else:
            raise NotImplementedError(
                f"PyTorch autograd optimization is not yet implemented for backend '{self.backend}'. "
                f"Supported backends: 'mace', 'allegro', 'nequip'"
            )
        
        original_positions_tensor = setup_result['original_positions']
        optimizer = torch.optim.Adam([displacement], lr=self.learning_rate)
        self.timer.stop("autograd_setup")

        # Initialize tracking variables
        best_loss = -float('inf')
        steps_without_improvement = 0
        trajectory = []

        print(f"[INFO] Starting optimization for parent ID: {parent_id}, gen: {generation} with PyTorch Autograd (patience={patience}, shake={shake})")

        # --- Optimization loop ---
        for step in tqdm(range(n_iterations), desc=f"Optimizing {struct_name}"):
            self.timer.start(f"step_{step}")

            # --- Patience and Shake Logic ---
            if steps_without_improvement >= patience:
                if shake:
                    print(f"\n[INFO] Step {step}: No improvement for {patience} steps. Applying random shake (std={shake_std} Å).")
                    with torch.no_grad():
                        noise = torch.randn_like(original_positions_tensor) * shake_std
                        original_positions_tensor.data.add_(noise)
                        displacement.data.zero_()
                    # Reinitialize optimizer state
                    optimizer = torch.optim.Adam([displacement], lr=self.learning_rate)
                    steps_without_improvement = 0
                else:
                    print(f"\n[INFO] Step {step}: No improvement for {patience} steps and shake=False. Stopping optimization.")
                    break

            optimizer.zero_grad()

            # --- Forward Pass ---
            self.timer.start("forward_pass")
            # Update positions using backend-specific approach
            new_positions_tensor = original_positions_tensor + displacement
            
            # Debug: Check if gradients are preserved
            if self.debug:
                print(f"[DEBUG] displacement.requires_grad: {displacement.requires_grad}")
                print(f"[DEBUG] new_positions_tensor.requires_grad: {new_positions_tensor.requires_grad}")
                print(f"[DEBUG] new_positions_tensor.grad_fn: {new_positions_tensor.grad_fn}")
            
            if setup_result['backend_type'] == 'mace':
                data = setup_result['data']
                data.positions = new_positions_tensor
                positions_for_distance_check = new_positions_tensor
            elif setup_result['backend_type'] == 'allegro':
                data = setup_result['data'].copy()  # Create a copy to avoid modifying original
                # Ensure the new positions tensor retains gradients
                data[AtomicDataDict.POSITIONS_KEY] = new_positions_tensor
                positions_for_distance_check = new_positions_tensor
            else:
                raise ValueError(f"Unknown backend type: {setup_result['backend_type']}")

            # --- Minimum Distance Check (with torch) ---
            if len(atoms) > 1:
                dists = torch.pdist(positions_for_distance_check)
                min_dist_val = torch.min(dists)
                if min_dist_val < min_distance:
                    if self.debug:
                        print(f"[DEBUG] Step {step}: Min dist violated: {min_dist_val:.3f} Å. Scaling back.")
                    with torch.no_grad():
                        # If displacement is zero, scaling won't help. Add a small random perturbation before scaling.
                        if torch.norm(displacement.data) < 1e-6:
                             displacement.data.add_(torch.randn_like(displacement.data) * 0.01)
                        
                        displacement.data *= 0.9
                    self.timer.stop("forward_pass")
                    self.timer.stop(f"step_{step}")
                    continue

            # --- Calculate forces and variance using autograd ---
            forces_list = []
            energy_list = [] # For probability calculation if needed
            
            if setup_result['backend_type'] == 'mace':
                for torch_model in self.models: # Iterate directly over raw torch models
                    # model expects a dict, not a Batch object.
                    # We set training=True to enable creation of the graph for second derivatives, which is required for autograd.
                    output = torch_model(data.to_dict(), training=True, compute_force=True)
                    forces_list.append(output['forces'])
                    # Always append energy for logging, even if probability is not used in loss
                    if 'energy' in output:
                        energy_list.append(output['energy'])
            elif setup_result['backend_type'] == 'allegro':
                for i, torch_model in enumerate(self.models): # Iterate directly over raw torch models
                    # Set model to training mode to enable gradient computation
                    torch_model.train()
                    
                    # Debug: Check data before model call
                    if self.debug and i == 0:  # Only debug for first model
                        pos_tensor = data[AtomicDataDict.POSITIONS_KEY]
                        print(f"[DEBUG] Model input positions.requires_grad: {pos_tensor.requires_grad}")
                        print(f"[DEBUG] Model input positions.grad_fn: {pos_tensor.grad_fn}")
                    
                    # For NequIP/Allegro, use the forces directly from model output
                    with torch.enable_grad():
                        # Ensure position tensor requires gradients
                        positions = data[AtomicDataDict.POSITIONS_KEY]
                        if not positions.requires_grad:
                            positions = positions.requires_grad_(True)
                            data[AtomicDataDict.POSITIONS_KEY] = positions
                        
                        # Get output with gradients enabled - model computes forces directly
                        output = torch_model(data)
                        
                        # Debug: Check what keys are available
                        if self.debug and i == 0:
                            print(f"[DEBUG] NequIP model output keys: {list(output.keys())}")
                        
                        # Get forces directly from model output (don't detach!)
                        if AtomicDataDict.FORCE_KEY in output:
                            forces = output[AtomicDataDict.FORCE_KEY]
                        elif 'forces' in output:
                            forces = output['forces']
                        else:
                            raise KeyError(f"No forces found in model output. Available keys: {list(output.keys())}")
                        
                        # Get energy for logging
                        if AtomicDataDict.TOTAL_ENERGY_KEY in output:
                            energy = output[AtomicDataDict.TOTAL_ENERGY_KEY]
                        elif 'total_energy' in output:
                            energy = output['total_energy']
                        elif 'energy' in output:
                            energy = output['energy']
                        else:
                            energy = None
                        
                        # Debug: Check force gradients
                        if self.debug and i == 0:
                            print(f"[DEBUG] Model forces.requires_grad: {forces.requires_grad}")
                            print(f"[DEBUG] Model forces.grad_fn: {forces.grad_fn}")
                            if energy is not None:
                                print(f"[DEBUG] Model energy.requires_grad: {energy.requires_grad}")
                        
                        forces_list.append(forces)
                        if energy is not None:
                            energy_list.append(energy)
                    
                    # Return model to eval mode
                    torch_model.eval()

            forces_tensor = torch.stack(forces_list)
            
            # --- Calculate Variance (Torch native) ---
            force_magnitudes = torch.linalg.norm(forces_tensor, dim=2, keepdim=True)
            force_magnitudes = torch.where(force_magnitudes < 1e-10, torch.tensor(1.0, device=self.device, dtype=self.dtype), force_magnitudes)
            normalized_forces = forces_tensor / force_magnitudes
            atom_variances = torch.var(normalized_forces, dim=0) # Variance across models
            total_atom_variances = torch.sum(atom_variances, dim=1) # Sum of x,y,z variances
            variance = torch.mean(total_atom_variances)

            # --- Calculate Loss ---
            mean_energy = None
            if self.include_probability:
                 if energy_list:
                     energy_tensor = torch.stack(energy_list)
                     mean_energy = torch.mean(energy_tensor)
                     # Note: probability calculation with autograd still needs full implementation
                     print("[WARN] probability calculation with autograd is not fully implemented. Ignoring probability term for loss.")
                 else:
                     print("[WARN] `include_probability` is True but energy list is empty. Ignoring probability term.")

            # Also compute mean_energy if it hasn't been, for logging purposes
            if mean_energy is None and energy_list:
                energy_tensor = torch.stack(energy_list)
                mean_energy = torch.mean(energy_tensor)
            
            loss_val = variance
            self.timer.stop("forward_pass")

            # Update best loss tracking and patience counter
            current_loss_item = loss_val.item()
            if current_loss_item > best_loss:
                best_loss = current_loss_item
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # --- Store step results in trajectory ---
            self.timer.start("store_trajectory_step")
            step_atoms = atoms.copy() # Create a fresh copy
            step_atoms.positions = new_positions_tensor.detach().cpu().numpy()
            step_atoms.calc = None # Always clear calculator

            # Store results in info
            step_atoms.info.update({
                'parent_id': parent_id,
                'generation': generation,
                'config_type': new_config_type,
                'step': step,
                'variance': variance.item(),
                'loss': current_loss_item,
                'energy': mean_energy.item() if mean_energy is not None else float('nan'),
                # 'probability': probability.item(), # if calculated
            })
            # Clean up inherited info
            step_atoms.info.pop('structure_id', None)
            step_atoms.info.pop('calculation_info', None)
            
            # Store mean forces in arrays
            mean_forces = torch.mean(forces_tensor, dim=0).detach().cpu().numpy()
            step_atoms.arrays['forces'] = mean_forces

            trajectory.append(copy.deepcopy(step_atoms))
            self.timer.stop("store_trajectory_step")

            # --- Backward Pass ---
            self.timer.start("backward_pass")
            # We want to MAXIMIZE variance/loss, so we minimize its NEGATIVE.
            objective = -loss_val
            objective.backward()
            self.timer.stop("backward_pass")
            
            # --- Optimizer Step ---
            self.timer.start("optimizer_step")
            optimizer.step()
            self.timer.stop("optimizer_step")

            self.timer.stop(f"step_{step}")
            
            # Log progress
            if self.debug and (step % 5 == 0 or step == n_iterations - 1):
                 print(f"[DEBUG] Step {step}: Var={variance.item():.6f}, Loss={current_loss_item:.6f}, LR={optimizer.param_groups[0]['lr']:.1e}, Disp_norm={torch.norm(displacement.data):.4f}")
            elif not self.debug and (step % 10 == 0 or step == n_iterations - 1):
                 # Reduced probability logging as it's not used in loss
                 print(f"Step {step}: Variance={variance.item():.6f}, Loss={current_loss_item:.6f}")


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

    def _setup_mace_autograd(self, atoms, displacement):
        """Setup MACE-specific data structures for autograd optimization."""
        if not MACE_AUTOGRAD_AVAILABLE:
            raise ImportError("MACE not available for autograd optimization")
        
        # Get parameters from the initialized calculator  
        ref_model = self.models[0]
        r_max = ref_model.r_max.item()
        z_table = self.calculator.z_table
        
        # Create a data object for MACE, which is then batched.
        config = config_from_atoms(atoms)
        data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
        # Move the data to the correct device after batching.
        data = Batch.from_data_list([data]).to(self.device)
        
        # Keep track of the original positions on the correct device.
        original_positions_tensor = data.positions.clone()
        
        return {
            'data': data,
            'original_positions': original_positions_tensor,
            'backend_type': 'mace'
        }
    
    def _setup_allegro_autograd(self, atoms, displacement):
        """Setup Allegro/NequIP-specific data structures for autograd optimization."""
        if not NEQUIP_AUTOGRAD_AVAILABLE:
            raise ImportError(
                "NequIP/Allegro not available for autograd optimization. "
                "Please install NequIP with: pip install nequip"
            )
        
        # Get the first calculator to access transforms and setup
        ref_calc = self.calculator._calculators[0]
        
        # Prepare data using NequIP's pipeline
        data = from_ase(atoms)
        for transform in ref_calc.transforms:
            data = transform(data)
        data = AtomicDataDict.to_(data, self.device)
        
        # Convert positions to tensor with proper dtype - we'll add gradients later
        # Use clone() instead of torch.tensor() to avoid warning
        positions_tensor = data[AtomicDataDict.POSITIONS_KEY]
        if isinstance(positions_tensor, torch.Tensor):
            original_positions_tensor = positions_tensor.detach().clone().to(device=self.device, dtype=self.dtype)
        else:
            original_positions_tensor = torch.tensor(
                positions_tensor, 
                device=self.device, 
                dtype=self.dtype,
                requires_grad=False
            )
        
        # IMPORTANT: Ensure all data tensors are in the correct dtype to avoid mismatch errors
        # Packaged models might load with float64, but we want consistent float32
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                # Convert floating point tensors to the target dtype
                data[key] = value.to(dtype=self.dtype, device=self.device)
        
        return {
            'data': data,
            'original_positions': original_positions_tensor,
            'backend_type': 'allegro',
            'ref_calc': ref_calc  # Store reference to calculator for transforms
        }

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
            # The use_cueq flag is specific to certain setups, so we handle it carefully.
            calc_kwargs = {
                'model_paths': model_path,
                'device': self.device,
                'default_dtype': self.default_dtype
            }
            if self.device == 'cuda':
                # This flag may not always be present or needed.
                # A try-except block could make this more robust if needed.
                # For now, assume it's a valid kwarg for the user's MACE version.
                try:
                    # Attempt to initialize with use_cueq
                    model = MACECalculator(**calc_kwargs, use_cueq=True)
                except TypeError:
                    # Fallback if use_cueq is not a valid argument
                    print("[INFO] MACECalculator does not accept 'use_cueq'. Initializing without it.")
                    model = MACECalculator(**calc_kwargs)
            else:
                model = MACECalculator(**calc_kwargs)
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
            # To prevent ASE from using cached results from the previous model,
            # we assign a new empty dictionary to .results.
            # This is safer than .clear() as it creates the attribute if it's missing.
            atoms.results = {}
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