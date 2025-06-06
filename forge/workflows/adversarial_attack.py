#!/usr/bin/env python
"""Workflow script for running gradient-based adversarial attacks."""

import numpy as np
import torch
import argparse
import os
import time
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from ase import Atoms
from ase.io import write

# --- Core Forge Imports ---
# Assume forge is installed or PYTHONPATH is set correctly
from forge.core.database import DatabaseManager
from forge.core.adversarial_attack import GradientAdversarialOptimizer
from mace.calculators import MACECalculator # Use the official calculator

# --- Helper: Timer Class (copied for simplicity or import if structure allows) ---
class Timer:
    """Simple timer class for performance monitoring."""
    def __init__(self, debug=False):
        self.debug = debug
        self.timers = {}
        self.starts = {}

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        if name in self.starts:
            elapsed = time.time() - self.starts[name]
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(elapsed)
            if self.debug:
                print(f"[DEBUG Timer] {name}: {elapsed:.4f} seconds")
            return elapsed
        return 0

    def summary(self):
        print("\n===== Performance Summary =====")
        for name, times in self.timers.items():
            total = sum(times)
            avg = total / len(times) if times else 0
            count = len(times)
            print(f"{name}: Total={total:.4f}s, Count={count}, Avg={avg:.4f}s")
        print("==============================\n")


# --- Main Workflow Function ---

def run_adversarial_attacks(
    db_manager: DatabaseManager, # Accept initialized DB manager
    structure_ids: List[int],
    model_paths: List[str],
    top_n: int,
    generation: int,
    n_iterations: int,
    learning_rate: float,
    temperature: float,
    include_probability: bool,
    min_distance: float,
    use_energy_per_atom: bool = True,
    device: Optional[str] = None,
    debug: bool = False,
    output_dir: str = '.',
    save_output: bool = False, # Added save_output flag back
    patience: int = 25, # Add patience parameter
    shake: bool = False, # Add shake parameter
    shake_std: float = 0.05, # Add shake_std for consistency
    ranking_metric: str = 'variance',
    reference_calculator: str = 'vasp'
) -> Union[Dict[int, List[Atoms]], None]:
    """
    Runs the gradient-based adversarial attack workflow using a provided DatabaseManager.

    Args:
        db_manager: An initialized instance of DatabaseManager.
        structure_ids: List of initial structure IDs from the database.
        model_paths: List of paths to MACE model files for the ensemble.
        top_n: Number of top variance structures to select and optimize.
        generation: Generation tag for the new structures (used in metadata).
        n_iterations: Number of optimization steps.
        learning_rate: Optimizer learning rate.
        temperature: Temperature (in eV) for Boltzmann weighting.
        include_probability: Whether to weight loss by Boltzmann probability.
        min_distance: Minimum allowed interatomic distance (Angstrom).
        use_energy_per_atom: Use energy per atom for probability calculations.
        device: Compute device ('cuda', 'cpu', or None for auto-detect).
        debug: Enable verbose debug printing.
        output_dir: Directory to save output files.
        save_output: If True, save trajectories and plots to output_dir and return None.
                      If False, return the dictionary of trajectories.
        shake: If True, apply random shake when optimizer patience is reached. If False, stop.
        shake_std: Standard deviation for the random shake if shake is True.
        ranking_metric: Metric to rank structures for attack ('variance' or 'force_rmse').
        reference_calculator: Calculator name to fetch for reference forces if ranking by RMSE.

    Returns:
        If save_output is False (default): Returns a dictionary where keys are parent IDs
        and values are lists of Atoms objects representing the optimization trajectory.
        If save_output is True: Saves each trajectory to `output_dir/structure_{parent_id}.xyz`
        and plots to `output_dir/plots/`, then returns None.
    """
    wf_timer = Timer(debug=debug)
    wf_timer.start("total_workflow")

    # --- Initialization ---
    wf_timer.start("init")
    print("--- Starting Adversarial Attack Workflow ---")
    if device is None:
        computed_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        computed_device = device
    print(f"Using device: {computed_device}")

    # --- Use the provided DatabaseManager ---
    # Ensure the provided object is valid (basic check)
    if not isinstance(db_manager, DatabaseManager):
        raise TypeError("db_manager must be an instance of DatabaseManager")
    if db_manager.dry_run:
        print("[INFO] Running workflow with DatabaseManager in DRY RUN mode.")

    # Initialize the official MACECalculator for the ranking stage
    print("Initializing MACE calculator for ranking...")
    try:
        ranking_calc = MACECalculator(
            model_paths=model_paths,
            device=computed_device,
            default_dtype='float32'
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize MACECalculator for ranking: {e}")
        return {}
    print(f"Initialized calculator with {len(ranking_calc.models)} models.")
    wf_timer.stop("init")

    # --- Fetch Initial Structures & Data ---
    wf_timer.start("fetch_data")
    print(f"\nFetching initial structures and latest calculations for {len(structure_ids)} IDs...")

    # Determine which calculation to fetch based on ranking metric
    calc_to_fetch = reference_calculator if ranking_metric == 'force_rmse' else None
    fetch_source = "latest (for energy)" if calc_to_fetch is None else reference_calculator
    print(f"Fetching reference calculations from '{fetch_source}' source...")

    # Use get_batch_atoms_with_calculation to get atoms + specific/latest calc data attached
    initial_atoms_list = db_manager.get_batch_atoms_with_calculation(structure_ids, calculator=calc_to_fetch)

    if not initial_atoms_list:
        print("[ERROR] No valid structures found for the given IDs in the database.")
        return {}

    print(f"Successfully fetched {len(initial_atoms_list)} structures.")

    # Prepare data for ranking and optimization
    structures_to_process = []
    initial_energies_for_norm = [] # List of energies (total or per atom)

    for atoms in initial_atoms_list:
        if not atoms: continue # Skip if fetching failed for some reason

        parent_id = atoms.info.get('structure_id')
        if parent_id is None:
             print(f"[WARN] Skipping structure with missing structure_id in info: {atoms.get_chemical_formula()}")
             continue

        # Extract config_type preferentially from attached calculation_info, fallback to atoms.info
        config_type = "unknown"
        if 'calculation_info' in atoms.info and isinstance(atoms.info['calculation_info'], dict):
             config_type = atoms.info['calculation_info'].get('config_type', config_type) # Check calc info first
        config_type = atoms.info.get('config_type', config_type) # Fallback to top-level info

        energy_key = 'energy_per_atom' if use_energy_per_atom else 'energy'
        energy_val = atoms.info.get(energy_key) # Check if energy/atom was already in info

        # If probability is needed, ensure we have the correct energy type for normalization
        if include_probability:
            current_energy = atoms.info.get('energy') # Total energy from latest calc
            num_atoms = len(atoms)
            if num_atoms > 0 and current_energy is not None:
                 if use_energy_per_atom:
                      initial_energies_for_norm.append(current_energy / num_atoms)
                 else:
                      initial_energies_for_norm.append(current_energy)
            elif debug:
                 print(f"[DEBUG] Missing energy or zero atoms for structure {parent_id}, cannot use for normalization.")


        structures_to_process.append({
            'id': parent_id,
            'atoms': atoms,
            'config_type': config_type
        })

    if not structures_to_process:
        print("[ERROR] No structures eligible for processing after initial fetch.")
        return {}

    if include_probability and not initial_energies_for_norm:
        print("[WARNING] `include_probability` is True, but no valid initial energies found for normalization. Proceeding deterministically (Q=1.0).")
        # We let the optimizer handle Q=1.0 internally
    elif include_probability:
        print(f"Using {len(initial_energies_for_norm)} initial energies for normalization constant.")

    wf_timer.stop("fetch_data")


    # --- Calculate Initial Variances and Rank ---
    wf_timer.start("rank_structures")
    initial_metrics = []

    if ranking_metric == 'force_rmse':
        print("\nCalculating initial force RMSE...")
        for i, data in enumerate(tqdm(structures_to_process, desc="Initial Force RMSE Calc")):
            atoms = data['atoms'].copy() # Use a copy to prevent side effects
            try:
                # 1. Get reference forces stored in the Atoms object from the database query.
                if not atoms.has("forces"):
                    # This means the database didn't return forces for this structure.
                    print(f"\n[DEBUG] `atoms.has('forces')` returned False. "
                          f"Atoms.info for structure {data['id']}: {atoms.info}")
                    raise ValueError(
                        "Reference forces not found in the Atoms object from the database."
                    )
                # Retrieve the reference forces directly from the arrays dictionary.
                ref_forces = atoms.arrays['forces']

                # 2. Get mean predicted forces from the ensemble using the ranking calculator.
                atoms.calc = ranking_calc
                model_forces = atoms.get_forces(apply_constraint=False) # Get mean forces
                # The MACECalculator with multiple models returns the average, so we don't need to calculate it.
                # However, for variance, we need individual forces. Let's adapt.
                # For RMSE, the mean is fine.

                # To be consistent, let's get all forces and take the mean here.
                # The .get_forces() on an ensemble calculator in ASE returns the mean.
                # To get all forces, we need to iterate, which is slow.
                # Let's assume for now the user wants RMSE against the *mean* of the ensemble.
                # This is a reasonable and fast approach.
                mean_model_forces = model_forces

                # 3. Calculate RMSE
                if ref_forces.shape != mean_model_forces.shape:
                    raise ValueError(f"Shape mismatch between reference forces {ref_forces.shape} and model forces {mean_model_forces.shape}")

                rmse = np.sqrt(np.mean((mean_model_forces - ref_forces)**2))
                initial_metrics.append({'id': data['id'], 'metric': float(rmse), 'index': i})

            except Exception as e:
                print(f"\n[WARN] Failed force RMSE calculation for structure {data['id']}: {e}")
                initial_metrics.append({'id': data['id'], 'metric': -1, 'index': i})  # Mark as failed

    elif ranking_metric == 'variance':
        print("\nCalculating initial variances...")
        for i, data in enumerate(tqdm(structures_to_process, desc="Initial Variance Calc")):
            atoms = data['atoms']
            try:
                 # To get variance, we must get forces from each model.
                 # The public API for MACECalculator averages them.
                 # We will access the models list directly for this step.
                 all_forces = []
                 with torch.no_grad():
                     for model in ranking_calc.models:
                         atoms.calc = model # Temporarily assign single model
                         all_forces.append(atoms.get_forces(apply_constraint=False))
                 
                 forces_array = np.array(all_forces)
                 force_magnitudes = np.linalg.norm(forces_array, axis=2, keepdims=True)
                 force_magnitudes = np.where(force_magnitudes < 1e-10, 1.0, force_magnitudes)
                 normalized_forces = forces_array / force_magnitudes
                 atom_variances = np.var(normalized_forces, axis=0)
                 mean_variance = float(np.mean(np.sum(atom_variances, axis=1))) if atom_variances.size > 0 else 0.0

                 initial_metrics.append({'id': data['id'], 'metric': mean_variance, 'index': i})
            except Exception as e:
                 print(f"\n[WARN] Failed initial variance calculation for structure {data['id']}: {e}")
                 initial_metrics.append({'id': data['id'], 'metric': -1, 'index': i}) # Mark as failed
    else:
        raise ValueError(f"Unknown ranking_metric: {ranking_metric}. Must be 'variance' or 'force_rmse'.")

    # Filter out failed calculations (-1 metric)
    valid_metrics = [v for v in initial_metrics if v['metric'] >= 0]
    if not valid_metrics:
        print(f"[ERROR] No valid initial '{ranking_metric}' values calculated. Exiting.")
        return {}

    # Sort by metric descending (higher variance or RMSE is "better" for attacking)
    valid_metrics.sort(key=lambda x: x['metric'], reverse=True)

    # Select top N structures
    num_to_select = min(top_n, len(valid_metrics))
    selected_indices = [v['index'] for v in valid_metrics[:num_to_select]]
    selected_structures_data = [structures_to_process[i] for i in selected_indices]

    print(f"\nSelected top {num_to_select} structures for optimization:")
    for i, data in enumerate(selected_structures_data):
        metric_val = valid_metrics[i]['metric']
        print(f"  {i+1}. ID: {data['id']}, Initial {ranking_metric.replace('_', ' ').title()}: {metric_val:.6f}")
    wf_timer.stop("rank_structures")


    # --- Initialize Optimizer ---
    wf_timer.start("optimizer_init")
    print("\nInitializing optimizer...")
    # Initialize the autograd optimizer. It will create its own internal calculator instance.
    optimizer = GradientAdversarialOptimizer(
        model_paths=model_paths, 
        device=computed_device,
        learning_rate=learning_rate,
        temperature=temperature,
        include_probability=include_probability,
        debug=debug,
        energy_list=initial_energies_for_norm,
        use_energy_per_atom=use_energy_per_atom
    )
    wf_timer.stop("optimizer_init")


    # --- Run Optimization Loop ---
    wf_timer.start("optimization_loop")
    print("\nStarting optimization runs...")
    all_trajectories: Dict[int, List[Atoms]] = {}
    
    plot_save_dir = Path(output_dir) / 'plots'
    print(f"[INFO] Plots will be saved within: {plot_save_dir.resolve()}")


    for data in tqdm(selected_structures_data, desc="Adversarial Attacks"):
        atoms_initial = data['atoms']
        parent_id = data['id']

        if debug:
             print(f"\n--- Optimizing Structure ID: {parent_id} ---")

        try:
            # Call the optimize method - it will use autograd for optimization
            generated_trajectory_for_parent = optimizer.optimize(
                atoms=atoms_initial,
                generation=generation, 
                n_iterations=n_iterations,
                min_distance=min_distance,
                output_dir=str(plot_save_dir),
                patience=patience, 
                shake=shake, 
                shake_std=shake_std
            )
            all_trajectories[parent_id] = generated_trajectory_for_parent
            if debug:
                 print(f"Finished optimization for {parent_id}. Generated trajectory with {len(generated_trajectory_for_parent)} steps.")

        except Exception as e:
            print(f"\n[ERROR] Optimization failed for structure {parent_id}: {e}")
            # Optionally add more robust error handling/logging here

    wf_timer.stop("optimization_loop")

    wf_timer.stop("total_workflow")
    print("\n--- Adversarial Attack Workflow Finished ---")
    if debug:
         wf_timer.summary()

    # --- Save trajectories to file or return dictionary --- 
    if save_output:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving trajectories to: {save_path.resolve()}")

        saved_files_count = 0
        initial_atoms_map = {data['id']: data['atoms'] for data in selected_structures_data}

        for parent_id, generated_trajectory in tqdm(all_trajectories.items(), desc="Saving Trajectories"):
            if parent_id in initial_atoms_map:
                atoms_initial = initial_atoms_map[parent_id]
                full_trajectory_to_save = [atoms_initial] + generated_trajectory
                for atom in full_trajectory_to_save:
                    atom.calc = None
                filename = save_path / f"structure_{parent_id}.xyz"
                try:
                    write(filename, full_trajectory_to_save, format='extxyz')
                    saved_files_count += 1
                except Exception as e:
                    print(f"\n[ERROR] Failed to save trajectory for parent ID {parent_id} to {filename}: {e}")
            else:
                 print(f"\n[WARN] Could not find initial atoms for parent ID {parent_id}. Skipping trajectory save.")

        print(f"\nSuccessfully saved {saved_files_count} trajectory files.")
        return None # Return None when saving
    else:
        # Return the collected trajectories dictionary
        return all_trajectories


# --- Command-Line Interface ---
# REMOVED CLI section 