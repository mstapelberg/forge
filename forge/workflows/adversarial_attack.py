#!/usr/bin/env python
"""Workflow script for running gradient-based adversarial attacks."""

import numpy as np
import torch
import argparse
import os
import time
import pickle
import hashlib
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from ase import Atoms
from ase.io import write
import matplotlib.pyplot as plt

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
        if self.debug:
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
    save_output: bool = False,
    patience: int = 25,
    shake: bool = False,
    shake_std: float = 0.05,
    ranking_metric: str = 'force_rmse',
    reference_calculator: str = 'vasp',
    cache_rmse: bool = True,
    plot_rmse_histogram: bool = True,
    rmse_cutoff: Optional[float] = None
) -> Union[Dict[int, List[Atoms]], None]:
    """
    Runs the gradient-based adversarial attack workflow using a provided DatabaseManager.

    NEW FEATURES:
        cache_rmse: If True, cache RMSE calculations to avoid recomputation
        plot_rmse_histogram: If True, plot histogram of RMSE values with statistics
        rmse_cutoff: If provided, show how many structures would be selected at this cutoff

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
        patience: Patience parameter for the optimizer.
        shake: If True, apply random shake when optimizer patience is reached. If False, stop.
        shake_std: Standard deviation for the random shake if shake is True.
        ranking_metric: Metric to rank structures for attack ('variance' or 'force_rmse').
        reference_calculator: Calculator name to fetch for reference forces if ranking by RMSE.
        cache_rmse: Whether to cache RMSE calculations for reuse
        plot_rmse_histogram: Whether to generate RMSE distribution histogram
        rmse_cutoff: Optional RMSE cutoff threshold for analysis

    Returns:
        If save_output is False (default): Returns a dictionary where keys are parent IDs
        and values are lists of Atoms objects representing the optimization trajectory.
        If save_output is True: Saves each trajectory to `output_dir/structure_{parent_id}.xyz`
        and plots to `output_dir/plots/`, then returns None.
    """
    wf_timer = Timer(debug=debug)
    wf_timer.start("total_workflow")

    # --- Create cache key for RMSE calculations ---
    if ranking_metric == 'force_rmse' and cache_rmse:
        # Create a hash based on model paths and reference calculator
        cache_key_string = f"{reference_calculator}_{'_'.join(sorted(model_paths))}"
        cache_key = hashlib.md5(cache_key_string.encode()).hexdigest()[:12]
        cache_dir = Path(output_dir) / "rmse_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"rmse_cache_{cache_key}.pkl"
        
        # Try to load existing cache
        rmse_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    rmse_cache = pickle.load(f)
                print(f"[INFO] Loaded RMSE cache with {len(rmse_cache)} entries from {cache_file}")
            except Exception as e:
                print(f"[WARN] Failed to load RMSE cache: {e}")
                rmse_cache = {}
    
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
        cached_count = 0
        computed_count = 0
        
        for i, data in enumerate(tqdm(structures_to_process, desc="Initial Force RMSE Calc")):
            structure_id = data['id']
            
            # Check cache first
            if cache_rmse and structure_id in rmse_cache:
                rmse = rmse_cache[structure_id]
                initial_metrics.append({'id': structure_id, 'metric': float(rmse), 'index': i})
                cached_count += 1
                continue
            
            atoms = data['atoms'].copy()
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
                initial_metrics.append({'id': structure_id, 'metric': float(rmse), 'index': i})
                
                # Cache the result
                if cache_rmse:
                    rmse_cache[structure_id] = float(rmse)
                computed_count += 1

            except Exception as e:
                print(f"\n[WARN] Failed force RMSE calculation for structure {structure_id}: {e}")
                initial_metrics.append({'id': structure_id, 'metric': -1, 'index': i})
        
        # Save updated cache
        if cache_rmse and computed_count > 0:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(rmse_cache, f)
                print(f"[INFO] Saved updated RMSE cache to {cache_file}")
            except Exception as e:
                print(f"[WARN] Failed to save RMSE cache: {e}")
        
        print(f"[INFO] RMSE calculations: {cached_count} from cache, {computed_count} newly computed")
        
        # Generate RMSE histogram and statistics
        if plot_rmse_histogram:
            valid_rmse_values = [m['metric'] for m in initial_metrics if m['metric'] >= 0]
            if valid_rmse_values:
                _plot_rmse_histogram(valid_rmse_values, top_n, rmse_cutoff, output_dir)
    
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


def _plot_rmse_histogram(rmse_values: List[float], top_n: int, rmse_cutoff: Optional[float], output_dir: str):
    """Plot histogram of RMSE values with statistics and selection info."""
    rmse_array = np.array(rmse_values)
    
    # Calculate statistics
    mean_rmse = np.mean(rmse_array)
    median_rmse = np.median(rmse_array)
    std_rmse = np.std(rmse_array)
    q25 = np.percentile(rmse_array, 25)
    q75 = np.percentile(rmse_array, 75)
    q90 = np.percentile(rmse_array, 90)
    q95 = np.percentile(rmse_array, 95)
    
    # Sort for top-N analysis
    sorted_rmse = np.sort(rmse_array)[::-1]  # Descending order
    
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    n_bins = min(50, len(rmse_values) // 10)
    n, bins, patches = plt.hist(rmse_array, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Calculate number of structures above each threshold
    n_above_mean = np.sum(rmse_array >= mean_rmse)
    n_above_median = np.sum(rmse_array >= median_rmse)
    n_above_q95 = np.sum(rmse_array >= q95)
    n_above_q90 = np.sum(rmse_array >= q90)
    n_above_q75 = np.sum(rmse_array >= q75)
    
    # Add vertical lines for statistics with structure counts
    plt.axvline(mean_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rmse:.4f} ({n_above_mean} structures)')
    plt.axvline(median_rmse, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rmse:.4f} ({n_above_median} structures)')
    plt.axvline(q95, color='orange', linestyle=':', linewidth=2, label=f'95th percentile: {q95:.4f} ({n_above_q95} structures)')
    plt.axvline(q90, color='brown', linestyle=':', linewidth=1, label=f'90th percentile: {q90:.4f} ({n_above_q90} structures)')
    plt.axvline(q75, color='purple', linestyle=':', linewidth=1, label=f'75th percentile: {q75:.4f} ({n_above_q75} structures)')
    
    # Show top-N threshold
    if top_n <= len(sorted_rmse):
        top_n_threshold = sorted_rmse[top_n - 1]
        plt.axvline(top_n_threshold, color='purple', linestyle='-', linewidth=3, 
                   label=f'Top {top_n} threshold: {top_n_threshold:.4f}')
    
    # Show custom cutoff if provided
    if rmse_cutoff is not None:
        n_selected_at_cutoff = np.sum(rmse_array >= rmse_cutoff)
        plt.axvline(rmse_cutoff, color='magenta', linestyle='-', linewidth=3,
                   label=f'Custom cutoff {rmse_cutoff:.4f} (selects {n_selected_at_cutoff})')
    
    plt.xlabel('Force RMSE (eV/Å)', fontsize=12)
    plt.ylabel('Number of Structures', fontsize=12)
    plt.title('Distribution of Force RMSE Values', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        f'Total structures: {len(rmse_values)}\n'
        f'Mean: {mean_rmse:.4f}\n'
        f'Median: {median_rmse:.4f}\n'
        f'Std: {std_rmse:.4f}\n'
        f'25th percentile: {q25:.4f}\n'
        f'75th percentile: {q75:.4f}\n'
        f'90th percentile: {q90:.4f}\n'
        f'95th percentile: {q95:.4f}'
    )
    
    if rmse_cutoff is not None:
        stats_text += f'\n\nAt cutoff {rmse_cutoff:.4f}:\n{n_selected_at_cutoff} structures selected'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save plot
    plot_path = Path(output_dir) / 'rmse_distribution.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== RMSE Distribution Statistics ===")
    print(f"Total structures: {len(rmse_values)}")
    print(f"Mean RMSE: {mean_rmse:.4f} eV/Å ({n_above_mean} structures above)")
    print(f"Median RMSE: {median_rmse:.4f} eV/Å ({n_above_median} structures above)")
    print(f"Standard deviation: {std_rmse:.4f} eV/Å")
    print(f"25th percentile: {q25:.4f} eV/Å ({len(rmse_values) - int(0.25 * len(rmse_values))} structures above)")
    print(f"75th percentile: {q75:.4f} eV/Å ({n_above_q75} structures above)")
    print(f"90th percentile: {q90:.4f} eV/Å ({n_above_q90} structures above)")
    print(f"95th percentile: {q95:.4f} eV/Å ({n_above_q95} structures above)")
    
    if top_n <= len(sorted_rmse):
        print(f"Top {top_n} threshold: {sorted_rmse[top_n - 1]:.4f} eV/Å")
    
    if rmse_cutoff is not None:
        print(f"Structures above {rmse_cutoff:.4f} eV/Å cutoff: {n_selected_at_cutoff}")
    
    print(f"Histogram saved to: {plot_path}")
    print("="*40)


# --- Command-Line Interface ---
# REMOVED CLI section 