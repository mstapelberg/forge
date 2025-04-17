#!/usr/bin/env python
"""Workflow script for running gradient-based adversarial attacks."""

import numpy as np
import torch
import argparse
import os
import time
from tqdm import tqdm
from typing import List, Dict, Optional, Any
from pathlib import Path

# --- Core Forge Imports ---
# Assume forge is installed or PYTHONPATH is set correctly
from forge.core.database import DatabaseManager
from forge.core.adversarial_attack import AdversarialCalculator, GradientAdversarialOptimizer

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
) -> List[int]:
    """
    Runs the gradient-based adversarial attack workflow using a provided DatabaseManager.

    Args:
        db_manager: An initialized instance of DatabaseManager.
        structure_ids: List of initial structure IDs from the database.
        model_paths: List of paths to MACE model files for the ensemble.
        top_n: Number of top variance structures to select and optimize.
        generation: Generation tag for the new structures.
        n_iterations: Number of optimization steps.
        learning_rate: Optimizer learning rate.
        temperature: Temperature (in eV) for Boltzmann weighting.
        include_probability: Whether to weight loss by Boltzmann probability.
        min_distance: Minimum allowed interatomic distance (Angstrom).
        use_energy_per_atom: Use energy per atom for probability calculations.
        device: Compute device ('cuda', 'cpu', or None for auto-detect).
        debug: Enable verbose debug printing.

    Returns:
        List of newly added structure IDs to the database (simulated if db_manager is in dry_run mode).
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

    # Initialize Adversarial Calculator
    try:
        adversarial_calc = AdversarialCalculator(
            model_paths=model_paths,
            device=computed_device,
            default_dtype='float32' # MACE default
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize AdversarialCalculator: {e}")
        return []
    wf_timer.stop("init")

    # --- Fetch Initial Structures & Data ---
    wf_timer.start("fetch_data")
    print(f"\nFetching initial structures and latest calculations for {len(structure_ids)} IDs...")
    # Use get_batch_atoms_with_calculation to get atoms + latest calc data attached
    # We fetch the latest calculation regardless of calculator type (calculator=None)
    # to ensure we get the most recent energy and config_type if available.
    initial_atoms_list = db_manager.get_batch_atoms_with_calculation(structure_ids, calculator=None)

    if not initial_atoms_list:
        print("[ERROR] No valid structures found for the given IDs in the database.")
        return []

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
        return []

    if include_probability and not initial_energies_for_norm:
        print("[WARNING] `include_probability` is True, but no valid initial energies found for normalization. Proceeding deterministically (Q=1.0).")
        # We let the optimizer handle Q=1.0 internally
    elif include_probability:
        print(f"Using {len(initial_energies_for_norm)} initial energies for normalization constant.")

    wf_timer.stop("fetch_data")


    # --- Calculate Initial Variances and Rank ---
    wf_timer.start("rank_structures")
    print("\nCalculating initial variances...")
    initial_variances = []

    # Need a temporary optimizer instance just to access _calculate_force_variance helper
    # Or better: Move variance calculation logic outside or make it static/part of calculator?
    # For now, create a dummy optimizer instance. This isn't ideal.
    # Let's use the calculator's forces and variance methods directly.
    for i, data in enumerate(tqdm(structures_to_process, desc="Initial Variance Calc")):
        atoms = data['atoms']
        try:
             forces = adversarial_calc.calculate_forces(atoms)
             atom_variances = adversarial_calc.calculate_normalized_force_variance(forces)
             mean_variance = float(np.mean(atom_variances)) if atom_variances.size > 0 else 0.0
             initial_variances.append({'id': data['id'], 'variance': mean_variance, 'index': i})
        except Exception as e:
             print(f"\n[WARN] Failed initial variance calculation for structure {data['id']}: {e}")
             initial_variances.append({'id': data['id'], 'variance': -1, 'index': i}) # Mark as failed

    # Filter out failed calculations (-1 variance)
    valid_variances = [v for v in initial_variances if v['variance'] >= 0]
    if not valid_variances:
        print("[ERROR] No valid initial variances calculated. Exiting.")
        return []

    # Sort by variance descending
    valid_variances.sort(key=lambda x: x['variance'], reverse=True)

    # Select top N structures
    num_to_select = min(top_n, len(valid_variances))
    selected_indices = [v['index'] for v in valid_variances[:num_to_select]]
    selected_structures_data = [structures_to_process[i] for i in selected_indices]

    print(f"\nSelected top {num_to_select} structures for optimization:")
    for i, data in enumerate(selected_structures_data):
        print(f"  {i+1}. ID: {data['id']}, Initial Variance: {valid_variances[i]['variance']:.6f}")
    wf_timer.stop("rank_structures")


    # --- Initialize Optimizer ---
    wf_timer.start("optimizer_init")
    print("\nInitializing optimizer...")
    # Use the energy list collected earlier
    optimizer = GradientAdversarialOptimizer(
        model_paths=model_paths, # Pass paths again (or pass calculator instance?)
        device=computed_device,
        learning_rate=learning_rate,
        temperature=temperature,
        include_probability=include_probability,
        debug=debug,
        energy_list=initial_energies_for_norm, # Pass collected energies
        use_energy_per_atom=use_energy_per_atom # Pass flag
    )
    # We passed model_paths, so the optimizer re-initializes its own calculator.
    # TODO: Refactor GradientAdversarialOptimizer to accept an existing AdversarialCalculator instance?
    # For now, this works but is slightly redundant.
    wf_timer.stop("optimizer_init")


    # --- Run Optimization Loop ---
    wf_timer.start("optimization_loop")
    print("\nStarting optimization runs...")
    all_generated_atoms = []
    model_source_str = ",".join(model_paths) # Simple way to store model paths identifier

    # Define output directory for plots (can be based on generation or a fixed path)
    # Example: Create a directory based on the generation number
    # Ensure this happens *before* the loop calling optimize
    plot_output_dir = Path(f'./adversarial_attack_gen_{generation}_plots')
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving plots to: {plot_output_dir.resolve()}")


    for data in tqdm(selected_structures_data, desc="Adversarial Attacks"):
        atoms_initial = data['atoms']
        parent_id = data['id']
        original_config_type = data['config_type'] # Keep this for potential logging if needed

        if debug:
             print(f"\n--- Optimizing Structure ID: {parent_id} ---")

        try:
            # Call the optimize method - it now requires generation and returns the full trajectory
            generated_trajectory_for_parent = optimizer.optimize(
                atoms=atoms_initial,
                generation=generation, # Pass generation explicitly
                n_iterations=n_iterations,
                min_distance=min_distance,
                output_dir=str(plot_output_dir) # Pass directory for plots
                # Removed parent_id, original_config_type, model_source_path, save_interval
            )
            # Extend the main list with all atoms from the returned trajectory
            all_generated_atoms.extend(generated_trajectory_for_parent)
            if debug:
                 print(f"Finished optimization for {parent_id}. Generated trajectory with {len(generated_trajectory_for_parent)} steps.")

        except Exception as e:
            print(f"\n[ERROR] Optimization failed for structure {parent_id}: {e}")
            # Optionally add more robust error handling/logging here

    wf_timer.stop("optimization_loop")


    # --- Add Generated Structures to Database ---
    wf_timer.start("db_add")
    added_structure_ids = []
    if all_generated_atoms:
        print(f"\nAdding {len(all_generated_atoms)} generated structures to the database...")
        # Use add_structure in a loop. Consider a batch method if performance is critical.
        for atoms_to_add in tqdm(all_generated_atoms, desc="Adding to DB"):
             try:
                  # Metadata is already in atoms_to_add.info
                  # source_type could indicate the origin, e.g., 'adversarial_gradient'
                  new_id = db_manager.add_structure(atoms_to_add, source_type='adversarial_gradient')
                  added_structure_ids.append(new_id)
             except Exception as e:
                  parent = atoms_to_add.info.get('parent_id', 'unknown')
                  step = atoms_to_add.info.get('step', 'unknown')
                  print(f"\n[ERROR] Failed to add generated structure (parent {parent}, step {step}) to database: {e}")

        print(f"Successfully added {len(added_structure_ids)} structures to the database.")
        if db_manager.dry_run:
             print("[INFO] (Dry Run Mode - IDs are simulated)")
    else:
        print("\nNo structures were generated during the optimization process.")
    wf_timer.stop("db_add")

    wf_timer.stop("total_workflow")
    print("\n--- Adversarial Attack Workflow Finished ---")
    if debug or True: # Always print summary for now
         wf_timer.summary()

    return added_structure_ids


# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gradient-based adversarial attacks workflow.")

    # Inputs
    parser.add_argument('--structure_ids', type=int, nargs='+', required=True, help='List of initial structure IDs from the database.')
    parser.add_argument('--model_paths', nargs='+', required=True, help='List of paths to MACE model files.')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the database file or connection string.')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top variance structures to optimize.')
    parser.add_argument('--generation', type=int, required=True, help='Generation identifier for the output structures.')

    # Optimization Parameters
    parser.add_argument('--n_iterations', type=int, default=50, help='Number of optimization iterations.')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for Adam optimizer.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature (in eV) for Boltzmann probability weighting.')
    parser.add_argument('--include_probability', action='store_true', help='Weight loss by Boltzmann probability (uses energy/atom by default).')
    parser.add_argument('--deterministic', action='store_true', help='Force deterministic mode (ignore probability). Overrides --include_probability.')
    # parser.add_argument('--use_total_energy', action='store_true', help='Use total energy instead of energy/atom for probability (if probability included).') # Option if needed
    parser.add_argument('--min_distance', type=float, default=1.5, help='Minimum allowed interatomic distance (Angstrom).')

    # Other Options
    parser.add_argument('--cpu', action='store_true', help='Force computation on CPU.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printing.')
    parser.add_argument('--dry_run_db', action='store_true', help='Simulate database writes without actual changes.')


    args = parser.parse_args()

    # --- Basic Input Validation ---
    if args.top_n <= 0: args.top_n = 1 # Ensure at least 1
    if args.generation < 0: raise ValueError("--generation must be non-negative.")
    if args.n_iterations <= 0: raise ValueError("--n_iterations must be positive.")
    if args.learning_rate <= 0: raise ValueError("--learning_rate must be positive.")
    if args.temperature <= 0: raise ValueError("--temperature must be positive.")
    if args.min_distance <= 0: raise ValueError("--min_distance must be positive.")
    if not all(os.path.exists(p) for p in args.model_paths):
        missing = [p for p in args.model_paths if not os.path.exists(p)]
        raise FileNotFoundError(f"One or more model paths do not exist: {missing}")
    # DB path check can be complex (file vs connection string), handled by DatabaseManager

    # Handle deterministic flag
    final_include_probability = args.include_probability and not args.deterministic

    # --- Initialize DatabaseManager for CLI execution ---
    db_manager_instance = None
    new_ids = []
    try:
        print(f"Initializing DatabaseManager (Dry Run: {args.dry_run_db})...")
        db_manager_instance = DatabaseManager(
            config_path=args.db_path,
            dry_run=args.dry_run_db,
            debug=args.debug
        )
        print("DatabaseManager initialized.")

        # --- Run the main workflow function ---
        new_ids = run_adversarial_attacks(
            db_manager=db_manager_instance, # Pass the instance
            structure_ids=args.structure_ids,
            model_paths=args.model_paths,
            top_n=args.top_n,
            generation=args.generation,
            n_iterations=args.n_iterations,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            include_probability=final_include_probability,
            min_distance=args.min_distance,
            use_energy_per_atom=True,
            device='cpu' if args.cpu else None,
            debug=args.debug,
        )

    except Exception as e:
        print(f"\n[ERROR] Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Ensure DB connection is closed ---
        if db_manager_instance:
            print("\nClosing database connection...")
            db_manager_instance.close_connection()
            print("Database connection closed.")

    print(f"\nWorkflow complete. Added {len(new_ids)} new structures.")
    if args.debug and new_ids:
         print(f"New structure IDs: {new_ids}") 