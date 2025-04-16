"""
SLURM job templates for adversarial attack workflows.

These functions generate SLURM submission script content tailored for
different stages of the AA workflow.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


# --- Variance Calculation Template (Placeholder) ---
def get_variance_calculation_script(
    output_dir_rel: str, # Relative path for variance results
    log_dir_rel: str, # Relative path for slurm logs
    model_dir_rel: str, # Relative path to models dir
    batch_script_rel_path: str, # Relative path template to batch xyz file
    array_range: str,
    n_models: int,
    compute_forces: bool = True, # Example parameter
    time: str = "12:00:00",
    cpus_per_task: int = 4,
    gpus_per_task: int = 1,
    account: Optional[str] = None,
    partition: Optional[str] = None,
) -> str:
    """
    Generate a SLURM script for variance calculation (PLACEHOLDER).

    Args:
        output_dir_rel: Relative path from workflow root to save variance results JSON.
        log_dir_rel: Relative path from workflow root for SLURM log files.
        model_dir_rel: Relative path from workflow root to the copied models directory.
        batch_script_rel_path: Relative path template (using ${...}) to the input XYZ file.
        array_range: SLURM array range string (e.g., "0-9").
        n_models: Number of models in the ensemble.
        compute_forces: Flag (example).
        time: Job time limit.
        cpus_per_task: CPUs per task.
        gpus_per_task: GPUs per task.
        account: SLURM account.
        partition: SLURM partition.

    Returns:
        SLURM script content as a string.
    """
    # TODO: Replace placeholder command with actual variance calculation script call
    #       when implemented. Arguments need to be defined for that script.
    variance_command = f"""
echo "--- Variance Calculation Placeholder ---"
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Input XYZ: {batch_script_rel_path}"
echo "Model Dir: {model_dir_rel}"
echo "Output Dir: {output_dir_rel}"
echo "Num Models: {n_models}"
# Example: Create dummy output file
touch "{output_dir_rel}/batch_${SLURM_ARRAY_TASK_ID}_variances.json"
echo "{{\\"struct_placeholder_1\\": 0.1, \\"struct_placeholder_2\\": 0.2}}" > "{output_dir_rel}/batch_${SLURM_ARRAY_TASK_ID}_variances.json"
echo "--- End Placeholder ---"
"""

    account_line = f"#SBATCH --account={account}" if account else ""
    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    # Assuming the script is run from the main workflow directory
    script = f"""#!/bin/bash
#SBATCH --job-name=aa_var_calc
#SBATCH --output={log_dir_rel}/var_calc_%A_%a.out
#SBATCH --error={log_dir_rel}/var_calc_%A_%a.err
#SBATCH --array={array_range}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-task={gpus_per_task}
{account_line}
{partition_line}

# --- Environment Setup (adjust as needed) ---
echo "Loading modules..."
module purge
module load anaconda3/2023.09 # Example module loading
echo "Activating environment..."
source activate forge # Example environment activation
echo "Environment activated."
# --- End Environment Setup ---

echo "Running Variance Calculation (Placeholder) Task $SLURM_ARRAY_TASK_ID..."

# Execute the placeholder command
{variance_command}

echo "Task $SLURM_ARRAY_TASK_ID finished."
"""
    return script


# --- Gradient AA Template (Updated) ---
def get_gradient_aa_script(
    batch_base_rel: str, # Relative path to base dir of batches (e.g., gradient_aa_optimization)
    log_dir_rel: str, # Relative path for slurm logs
    model_dir_rel: str, # Relative path to models dir
    structure_file_rel: str, # Relative path template to batch xyz file from batch_base_rel
    engine_output_dir_rel: str, # Relative path template for engine output (batch dir)
    array_range: str,
    n_iterations: int,
    learning_rate: float,
    min_distance: float,
    include_probability: bool,
    temperature: float, # eV
    device: str,
    save_trajectory: bool = True,
    database_id: Optional[int] = None, # Pass parent ID if optimizing single known structure
    time: str = "12:00:00",
    cpus_per_task: int = 8,
    gpus_per_task: int = 1, # Should match device='cuda'
    account: Optional[str] = None,
    partition: Optional[str] = None,
) -> str:
    """
    Generate a SLURM script for gradient-based adversarial attack optimization.

    Args:
        batch_base_rel: Relative path from workflow root to the AA batch base directory.
        log_dir_rel: Relative path from workflow root for SLURM log files.
        model_dir_rel: Relative path from workflow root to the copied models directory.
        structure_file_rel: Relative path template from batch_base_rel to the input XYZ file.
        engine_output_dir_rel: Relative path template from batch_base_rel where the
                               optimization engine should save its results.
        array_range: SLURM array range string.
        n_iterations: Number of optimization iterations.
        learning_rate: Gradient step size.
        min_distance: Minimum allowed interatomic distance.
        include_probability: Whether to use probability weighting.
        temperature: Temperature (eV) for probability weighting.
        device: Device ('cpu' or 'cuda').
        save_trajectory: Whether the engine should save trajectories.
        database_id: Optional parent structure ID (for single structure optimization).
        time: Job time limit.
        mem: Memory allocation.
        cpus_per_task: CPUs per task.
        gpus_per_task: GPUs per task (should be >0 if device='cuda').
        account: SLURM account.
        partition: SLURM partition.

    Returns:
        SLURM script content as string.
    """
    account_line = f"#SBATCH --account={account}" if account else ""
    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    # Construct paths relative to workflow root where script is launched
    xyz_input_path = f"{batch_base_rel}/{structure_file_rel}"
    output_path = f"{batch_base_rel}/{engine_output_dir_rel}"

    # Build the command line arguments for optimization_engine.py
    cmd_args = [
        f'"{xyz_input_path}"', # Positional arg 1
        f'"{output_path}"',   # Positional arg 2
        f'--model_dir "{model_dir_rel}"',
        "--gradient", # Flag to select gradient method
        f"--n_iterations {n_iterations}",
        f"--learning_rate {learning_rate}",
        f"--min_distance {min_distance}",
        f"--temperature {temperature}", # Pass temp (eV)
        f"--device {device}",
    ]
    if include_probability:
        cmd_args.append("--include_probability")
    if save_trajectory:
        cmd_args.append("--save-trajectory")
    else:
         cmd_args.append("--no-save-trajectory") # Use the BooleanOptionalAction
    if database_id is not None:
        cmd_args.append(f"--database_id {database_id}")
    # Add flags for database saving if needed, e.g. --save_to_database

    # Join arguments, handling spaces in paths with quotes
    engine_command = f"python -m forge.workflows.adversarial_attack.optimization_engine \\\n    " + " \\\n    ".join(cmd_args)


    script = f"""#!/bin/bash
#SBATCH --job-name=aa_grad
#SBATCH --output={log_dir_rel}/aa_grad_%A_%a.out
#SBATCH --error={log_dir_rel}/aa_grad_%A_%a.err
#SBATCH --array={array_range}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-task={gpus_per_task} # Request GPU if device=cuda
{account_line}
{partition_line}

# --- Environment Setup (adjust as needed) ---
echo "Loading modules..."
module purge
module load anaconda3/2023.09 # Example
echo "Activating environment..."
source activate forge # Example
echo "Environment activated."
# --- End Environment Setup ---

echo "Running Gradient AA Engine Task $SLURM_ARRAY_TASK_ID..."
echo "Input XYZ: {xyz_input_path}"
echo "Output Dir: {output_path}"
echo "Models: {model_dir_rel}"

# Execute the optimization engine script
{engine_command}

echo "Task $SLURM_ARRAY_TASK_ID finished."

# Check exit code
if [ $? -ne 0 ]; then
  echo "Error occurred in Task $SLURM_ARRAY_TASK_ID." >&2
  # Potentially add cleanup or notification here
  exit 1
fi
"""
    return script


# --- Monte Carlo AA Template (Updated) ---
def get_monte_carlo_aa_script(
    batch_base_rel: str, # Relative path to base dir of batches (e.g., monte_carlo_aa_optimization)
    log_dir_rel: str, # Relative path for slurm logs
    model_dir_rel: str, # Relative path to models dir
    structure_file_rel: str, # Relative path template to batch xyz file from batch_base_rel
    engine_output_dir_rel: str, # Relative path template for engine output (batch dir)
    array_range: str,
    max_steps: int,
    patience: int,
    temperature: float, # K
    min_distance: float,
    max_displacement: float, # Added
    mode: str,
    device: str,
    save_trajectory: bool = True,
    database_id: Optional[int] = None,
    time: str = "12:00:00",
    cpus_per_task: int = 8,
    gpus_per_task: int = 1, # Should match device='cuda'
    account: Optional[str] = None,
    partition: Optional[str] = None,
) -> str:
    """
    Generate a SLURM script for Monte Carlo adversarial attack optimization.

    Args:
        batch_base_rel: Relative path from workflow root to the AA batch base directory.
        log_dir_rel: Relative path from workflow root for SLURM log files.
        model_dir_rel: Relative path from workflow root to the copied models directory.
        structure_file_rel: Relative path template from batch_base_rel to the input XYZ file.
        engine_output_dir_rel: Relative path template from batch_base_rel where the
                               optimization engine should save its results.
        array_range: SLURM array range string.
        max_steps: Maximum number of MC steps.
        patience: Patience parameter for MC stopping.
        temperature: Temperature (K) for Metropolis criterion.
        min_distance: Minimum allowed interatomic distance.
        max_displacement: Maximum atomic displacement per step.
        mode: MC displacement mode ('all' or 'single').
        device: Device ('cpu' or 'cuda').
        save_trajectory: Whether the engine should save trajectories.
        database_id: Optional parent structure ID.
        time: Job time limit.
        cpus_per_task: CPUs per task.
        gpus_per_task: GPUs per task.
        account: SLURM account.
        partition: SLURM partition.

    Returns:
        SLURM script content as string.
    """
    account_line = f"#SBATCH --account={account}" if account else ""
    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    xyz_input_path = f"{batch_base_rel}/{structure_file_rel}"
    output_path = f"{batch_base_rel}/{engine_output_dir_rel}"

    # Build the command line arguments for optimization_engine.py (NO --gradient flag)
    cmd_args = [
        f'"{xyz_input_path}"', # Positional arg 1
        f'"{output_path}"',   # Positional arg 2
        f'--model_dir "{model_dir_rel}"',
        # No --gradient flag here
        f"--max_steps {max_steps}",
        f"--patience {patience}",
        f"--temperature {temperature}", # Pass temp (K)
        f"--min_distance {min_distance}",
        f"--max_displacement {max_displacement}", # Pass max_displacement
        f"--mode {mode}",
        f"--device {device}",
    ]
    # BooleanOptionalAction flag for trajectory
    if save_trajectory:
        cmd_args.append("--save-trajectory")
    else:
        cmd_args.append("--no-save-trajectory")
    if database_id is not None:
        cmd_args.append(f"--database_id {database_id}")
    # Add flags for database saving if needed, e.g. --save_to_database

    engine_command = f"python -m forge.workflows.adversarial_attack.optimization_engine \\\n    " + " \\\n    ".join(cmd_args)

    script = f"""#!/bin/bash
#SBATCH --job-name=aa_mc
#SBATCH --output={log_dir_rel}/aa_mc_%A_%a.out
#SBATCH --error={log_dir_rel}/aa_mc_%A_%a.err
#SBATCH --array={array_range}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-task={gpus_per_task} # Request GPU if device=cuda
{account_line}
{partition_line}

# --- Environment Setup (adjust as needed) ---
echo "Loading modules..."
module purge
module load anaconda3/2023.09 # Example
echo "Activating environment..."
source activate forge # Example
echo "Environment activated."
# --- End Environment Setup ---

echo "Running Monte Carlo AA Engine Task $SLURM_ARRAY_TASK_ID..."
echo "Input XYZ: {xyz_input_path}"
echo "Output Dir: {output_path}"
echo "Models: {model_dir_rel}"

# Execute the optimization engine script
{engine_command}

echo "Task $SLURM_ARRAY_TASK_ID finished."

# Check exit code
if [ $? -ne 0 ]; then
  echo "Error occurred in Task $SLURM_ARRAY_TASK_ID." >&2
  exit 1
fi
"""
    return script