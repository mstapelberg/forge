"""
SLURM job templates for adversarial attack workflows.

These functions generate SLURM submission script content tailored for
different stages of the AA workflow.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


# --- Variance Calculation Template (Updated) ---
def get_variance_calculation_script(
    output_dir_rel: str, # Relative path for variance results (e.g., variance_results)
    log_dir_rel: str, # Relative path for slurm logs (e.g., variance_calculations/slurm_logs)
    model_dir_rel: str, # Relative path to models dir (e.g., models)
    batch_script_rel_path: str, # Relative path template to batch xyz file (e.g., variance_calculations/batch_${SLURM_ARRAY_TASK_ID}/batch_${SLURM_ARRAY_TASK_ID}.xyz)
    array_range: str,
    n_models: int, # Keep for info/logging if needed, not directly used by engine
    hpc_profile: dict, # Pass loaded profile dict
    time: str = "01:00:00",
    cpus_per_task: int = 1,
    gpus_per_task: int = 0,
    account: Optional[str] = None,
    partition: Optional[str] = None,
) -> str:
    """
    Generate a SLURM script for variance calculation using optimization_engine.py.

    Args:
        output_dir_rel: Relative path from workflow root where result JSONs will be stored.
        log_dir_rel: Relative path from workflow root for SLURM log files.
        model_dir_rel: Relative path from workflow root to the copied models directory.
        batch_script_rel_path: Relative path template (using ${{SLURM_ARRAY_TASK_ID}}) to the input XYZ file.
        array_range: SLURM array range string (e.g., "0-9").
        n_models: Number of models in the ensemble (for logging).
        hpc_profile: Dictionary containing HPC profile settings (including slurm_directives
                     and environment_setup). Keys like 'time', 'cpus-per-task', 'gpus',
                     'account', 'partition' will be extracted from 'slurm_directives'.
        time: Job time limit (extracted from HPC profile).
        cpus_per_task: CPUs per task (extracted from HPC profile).
        gpus_per_task: GPUs per task (extracted from HPC profile).
        account: SLURM account (extracted from HPC profile).
        partition: SLURM partition (extracted from HPC profile).

    Returns:
        SLURM script content as a string.
    """
    slurm_directives = hpc_profile.get("slurm_directives", {})
    job_time = slurm_directives.get("time", time)
    job_cpus = int(slurm_directives.get("cpus-per-task", cpus_per_task)) # Ensure int

    job_gpus = gpus_per_task # Start with default
    if "gpus" in slurm_directives:
         try:
             job_gpus = int(slurm_directives["gpus"])
         except (ValueError, TypeError): pass # Keep default if parse fails
    elif "gres" in slurm_directives and isinstance(slurm_directives["gres"], str) and "gpu" in slurm_directives["gres"]:
         try:
             parts = slurm_directives["gres"].split(":")
             job_gpus = int(parts[-1])
         except (ValueError, TypeError, IndexError): pass # Keep default if parse fails

    job_account = slurm_directives.get("account", account)
    job_partition = slurm_directives.get("partition", partition)

    account_line = f"#SBATCH --account={job_account}" if job_account else ""
    partition_line = f"#SBATCH --partition={job_partition}" if job_partition else ""
    gpu_line = f"#SBATCH --gpus-per-task={job_gpus}" if job_gpus > 0 else "#SBATCH --gpus-per-task=0" # Explicitly set 0 if no GPUs

    env_setup_lines = hpc_profile.get('environment_setup', [])
    if isinstance(env_setup_lines, str):
         env_setup_lines = env_setup_lines.strip().split('\n')
    elif not isinstance(env_setup_lines, list):
         print(f"[WARN] Unexpected type for environment_setup in profile: {type(env_setup_lines)}. Expected list or string.")
         env_setup_lines = [] # Default to empty list

    env_setup_block = '\n'.join([line for line in env_setup_lines if isinstance(line, str)])

    script = f"""#!/bin/bash
#SBATCH --job-name=aa_var_calc
#SBATCH --output={log_dir_rel}/var_calc_%A_%a.out
#SBATCH --error={log_dir_rel}/var_calc_%A_%a.err
#SBATCH --array={array_range}
#SBATCH --time={job_time}
#SBATCH --cpus-per-task={job_cpus}
{gpu_line}
{account_line}
{partition_line}

# --- Environment Setup (from HPC Profile: {hpc_profile.get('name', 'unknown')}) ---
{env_setup_block}
# --- End Environment Setup ---

# Define SLURM variables for clarity in paths
export BATCH_ID=$SLURM_ARRAY_TASK_ID

# Construct paths within the script - use bash variables
# Use printf to handle potential spaces in paths safely
printf -v INPUT_XYZ_PATH %q "{batch_script_rel_path}" # Use template directly
printf -v MODEL_DIR_PATH %q "{model_dir_rel}"
# Output JSON path needs to use the relative output dir
printf -v OUTPUT_JSON_PATH %q "{output_dir_rel}/batch_${{BATCH_ID}}_variances.json"

echo "Running Variance Calculation Task $BATCH_ID..."
echo "Input XYZ: $INPUT_XYZ_PATH"
echo "Model Dir: $MODEL_DIR_PATH"
echo "Output JSON: $OUTPUT_JSON_PATH"
echo "Num Models: {n_models}"

# Determine device based on GPU request
DEVICE="cpu"
if [ {job_gpus} -gt 0 ]; then
    # Basic check if CUDA is available (heuristic, might need refinement)
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    else
        echo "[WARNING] GPUs requested ({job_gpus}), but CUDA/nvidia-smi not detected. Falling back to CPU."
    fi
fi
echo "Using device: $DEVICE"

# Execute the optimization engine script in variance mode using bash variables
# Use eval to correctly handle the quoted paths from printf -v
eval python -m forge.workflows.adversarial_attack.optimization_engine \
    "$INPUT_XYZ_PATH" \
    --calculate_variance \
    --output_json "$OUTPUT_JSON_PATH" \
    --model_dir "$MODEL_DIR_PATH" \
    --device "$DEVICE"
    # Add --debug if needed

echo "Task $BATCH_ID finished."

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Error occurred in Task $BATCH_ID (Exit Code: $EXIT_CODE)." >&2
  # Potentially add cleanup or notification here
  exit $EXIT_CODE
fi
"""
    return script


# --- Gradient AA Template (Updated) ---
def get_gradient_aa_script(
    batch_base_rel: str, # Relative path to base dir of batches (e.g., gradient_aa_optimization)
    log_dir_rel: str, # Relative path for slurm logs
    model_dir_rel: str, # Relative path to models dir
    structure_file_rel: str, # Relative path template to batch xyz file from batch_base_rel, uses ${BATCH_ID}
    engine_output_dir_rel: str, # Relative path template for engine output (batch dir), uses ${BATCH_ID}
    array_range: str,
    n_iterations: int,
    learning_rate: float,
    min_distance: float,
    include_probability: bool,
    temperature: float, # eV
    device: str,
    save_trajectory: bool,
    time: str, # Pass from profile
    cpus_per_task: int, # Pass from profile
    gpus_per_task: int, # Pass from profile
    hpc_profile: dict, # Pass loaded profile dict
    account: Optional[str] = None, # Pass from profile
    partition: Optional[str] = None, # Pass from profile
) -> str:
    """
    Generate a SLURM script for gradient-based adversarial attack optimization.

    Args:
        batch_base_rel: Relative path from workflow root to the AA batch base directory.
        log_dir_rel: Relative path from workflow root for SLURM log files.
        model_dir_rel: Relative path from workflow root to the copied models directory.
        structure_file_rel: Relative path template from batch_base_rel to the input XYZ file (uses ${{BATCH_ID}}).
        engine_output_dir_rel: Relative path template from batch_base_rel where the
                               optimization engine should save its results (uses ${{BATCH_ID}}).
        array_range: SLURM array range string.
        n_iterations: Number of optimization iterations.
        learning_rate: Gradient step size.
        min_distance: Minimum allowed interatomic distance.
        include_probability: Whether to use probability weighting.
        temperature: Temperature (eV) for probability weighting.
        device: Device ('cpu' or 'cuda').
        save_trajectory: Whether the engine should save trajectories.
        time: Job time limit (from HPC profile).
        cpus_per_task: CPUs per task (from HPC profile).
        gpus_per_task: GPUs per task (from HPC profile).
        hpc_profile: Dictionary containing HPC profile settings (including environment_setup).
        account: SLURM account (from HPC profile).
        partition: SLURM partition (from HPC profile).

    Returns:
        SLURM script content as string.
    """
    account_line = f"#SBATCH --account={account}" if account else ""
    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    # Get environment setup lines from profile
    env_setup_lines = hpc_profile.get('environment_setup', [])
    env_setup_block = '\n'.join(env_setup_lines)

    # Build the command line arguments for optimization_engine.py flags
    cmd_flags = [
        "--gradient", # Flag to select gradient method
        f"--n_iterations {n_iterations}",
        f"--learning_rate {learning_rate}",
        f"--min_distance {min_distance}",
        f"--temperature {temperature}", # Pass temp (eV)
        f"--device {device}",
    ]
    if include_probability:
        cmd_flags.append("--include_probability")
    if save_trajectory:
        cmd_flags.append("--save-trajectory")
    else:
         cmd_flags.append("--no-save-trajectory") # Use the BooleanOptionalAction
    # Add flags for database saving if needed (removed)
    # if database_id is not None:
    #     cmd_flags.append(f"--database_id {database_id}")

    # Join flags
    flags_str = " \\\n    ".join(cmd_flags)

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

# --- Environment Setup (from HPC Profile: {hpc_profile.get('name', 'unknown')}) ---
{env_setup_block}
# --- End Environment Setup ---

# Define SLURM variables for clarity in paths
export BATCH_ID=$SLURM_ARRAY_TASK_ID

# Construct paths using bash variables
printf -v BASE_DIR %q "{batch_base_rel}"
printf -v MODEL_DIR_PATH %q "{model_dir_rel}"
printf -v INPUT_XYZ_PATH %q "${{BASE_DIR}}/{structure_file_rel.replace('${BATCH_ID}', '${BATCH_ID}')}"
printf -v OUTPUT_DIR_PATH %q "${{BASE_DIR}}/{engine_output_dir_rel.replace('${BATCH_ID}', '${BATCH_ID}')}"

echo "Running Gradient AA Engine Task $BATCH_ID..."
echo "Input XYZ: $INPUT_XYZ_PATH"
echo "Output Dir: $OUTPUT_DIR_PATH"
echo "Models: $MODEL_DIR_PATH"

# Execute the optimization engine script using bash variables
# Use eval to handle quoted paths correctly
eval python -m forge.workflows.adversarial_attack.optimization_engine \\
    "$INPUT_XYZ_PATH" \\
    "$OUTPUT_DIR_PATH" \\
    --model_dir "$MODEL_DIR_PATH" \\
    {flags_str}
    # Add --debug if needed

echo "Task $BATCH_ID finished."

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Error occurred in Task $BATCH_ID (Exit Code: $EXIT_CODE)." >&2
  # Potentially add cleanup or notification here
  exit $EXIT_CODE
fi
"""
    return script


# --- Monte Carlo AA Template (Updated) ---
def get_monte_carlo_aa_script(
    batch_base_rel: str, # Relative path to base dir of batches (e.g., monte_carlo_aa_optimization)
    log_dir_rel: str, # Relative path for slurm logs
    model_dir_rel: str, # Relative path to models dir
    structure_file_rel: str, # Relative path template to batch xyz file from batch_base_rel, uses ${BATCH_ID}
    engine_output_dir_rel: str, # Relative path template for engine output (batch dir), uses ${BATCH_ID}
    array_range: str,
    max_steps: int,
    patience: int,
    temperature: float, # K
    min_distance: float,
    max_displacement: float, # Added
    mode: str,
    device: str,
    save_trajectory: bool,
    time: str, # Pass from profile
    cpus_per_task: int, # Pass from profile
    gpus_per_task: int, # Pass from profile
    hpc_profile: dict, # Pass loaded profile dict
    account: Optional[str] = None, # Pass from profile
    partition: Optional[str] = None, # Pass from profile
) -> str:
    """
    Generate a SLURM script for Monte Carlo adversarial attack optimization.

    Args:
        batch_base_rel: Relative path from workflow root to the AA batch base directory.
        log_dir_rel: Relative path from workflow root for SLURM log files.
        model_dir_rel: Relative path from workflow root to the copied models directory.
        structure_file_rel: Relative path template from batch_base_rel to the input XYZ file (uses ${{BATCH_ID}}).
        engine_output_dir_rel: Relative path template from batch_base_rel where the
                               optimization engine should save its results (uses ${{BATCH_ID}}).
        array_range: SLURM array range string.
        max_steps: Maximum number of MC steps.
        patience: Patience parameter for MC stopping.
        temperature: Temperature (K) for Metropolis criterion.
        min_distance: Minimum allowed interatomic distance.
        max_displacement: Maximum atomic displacement per step.
        mode: MC displacement mode ('all' or 'single').
        device: Device ('cpu' or 'cuda').
        save_trajectory: Whether the engine should save trajectories.
        time: Job time limit (from HPC profile).
        cpus_per_task: CPUs per task (from HPC profile).
        gpus_per_task: GPUs per task (from HPC profile).
        hpc_profile: Dictionary containing HPC profile settings (including environment_setup).
        account: SLURM account (from HPC profile).
        partition: SLURM partition (from HPC profile).

    Returns:
        SLURM script content as string.
    """
    account_line = f"#SBATCH --account={account}" if account else ""
    partition_line = f"#SBATCH --partition={partition}" if partition else ""

    # Get environment setup lines from profile
    env_setup_lines = hpc_profile.get('environment_setup', [])
    env_setup_block = '\n'.join(env_setup_lines)

    # Build the command line arguments for optimization_engine.py flags (NO --gradient flag)
    cmd_flags = [
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
        cmd_flags.append("--save-trajectory")
    else:
        cmd_flags.append("--no-save-trajectory")
    # if database_id is not None: # Removed
    #     cmd_flags.append(f"--database_id {database_id}")

    flags_str = " \\\n    ".join(cmd_flags)

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

# --- Environment Setup (from HPC Profile: {hpc_profile.get('name', 'unknown')}) ---
{env_setup_block}
# --- End Environment Setup ---

# Define SLURM variables for clarity in paths
export BATCH_ID=$SLURM_ARRAY_TASK_ID

# Construct paths using bash variables
printf -v BASE_DIR %q "{batch_base_rel}"
printf -v MODEL_DIR_PATH %q "{model_dir_rel}"
printf -v INPUT_XYZ_PATH %q "${{BASE_DIR}}/{structure_file_rel.replace('${BATCH_ID}', '${BATCH_ID}')}"
printf -v OUTPUT_DIR_PATH %q "${{BASE_DIR}}/{engine_output_dir_rel.replace('${BATCH_ID}', '${BATCH_ID}')}"

echo "Running Monte Carlo AA Engine Task $BATCH_ID..."
echo "Input XYZ: $INPUT_XYZ_PATH"
echo "Output Dir: $OUTPUT_DIR_PATH"
echo "Models: $MODEL_DIR_PATH"

# Execute the optimization engine script using bash variables
# Use eval to handle quoted paths correctly
eval python -m forge.workflows.adversarial_attack.optimization_engine \\
    "$INPUT_XYZ_PATH" \\
    "$OUTPUT_DIR_PATH" \\
    --model_dir "$MODEL_DIR_PATH" \\
    {flags_str}
    # Add --debug if needed

echo "Task $BATCH_ID finished."

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Error occurred in Task $BATCH_ID (Exit Code: $EXIT_CODE)." >&2
  exit $EXIT_CODE
fi
"""
    return script