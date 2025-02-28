"""
SLURM job templates for adversarial attack workflows.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


def get_variance_calculation_script(
    output_dir: str,
    ensemble_path: str,
    structure_file: str,
    n_models: int,
    array_range: str,
    compute_forces: bool = True,
    time: str = "4:00:00",
    mem: str = "16G",
    cpus_per_task: int = 4,
    gpus_per_task: int = 1,
    account: Optional[str] = None,
    partition: Optional[str] = None,
) -> str:
    """
    Generate a SLURM script for variance calculation.
    
    Args:
        output_dir: Directory to save outputs
        ensemble_path: Path to model ensemble
        structure_file: Path to structure file
        n_models: Number of models in ensemble
        array_range: SLURM array range (e.g., "0-4")
        compute_forces: Whether to compute forces
        time: Job time limit
        mem: Memory allocation
        cpus_per_task: CPUs per task
        gpus_per_task: GPUs per task
        account: SLURM account
        partition: SLURM partition
        
    Returns:
        SLURM script as string
    """
    account_line = f"#SBATCH --account={account}" if account else ""
    partition_line = f"#SBATCH --partition={partition}" if partition else ""
    
    script = f"""#!/bin/bash
#SBATCH --job-name=var_calc
#SBATCH --output={output_dir}/slurm_logs/var_calc_%A_%a.out
#SBATCH --error={output_dir}/slurm_logs/var_calc_%A_%a.err
#SBATCH --array={array_range}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-task={gpus_per_task}
{account_line}
{partition_line}

# Load modules
module purge
module load anaconda3/2023.09

# Activate environment
source activate forge

# Run the variance calculation
python -m forge.workflows.adversarial_attack run-variance-calculation \\
    --ensemble-path {ensemble_path} \\
    --structure-file {structure_file} \\
    --model-idx $SLURM_ARRAY_TASK_ID \\
    --output-dir {output_dir} \\
    --n-models {n_models} \\
    {"--compute-forces" if compute_forces else ""}
"""
    return script


def get_gradient_aa_script(
    output_dir: str,
    ensemble_path: str,
    structure_file: str,
    n_steps: int = 100,
    step_size: float = 0.01,
    array_range: str = "0",
    time: str = "8:00:00",
    mem: str = "32G",
    cpus_per_task: int = 8,
    gpus_per_task: int = 1,
    account: Optional[str] = None,
    partition: Optional[str] = None,
    use_probability_weighting: bool = False,
    temperature: float = 300.0,
    force_only: bool = True,
    save_trajectory: bool = True,
    save_forces: bool = True,
    database_id: Optional[int] = None,
) -> str:
    """
    Generate a SLURM script for gradient-based adversarial attack.
    
    Args:
        output_dir: Directory to save outputs
        ensemble_path: Path to model ensemble
        structure_file: Path to structure file
        n_steps: Number of optimization steps
        step_size: Gradient step size
        array_range: SLURM array range
        time: Job time limit
        mem: Memory allocation
        cpus_per_task: CPUs per task
        gpus_per_task: GPUs per task
        account: SLURM account
        partition: SLURM partition
        use_probability_weighting: Whether to use probability weighting
        temperature: Temperature for probability weighting
        force_only: Whether to use only force variance
        save_trajectory: Whether to save trajectory
        save_forces: Whether to save forces
        database_id: Structure ID in database
        
    Returns:
        SLURM script as string
    """
    account_line = f"#SBATCH --account={account}" if account else ""
    partition_line = f"#SBATCH --partition={partition}" if partition else ""
    database_arg = f"--database-id {database_id}" if database_id is not None else ""
    
    script = f"""#!/bin/bash
#SBATCH --job-name=aa_gen
#SBATCH --output={output_dir}/slurm_logs/aa_gen_%A_%a.out
#SBATCH --error={output_dir}/slurm_logs/aa_gen_%A_%a.err
#SBATCH --array={array_range}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-task={gpus_per_task}
{account_line}
{partition_line}

# Load modules
module purge
module load anaconda3/2023.09

# Activate environment
source activate forge

# Run the gradient-based adversarial attack
python -m forge.workflows.adversarial_attack run-gradient-aa \\
    --ensemble-path {ensemble_path} \\
    --structure-file {structure_file} \\
    --n-steps {n_steps} \\
    --step-size {step_size} \\
    --output-dir {output_dir} \\
    {"--use-probability-weighting" if use_probability_weighting else ""} \\
    --temperature {temperature} \\
    {"--force-only" if force_only else ""} \\
    {"--save-trajectory" if save_trajectory else ""} \\
    {"--save-forces" if save_forces else ""} \\
    {database_arg}
"""
    return script


def get_monte_carlo_aa_script(
    output_dir: str,
    ensemble_path: str,
    structure_file: str,
    n_steps: int = 1000,
    max_displacement: float = 0.1,
    array_range: str = "0",
    time: str = "8:00:00",
    mem: str = "32G",
    cpus_per_task: int = 8,
    gpus_per_task: int = 1,
    account: Optional[str] = None,
    partition: Optional[str] = None,
    temperature: float = 300.0,
    force_only: bool = True,
    save_trajectory: bool = True,
    database_id: Optional[int] = None,
) -> str:
    """
    Generate a SLURM script for Monte Carlo adversarial attack.
    
    Args:
        output_dir: Directory to save outputs
        ensemble_path: Path to model ensemble
        structure_file: Path to structure file
        n_steps: Number of optimization steps
        max_displacement: Maximum atomic displacement
        array_range: SLURM array range
        time: Job time limit
        mem: Memory allocation
        cpus_per_task: CPUs per task
        gpus_per_task: GPUs per task
        account: SLURM account
        partition: SLURM partition
        temperature: Temperature for Metropolis criterion
        force_only: Whether to use only force variance
        save_trajectory: Whether to save trajectory
        database_id: Structure ID in database
        
    Returns:
        SLURM script as string
    """
    account_line = f"#SBATCH --account={account}" if account else ""
    partition_line = f"#SBATCH --partition={partition}" if partition else ""
    database_arg = f"--database-id {database_id}" if database_id is not None else ""
    
    script = f"""#!/bin/bash
#SBATCH --job-name=aa_mc
#SBATCH --output={output_dir}/slurm_logs/aa_mc_%A_%a.out
#SBATCH --error={output_dir}/slurm_logs/aa_mc_%A_%a.err
#SBATCH --array={array_range}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-task={gpus_per_task}
{account_line}
{partition_line}

# Load modules
module purge
module load anaconda3/2023.09

# Activate environment
source activate forge

# Run the Monte Carlo adversarial attack
python -m forge.workflows.adversarial_attack run-aa \\
    --ensemble-path {ensemble_path} \\
    --structure-file {structure_file} \\
    --n-steps {n_steps} \\
    --max-displacement {max_displacement} \\
    --output-dir {output_dir} \\
    --temperature {temperature} \\
    {"--force-only" if force_only else ""} \\
    {"--save-trajectory" if save_trajectory else ""} \\
    {database_arg}
"""
    return script