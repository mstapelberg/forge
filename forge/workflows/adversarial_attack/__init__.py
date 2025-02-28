"""Adversarial Attack workflow module for Forge."""

from pathlib import Path
from typing import List, Optional

def register_workflow_command(name):
    """Simple decorator to register workflow commands."""
    def decorator(func):
        return func
    return decorator

@register_workflow_command("create-aa-jobs")
def create_aa_jobs(
    output_dir: str,
    model_dir: str,
    elements: List[str],
    n_batches: int,
    structure_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    structure_ids: Optional[List[int]] = None,
    debug: bool = False,
):
    """Create adversarial attack workflow directory and initial variance calculation jobs.
    
    Args:
        output_dir: Directory to create workflow in
        model_dir: Directory containing MACE model files
        elements: List of elements to filter structures by
        n_batches: Number of batches to split calculations into
        structure_type: Optional structure type filter
        composition_constraints: Optional JSON string for composition constraints
        structure_ids: Optional list of specific structure IDs
        debug: Enable debug output
    """
    from .aa_driver import prepare_aa_workflow
    return prepare_aa_workflow(
        output_dir=output_dir,
        model_dir=model_dir,
        elements=elements,
        n_batches=n_batches,
        structure_type=structure_type,
        composition_constraints=composition_constraints,
        structure_ids=structure_ids,
        debug=debug
    )

@register_workflow_command("run-aa-jobs")
def run_aa_jobs(
    input_directory: str,
    model_dir: str,
    n_structures: int,
    n_batches: int,
    temperature: float = 1200.0,
    max_steps: int = 50,
    patience: int = 25,
    min_distance: float = 2.0,
    mode: str = "all",
    device: str = "cuda",
    debug: bool = False,
):
    """Run adversarial attack optimization on highest-variance structures.
    
    Args:
        input_directory: Directory containing variance calculation results
        model_dir: Directory containing model files
        n_structures: Number of highest-variance structures to select
        n_batches: Number of batches to split calculations into
        temperature: Temperature for adversarial optimization (K)
        max_steps: Maximum optimization steps per structure
        patience: Stop if no improvement after this many steps
        min_distance: Minimum allowed distance between atoms (Å)
        mode: Optimization mode ('all' or 'single' atom)
        device: Device to run on (cpu/cuda)
        debug: Enable debug output
    """
    from .aa_driver import prepare_aa_optimization
    return prepare_aa_optimization(
        input_directory=input_directory,
        model_dir=model_dir,
        n_structures=n_structures,
        n_batches=n_batches,
        temperature=temperature,
        max_steps=max_steps,
        patience=patience,
        min_distance=min_distance,
        mode=mode,
        device=device,
        debug=debug
    )

@register_workflow_command("create-aa-vasp-jobs")
def create_aa_vasp_jobs(
    input_directory: str,
    output_directory: str,
    hpc_profile: str = "default",
    debug: bool = False,
):
    """Create VASP jobs for optimized structures from adversarial attack.
    
    Args:
        input_directory: Directory containing AA optimization results
        output_directory: Directory to create VASP jobs in
        hpc_profile: HPC profile to use for job settings
        debug: Enable debug output
    """
    from .aa_driver import prepare_vasp_jobs
    return prepare_vasp_jobs(
        input_directory=input_directory,
        output_directory=output_directory,
        hpc_profile=hpc_profile,
        debug=debug
    )

@register_workflow_command("run-gradient-aa-jobs")
def run_gradient_aa_jobs(
    input_directory: str,
    model_dir: str,
    n_structures: int,
    n_batches: int,
    learning_rate: float = 0.01,
    n_iterations: int = 60,
    min_distance: float = 1.5,
    include_probability: bool = False,
    temperature: float = 0.86,
    device: str = "cuda",
    debug: bool = False,
):
    """Run gradient-based adversarial attack optimization on highest-variance structures.
    
    Args:
        input_directory: Directory containing variance calculation results
        model_dir: Directory containing model files
        n_structures: Number of highest-variance structures to select
        n_batches: Number of batches to split calculations into
        learning_rate: Learning rate for gradient ascent
        n_iterations: Number of optimization iterations
        min_distance: Minimum allowed distance between atoms (Å)
        include_probability: Whether to include the probability term in the loss
        temperature: Temperature for probability weighting (eV)
        device: Device to run on (cpu/cuda)
        debug: Enable debug output
    """
    from .aa_driver import prepare_gradient_aa_optimization
    return prepare_gradient_aa_optimization(
        input_directory=input_directory,
        model_dir=model_dir,
        n_structures=n_structures,
        n_batches=n_batches,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        min_distance=min_distance,
        include_probability=include_probability,
        temperature=temperature,
        device=device,
        debug=debug
    )

@register_workflow_command("calculate-variance")
def calculate_variance(xyz_file: str, output_dir: str, model_paths: list[str], device: str = "cpu"):
    """Calculate model variance across an ensemble of models.
    
    Args:
        xyz_file: Path to input XYZ file
        output_dir: Directory to save results
        model_paths: List of paths to model files
        device: Device to run on (cpu/cuda)
    """
    from .aa_driver import calculate_model_variance
    return calculate_model_variance(xyz_file, output_dir, model_paths, device)

@register_workflow_command("create-vasp-jobs-from-aa")
def create_vasp_jobs_from_aa(
    input_dir: str, 
    output_dir: str = None, 
    vasp_profile: str = "static", 
    hpc_profile: str = "default",
    structures_per_traj: int = 5
):
    """Create VASP jobs for structures optimized with adversarial attack.
    
    Args:
        input_dir: Directory containing AA optimization results
        output_dir: Directory to create VASP jobs in (defaults to input_dir/vasp_jobs)
        vasp_profile: VASP settings profile to use
        hpc_profile: HPC profile to use
        structures_per_traj: Number of structures to select per trajectory
    """
    from .run_aa import create_vasp_jobs_from_aa_results
    return create_vasp_jobs_from_aa_results(
        input_dir=input_dir,
        output_dir=output_dir,
        vasp_profile=vasp_profile,
        hpc_profile=hpc_profile,
        structures_per_traj=structures_per_traj
    )

@register_workflow_command("test-aa-database-workflow")
def test_aa_database_workflow(
    elements: List[str] = None,
    structure_type: str = None,
    num_structures: int = 2,
    output_dir: str = "aa_test_workflow",
    model_dir: str = None
):
    """Test the full AA database workflow from structure selection to VASP job creation.
    
    Args:
        elements: List of elements to filter by
        structure_type: Structure type to filter by
        num_structures: Number of structures to process
        output_dir: Directory to save workflow results
        model_dir: Directory containing MACE model files
    """
    import sys
    import os
    from pathlib import Path
    
    # We need to pass through command line arguments
    # Build argument list for db_test_workflow_main
    sys.argv = [sys.argv[0]]
    if elements:
        sys.argv.extend(["--elements"] + elements)
    if structure_type:
        sys.argv.extend(["--structure_type", structure_type])
    sys.argv.extend(["--num_structures", str(num_structures)])
    sys.argv.extend(["--output_dir", output_dir])
    if model_dir:
        sys.argv.extend(["--model_dir", model_dir])
    else:
        # Try to find model dir in the package
        model_dir = os.path.join(os.path.dirname(__file__), "../../tests/resources/potentials/mace")
        if os.path.exists(model_dir):
            sys.argv.extend(["--model_dir", model_dir])
        else:
            raise ValueError("No model_dir provided and couldn't find default models. Please specify model_dir.")
    
    # Import torch here to check for CUDA
    import torch
    from .run_aa import db_test_workflow_main
    return db_test_workflow_main() 