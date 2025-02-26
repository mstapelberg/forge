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