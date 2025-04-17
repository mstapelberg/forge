"""Adversarial Attack workflow module for Forge."""

from pathlib import Path
from typing import List, Optional

def register_workflow_command(name):
    """Simple decorator to register workflow commands."""
    def decorator(func):
        func.command_name = name
        return func
    return decorator

@register_workflow_command("create-aa-jobs")
def create_aa_jobs(
    output_dir: str,
    model_dir: str,
    elements: Optional[List[str]] = None,
    n_batches: int = 1,
    structure_type: Optional[str] = None,
    composition_constraints: Optional[str] = None,
    structure_ids: Optional[List[int]] = None,
    hpc_profile: str = "default",
    debug: bool = False,
    n_structures_aa: int = 10,
    hpc_profile_vasp: str = "default",
    example_aa_n_batches: int = 1,
    example_selection_mode: str = 'total',
    example_selection_value: int = 1,
):
    """Create adversarial attack workflow directory and initial variance calculation jobs.
    
    This command sets up the initial structure files, model ensemble copies,
    and a SLURM script (using the specified HPC profile) needed to calculate
    the initial force variance across the selected structures using the model ensemble.

    Requires either 'elements' or 'structure_ids' to select input structures.

    Args:
        output_dir: Directory to create the workflow structure in.
        model_dir: Directory containing the MACE model ensemble files (*.model).
        elements: List of elements to filter structures by from the database.
                  Ignored if 'structure_ids' is provided.
        n_batches: Number of batches (SLURM jobs) to split variance calculations into.
        structure_type: Optional structure type filter (e.g., 'bulk', 'surface')
                        applied when querying the database with 'elements'.
        composition_constraints: Optional JSON string specifying composition constraints
                                 (e.g., '{"Ti": [0, 1], "O": [2, 2]}') applied when
                                 querying the database with 'elements'.
        structure_ids: Optional list of specific structure IDs from the database
                       to use as input, bypassing element/type/composition search.
        hpc_profile: Name of the HPC profile to use for SLURM script generation
                     for the *variance calculation* step.
        debug: Enable detailed debug output during setup.
        n_structures_aa: Number of structures planned for AA step (for README).
        hpc_profile_vasp: Name of HPC profile for VASP jobs (for README).
        example_aa_n_batches: Example number of AA batches (for README).
        example_selection_mode: Example selection mode for VASP jobs (for README).
        example_selection_value: Example selection value for VASP jobs (for README).
    """
    from .workflow_setup import prepare_aa_workflow
    if not elements and not structure_ids:
         raise ValueError("Either --elements or --structure_ids must be provided.")
    return prepare_aa_workflow(
        output_dir=output_dir,
        model_dir=model_dir,
        elements=elements,
        n_batches_variance=n_batches,
        structure_type=structure_type,
        composition_constraints=composition_constraints,
        structure_ids=structure_ids,
        hpc_profile_name=hpc_profile,
        debug=debug,
        n_structures_aa=n_structures_aa,
        hpc_profile_vasp=hpc_profile_vasp,
        example_aa_n_batches=example_aa_n_batches,
        example_selection_mode=example_selection_mode,
        example_selection_value=example_selection_value,
    )

@register_workflow_command("run-monte-carlo-aa-jobs")
def run_monte_carlo_aa_jobs(
    input_directory: str,
    model_dir: str,
    n_structures: int,
    n_batches: int = 1,
    temperature: float = 1200.0,
    max_steps: int = 50,
    patience: int = 25,
    min_distance: float = 2.0,
    max_displacement: float = 0.1,
    mode: str = "all",
    device: str = "cuda",
    hpc_profile: str = "default",
    debug: bool = False,
):
    """Prepare Monte Carlo adversarial attack optimization jobs.

    This command takes the results from the variance calculation step, selects
    the top N structures with the highest variance, and prepares batch files
    and a SLURM script (using the specified HPC profile) to run the
    Monte Carlo AA optimization engine on them.

    Args:
        input_directory: Directory containing variance calculation results
                         (typically 'variance_results' from create-aa-jobs).
        model_dir: Directory containing the MACE model ensemble files (*.model).
        n_structures: Number of highest-variance structures to select for optimization.
        n_batches: Number of batches (SLURM jobs) to split optimization into.
        temperature: Temperature (K) for the Metropolis acceptance criterion in MC.
        max_steps: Maximum number of Monte Carlo steps per structure.
        patience: Stop optimization if max variance doesn't increase for this many steps.
        min_distance: Minimum allowed interatomic distance (Å).
        max_displacement: Maximum distance (Å) an atom can be moved per MC step.
        mode: Atom displacement mode for MC ('all' or 'single').
        device: Device to request for computation ('cpu' or 'cuda').
        hpc_profile: Name of the HPC profile to use for SLURM script generation.
        debug: Enable detailed debug output during setup.
    """
    from .workflow_setup import prepare_monte_carlo_aa_optimization
    return prepare_monte_carlo_aa_optimization(
        input_directory=input_directory,
        model_dir=model_dir,
        n_structures=n_structures,
        n_batches=n_batches,
        temperature=temperature,
        max_steps=max_steps,
        patience=patience,
        min_distance=min_distance,
        max_displacement=max_displacement,
        mode=mode,
        device=device,
        hpc_profile_name=hpc_profile,
        debug=debug
    )

@register_workflow_command("create-aa-vasp-jobs")
def create_aa_vasp_jobs(
    input_directory: str,
    output_directory: str,
    vasp_profile: str = "static",
    hpc_profile: str = "default",
    selection_mode: str = 'total',
    selection_value: int = 1,
    generation: Optional[int] = None,
    debug: bool = False,
):
    """Create VASP jobs for structures resulting from AA optimization.

    This command processes the results from the AA optimization step
    (gradient-based or Monte Carlo), selects structures based on the chosen mode
    (e.g., final structure, or multiple steps from trajectory), adds them to the
    database (if not already present), and prepares VASP calculation directories
    using specified profiles.

    Args:
        input_directory: Directory containing AA optimization results (e.g.,
                         'gradient_aa_optimization' or 'aa_optimization').
        output_directory: Directory where the VASP job directories will be created.
        vasp_profile: Name of the VASP settings profile to use (defined in forge config).
        hpc_profile: Name of the HPC profile to use for job submission scripts
                     (defined in forge config).
        selection_mode: How to select structures from trajectories ('total' or 'every_n').
        selection_value: N value for the chosen selection mode.
        generation: Optional integer to assign as the generation number in metadata.
        debug: Enable detailed debug output during VASP job setup.
    """
    from .workflow_setup import prepare_vasp_jobs
    return prepare_vasp_jobs(
        input_directory=input_directory,
        output_directory=output_directory,
        vasp_profile=vasp_profile,
        hpc_profile=hpc_profile,
        selection_mode=selection_mode,
        selection_value=selection_value,
        generation=generation,
        debug=debug
    )

@register_workflow_command("run-gradient-aa-jobs")
def run_gradient_aa_jobs(
    input_directory: str,
    model_dir: str,
    n_structures: int,
    n_batches: int = 1,
    learning_rate: float = 0.01,
    n_iterations: int = 60,
    min_distance: float = 1.5,
    include_probability: bool = False,
    temperature: float = 0.86,
    device: str = "cuda",
    hpc_profile: str = "PSFC-GPU-AA",
    debug: bool = False,
):
    """Prepare Gradient-Based adversarial attack optimization jobs.

     This command takes the results from the variance calculation step, selects
     the top N structures with the highest variance, and prepares batch files
     and a SLURM script (using the specified HPC profile) to run the
     Gradient-Based AA optimization engine on them.

    Args:
        input_directory: Directory containing variance calculation results
                         (typically 'variance_results' from create-aa-jobs).
        model_dir: Directory containing the MACE model ensemble files (*.model).
        n_structures: Number of highest-variance structures to select for optimization.
        n_batches: Number of batches (SLURM jobs) to split optimization into.
        learning_rate: Learning rate for gradient ascent steps.
        n_iterations: Number of gradient ascent iterations.
        min_distance: Minimum allowed interatomic distance (Å).
        include_probability: Include Boltzmann probability weighting in the loss term.
                             Requires 'temperature'.
        temperature: Temperature (in eV) for probability weighting term.
        device: Device to request for computation ('cpu' or 'cuda').
        hpc_profile: Name of the HPC profile to use for SLURM script generation.
        debug: Enable detailed debug output during setup.
    """
    from .workflow_setup import prepare_gradient_aa_optimization
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
        hpc_profile_name=hpc_profile,
        debug=debug
    )

# @register_workflow_command("calculate-variance")
# def calculate_variance(xyz_file: str, output_dir: str, model_paths: list[str], device: str = "cpu"):
#     from .optimization_engine import calculate_model_variance
#     return calculate_model_variance(xyz_file, output_dir, model_paths, device)

# @register_workflow_command("create-vasp-jobs-from-aa")
# def create_vasp_jobs_from_aa(
#     input_dir: str, 
#     output_dir: str = None, 
#     vasp_profile: str = "static", 
#     hpc_profile: str = "default",
#     structures_per_traj: int = 5
# ):
#     from .run_aa import create_vasp_jobs_from_aa_results
#     return create_vasp_jobs_from_aa_results(
#         input_dir=input_dir,
#         output_dir=output_dir,
#         vasp_profile=vasp_profile,
#         hpc_profile=hpc_profile,
#         structures_per_traj=structures_per_traj
#     )

# @register_workflow_command("test-aa-database-workflow")
# def test_aa_database_workflow(
#     elements: List[str] = None,
#     structure_type: str = None,
#     num_structures: int = 2,
#     output_dir: str = "aa_test_workflow",
#     model_dir: str = None
# ):
#     import sys
#     import os
#     from pathlib import Path
#     
#     # We need to pass through command line arguments
#     # Build argument list for db_test_workflow_main
#     sys.argv = [sys.argv[0]]
#     if elements:
#         sys.argv.extend(["--elements"] + elements)
#     if structure_type:
#         sys.argv.extend(["--structure_type", structure_type])
#     sys.argv.extend(["--num_structures", str(num_structures)])
#     sys.argv.extend(["--output_dir", output_dir])
#     if model_dir:
#         sys.argv.extend(["--model_dir", model_dir])
#     else:
#         # Try to find model dir in the package
#         model_dir = os.path.join(os.path.dirname(__file__), "../../tests/resources/potentials/mace")
#         if os.path.exists(model_dir):
#             sys.argv.extend(["--model_dir", model_dir])
#         else:
#             raise ValueError("No model_dir provided and couldn't find default models. Please specify model_dir.")
#     
#     # Import torch here to check for CUDA
#     import torch
#     from .run_aa import db_test_workflow_main
#     return db_test_workflow_main() 