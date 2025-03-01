"""Functions for preparing VASP calculations from structures."""

import os
import shutil
from datetime import datetime
from pathlib import Path

from ase.atoms import Atoms
from ase.io import read,write
from ase.mep import DyNEB

from forge.core.database import DatabaseManager
from forge.workflows.helpers import determine_kpoint_grid
from forge.workflows.profiles import ProfileManager


def is_job_already_pending(
    db_manager: DatabaseManager, structure_id: int, profile_name: str
) -> bool:
    """Check metadata to see if a pending job for 'profile_name' is already marked."""
    metadata = db_manager.get_structure_metadata(structure_id)
    jobs_meta = metadata.get("jobs", {})
    info = jobs_meta.get(profile_name, {})
    return info.get("status") == "pending"


def mark_job_pending(
    db_manager: DatabaseManager, structure_id: int, profile_name: str
) -> None:
    """Mark a structure as having a pending job in the metadata for a given HPC profile."""
    metadata = db_manager.get_structure_metadata(structure_id)
    jobs_meta = metadata.get("jobs", {})
    jobs_meta[profile_name] = {
        "status": "pending",
        "timestamp": datetime.now().isoformat(),
    }
    metadata["jobs"] = jobs_meta
    db_manager.update_structure_metadata(structure_id, metadata)


def _write_incar(incar_dict: dict, filepath: str) -> None:
    """Write INCAR from a dictionary of tags."""
    with open(filepath, "w") as f:
        for key, value in incar_dict.items():
            f.write(f"{key} = {value}\n")


def _write_potcar(
    symbols: list, potcar_map: dict, potcar_dir: str, output_path: str
) -> None:
    """Write or concatenate POTCAR files in a specific order.

    Args:
        symbols: Sorted list of unique chemical symbols
        potcar_map: Element to potcar label mapping (i.e. 'V': 'V_pv', etc.)
        potcar_dir: Directory containing POTCAR files (typically from VASP_PP_PATH)
        output_path: Path to write concatenated POTCAR
    """
    with open(output_path, "wb") as out_potcar:
        for elem in symbols:
            pot_label = potcar_map.get(elem)
            if not pot_label:
                raise ValueError(f"No POTCAR mapping found for element {elem}")
            potcar_path = os.path.join(potcar_dir, pot_label, "POTCAR")
            if not os.path.exists(potcar_path):
                raise FileNotFoundError(
                    f"POTCAR not found for {pot_label} at {potcar_path}"
                )
            with open(potcar_path, "rb") as pfile:
                out_potcar.write(pfile.read())


def _write_kpoints(filepath: str, kpoints: tuple, gamma_centered: bool) -> None:
    """Write a basic KPOINTS file.

    Args:
        filepath: Path to write KPOINTS file
        kpoints: (kx, ky, kz) tuple
        gamma_centered: Whether to use gamma-centered grid
    """
    with open(filepath, "w") as f:
        f.write("KPOINTS generated by db_to_vasp\n")
        f.write("0\n")  # Automatically generated k-points
        f.write("Gamma\n" if gamma_centered else "Monkhorst\n")
        f.write(f"{kpoints[0]} {kpoints[1]} {kpoints[2]}\n")
        f.write("0 0 0\n")


def _create_slurm_script(
    hpc_profile: dict, 
    output_dir: str,
    job_name: str = None,
) -> str:
    """Build the Slurm script from HPC profile JSON/dict structure.

    Args:
        hpc_profile: HPC profile dictionary containing slurm settings
        output_dir: Directory for job output files
        job_name: Optional custom job name (defaults to profile's job-name)
    """
    slurm_directives = hpc_profile.get("slurm_directives", {})
    module_load = hpc_profile.get("module_load", "")
    run_cmd = hpc_profile.get("run_command", "")
    tasks_per_node = hpc_profile.get("tasks_per_node", 1)
    environment = hpc_profile.get("environment", {})

    # Use provided job_name or fall back to profile default
    if job_name is None:
        job_name = slurm_directives.get("job-name", "vasp")

    # Build the #SBATCH lines
    sbatch_lines = ["#!/bin/bash"]
    
    # Handle special output/error file naming with job name substitution
    output_file = slurm_directives.get("output", f"{job_name}.out")
    error_file = slurm_directives.get("error", f"{job_name}.err")
    
    # Add job name and output/error files first
    sbatch_lines.extend([
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_file}",
        f"#SBATCH --error={error_file}",
    ])

    # Add other directives
    for key, val in slurm_directives.items():
        if key in ["job-name", "output", "error"]:
            continue  # already handled
        if val is None:  # Handle flags without values (like --exclusive)
            sbatch_lines.append(f"#SBATCH --{key}")
        else:
            sbatch_lines.append(f"#SBATCH --{key}={val}")

    # Build complete script
    script_lines = sbatch_lines + [
        "",
        "# Module loading",
        module_load,
        "",
        "# Environment setup",
    ]
    if environment:
        for var, value in environment.items():
            script_lines.append(f"export {var}={value}")

    script_lines.extend([
        "",
        "# Job execution",
    ])

    # Calculate total cores and handle variable substitution
    node_str = slurm_directives.get("nodes", 1)
    try:
        num_nodes = int(node_str)
    except ValueError:
        num_nodes = 1
    total_cores = num_nodes * tasks_per_node
    
    # Replace variables in run command
    if "${TOTAL_CORES}" in run_cmd:
        run_cmd = run_cmd.replace("${TOTAL_CORES}", str(total_cores))
    if "${NODES}" in run_cmd:
        run_cmd = run_cmd.replace("${NODES}", str(num_nodes))
    if "${GPUS}" in run_cmd:
        gpus = slurm_directives.get("gpus", 0)
        run_cmd = run_cmd.replace("${GPUS}", str(gpus))

    script_lines.append("cd $SLURM_SUBMIT_DIR")
    script_lines.append(run_cmd)

    return "\n".join(script_lines) + "\n"


def prepare_vasp_job_from_ase(
    atoms: Atoms,
    vasp_profile_name: str,
    hpc_profile_name: str,
    output_dir: str,
    auto_kpoints: bool = False,
    DEBUG: bool = False,
    job_name: str = None,
) -> None:
    """Create a VASP job folder from an ASE Atoms object.

    Args:
        atoms: ASE Atoms object to run calculations on
        vasp_profile_name: Name of VASP settings profile to use
        hpc_profile_name: Name of HPC profile to use
        output_dir: Directory to create job files in
        auto_kpoints: Whether to automatically determine k-points
        DEBUG: Whether to print debug information
        job_name: Optional custom job name (defaults to profile's job-name)
    """
    # Get the forge package root directory
    forge_root = Path(__file__).parent.parent.parent

    # Debug: Print paths
    hpc_profile_dir = forge_root / "forge" / "workflows" / "hpc_profiles"
    vasp_profile_dir = forge_root / "forge" / "workflows" / "vasp_settings"
    if DEBUG:
        print(f"[DEBUG] Looking for HPC profile in: {hpc_profile_dir}")
        print(f"[DEBUG] Looking for VASP profile in: {vasp_profile_dir}")
        print(f"[DEBUG] HPC profile directory exists: {hpc_profile_dir.exists()}")
        print(f"[DEBUG] VASP profile directory exists: {vasp_profile_dir.exists()}")
        if hpc_profile_dir.exists():
            print(f"[DEBUG] Available HPC profiles: {list(hpc_profile_dir.glob('*.json'))}")
        if vasp_profile_dir.exists():
            print(f"[DEBUG] Available VASP profiles: {list(vasp_profile_dir.glob('*.json'))}")

    # Use absolute paths for profile directories
    hpc_manager = ProfileManager(hpc_profile_dir)
    vasp_manager = ProfileManager(vasp_profile_dir)

    # Load and retrieve profiles
    hpc_manager.load_profile(hpc_profile_name)
    vasp_manager.load_profile(vasp_profile_name)
    hpc_profile = hpc_manager.get_profile(hpc_profile_name)
    vasp_profile = vasp_manager.get_profile(vasp_profile_name)

    os.makedirs(output_dir, exist_ok=True)

    # Sort atoms by chemical symbol
    unique_species = sorted(set(atoms.get_chemical_symbols()))
    sorted_indices = []

    if DEBUG:
        print("\n[DEBUG] Original atom ordering:")
        symbols = atoms.get_chemical_symbols()
        elem_counts = {sym: symbols.count(sym) for sym in unique_species}
        for elem, count in elem_counts.items():
            print(f"  {elem}: {count} atoms")

    for symbol in unique_species:
        indices = [
            i
            for i, s in enumerate(atoms.get_chemical_symbols())
            if s == symbol
        ]
        sorted_indices.extend(indices)

    # Create a new sorted Atoms object
    sorted_atoms = atoms[sorted_indices]

    if DEBUG:
        print("\n[DEBUG] Sorted atom ordering:")
        symbols = sorted_atoms.get_chemical_symbols()
        elem_counts = {sym: symbols.count(sym) for sym in unique_species}
        for elem, count in elem_counts.items():
            print(f"  {elem}: {count} atoms")
        print(
            f"\n[DEBUG] POTCAR will be concatenated in this order: {' + '.join(unique_species)}"
        )

        # Verify sorting worked correctly
        prev_symbol = None
        is_sorted = True
        for symbol in sorted_atoms.get_chemical_symbols():
            if prev_symbol and symbol < prev_symbol:
                is_sorted = False
                break
            prev_symbol = symbol
        print(f"[DEBUG] Atoms are properly sorted: {is_sorted}")

    # Write sorted POSCAR
    write(os.path.join(output_dir, "POSCAR"), sorted_atoms, format="vasp")

    # Write INCAR
    _write_incar(vasp_profile["incar"], os.path.join(output_dir, "INCAR"))

    # Determine KPOINTS
    base_kpts = vasp_profile["kpoints"].get("base_kpts", [4, 4, 4])
    gamma_center = vasp_profile["kpoints"].get("gamma", True)
    kpts, gamma_flag = determine_kpoint_grid(
        sorted_atoms,
        auto_kpoints=auto_kpoints,
        base_kpts=tuple(base_kpts),
        gamma=gamma_center,
    )
    _write_kpoints(os.path.join(output_dir, "KPOINTS"), kpts, gamma_flag)

    # Write POTCAR
    potcar_map = vasp_profile["potcars"]
    potcar_dir = os.environ.get("VASP_PP_PATH")
    if potcar_dir is None:
        raise ValueError("VASP_PP_PATH environment variable is not set.")
    _write_potcar(
        unique_species, potcar_map, potcar_dir, os.path.join(output_dir, "POTCAR")
    )

    # Create Slurm script
    slurm_script = _create_slurm_script(
        hpc_profile, 
        output_dir,
        job_name=job_name
    )
    with open(os.path.join(output_dir, "submit.sh"), "w") as f:
        f.write(slurm_script)

    print(
        f"[INFO] Created VASP job in {output_dir} using HPC={hpc_profile_name}, VASP={vasp_profile_name}"
    )


def prepare_vasp_job_from_db(
    db_manager: DatabaseManager,
    structure_id: int,
    vasp_profile_name: str,
    hpc_profile_name: str,
    output_dir: str = None,
    auto_kpoints: bool = False,
) -> None:
    """Create a VASP job folder from a structure in the DB.

    Args:
        db_manager: DatabaseManager instance
        structure_id: ID of structure in database
        vasp_profile_name: Name of VASP settings profile to use
        hpc_profile_name: Name of HPC profile to use
        output_dir: Directory to create job files in (defaults to job_{structure_id})
        auto_kpoints: Whether to automatically determine k-points
    """
    # Check if job is already pending
    if is_job_already_pending(db_manager, structure_id, hpc_profile_name):
        print(
            f"[WARNING] Structure {structure_id} already has a pending job for {hpc_profile_name}"
        )
        return

    # Mark job as pending in database
    mark_job_pending(db_manager, structure_id, hpc_profile_name)

    # Set default output directory
    if output_dir is None:
        output_dir = f"job_{structure_id}"

    # Get structure from database
    atoms = db_manager.get_structure(structure_id)

    # Use the ASE-based job preparation
    prepare_vasp_job_from_ase(
        atoms=atoms,
        vasp_profile_name=vasp_profile_name,
        hpc_profile_name=hpc_profile_name,
        output_dir=output_dir,
        auto_kpoints=auto_kpoints,
    )

    print(f"[INFO] Created VASP job for structure {structure_id} in {output_dir}")


def prepare_neb_vasp_job(
    start_outcar: str,
    end_outcar: str,
    n_images: int,
    vasp_profile_name: str,
    hpc_profile_name: str = "Perlmutter-GPU-NEB",
    output_dir: str = "neb_job",
    DEBUG: bool = False,
    job_name: str = None,
) -> None:
    """Create a VASP NEB job folder from start and end OUTCAR files.

    Args:
        start_outcar: Path to starting structure OUTCAR
        end_outcar: Path to ending structure OUTCAR
        n_images: Number of intermediate images (total images will be n_images + 2)
        vasp_profile_name: Name of VASP settings profile to use (e.g., "neb" or "neb-vtst")
        hpc_profile_name: Name of HPC profile to use
        output_dir: Directory to create NEB job files in
        DEBUG: Whether to print debug information
        job_name: Optional custom job name (defaults to profile's job-name)
    """
    # Read structures
    start_atoms = read(start_outcar)
    end_atoms = read(end_outcar)

    # Validate structures
    if len(start_atoms) != len(end_atoms):
        raise ValueError("Start and end structures must have the same number of atoms")
    if start_atoms.get_chemical_formula() != end_atoms.get_chemical_formula():
        raise ValueError("Start and end structures must have the same composition")

    # Create NEB images
    images = [start_atoms.copy() for _ in range(n_images + 2)]  # +2 for start and end
    images[-1] = end_atoms.copy()

    # Interpolate
    neb = DyNEB(images)
    neb.interpolate(mic=True)

    # Create main job directory
    os.makedirs(output_dir, exist_ok=True)

    # Write interpolation trajectory for visualization
    write(os.path.join(output_dir, "neb_path.xyz"), images)

    # Setup image directories and files
    for i in range(len(images)):
        # Create image directory
        image_dir = os.path.join(output_dir, f"{i:02d}")
        os.makedirs(image_dir, exist_ok=True)

        # Write POSCAR for each image
        write(os.path.join(image_dir, "POSCAR"), images[i], format="vasp")

        # Copy OUTCAR for endpoints
        if i == 0:
            shutil.copy2(start_outcar, os.path.join(image_dir, "OUTCAR"))
        elif i == len(images) - 1:
            shutil.copy2(end_outcar, os.path.join(image_dir, "OUTCAR"))

    # Get the forge package root directory
    forge_root = Path(__file__).parent.parent.parent

    # Setup profile managers
    hpc_profile_dir = forge_root / "forge" / "workflows" / "hpc_profiles"
    vasp_profile_dir = forge_root / "forge" / "workflows" / "vasp_settings"

    if DEBUG:
        print(f"[DEBUG] Looking for HPC profile in: {hpc_profile_dir}")
        print(f"[DEBUG] Looking for VASP profile in: {vasp_profile_dir}")

    # Load and retrieve profiles
    hpc_manager = ProfileManager(hpc_profile_dir)
    vasp_manager = ProfileManager(vasp_profile_dir)

    hpc_manager.load_profile(hpc_profile_name)
    vasp_manager.load_profile(vasp_profile_name)
    hpc_profile = hpc_manager.get_profile(hpc_profile_name)
    vasp_profile = vasp_manager.get_profile(vasp_profile_name)

    # Write shared input files in main directory
    _write_incar(vasp_profile["incar"], os.path.join(output_dir, "INCAR"))

    # Get unique species from first image (all should be same)
    unique_species = sorted(set(images[0].get_chemical_symbols()))

    # Write KPOINTS
    base_kpts = vasp_profile["kpoints"].get("base_kpts", [4, 4, 4])
    gamma_center = vasp_profile["kpoints"].get("gamma", True)
    kpts, gamma_flag = determine_kpoint_grid(
        images[0],
        auto_kpoints=False,
        base_kpts=tuple(base_kpts),
        gamma=gamma_center,
    )
    _write_kpoints(os.path.join(output_dir, "KPOINTS"), kpts, gamma_flag)

    # Write POTCAR
    potcar_map = vasp_profile["potcars"]
    potcar_dir = os.environ.get("VASP_PP_PATH")
    if potcar_dir is None:
        raise ValueError("VASP_PP_PATH environment variable is not set.")
    _write_potcar(
        unique_species, potcar_map, potcar_dir, os.path.join(output_dir, "POTCAR")
    )

    # Create Slurm script
    slurm_script = _create_slurm_script(
        hpc_profile, 
        output_dir,
        job_name=job_name
    )
    with open(os.path.join(output_dir, "submit.sh"), "w") as f:
        f.write(slurm_script)

    print(
        f"[INFO] Created NEB job in {output_dir} using HPC={hpc_profile_name}, VASP={vasp_profile_name}"
    )
    print(
        f"[INFO] NEB path visualization saved to {os.path.join(output_dir, 'neb_path.xyz')}"
    )


# Alias for backward compatibility
prepare_vasp_job = prepare_vasp_job_from_db
