import os
from pathlib import Path
import numpy as np 
from ase.io import read
from ase.calculators.vasp import Vasp
from ase.build import bulk

def determine_kpoint_grid(atoms, auto_kpoints=False, base_kpts=(4,4,4), gamma=True):
    """
    Determine k-point grid by scaling relative to a reference cell size.
    For example, if you use 4x4x4 for a 128-atom cell, it will scale
    appropriately for larger/smaller cells.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        The atomic structure for which we want to compute a k-point grid.
    auto_kpoints : bool
        If True, scales k-points based on cell size.
        If False, returns base_kpts (default 4,4,4).
    base_kpts : tuple(int, int, int)
        Reference k-point mesh for your reference system
        (e.g., 4,4,4 for a 128-atom cell)
    gamma : bool
        Whether to use gamma-centered k-points.
    
    Returns:
    --------
    (kpts, gamma_flag) : ((int,int,int), bool)
        K-point mesh and gamma-centering flag
    """
    if not auto_kpoints:
        return base_kpts, gamma
    
    # Get cell vectors
    cell = atoms.get_cell()
    
    # Calculate lengths of cell vectors
    lengths = np.sqrt(np.sum(cell**2, axis=1))
    
    # Reference cell size (e.g., for 128-atom BCC cell)
    ref_atoms = bulk('V', crystalstructure='bcc', a=3.01, cubic=True)
    ref_supercell = ref_atoms * (4,4,4)
    ref_lengths = np.sqrt(np.sum(ref_supercell.get_cell()**2, axis=1))
    print(ref_lengths)
    #ref_lengths = np.array([15.0, 15.0, 15.0])  # Example for 128-atom BCC
    
    # Scale k-points inversely with cell size
    # Larger cell → fewer k-points
    # Smaller cell → more k-points
    scaling = ref_lengths / lengths
    
    # Calculate new k-points, ensuring minimum of 1
    kpts = np.maximum(1, np.round(scaling * np.array(base_kpts))).astype(int)
    
    return tuple(kpts), gamma


def run_static_vasp(atoms, output_dir, create_inputs_only=True, auto_kpoints=False):
    """Create VASP input files for a static calculation.
    
    Args:
        atoms: ASE Atoms object
        output_dir: Directory to write VASP files
        create_inputs_only: Ignored, kept for backwards compatibility 
        auto_kpoints: Whether to automatically scale k-points based on cell size
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check pseudopotentials before setting up the calculator
    unique_elements = set(atoms.get_chemical_symbols())
    #check_pseudopotentials(unique_elements)
    
    # Determine appropriate setups for each element
    #setups = {symbol: get_vasp_potential(symbol) for symbol in unique_elements}
    
    my_kpts, my_gamma = determine_kpoint_grid(atoms, auto_kpoints=auto_kpoints, base_kpts=(4,4,4), gamma=True)

    # VASP calculator settings
    calc = Vasp(
        prec='Accurate',
        encut=520,
        ediff=1e-6,
        ediffg=-0.01,
        nelm=100,
        nsw=0,
        ibrion=-1,
        ismear=1,
        sigma=0.2,
        lcharg=True,
        lwave=False,
        lreal=False,
        lorbit=11,
        xc='PBE',
        kpts=my_kpts,
        gamma=my_gamma,
        #setups=setups,
        directory=output_dir,
        algo='Normal',
    )
    
    atoms.calc = calc

    # Create input files
    calc.initialize(atoms)
    calc.write_input(atoms)
    print(f"VASP input files created in {output_dir}")

def create_slurm_script(profile, job_name, output_dir, resources):
    """Create a SLURM job submission script."""
    
#TODO Need to change the scripts to work for a list of jobs within a single generation. 
    if profile == 'Perlmutter-GPU':
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/{job_name}.out
#SBATCH --error={output_dir}/{job_name}.err
#SBATCH --nodes={resources['nodes']}
#SBATCH --ntasks-per-node={resources['tasks_per_node']}
#SBATCH --time={resources['time']}
#SBATCH --partition={resources['partition']}

module load vasp-tpc/6.4.2-gpu
cd {output_dir}
mpirun vasp_std
"""
    elif profile == 'Perlmutter-CPU':
        script_content = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --output={output_dir}/{job_name}.out
#SBATCH --error={output_dir}/{job_name}.err
#SBATCH --nodes={resources['nodes']}
#SBATCH --time={resources['time']}
#SBATCH --partition={resources['partition']}

module load vasp-tpc/6.4.2-cpu

export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n {resources['nodes']*64} -c 4 vasp_std
"""

    elif profile == 'PSFC-GPU':
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/{job_name}.out
#SBATCH --error={output_dir}/{job_name}.err
#SBATCH --nodes={resources['nodes']}
#SBATCH --ntasks-per-node={resources['tasks_per_node']}
#SBATCH --time={resources['time']}
#SBATCH --partition={resources['partition']}

cd {output_dir}
mpirun /home/myless/VASP/vasp.6.4.2/bin/vasp_std
"""
    else:
        raise ValueError(f"Unknown profile: {profile}")

    return script_content
