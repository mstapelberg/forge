from pathlib import Path
import os
from ase.io import read
from forge.core.database import DatabaseManager
from pymatgen.io.vasp import Vasprun

def parse_vasp_output(output_dir: str):
    """
    Check the output_dir for OUTCAR, vasprun.xml, or CONTCAR,
    parse results, and return relevant data in a dictionary.
    """

    # Parse vasprun.xml file for a static calculation
    vasprun = Vasprun(os.path.join(output_dir, 'vasprun.xml'))
    
    # Get final energy without entropy contributions
    energy = vasprun.final_energy
    
    # Get forces from last ionic step
    forces = vasprun.ionic_steps[-1]['forces']
    
    # Get stress tensor from last ionic step (in kB, convert to eV/Å³)
    stress = vasprun.ionic_steps[-1]['stress']
    stress = stress * 0.0006241509125883258 # Convert kB to eV/Å³
    
    calc_data = {
        "model_type": "vasp",
        "model_path": os.environ.get('VASP_PATH', 'vasp'),
        "energy": float(energy), 
        "forces": forces,
        "stress": stress,
        "status": "completed",
        "calculation_type": "static"
    }
    return calc_data

def add_vasp_results_to_db(db_manager: DatabaseManager, structure_id: int, output_dir: str):
    """
    Parse VASP results from output_dir, then add them to the DB.
    """
    calc_data = parse_vasp_output(output_dir)
    db_manager.add_calculation(structure_id, calc_data)