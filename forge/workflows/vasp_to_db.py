import os
from pathlib import Path
from pymatgen.io.vasp import Vasprun
from ase.io.vasp import read_vasp_out
from forge.core.database import DatabaseManager
import numpy as np

def parse_vasp_output(output_dir: str, calculation_type="static"):
    """
    Attempt to parse results for final energy, forces, stress.
    Uses ase.read_vasp_out to parse OUTCAR if available,
    otherwise tries vasprun.xml with pymatgen.
    """
    outcar_path = os.path.join(output_dir, 'OUTCAR')
    vasprun_path = os.path.join(output_dir, 'vasprun.xml')
    energy = None
    forces = None
    stress = None

    # First try ASE parser - this is our preferred method
    if os.path.exists(outcar_path):
        try:
            final_atoms = read_vasp_out(outcar_path)
            energy = final_atoms.get_potential_energy()
            forces = final_atoms.get_forces()
            stress = final_atoms.get_stress(voigt=False)  # 9-component vector in ASE
            return {
                "model_type": "vasp",
                "calculation_type": calculation_type,
                "energy": float(energy),
                "forces": forces.tolist() if hasattr(forces, 'tolist') else forces,
                "stress": stress.tolist() if hasattr(stress, 'tolist') else stress,
                "status": "completed",
                "parser": "ase"
            }
        except Exception as e:
            print(f"Warning: Could not parse OUTCAR with ASE: {e}")
    
    # Fallback to vasprun.xml with validation against OUTCAR
    if os.path.exists(vasprun_path):
        try:
            vasprun = Vasprun(vasprun_path)
            energy = vasprun.final_energy
            last_step = vasprun.ionic_steps[-1]
            forces = last_step['forces']
            raw_stress = last_step['stress']
            # Start of Selection
            # Convert raw_stress from Voigt notation to a full 9-component tensor and apply conversion factor
            stress_voigt = np.array(raw_stress) * 0.0006241509125883258  # Convert kB to eV/Å³
            if len(stress_voigt) == 6:
                stress = np.array([
                    [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                    [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                    [stress_voigt[4], stress_voigt[3], stress_voigt[2]]
                ]).flatten()
            else:
                stress = stress_voigt
            
            # Validate against ASE parser if OUTCAR exists
            if os.path.exists(outcar_path):
                try:
                    ase_atoms = read_vasp_out(outcar_path)
                    ase_energy = ase_atoms.get_potential_energy()
                    ase_forces = ase_atoms.get_forces()
                    ase_stress = ase_atoms.get_stress()
                    
                    # Compare results (with reasonable tolerances)
                    energy_diff = abs(energy - ase_energy)
                    forces_diff = np.max(np.abs(np.array(forces) - ase_forces))
                    stress_diff = np.max(np.abs(np.array(stress) - ase_stress))
                    
                    if energy_diff > 1e-3 or forces_diff > 1e-3 or stress_diff > 1e-3:
                        print("Warning: Discrepancy between ASE and pymatgen parsing:")
                        print(f"Energy diff: {energy_diff:.6f} eV")
                        print(f"Max force diff: {forces_diff:.6f} eV/Å")
                        print(f"Max stress diff: {stress_diff:.6f} eV/Å³")
                except Exception as e:
                    print(f"Warning: Validation against ASE parser failed: {e}")
            
            return {
                "model_type": "vasp",
                "calculation_type": calculation_type,
                "energy": float(energy),
                "forces": forces.tolist() if hasattr(forces, 'tolist') else forces,
                "stress": stress.tolist() if hasattr(stress, 'tolist') else stress,
                "status": "completed",
                "parser": "pymatgen"
            }
        except Exception as e:
            raise RuntimeError(f"Unable to parse vasp outputs in {output_dir}: {e}")
    else:
        raise FileNotFoundError(f"No OUTCAR or vasprun.xml found in {output_dir}")

def add_vasp_results_to_db(db_manager: DatabaseManager, structure_id: int, output_dir: str, calculation_type="static"):
    """
    Parse VASP results and add them to the 'calculations' table.
    Update the structure's metadata from 'pending' to 'completed'.
    """
    calc_data = parse_vasp_output(output_dir, calculation_type=calculation_type)
    db_manager.add_calculation(structure_id, calc_data)
    
    # Mark job as completed in metadata.
    # We assume there's a key "jobs" -> {profile: {status: "pending"}}
    metadata = db_manager.get_structure_metadata(structure_id)
    jobs_meta = metadata.get("jobs", {})
    for profile, info in jobs_meta.items():
        if info.get("status") == "pending":
            info["status"] = "completed"
    metadata["jobs"] = jobs_meta
    db_manager.update_structure_metadata(structure_id, metadata)
    print(f"Uploaded VASP results to DB for structure {structure_id} and marked job as completed.")
