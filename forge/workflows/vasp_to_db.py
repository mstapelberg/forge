# vasp_to_db.py
import os
import re # Keep re if needed elsewhere, otherwise VaspParser handles it
from pathlib import Path
# from pymatgen.io.vasp import Vasprun # No longer needed
# from ase.io.vasp import read_vasp_out # Handled by VaspParser
from forge.core.database import DatabaseManager
from forge.workflows.vasp_parser import VaspParser # Import the new class
import numpy as np # Keep for type hints or other uses if any
from datetime import datetime
from typing import Dict, Optional, Generator, Tuple, List # Added Generator, List
from dataclasses import dataclass # Added dataclass
from tqdm import tqdm # Added tqdm
import ase # Added ase

# --- Added from convert_outcars.py ---
@dataclass
class StructureMetadata:
    """Container for structure metadata parsed from paths."""
    generation: Optional[int] = None # Made fields optional
    structure_type: Optional[str] = None
    structure_index: Optional[int] = None
    composition: Optional[str] = None
    temperature: Optional[int] = None
    config_type: Optional[str] = None
    batch_id: Optional[int] = None
    adversarial_step: Optional[int] = None
    source_path: Optional[str] = None # Changed Path to str for JSON compatibility

def parse_path_metadata(path: Path, atoms: Optional[ase.Atoms] = None) -> Dict[str, any]:
    """
    Simplified metadata extraction from directory structure for Workflow A.
    Focuses on generation, config_type, and uses Atoms for composition.
    """
    path_str = str(path)
    metadata = {'source_path': path_str} # Store the source path

    # --- Generation ---
    gen_match = re.search(r'job_gen_(\d+)', path_str)
    if gen_match:
        metadata['generation'] = int(gen_match.group(1))
    else:
        metadata['generation'] = 0 # Or some default if not found

    struct_idx_match = re.search(r'_idx_(\d+)', path_str)
    if struct_idx_match:
        metadata['structure_index'] = int(struct_idx_match.group(1))

    # --- Composition String ---
    if atoms:
         metadata['composition_str'] = atoms.get_chemical_formula()
    else:
         # Fallback: Try parsing from path (less reliable)
         comp_match = re.search(r'/([A-Z][a-z]?\d+)_', path_str) # Example: /Cr2_...
         if comp_match:
              metadata['composition_str_from_path'] = comp_match.group(1)

    config_match = re.search(r'_([a-zA-Z0-9_-]+)_idx_\d+', path_str)
    config_type = config_match.group(1) if config_match else None
    if config_type:
        metadata['config_type'] = config_type
    else:
        metadata['config_type'] = None

    return metadata

def add_vasp_results_to_db(db_manager: DatabaseManager, structure_id: int, output_dir: str, calculation_type="static", vasp_profile_name: Optional[str] = None, hpc_profile_name: Optional[str] = None):
    """
    Parses VASP results from an OUTCAR using VaspParser and adds them
    to the 'calculations' table associated with the structure_id.
    Updates the structure's metadata job status.

    Args:
        db_manager: Instance of DatabaseManager.
        structure_id: The ID of the structure this calculation belongs to.
        output_dir: The path to the VASP calculation directory.
        calculation_type: Type of VASP calculation (e.g., 'static', 'relax').
    """
    print(f"[INFO] Processing VASP results for initial structure {structure_id} in: {output_dir} (Type: {calculation_type})")
    parser = VaspParser(output_dir, calculation_type=calculation_type)

    job_key = hpc_profile_name or vasp_profile_name or calculation_type # Key for metadata update

    if not parser.is_successful:
        print(f"[ERROR] Failed to parse core VASP results for structure {structure_id} in {output_dir}. Error: {parser.error_message}")
        # Update structure metadata to reflect the error
        try:
            metadata = db_manager.get_structure_metadata(structure_id)
            if metadata:
                 jobs_meta = metadata.get("jobs", {})
                 job_info = jobs_meta.get(job_key, {})
                 job_info["status"] = "parse_error" # Or potentially 'vasp_error' if detectable
                 job_info["error"] = parser.error_message
                 job_info["completed_timestamp"] = datetime.now().isoformat()
                 jobs_meta[job_key] = job_info # Ensure update
                 metadata["jobs"] = jobs_meta
                 db_manager.update_structure_metadata(structure_id, metadata)
                 print(f"[INFO] Updated initial structure {structure_id} metadata with parse error for job '{job_key}'.")
        except Exception as e:
            print(f"[WARN] Could not update initial structure {structure_id} metadata with error status: {e}")
        return # Stop processing this directory

    # Get the formatted data dictionary
    calc_data_dict = parser.get_calculation_data() # Contains energy, forces, stress, metadata from OUTCAR

    if calc_data_dict is None: # Should not happen if is_successful is True
         print(f"[ERROR] Parser reported success but failed to generate calculation data for {output_dir}")
         # Optionally update status to an internal error
         return

    target_structure_id = structure_id # Default: link calc to initial structure
    final_status = f"completed_{calculation_type}"
    new_structure_id = None

    # --- Handle Relaxation Specifics ---
    if calculation_type == "relax":
        try:
            relaxed_atoms = parser.atoms # Get the final Atoms object from OUTCAR
            if relaxed_atoms:
                # Add the relaxed structure as a new entry
                print(f"[INFO] Adding final relaxed structure from {output_dir} to DB.")
                # Combine path and OUTCAR metadata for the new structure
                path_meta = parse_path_metadata(Path(output_dir))
                # OUTCAR metadata is already *inside* calc_data_dict['metadata']
                meta_for_relaxed = {
                    "source": "vasp-relax",
                    "parent_structure_id": structure_id, # Link back to the original
                    **path_meta, # Add path info
                    "vasp_metadata": calc_data_dict.get("metadata", {}) # Embed VASP specific meta
                 }


                new_structure_id = db_manager.add_structure(
                    relaxed_atoms,
                    source_type='vasp-relax',
                    parent_id=structure_id, # Explicit parent ID
                    metadata=meta_for_relaxed
                )
                target_structure_id = new_structure_id # Link calculation to the *new* relaxed structure
                print(f"[INFO] Added relaxed structure with ID: {new_structure_id}, parent: {structure_id}")
                final_status = "completed_relax" # More specific status
            else:
                print(f"[WARN] Calculation type is 'relax' but could not get relaxed Atoms object from parser for {output_dir}.")
                final_status = "relax_postprocess_warn" # Indicate issue

        except Exception as e:
            print(f"[ERROR] Failed to add relaxed structure for {structure_id} from {output_dir}: {e}")
            final_status = "relax_postprocess_error" # Indicate failure during post-processing

    # --- Add the calculation to the database ---
    try:
        # Ensure calc_data_dict does NOT contain the nested 'metadata' if it was extracted for the structure
        # (Adjust based on how add_calculation expects data)
        calc_data_for_db = calc_data_dict.copy()
        # If add_calculation expects metadata directly, keep it, otherwise remove if stored with structure
        # calc_data_for_db.pop('metadata', None) # Example: Remove if structure handles it

        # Add calculation linked to the appropriate structure ID (initial or relaxed)
        calc_id = db_manager.add_calculation(
            structure_id=target_structure_id,
            calc_data=calc_data_for_db # Use potentially modified dict
        )
        print(f"[INFO] Uploaded VASP calculation {calc_id} to DB for structure {target_structure_id}")

        # Update the *initial* structure's metadata to mark job as completed
        metadata = db_manager.get_structure_metadata(structure_id)
        if metadata:
            jobs_meta = metadata.get("jobs", {})
            job_info = jobs_meta.get(job_key, {}) # Get existing or new dict
            job_info["status"] = final_status
            job_info["completed_timestamp"] = datetime.now().isoformat()
            if new_structure_id: # Store the ID of the final structure if relaxed
                job_info["final_structure_id"] = new_structure_id
            if "error" in job_info: # Clear previous error if successful now
                del job_info["error"]

            jobs_meta[job_key] = job_info # Update the specific job entry
            metadata["jobs"] = jobs_meta # Put the updated jobs dict back
            db_manager.update_structure_metadata(structure_id, metadata)
            print(f"[INFO] Updated initial structure {structure_id} metadata status to '{final_status}' for job '{job_key}'.")
        else:
             print(f"[WARN] Could not retrieve metadata to update status for initial structure {structure_id}.")

    except Exception as e:
        print(f"[ERROR] Failed to add calculation or update metadata for structure {structure_id}/{target_structure_id}: {e}")
        # Optionally update status to indicate DB error


# --- BATCH PROCESSING FUNCTION (WORKFLOW A) ---
def process_vasp_directory_and_add(db_manager: DatabaseManager, base_dir: str, skip_duplicates: bool = True, default_source_type: str = 'vasp'):
    """
    Walks through VASP directories, parses results, and adds structure+calculation to DB.
    (Workflow A: Assumes structures are NOT already in DB; uses simplified path parsing)

    Args:
        db_manager: Instance of DatabaseManager.
        base_dir: The root directory containing VASP job subfolders.
        skip_duplicates: If True, check DB for duplicates before adding.
        default_source_type: The source_type to assign to structures added via this function.
    """
    base_path = Path(base_dir)
    print(f"[INFO] Starting WF-A batch processing of VASP directories in: {base_path}")
    processed_count = 0
    added_count = 0
    skipped_duplicate_count = 0
    failed_db_add_count = 0
    failed_parse_count = 0
    outcar_paths = list(base_path.rglob('OUTCAR'))
    total_dirs = len(outcar_paths)
    print(f"[INFO] Found {total_dirs} potential VASP calculation directories.")


    with tqdm(total=total_dirs, desc="Processing VASP dirs (WF-A)") as pbar:
        for outcar_path in outcar_paths:
            job_dir = outcar_path.parent
            pbar.set_postfix_str(f"Processing: ...{str(job_dir)[-40:]}", refresh=True)
            try:
                # 1. Parse OUTCAR
                parser = VaspParser(str(job_dir))
                if not parser.is_successful:
                    failed_parse_count += 1
                    pbar.update(1)
                    continue

                # 2. Get parsed data
                final_atoms = parser.atoms
                calc_data_from_parser = parser.get_calculation_data()

                if not final_atoms or calc_data_from_parser is None:
                     failed_parse_count += 1 # Count as parse fail if data missing
                     pbar.update(1)
                     continue

                processed_count += 1 # Increment successfully parsed counter


                # 3. Parse path metadata (SIMPLIFIED)
                path_meta = parse_path_metadata(job_dir, final_atoms)
                # structure_source_type is now set via function argument default_source_type


                # 4. Check for duplicates (optional)
                is_duplicate = False
                if skip_duplicates:
                    try:
                        if db_manager.check_duplicate_structure(final_atoms):
                            is_duplicate = True
                            skipped_duplicate_count += 1
                    except Exception as e:
                        pass # Log error? Count as fail? For now, proceed.

                # 5. Add to Database if not duplicate
                if not is_duplicate:
                    struct_id = None # Keep track in case calc add fails
                    try:
                        # --- Prepare Structure Data ---
                        # Metadata now only contains source_path, generation (optional), composition_str
                        structure_metadata_for_db = path_meta.copy()
                        structure_metadata_for_db['date_added_to_db'] = datetime.now().isoformat()

                        struct_id = db_manager.add_structure(
                            atoms=final_atoms,
                            source_type=default_source_type, # Use the default passed to function
                            metadata=structure_metadata_for_db # Simplified metadata
                        )

                        # --- Prepare Calculation Data ---
                        calc_data_for_db = {
                             'calculator': 'vasp',
                             'calculation_source_path': str(job_dir),
                             'energy': calc_data_from_parser.get('energy'),
                             'forces': calc_data_from_parser.get('forces'),
                             'stress': calc_data_from_parser.get('stress'),
                             'metadata': calc_data_from_parser.get('metadata', {}) # VASP run details
                         }


                        calc_id = db_manager.add_calculation(
                            structure_id=struct_id,
                            calc_data=calc_data_for_db
                        )
                        added_count += 1

                    except Exception as e:
                        print(f"\n[ERROR] Failed DB add for {job_dir} (Struct/Calc): {e}")
                        failed_db_add_count += 1
                        # Optional: If structure was added but calc failed, remove structure?
                pbar.update(1)

            except Exception as e:
                print(f"\n[ERROR] Unhandled exception for {job_dir}: {e}")
                import traceback
                traceback.print_exc()
                failed_parse_count += 1 # Count unhandled as parse fail
                pbar.update(1)

    # Updated summary print
    print(f"\n[INFO] Finished WF-A batch processing.")
    print(f"  - Successfully processed (OUTCAR parsed): {processed_count}")
    print(f"  - Successfully added to DB (Structure + Calc): {added_count}")
    print(f"  - Skipped as duplicate: {skipped_duplicate_count}")
    print(f"  - Failed (Parse Error or Missing Data): {failed_parse_count}")
    print(f"  - Failed (DB Add Error): {failed_db_add_count}")

# Example of how you might call the batch processing function
if __name__ == "__main__":
    # This is example usage, replace with your actual DB setup and directory
    db_config = {
        'database': {
            'dbname': 'your_db_name',
            'user': 'your_user',
            'password': 'your_password',
            'host': 'your_host',
            'port': 5432
        }
    }
    dbm = DatabaseManager(config_dict=db_config)

    # --- Example 1: Process a single known directory ---
    # structure_id_for_single_job = 3129
    # single_job_directory = "/path/to/your/data/job_gen_7_2025-03-28/Cr/Cr2_distorted_bcc_idx_3129"
    # add_vasp_results_to_db(dbm, structure_id_for_single_job, single_job_directory)

    # --- Example 2: Process a base directory containing multiple jobs ---
    # base_vasp_directory = "/path/to/your/data/job_gen_7_2025-03-28/"
    # process_vasp_directory(dbm, base_vasp_directory)

    print("\nScript finished.")
