# vasp_to_db.py
import os
import re # Keep re if needed elsewhere, otherwise VaspParser handles it
import json
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
import multiprocessing
import traceback

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

def add_vasp_results_to_db(db_manager: DatabaseManager, 
                           structure_id: int, 
                           output_dir: str, 
                           calculation_type="static", 
                           vasp_profile_name: Optional[str] = None, 
                           hpc_profile_name: Optional[str] = None):
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


# The legacy `process_vasp_directory_and_add` function has been removed.
# Please use the `process_vasp_jobs` function as the standard method
# for importing new VASP jobs from directories.


def _parse_single_vasp_job(args: Tuple) -> Dict:
    """
    Helper function for parallel processing. Parses a single VASP job directory.
    """
    job_dir, calculation_type, default_config_type = args
    try:
        metadata_path = job_dir / "metadata.json"
        if not metadata_path.exists():
            return {'status': 'missing_meta', 'path': str(job_dir)}

        parser = VaspParser(str(job_dir), calculation_type=calculation_type)
        if not parser.is_successful:
            return {'status': 'parse_fail', 'path': str(job_dir), 'error': parser.error_message}

        final_atoms = parser.atoms
        calc_data_from_parser = parser.get_calculation_data()

        if not final_atoms or calc_data_from_parser is None:
            return {'status': 'parse_fail', 'path': str(job_dir), 'error': 'Could not extract atoms or calculation data from OUTCAR.'}

        with open(metadata_path, 'r') as f:
            metadata_from_json = json.load(f)

        config_type = metadata_from_json.get('config_type')
        if not config_type:
            if default_config_type:
                metadata_from_json['config_type'] = default_config_type
            else:
                return {'status': 'parse_fail', 'path': str(job_dir), 'error': f"'config_type' not found in {metadata_path} and no 'default_config_type' was provided."}
        
        return {
            'status': 'success',
            'atoms': final_atoms,
            'calc_data': calc_data_from_parser,
            'metadata': metadata_from_json,
            'job_dir': str(job_dir),
        }
    except Exception as e:
        return {'status': 'exception', 'path': str(job_dir), 'error': str(e), 'traceback': traceback.format_exc()}


def process_vasp_jobs(
    db_manager: DatabaseManager,
    base_dir: str,
    generation_tag: int,
    calculation_type: str = 'static',
    default_config_type: Optional[str] = None,
    skip_duplicates: bool = True
):
    """
    Processes VASP jobs, reads metadata from JSON, and adds them to the database.

    This is the standard workflow for importing new VASP calculations. It scans a
    directory for jobs, each expecting an OUTCAR and an accompanying `metadata.json`
    file. It uses multiprocessing to parse jobs in parallel and batch database
    insertions for high performance.

    Args:
        db_manager (DatabaseManager): Instance of the database manager.
        base_dir (str): The root directory to search for VASP jobs.
        generation_tag (int): The generation number to assign to all new structures.
        calculation_type (str, optional): The type of VASP calculation, e.g.,
            'static' or 'relax'. Defaults to 'static'.
        default_config_type (Optional[str], optional): A fallback config type to use
            if 'config_type' is not found in the `metadata.json`. If this is not
            provided and 'config_type' is missing, an error will be raised.
            This value is stored within the structure's metadata. Defaults to None.
        skip_duplicates (bool, optional): If True, checks for duplicates in the
            database before adding new structures. Defaults to True.
    """
    base_path = Path(base_dir)
    print(f"[INFO] Starting batch processing of VASP jobs in: {base_path}")
    print(f"[INFO] Assigning all new structures to Generation: {generation_tag}")

    skipped_duplicate_count = 0
    failed_parse_count = 0
    missing_meta_count = 0

    outcar_paths = list(base_path.rglob('OUTCAR'))
    job_dirs = [p.parent for p in outcar_paths]
    total_dirs = len(job_dirs)
    print(f"[INFO] Found {total_dirs} potential VASP calculation directories.")

    # --- Stage 1: Parse all structures in parallel ---
    parsed_results = []
    print("\n[INFO] Stage 1: Parsing VASP jobs in parallel...")
    with multiprocessing.Pool() as pool:
        args_list = [(job_dir, calculation_type, default_config_type) for job_dir in job_dirs]
        results_iterator = pool.imap_unordered(_parse_single_vasp_job, args_list)
        
        for result in tqdm(results_iterator, total=total_dirs, desc="Parsing VASP jobs"):
            if result['status'] == 'success':
                parsed_results.append(result)
            elif result['status'] == 'missing_meta':
                missing_meta_count += 1
            else: # parse_fail or exception
                failed_parse_count += 1
                print(f"\n[WARN] Failed to parse {result['path']}: {result['error']}")
                if 'traceback' in result:
                    print(result['traceback'])
    
    processed_count = len(parsed_results)

    # --- Stage 2: Filter duplicates and prepare for batch insert ---
    print("\n[INFO] Stage 2: Filtering duplicates and preparing data...")
    structures_to_add = []
    calculations_to_prepare = []
    
    # Pre-calculate duplicate flags in a single batch call if required
    is_duplicate_list = [False] * len(parsed_results)
    if skip_duplicates and parsed_results:
        print("[INFO] Performing batch duplicate check against database...")
        atoms_to_check = [res['atoms'] for res in parsed_results]
        is_duplicate_list = db_manager.batch_check_duplicates(atoms_to_check)

    for i, result in enumerate(tqdm(parsed_results, desc="Preparing structures")):
        if is_duplicate_list[i]:
            skipped_duplicate_count += 1
            continue

        # This structure is not a duplicate, prepare it for insertion
        metadata_from_json = result['metadata']
        calc_data_from_parser = result['calc_data']

        structure_metadata_for_db = metadata_from_json.copy()
        structure_metadata_for_db['generation'] = generation_tag
        structure_metadata_for_db['date_added_to_db'] = datetime.now().isoformat()
        
        structures_to_add.append({
            'atoms': result['atoms'],
            'source_type': 'vasp-from-metadata',
            'parent_id': metadata_from_json.get('parent_id'),
            'metadata': structure_metadata_for_db
        })

        calculations_to_prepare.append({
            'calculator': 'vasp',
            'calculation_type': calculation_type,
            'calculation_source_path': result['job_dir'],
            'energy': calc_data_from_parser.get('energy'),
            'forces': calc_data_from_parser.get('forces'),
            'stress': calc_data_from_parser.get('stress'),
            'metadata': calc_data_from_parser.get('metadata', {})
        })

    # --- Stage 3: Batch insert structures and calculations ---
    added_count = 0
    failed_db_add_count = 0
    if not structures_to_add:
        print("\n[INFO] No new, non-duplicate structures found to add.")
    else:
        print(f"\n[INFO] Stage 3: Batch inserting {len(structures_to_add)} structures...")
        try:
            new_structure_ids = db_manager.batch_add_structures(structures_to_add)
            
            print(f"\n[INFO] Stage 4: Batch inserting {len(new_structure_ids)} corresponding calculations...")
            
            calculations_to_add = []
            for i, struct_id in enumerate(new_structure_ids):
                calculations_to_add.append({
                    'structure_id': struct_id,
                    'calc_data': calculations_to_prepare[i]
                })

            new_calc_ids = db_manager.batch_add_calculations(calculations_to_add)
            added_count = len(new_calc_ids)
            if len(new_structure_ids) != added_count:
                failed_db_add_count = len(new_structure_ids) - added_count

        except Exception as e_struct:
            print(f"\n[ERROR] A critical error occurred during a batch database operation: {e_struct}")
            failed_db_add_count = len(structures_to_add)

    # Summary print
    print("\n[INFO] Finished batch processing.")
    print(f"  - Successfully processed (OUTCAR parsed): {processed_count}")
    print(f"  - Successfully added to DB (Structure + Calc): {added_count}")
    print(f"  - Skipped (Duplicate): {skipped_duplicate_count}")
    print(f"  - Skipped (Missing metadata.json): {missing_meta_count}")
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
