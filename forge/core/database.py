# Core database interface (core/database.py)
from typing import Any, Dict, List, Optional, Union, Tuple
import psycopg2
from psycopg2.extras import Json, execute_values
from ase import Atoms
import numpy as np
import yaml
from pathlib import Path
import os
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
import hashlib
from tqdm import tqdm

def _compute_composition_hash(comp_dict: Dict, decimal: int = 4) -> str:
    """
    Create a hash of composition dictionary, rounding fractions to specified decimals.
    """
    # Sort elements and round their fractions
    sorted_comp = sorted([
        (elem, round(data['at_frac'], decimal))
        for elem, data in comp_dict.items()
    ])
    
    # Create a string representation and hash it
    comp_str = '_'.join([f"{elem}{frac}" for elem, frac in sorted_comp])
    return hashlib.md5(comp_str.encode('utf-8')).hexdigest()

def fix_numpy(obj):
    """
    Recursively convert numpy types to Python native types.
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: fix_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [fix_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(fix_numpy(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return fix_numpy(obj.tolist())
    elif isinstance(obj, (np.intc, np.intp, np.int8, np.int16, 
                        np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        try:
            # Try to convert to a basic type if possible
            return obj.item() if hasattr(obj, 'item') else obj
        except:
            return str(obj)  # Last resort: convert to string

class DatabaseManager:
    """PostgreSQL database manager for FORGE."""
    
    def __init__(self, config_path: Union[str, Path] = None, config_dict: Dict = None, dry_run: bool = False, debug: bool = False):
        """
        Initialize database connection using configuration.
        
        Args:
            config_path: Path to database configuration YAML
            config_dict: Dictionary containing database configuration
            dry_run: If True, simulate database operations without actually writing
            debug: If True, print debug messages during initialization and potentially other operations.
            
        Note: If both config_path and config_dict are provided, config_dict takes precedence
        """
        self.dry_run = dry_run
        self.debug = debug

        if not dry_run:
            if self.debug: print("[DEBUG] Loading configuration...")
            self.config = self._load_config(config_path, config_dict)
            if self.debug: print("[DEBUG] Initializing database connection...")
            self.conn = self._initialize_connection()
            if self.debug: print("[DEBUG] Connection established.")
            self._initialize_tables()
        else:
            self.conn = None
            self._fake_id_counter = 1
            print("[INFO] DatabaseManager initialized in DRY RUN mode. No changes will be made.")
    
    def _load_config(self, config_path: Union[str, Path] = None, config_dict: Dict = None) -> Dict:
        """
        Load database configuration from YAML file or dictionary.
        
        Args:
            config_path: Path to database configuration YAML
            config_dict: Dictionary containing database configuration
            
        Returns:
            Dict: Database configuration
        """
        if config_dict is not None:
            return config_dict
        
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'database.yaml'
        
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _initialize_connection(self) -> psycopg2.extensions.connection:
        """Initialize PostgreSQL connection."""
        return psycopg2.connect(**self.config['database'])
    
    def _initialize_tables(self) -> None:
        """Create database tables and necessary indices if they don't exist."""
        if self.debug: print("[DEBUG] Entering _initialize_tables...")
        try:
            with self.conn.cursor() as cur:
                if self.debug: print("[DEBUG] Creating structures table IF NOT EXISTS...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS structures (
                        structure_id SERIAL PRIMARY KEY,
                        formula TEXT NOT NULL,
                        composition JSONB NOT NULL,
                        positions JSONB NOT NULL,
                        cell JSONB NOT NULL,
                        pbc BOOLEAN[] NOT NULL,
                        parent_structure_id INTEGER REFERENCES structures(structure_id),
                        source_type TEXT,
                        metadata JSONB,
                        composition_hash TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                if self.debug: print("[DEBUG] Creating structures composition_hash index IF NOT EXISTS...")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_structures_composition_hash
                    ON structures (composition_hash);
                """)
                if self.debug: print("[DEBUG] Creating structures parent_id index IF NOT EXISTS...")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_structures_parent_id
                    ON structures (parent_structure_id); -- Index for parent lookups
                """)

                if self.debug: print("[DEBUG] Creating calculations table IF NOT EXISTS...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS calculations (
                        calculation_id SERIAL PRIMARY KEY,
                        structure_id INTEGER REFERENCES structures(structure_id) ON DELETE CASCADE NOT NULL,
                        calculator TEXT DEFAULT 'vasp', -- Renamed, Default added
                        calculation_source_path TEXT, -- Renamed, Allows NULL
                        energy REAL,                  -- Changed type
                        forces JSONB,
                        stress JSONB,
                        -- Removed model_generation, ensemble_variance
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                if self.debug: print("[DEBUG] Creating calculations structure_id index IF NOT EXISTS...")
                cur.execute("""
                     CREATE INDEX IF NOT EXISTS idx_calculations_structure_id
                     ON calculations (structure_id);
                """)
                if self.debug: print("[DEBUG] Creating calculations calculator index IF NOT EXISTS...")
                cur.execute("""
                     CREATE INDEX IF NOT EXISTS idx_calculations_calculator
                     ON calculations (calculator); -- Index for calculator type
                """)

                if self.debug: print("[DEBUG] Creating mlip_models table IF NOT EXISTS...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS mlip_models (
                        mlip_model_id SERIAL PRIMARY KEY,
                        model_generation INTEGER NOT NULL,
                        mlip_type TEXT,
                        train_structure_ids INTEGER[],
                        validation_structure_ids INTEGER[],
                        test_structure_ids INTEGER[],
                        model_parameters JSONB,
                        wandb_link TEXT,
                        model_file_path TEXT NOT NULL, -- Stores local path or S3 URI
                        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        notes TEXT
                    )
                """)
                if self.debug: print("[DEBUG] Creating mlip_models generation index IF NOT EXISTS...")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mlip_models_generation ON mlip_models (model_generation);
                """)
                if self.debug: print("[DEBUG] Creating mlip_models type index IF NOT EXISTS...")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mlip_models_type ON mlip_models (mlip_type);
                """)
            if self.debug: print("[DEBUG] Committing transaction...")
            self.conn.commit()
            if self.debug: print("[DEBUG] Transaction committed.")
        except Exception as e:
            print(f"[ERROR] Exception during _initialize_tables: {e}")
            try:
                self.conn.rollback()
            except Exception as rb_e:
                print(f"[ERROR] Failed to rollback transaction: {rb_e}")
            raise
        if self.debug: print("[DEBUG] Exiting _initialize_tables normally.")
    
    def batch_add_structures(self, structures_data: List[Dict[str, Any]]) -> List[int]:
        """Adds a batch of new atomic structures to the database efficiently.

        This method takes a list of structure data dictionaries and inserts them
        in a single transaction using `psycopg2.extras.execute_values` for
        high performance.

        Args:
            structures_data: A list where each dictionary contains the data
                for a single structure. Each dictionary should contain the
                following keys:
                - 'atoms' (ase.Atoms): The ASE Atoms object for the structure. This key
                  is required.
                - 'source_type' (str, optional): A string indicating the origin or
                  provenance of the structure. This helps track how a structure
                  entered the database. Examples: 'vasp-relax', 'xyz_import',
                  'user_script', 'adversarial_attack'. This is distinct from
                  'config_type', which is typically stored in the metadata and
                  describes the physical configuration (e.g., 'liquid', 'defect').
                  Defaults to 'vasp'.
                - 'parent_id' (int, optional): The `structure_id` of a parent
                  structure if this one was derived from it (e.g., via relaxation
                  or an adversarial attack). Defaults to None.
                - 'metadata' (dict, optional): A dictionary of additional
                  data to store with the structure. This is merged with the
                  `atoms.info` dictionary (with `metadata` taking precedence)
                  and stored in a JSONB field. Defaults to None.

        Returns:
            List[int]: A list of the unique `structure_id`s for the newly added
                       structures, in the same order as the input list.

        Raises:
            ValueError: If `structures_data` is empty or if any dictionary is
                        missing the 'atoms' key.
        """
        if not structures_data:
            return []

        if self.dry_run:
            num_structures = len(structures_data)
            fake_ids = list(range(self._fake_id_counter, self._fake_id_counter + num_structures))
            self._fake_id_counter += num_structures
            print(f"[DRY RUN] Would add batch of {num_structures} structures to database with IDs: {fake_ids}")
            return fake_ids

        records_to_insert = []
        for data in structures_data:
            atoms = data.get('atoms')
            if not isinstance(atoms, Atoms):
                raise ValueError("Each dictionary in structures_data must contain an 'atoms' key with an ASE Atoms object.")
            
            source_type = data.get('source_type', 'vasp')
            parent_id = data.get('parent_id')
            metadata = data.get('metadata')
            
            if parent_id is not None:
                parent_id = int(parent_id)
            
            formula_str = atoms.get_chemical_formula()

            combined_metadata = atoms.info.copy()
            if metadata:
                combined_metadata.update(metadata)
            combined_metadata['formula_string'] = formula_str

            if parent_id is None and 'parent_id' in combined_metadata:
                try:
                    parent_id = int(combined_metadata['parent_id'])
                except (ValueError, TypeError):
                    parent_id = None
            
            symbols = atoms.get_chemical_symbols()
            if not symbols:
                raise ValueError("Cannot add structure with no atoms.")
            total_atoms = len(symbols)
            composition = {
                symbol: {"at_frac": symbols.count(symbol) / total_atoms, "num_atoms": symbols.count(symbol)}
                for symbol in set(symbols)
            }

            composition_hash = _compute_composition_hash(composition, decimal=4)

            safe_data = {
                'formula': formula_str,
                'composition': fix_numpy(composition),
                'positions': fix_numpy(atoms.positions),
                'cell': fix_numpy(atoms.cell.tolist()),
                'pbc': fix_numpy(atoms.pbc.tolist()),
                'metadata': fix_numpy(combined_metadata)
            }
            
            records_to_insert.append((
                safe_data['formula'],
                Json(safe_data['composition']),
                Json(safe_data['positions']),
                Json(safe_data['cell']),
                safe_data['pbc'],
                parent_id,
                source_type,
                Json(safe_data['metadata']) if safe_data['metadata'] else None,
                composition_hash
            ))
        
        with self.conn.cursor() as cur:
            insert_query = """
                INSERT INTO structures (
                    formula, composition, positions, cell, pbc,
                    parent_structure_id, source_type, metadata, composition_hash
                )
                VALUES %s
                RETURNING structure_id
            """
            
            results = execute_values(
                cur, insert_query, records_to_insert,
                template=None, page_size=100, fetch=True
            )
            structure_ids = [row[0] for row in results]

        self.conn.commit()
        return structure_ids

    def add_structure(self, atoms: Atoms, source_type: str = 'vasp',
                     parent_id: Optional[int] = None, metadata: Optional[Dict] = None) -> int:
        """Adds a single atomic structure to the database.

        This method is a convenience wrapper around `batch_add_structures` for
        adding a single item. For adding multiple structures, using
        `batch_add_structures` directly is more performant.

        Args:
            atoms (Atoms): The ASE Atoms object representing the structure.
            source_type (str, optional): A string indicating the origin or
                provenance of the structure. Defaults to 'vasp'.
            parent_id (Optional[int], optional): The `structure_id` of a parent
                structure. Defaults to None.
            metadata (Optional[Dict], optional): A dictionary of additional
                data to store with the structure. Defaults to None.

        Returns:
            int: The unique `structure_id` of the newly added structure.
        """
        structure_data = [{
            'atoms': atoms,
            'source_type': source_type,
            'parent_id': parent_id,
            'metadata': metadata
        }]
        
        new_ids = self.batch_add_structures(structure_data)
        return new_ids[0]
    
    def _get_composition_dict(self, atoms: Atoms) -> Dict[str, float]:
        """Convert Atoms object to composition dictionary."""
        symbols = atoms.get_chemical_symbols()
        unique_symbols = set(symbols)
        total_atoms = len(symbols)
        
        return {
            symbol: symbols.count(symbol) / total_atoms
            for symbol in unique_symbols
        }
    
    def get_structure(self, structure_id: int) -> Atoms:
        """
        Retrieve structure from database.
        
        Args:
            structure_id: Database ID of structure
            
        Returns:
            Atoms: ASE Atoms object with structure_id in info dict.
        """
        if self.dry_run:
            # Return a dummy structure in dry run mode
            from ase.build import bulk
            atoms = bulk('Ti', 'bcc', a=3.3)
            atoms.info['structure_id'] = structure_id # Add ID even for dummy
            atoms.info['structure_name'] = f'test_struct_{structure_id}'
            return atoms
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT positions, cell, pbc, formula, metadata
                FROM structures
                WHERE structure_id = %s
            """, (structure_id,))
            
            result = cur.fetchone() # Get the result row
            if result is None:
                 raise ValueError(f"Structure with ID {structure_id} not found.")

            positions, cell, pbc, formula, metadata = result # Unpack the result
            
            atoms = Atoms(
                symbols=formula,
                positions=np.array(positions),
                cell=np.array(cell),
                pbc=pbc
            )

            # Update with metadata first (if any)
            if metadata:
                atoms.info.update(metadata)

            # Explicitly add/overwrite structure_id in the info dict
            atoms.info['structure_id'] = structure_id

            return atoms

    def batch_add_calculations(self, calculations_data: List[Dict[str, Any]]) -> List[int]:
        """Adds a batch of calculation results to the database.

        Args:
            calculations_data (List[Dict[str, Any]]): A list where each dictionary
                contains the data for a single calculation. Required keys:
                - 'structure_id' (int): The ID of the structure for this calculation.
                - 'calc_data' (dict): A dictionary of calculation results,
                  which can contain 'energy', 'forces', 'stress', 'calculator',
                  'calculation_source_path', and 'metadata'.
        Returns:
            List[int]: A list of the new calculation_ids.
        Raises:
            ValueError: if input list is empty or data is malformed.
        """
        if not calculations_data:
            return []
        
        if self.dry_run:
            num_calcs = len(calculations_data)
            fake_ids = list(range(self._fake_id_counter, self._fake_id_counter + num_calcs))
            self._fake_id_counter += num_calcs
            print(f"[DRY RUN] Would add batch of {num_calcs} calculations to database with IDs: {fake_ids}")
            return fake_ids

        records_to_insert = []
        for item in calculations_data:
            structure_id = item.get('structure_id')
            calc_data = item.get('calc_data')
            if structure_id is None or calc_data is None:
                raise ValueError("Each item in calculations_data must have 'structure_id' and 'calc_data' keys.")

            safe_data = fix_numpy(calc_data)
            
            calculator = safe_data.get('calculator', safe_data.get('model_type', 'vasp'))
            calculation_source_path = safe_data.get('calculation_source_path', safe_data.get('model_path'))
            energy_val = safe_data.get('energy')
            forces_val = safe_data.get('forces')
            stress_val = safe_data.get('stress')

            db_energy = None
            if isinstance(energy_val, (int, float)):
                db_energy = float(energy_val)
            elif isinstance(energy_val, list) and len(energy_val) == 1 and isinstance(energy_val[0], (int, float)):
                db_energy = float(energy_val[0])
            elif energy_val is not None:
                try:
                    db_energy = float(energy_val)
                except (ValueError, TypeError):
                    raise ValueError(f"Unexpected energy format for structure {structure_id}: {energy_val}. Batch aborted.")
            
            column_keys = {'structure_id', 'calculator', 'calculation_source_path', 'energy', 'forces', 'stress'}
            old_column_keys = {'model_type', 'model_path', 'model_generation', 'ensemble_variance'}
            exclude_keys = column_keys.union(old_column_keys).union({'metadata'})

            metadata_dict = {k: v for k, v in safe_data.items() if k not in exclude_keys}
            if 'metadata' in safe_data and isinstance(safe_data['metadata'], dict):
                metadata_dict.update(safe_data['metadata'])

            records_to_insert.append((
                structure_id,
                calculator,
                calculation_source_path,
                db_energy,
                Json(forces_val) if forces_val is not None else None,
                Json(stress_val) if stress_val is not None else None,
                Json(metadata_dict) if metadata_dict else None
            ))

        with self.conn.cursor() as cur:
            insert_query = """
                INSERT INTO calculations (
                    structure_id, calculator, calculation_source_path,
                    energy, forces, stress, metadata
                )
                VALUES %s
                RETURNING calculation_id
            """
            results = execute_values(cur, insert_query, records_to_insert, fetch=True, page_size=100)
            calc_ids = [row[0] for row in results]

        self.conn.commit()
        return calc_ids

    def add_calculation(self, structure_id: int, calc_data: Dict) -> int:
        """Add calculation results to database (new schema)."""
        calculations_data = [{
            'structure_id': structure_id,
            'calc_data': calc_data
        }]
        new_ids = self.batch_add_calculations(calculations_data)
        return new_ids[0]

    def get_calculations(self, structure_id: int, calculator: Optional[str] = None) -> List[Dict]:
        """
        Retrieve calculations for a given structure, optionally filtering by calculator.

        Args:
            structure_id: ID of the structure.
            calculator: Name of the calculator (e.g., 'vasp', 'mace'). Wildcards not supported directly here.

        Returns:
            List of calculation dictionaries.
        """
        query = """
            SELECT calculation_id, calculator, calculation_source_path,
                   energy, forces, stress, metadata
            FROM calculations
            WHERE structure_id = %s
        """
        params: List[Union[int, str]] = [structure_id]

        if calculator:
            query += " AND calculator = %s"
            params.append(calculator)

        query += " ORDER BY created_at DESC" # Default sort: newest first

        results = []
        with self.conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            for row in rows:
                calc_dict = {
                    'calculation_id': row[0],
                    'structure_id': structure_id, # Add structure_id back for context
                    'calculator': row[1],
                    'calculation_source_path': row[2],
                    'energy': row[3], # Already REAL or None
                    'forces': np.array(row[4]) if row[4] else None, # Convert back to numpy
                    'stress': np.array(row[5]) if row[5] else None, # Convert back to numpy
                    'metadata': row[6] if row[6] else {} # Ensure metadata is a dict
                }
                # Unpack metadata contents into the main dict if desired, be careful of key collisions
                # if row[6]: calc_dict.update(row[6])
                results.append(calc_dict)

        return results

    def find_structures(self, elements: List[str] = None, 
                    structure_type: str = None,
                    min_lattice_parameter: float = None,
                    max_lattice_parameter: float = None,
                    composition_constraints: Dict[str, Tuple[float, float]] = None,
                    debug: bool = False) -> List[int]:
        """Search for structures matching criteria."""
        query = "SELECT structure_id FROM structures WHERE 1=1"
        params = []
        
        if elements:
            elements_array = '{' + ','.join(elements) + '}'
            query += " AND composition ?| %s"
            params.append(elements_array)
            if debug:
                print(f"[DEBUG] Elements filter applied: {elements_array}")

        # Add composition fraction constraints
        if composition_constraints:
            for element, (min_frac, max_frac) in composition_constraints.items():
                query += f" AND (composition->'{element}'->>'at_frac')::float >= %s"
                query += f" AND (composition->'{element}'->>'at_frac')::float <= %s"
                params.extend([min_frac, max_frac])
                if debug:
                    print(f"[DEBUG] Composition constraint applied for {element}: >= {min_frac}, <= {max_frac}")

        if structure_type:
            query += " AND metadata->>'structure_type' = %s"
            params.append(structure_type)
            if debug:
                print(f"[DEBUG] Structure type filter applied: {structure_type}")

        if min_lattice_parameter or max_lattice_parameter:
            query += """ 
                AND (
                    SELECT POWER(ABS(cell->0->0 * (cell->1->1 * cell->2->2 - 
                    cell->1->2 * cell->2->1)), 1.0/3.0)
                )
            """
            if min_lattice_parameter:
                query += " >= %s"
                params.append(min_lattice_parameter)
                if debug:
                    print(f"[DEBUG] Minimum lattice parameter filter applied: {min_lattice_parameter}")
            if max_lattice_parameter:
                query += " <= %s"
                params.append(max_lattice_parameter)
                if debug:
                    print(f"[DEBUG] Maximum lattice parameter filter applied: {max_lattice_parameter}")

        if debug:
            print(f"[DEBUG] Executing query: {query}")  # Debug
            print(f"[DEBUG] With params: {params}")  # Debug
                
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            results = [row[0] for row in cur.fetchall()]
            if debug:
                print(f"[DEBUG] Query returned {len(results)} matching structures")  # Debug
            return results
    
    def find_structures_without_calculation(self, calculator: Optional[str] = None) -> List[int]:
        """
        Return a list of structure_ids that do NOT have a calculation
        with the given calculator.
        If calculator is None, finds structures with no calculations at all.
        """
        if calculator is None:
            query = """
                SELECT s.structure_id
                FROM structures s
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM calculations c
                    WHERE c.structure_id = s.structure_id
                )
            """
            params = []
        else:
            # replace * with % for SQL LIKE pattern matching 
            calculator_pattern = calculator.replace('*', '%')
            query = """
                SELECT s.structure_id
                FROM structures s
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM calculations c
                    WHERE c.structure_id = s.structure_id
                    AND c.calculator LIKE %s
                )
            """
            params = [calculator_pattern]

        with self.conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

        return [r[0] for r in rows]

    def add_structures_from_xyz(
        self,
        xyz_file: str,
        skip_duplicates: bool = True,
        default_calculator: str = "vasp", # Changed default
    ) -> List[int]:
        """
        Reads frames from a .xyz file, adds structures, and optionally adds calculations
        if energy/forces/stress are present in the XYZ frame's info/arrays.

        Args:
            xyz_file: Path to the .xyz file containing frames.
            skip_duplicates: Whether to skip duplicates by position and composition hash.
            default_calculator: The calculator name to use if adding associated calculation data.

        Returns:
            List of structure_ids added to the database.
        """

        if not os.path.isfile(xyz_file):
            print(f"[WARN] XYZ file not found: {xyz_file}")
            return []

        added_ids = []
        frames = read(xyz_file, index=":")
        print(f"[INFO] Found {len(frames)} frames in {xyz_file}")

        for i, atoms in enumerate(tqdm(frames, desc="Processing XYZ frames")):

            # If skipping duplicates, do a quick check
            if skip_duplicates: # TODO: Need to implement hashing for checking duplicate structures based on the atom positions and composition
                if self.check_duplicate_structure(atoms): # Use default tolerances
                    # print("[INFO] Skipping duplicate structure.") # Too verbose
                    continue

            # Extract potential calculation data BEFORE adding structure
            # to avoid modifying atoms.info permanently if calc add fails
            calc_data_to_add = None
            has_calc_data = False
            temp_calc_data = {'calculator': default_calculator}

            if 'energy' in atoms.info:
                 temp_calc_data['energy'] = atoms.info['energy']
                 has_calc_data = True
            if 'forces' in atoms.arrays:
                 temp_calc_data['forces'] = atoms.arrays['forces']
                 has_calc_data = True
            if 'stress' in atoms.info:
                 temp_calc_data['stress'] = atoms.info['stress']
                 has_calc_data = True

            # Add other relevant info from atoms.info to calc metadata?
            calc_meta = {}
            # Example: Copy specific keys you might find in XYZ info
            # for key in ['source_comment', 'xyz_line_number']:
            #      if key in atoms.info: calc_meta[key] = atoms.info[key]
            if calc_meta:
                 temp_calc_data['metadata'] = calc_meta


            if has_calc_data:
                 calc_data_to_add = temp_calc_data


            # --- Add the structure ---
            try:
                 # Prepare structure metadata (extract from atoms.info, excluding calc data)
                 struct_meta = atoms.info.copy()
                 struct_meta.pop('energy', None)
                 struct_meta.pop('stress', None)
                 struct_meta['source_xyz_file'] = xyz_file
                 struct_meta['source_xyz_index'] = i

                 # Add structure (atoms.info might contain extra stuff, add_structure handles it)
                 # We pass struct_meta explicitly to ensure calc data isn't in structure metadata
                 struct_id = self.add_structure(atoms, metadata=struct_meta, source_type='xyz_import')
                 added_ids.append(struct_id)
                 # print(f"[INFO] Added structure, ID={struct_id}") # Too verbose

                 # --- Add calculation if data was found ---
                 if calc_data_to_add:
                      try:
                           calc_id = self.add_calculation(struct_id, calc_data_to_add)
                           # print(f"[INFO] Added calculation, ID={calc_id}") # Too verbose
                      except Exception as calc_e:
                           print(f"\n[WARN] Added structure {struct_id} but failed to add calculation from frame {i}: {calc_e}")

            except Exception as struct_e:
                 print(f"\n[ERROR] Failed to add structure from frame {i}: {struct_e}")


        print(f"[INFO] Finished adding frames from {xyz_file}. Added {len(added_ids)} new structures.")
        return added_ids

    def check_duplicate_structure(self, atoms: Atoms,
                                  decimal: int = 4, position_tol: float = 1e-5) -> bool:
        """
        Check if 'atoms' is a duplicate of any structure in the DB.
        This is a convenience wrapper around the more performant batch_check_duplicates.
        """
        results = self.batch_check_duplicates([atoms], decimal=decimal, position_tol=position_tol)
        return results[0]

    def batch_check_duplicates(self, atoms_list: List[Atoms],
                               decimal: int = 4, position_tol: float = 1e-5) -> List[bool]:
        """
        Checks a batch of Atoms objects for duplicates in the database.

        This method first finds all structures in the database that share a
        composition hash with any of the input structures, retrieving them in a
        single query. It then performs position comparisons in memory.

        Args:
            atoms_list: A list of ASE Atoms objects to check.
            decimal: Number of decimal places to round composition fractions for hashing.
            position_tol: Tolerance for comparing atomic positions.

        Returns:
            A list of booleans of the same length as atoms_list, where True
            indicates the corresponding Atoms object is a duplicate of an
            existing structure in the database.
        """
        if not atoms_list:
            return []

        # 1. Compute hashes for all input structures
        input_hashes = {}  # {hash: [indices_in_atoms_list]}
        for i, atoms in enumerate(atoms_list):
            symbols = atoms.get_chemical_symbols()
            if not symbols: continue
            total_atoms = len(symbols)
            comp_dict = {
                symbol: {"at_frac": symbols.count(symbol) / total_atoms}
                for symbol in set(symbols)
            }
            composition_hash = _compute_composition_hash(comp_dict, decimal=decimal)
            if composition_hash not in input_hashes:
                input_hashes[composition_hash] = []
            input_hashes[composition_hash].append(i)

        # 2. Fetch all potential duplicates from DB in one query
        if not input_hashes:
            return [False] * len(atoms_list)
            
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT composition_hash, structure_id, positions
                FROM structures
                WHERE composition_hash = ANY(%s)
            """, (list(input_hashes.keys()),))
            potential_duplicates = cur.fetchall()

        # 3. Group DB results by hash for quick lookup
        db_dupes_by_hash = {}
        for comp_hash, struct_id, db_positions_json in potential_duplicates:
            if comp_hash not in db_dupes_by_hash:
                db_dupes_by_hash[comp_hash] = []
            db_dupes_by_hash[comp_hash].append(np.array(db_positions_json))

        # 4. Perform comparisons in memory
        is_duplicate_list = [False] * len(atoms_list)
        pos_decimals = int(abs(np.log10(position_tol)))

        # Create a list of items to iterate over with a progress bar
        # This list contains tuples of (hash, index_in_original_list)
        items_to_check = []
        for comp_hash, indices in input_hashes.items():
            if comp_hash in db_dupes_by_hash:
                for index in indices:
                    items_to_check.append((comp_hash, index))
        
        # This is where the majority of the time will be spent if there are many potential matches
        for comp_hash, index in tqdm(items_to_check, desc="Comparing positions", leave=False):
            db_positions_list = db_dupes_by_hash[comp_hash]
            input_atoms = atoms_list[index]
            new_positions_rounded = np.round(input_atoms.positions, decimals=pos_decimals)

            for db_positions in db_positions_list:
                db_positions_rounded = np.round(db_positions, decimals=pos_decimals)
                if new_positions_rounded.shape == db_positions_rounded.shape and \
                   np.allclose(new_positions_rounded, db_positions_rounded, atol=position_tol):
                    is_duplicate_list[index] = True
                    break  # Found a match for this input_atoms, move to next one

        return is_duplicate_list

    def remove_duplicate_structures(
        self, decimal: int = 5, position_tol: float = 1e-5
    ) -> Dict[int, List[int]]:
        """
        Remove duplicate structures by hashing positions (rounded to `decimal` places)
        and symbols. Keeps only the earliest structure_id in each hash group, removes
        the rest. Optionally, you can do a final position check among those with the
        same composition_hash if you like.

        Args:
            decimal: Number of decimal places to round positions for the hash.
            position_tol: Tolerance for position comparison.

        Returns:
            A dict of {kept_structure_id: [duplicate_ids]}.
        """
        # Check if running in dry_run mode first
        if self.dry_run:
            print("[DRY RUN] Would identify and simulate removal of duplicate structures.")
            # In dry run, we might still want to identify potential duplicates
            # but not perform the DELETE operations. For now, just returning empty.
            # A more sophisticated dry run could mimic the identification logic.
            return {}
            
        # Step 1: Gather all structure_ids
        with self.conn.cursor() as cur:
            cur.execute("SELECT structure_id FROM structures ORDER BY structure_id ASC")
            all_structure_ids = [row[0] for row in cur.fetchall()]
        
        # Step 2: Build a hash for each structure
        hash_map = {}  # {hash_value: [structure_ids]}
        for struct_id in all_structure_ids:
            try:
                atoms = self.get_structure(struct_id)
            except Exception as e:
                print(f"[WARN] Could not retrieve structure {struct_id}. Error: {e}")
                continue

            # Generate hash (position-based)
            hval = self._get_structure_hash(atoms, decimal=decimal)
            
            if hval not in hash_map:
                hash_map[hval] = [struct_id]
            else:
                hash_map[hval].append(struct_id)
        
        # Step 3: Identify duplicates in each hash group
        duplicates_map = {}  # {kept_id: [removed_ids]}
        for hval, ids in hash_map.items():
            if len(ids) > 1:
                # Keep the first ID in ascending order, remove others
                kept_id = ids[0]
                removed_ids = ids[1:]
                duplicates_map[kept_id] = removed_ids

        # Step 4: Remove duplicates from DB
        if duplicates_map:
            print("[INFO] Removing duplicate structures by hash...")
            with self.conn.cursor() as cur:
                for kept_id, duplicate_ids in duplicates_map.items():
                    # You might do a final position check here if you want,
                    # but if you're confident in the hash approach, skip it.

                    # 4A) Remove calculations for duplicates
                    cur.execute("""
                        DELETE FROM calculations
                        WHERE structure_id = ANY(%s)
                        RETURNING calculation_id
                    """, (duplicate_ids,))
                    deleted_calcs = cur.fetchall()
                    
                    # 4B) Remove the duplicate structures
                    cur.execute("""
                        DELETE FROM structures
                        WHERE structure_id = ANY(%s)
                        RETURNING structure_id
                    """, (duplicate_ids,))
                    deleted_structs = cur.fetchall()
                    
                    print(f"[INFO] Kept {kept_id}, removed duplicates {duplicate_ids}")
                    print(f"       Removed {len(deleted_calcs)} calculations and {len(deleted_structs)} structures")
            
            self.conn.commit()
        else:
            print("[INFO] No duplicates found by hash.")
        
        return duplicates_map

    def get_duplicate_summary(self, duplicates_map: Dict[int, List[int]]) -> None:
        """
        Print a summary of duplicates removed by remove_duplicate_structures.

        Args:
            duplicates_map: dict of {kept_structure_id: [removed_structure_ids]}.
        """
        if not duplicates_map:
            print("[INFO] No duplicates to summarize!")
            return
        
        print("\n[INFO] Duplicate Structure Summary (Hash-Based):")
        print("-----------------------------------------")
        total_removed = sum(len(dups) for dups in duplicates_map.values())
        print(f"Total duplicate structures removed: {total_removed}")
        print(f"Number of unique structure groups: {len(duplicates_map)}")
        
        for kept_id, removed_ids in duplicates_map.items():
            # Optionally retrieve info about the kept structure
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT formula, metadata
                    FROM structures
                    WHERE structure_id = %s
                """, (kept_id,))
                row = cur.fetchone()
                if row:
                    print(f"  Kept ID {kept_id}: Formula={row[0]}, Metadata={row[1]}")
                else:
                    print(f"  Kept ID {kept_id}: (not found in DB)")
            print(f"    Removed duplicates: {removed_ids}")

    def _get_structure_hash(self, atoms: Atoms, decimal: int = 5) -> str:
        """
        Compute a hash string for a structure by rounding each (x, y, z) to `decimal` places,
        then concatenating the element and positions in a deterministic order.
        This does NOT account for permutations, only the exact ordering of atoms as in `atoms`.
        """
        coords = []
        for atom in atoms:
            x, y, z = np.round(atom.position, decimals=decimal)
            coords.append(f"{atom.symbol}:{x:.5f}:{y:.5f}:{z:.5f}")
        full_string = "|".join(coords)
        return hashlib.md5(full_string.encode("utf-8")).hexdigest()

    def get_structure_metadata(self, structure_id: int) -> Dict:
        """Get structure metadata from the database.
        
        Args:
            structure_id: Structure ID
            
        Returns:
            dict: Structure metadata
        """
        if self.dry_run:
            # Return dummy metadata in dry run mode
            return {
                'structure_id': structure_id,
                'config_type': 'test',
                'creation_time': '2024-03-19T00:00:00'
            }
            
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT metadata
                FROM structures
                WHERE structure_id = %s
            """, (structure_id,))
            metadata = cur.fetchone()[0]
            return metadata

    def update_structure_metadata(self, structure_id: int, metadata: Dict) -> None:
        """Update structure metadata in the database. WARNING: This will overwrite the existing metadata JSONB
        field in the database with the provided dictionary.
        
        Args:
            structure_id: The ID of the structure whose metadata to update.
            metadata: The new metadata dictionary to store.
        """
        if self.dry_run:
            print(f"[DRY RUN] Would update metadata for structure {structure_id}")
            return
            
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE structures
                SET metadata = %s
                WHERE structure_id = %s
            """, (Json(metadata), structure_id))
        self.conn.commit()

    def get_atoms_with_calculation(self, structure_id: int, calculator: Optional[str] = 'vasp') -> Optional[Atoms]:
        """
        DEPRECATED: Use get_batch_atoms_with_calculation instead for better performance,
                    especially when fetching multiple structures.

        Get an Atoms object with the latest calculation results attached.

        Args:
            structure_id: ID of the structure to retrieve.
            calculator: Name of calculator results to attach (e.g., 'vasp'). Defaults to 'vasp'.

        Returns:
            Atoms object with calculation results in .info/.arrays, or None if structure/calc not found.
        """
        print("[WARN] get_atoms_with_calculation is deprecated. Use get_batch_atoms_with_calculation.")
        # Keep old implementation for now, or call the new batch method with single ID
        batch_result = self.get_batch_atoms_with_calculation([structure_id], calculator=calculator)
        return batch_result[0] if batch_result else None

    def get_structures_batch(self, structure_ids: List[int]) -> Dict[int, Atoms]:
        """Retrieves multiple structures efficiently in a single query.

        Args:
            structure_ids: A list of structure IDs to retrieve.

        Returns:
            A dictionary mapping each found structure_id to its ASE Atoms object.
            IDs not found in the database will be omitted from the result.
        """
        if not structure_ids:
            return {}
        if self.dry_run:
            # Return dummy structures in dry run mode
            from ase.build import bulk
            atoms_dict = {}
            for sid in structure_ids:
                atoms = bulk('Ti', 'bcc', a=3.3)
                atoms.info['structure_id'] = sid
                atoms.info['structure_name'] = f'test_struct_{sid}'
                atoms_dict[sid] = atoms
            print(f"[DRY RUN] Would retrieve batch of {len(structure_ids)} structures.")
            return atoms_dict

        atoms_map = {}
        query = """
            SELECT structure_id, positions, cell, pbc, formula, metadata
            FROM structures
            WHERE structure_id = ANY(%s)
        """
        try:
            with self.conn.cursor() as cur:
                # Pass structure_ids as a list/tuple directly for ANY()
                cur.execute(query, (list(structure_ids),))
                results = cur.fetchall()

                for row in results:
                    struct_id, positions, cell, pbc, formula, metadata = row
                    atoms = Atoms(
                        symbols=formula,
                        positions=np.array(positions),
                        cell=np.array(cell),
                        pbc=pbc
                    )
                    # Update with metadata first (if any)
                    if metadata:
                        atoms.info.update(metadata)
                    # Explicitly add/overwrite structure_id
                    atoms.info['structure_id'] = struct_id
                    atoms_map[struct_id] = atoms

        except Exception as e:
            print(f"[ERROR] Failed to retrieve structures batch: {e}")
            # Optionally re-raise or return partial results depending on desired behavior
            if self.conn: # Check if connection exists before rolling back
                 self.conn.rollback()
            raise # Re-raise by default

        return atoms_map

    def get_calculations_batch(self, structure_ids: List[int], calculator: Optional[str] = None) -> Dict[int, Dict]:
        """Retrieves the latest calculation for multiple structures efficiently.

        Uses a window function (`ROW_NUMBER`) to select only the most recent
        calculation per structure_id, optionally filtering by calculator type.

        Args:
            structure_ids: List of structure IDs to fetch calculations for.
            calculator: Optional calculator name to filter by.

        Returns:
            A dictionary mapping each structure_id (that has a matching calculation)
            to its latest calculation dictionary.
        """
        if not structure_ids:
            return {}

        params: List[Union[List[int], str]] = [list(structure_ids)] # Start with list of IDs for ANY()

        # Base query using ROW_NUMBER() to get the latest calculation per structure_id
        query = """
        WITH RankedCalculations AS (
            SELECT
                calculation_id, structure_id, calculator, calculation_source_path,
                energy, forces, stress, metadata, created_at,
                ROW_NUMBER() OVER(PARTITION BY structure_id ORDER BY created_at DESC) as rn
            FROM calculations
            WHERE structure_id = ANY(%s)
        """

        # Add calculator filter if provided
        if calculator:
            query += " AND calculator = %s"
            params.append(calculator)

        query += """
        )
        SELECT
            calculation_id, structure_id, calculator, calculation_source_path,
            energy, forces, stress, metadata
        FROM RankedCalculations
        WHERE rn = 1;
        """

        results_map = {}
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                for row in rows:
                    struct_id = row[1]
                    calc_dict = {
                        "calculation_id": row[0],
                        "structure_id": struct_id,
                        "calculator": row[2],
                        "calculation_source_path": row[3],
                        "energy": row[4],
                        "forces": np.array(row[5]) if row[5] else None,
                        "stress": np.array(row[6]) if row[6] else None,
                        "metadata": row[7] if row[7] else {}
                    }
                    results_map[struct_id] = calc_dict
        except Exception as e:
             print(f"[ERROR] Failed to retrieve calculations batch: {e}")
             raise # Re-raise by default

        return results_map
    
    def get_batch_atoms_with_calculation(
        self, structure_ids: List[int], calculator: Optional[str] = "vasp"
    ) -> List[Atoms]:
        """Efficiently retrieves multiple Atoms objects with their latest calculation attached.

        Combines batch fetching of structures and calculations to minimize
        database queries.

        Args:
            structure_ids: List of structure IDs to retrieve.
            calculator: Name of the calculator results to attach (e.g., 'vasp').
                        Defaults to 'vasp'. Pass None to get latest calculation
                        regardless of calculator type.

        Returns:
            A list of ASE Atoms objects, each populated with data from its latest
            matching calculation (if found) in `atoms.info` and `atoms.arrays`.
            Structures for which no matching calculation is found will still be
            included but without calculation data attached. The order corresponds
            to the input `structure_ids`, but only includes found structures.
        """
        if not structure_ids:
            return []

        # 1. Batch get structures
        atoms_map = self.get_structures_batch(structure_ids)
        if not atoms_map:
            return [] # No structures found for the given IDs

        # 2. Batch get latest calculations for the *found* structure IDs
        found_ids = list(atoms_map.keys())
        calculations_map = self.get_calculations_batch(found_ids, calculator=calculator)

        # 3. Combine data in Python
        final_atoms_list = []
        for sid in structure_ids: # Iterate original list to preserve order somewhat
            if sid in atoms_map:
                atoms = atoms_map[sid]
                calc_data = calculations_map.get(sid) # Get the latest calc if available

                # Clear any previous calc info first
                atoms.info.pop("energy", None)
                atoms.info.pop("stress", None)
                atoms.arrays.pop("forces", None)
                atoms.info.pop("calculation_info", None)

                if calc_data:
                    # Attach energy/stress to info
                    if calc_data.get("energy") is not None:
                         # Should already be float or None from get_calculations_batch
                        atoms.info["energy"] = calc_data["energy"]
                    if calc_data.get("stress") is not None:
                         # Should already be numpy array or None
                        atoms.info["stress"] = calc_data["stress"]

                    # Attach forces to arrays (with shape check)
                    if calc_data.get("forces") is not None:
                        forces = calc_data["forces"] # Should be numpy array or None
                        if forces is not None and forces.shape == (len(atoms), 3):
                            atoms.arrays["forces"] = forces
                        elif forces is not None:
                             print(f"[WARN] Mismatched forces shape for structure {sid}, "
                                   f"calc {calc_data.get('calculation_id')}. "
                                   f"Expected {(len(atoms), 3)}, got {forces.shape}. "
                                   "Forces not attached.")

                    # Attach calculation metadata
                    atoms.info["calculation_info"] = {
                        "calculation_id": calc_data.get("calculation_id"),
                        "calculator": calc_data.get("calculator"),
                        "calculation_source_path": calc_data.get(
                            "calculation_source_path"
                        ),
                        **(calc_data.get("metadata", {})),
                    }
                final_atoms_list.append(atoms)

        return final_atoms_list

    def find_structures_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        operator: str = "exact",
        debug: bool = False,
    ) -> List[int]:
        """Searches for structures based on key-value pairs in their metadata.

        Supports matching exact values, checking if a string contains a substring,
        or comparing numeric values ('>', '<', '>=', '<='). Handles nested
        metadata keys using dot notation (e.g., 'potential.name').

        Args:
            metadata_filters: Dictionary of {metadata_key: value} criteria.
                              Keys can use dot notation for nested access.
            operator: Comparison operator: 'exact' (default), 'contains' (for strings),
                      '>', '<', '>=', '<=' (for numbers).
            debug: If True, print the generated SQL query and parameters.

        Returns:
            A list of structure IDs matching the metadata criteria.
        """
        if self.dry_run:
            print(f"[DRY RUN] Would search structures by metadata: {metadata_filters} ({operator})")
            return [4, 5, 6]  # Dummy IDs

        if self.conn is None:
            raise ConnectionError("Database is not connected.")

        query = "SELECT structure_id FROM structures WHERE 1=1"
        params = []

        for key, value in metadata_filters.items():
            # Build JSON path expression (handling potential nesting)
            parts = key.split(".")
            # Use #>> for text extraction at the final path element
            json_path_op = "->".join([f"'{part}'" for part in parts[:-1]])
            if json_path_op:
                 db_key_base = f"metadata->{json_path_op}"
            else:
                 db_key_base = "metadata" # Top level key
            db_key = f"{db_key_base}->>'{parts[-1]}'" # Final text extraction


            # Add condition based on operator and value type
            if operator == "exact":
                if value is None:
                    # Check if the final key exists and is explicitly NULL, or if path doesn't exist
                    # This logic might need refinement based on exact desired NULL handling for nested keys
                     query += f" AND ({db_key} IS NULL)" # Simpler check: is the extracted text value NULL?
                else:
                    query += f" AND {db_key} = %s"
                    params.append(str(value)) # Compare as text
            elif operator == "contains" and isinstance(value, str):
                query += f" AND {db_key} LIKE %s"
                params.append(f"%{value}%")
            elif operator in (">", "<", ">=", "<=") and isinstance(value, int | float):
                # Attempt direct cast to numeric for comparison
                query += f" AND ({db_key})::numeric {operator} %s"
                params.append(value)
            elif operator in (">", "<", ">=", "<=") and value is None:
                 raise ValueError(f"Comparison operators ('{operator}') cannot be used with None value.")
            else:
                raise ValueError(
                    f"Unsupported operator '{operator}' for value type {type(value)} "
                    f"or unsupported value type for operator."
                )

        if debug:
             with self.conn.cursor() as temp_cur:
                  print(f"[DEBUG] Executing query: {temp_cur.mogrify(query, tuple(params)).decode()}")

        results = []
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, tuple(params))
                results = [row[0] for row in cur.fetchall()]
        except psycopg2.Error as e:
            print(f"[ERROR] Failed to find structures by metadata: {e}")
            if self.conn: # Check if connection exists before rolling back
                 self.conn.rollback()
            raise

        if debug:
            print(f"[DEBUG] Query returned {len(results)} matching structures")

        return results

    def remove_structure(self, structure_id: int, dry_run_override: Optional[bool] = None) -> None:
        """Removes a single structure and its calculations by wrapping the batch method."""
        self.remove_structures_batch([structure_id], dry_run_override=dry_run_override)

    def remove_structures_batch(self, structure_ids: List[int], dry_run_override: Optional[bool] = None) -> None:
        """
        Remove a batch of structures and their associated calculations from the database.

        This method first explicitly deletes calculations linked to the given structure
        IDs, then deletes the structures themselves. This is more robust than relying
        on the database's `ON DELETE CASCADE` feature.

        Args:
            structure_ids: A list of structure IDs to remove.
            dry_run_override: If provided, overrides the instance's dry_run setting for this operation.

        Raises:
            ValueError: If the list of structure_ids is empty.
            Exception: Propagates database errors during deletion.
        """
        is_dry_run = self.dry_run if dry_run_override is None else dry_run_override

        if not structure_ids:
            print("[WARN] remove_structures_batch called with an empty list. No action taken.")
            return

        if is_dry_run:
            print(f"[DRY RUN] Would attempt to remove {len(structure_ids)} structures and their calculations:")
            print(f"[DRY RUN] Structure IDs: {structure_ids}")
            return

        if not self.conn:
            print("[ERROR] Database connection is not available. Cannot remove structures.")
            return

        try:
            with self.conn.cursor() as cur:
                # Step 1: Explicitly delete associated calculations
                cur.execute(
                    "DELETE FROM calculations WHERE structure_id = ANY(%s) RETURNING calculation_id;",
                    (structure_ids,)
                )
                deleted_calc_ids = [row[0] for row in cur.fetchall()]
                if deleted_calc_ids:
                    print(f"[INFO] Removed {len(deleted_calc_ids)} associated calculations.")

                # Step 2: Delete the structures
                cur.execute(
                    "DELETE FROM structures WHERE structure_id = ANY(%s) RETURNING structure_id;",
                    (structure_ids,)
                )
                deleted_ids = [row[0] for row in cur.fetchall()]
                
                if len(deleted_ids) < len(structure_ids):
                    missing_ids = set(structure_ids) - set(deleted_ids)
                    print(f"[WARN] Some requested structure IDs were not found or not deleted: {list(missing_ids)}")

                if deleted_ids:
                    print(f"[INFO] Successfully removed {len(deleted_ids)} structures: {deleted_ids}")
                else:
                    print("[INFO] No structures were removed (possibly none of the provided IDs existed).")
                
                self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR] Error during batch removal of structures {structure_ids}: {e}")
            raise

    def remove_calculation(self, calculation_id: int, dry_run_override: Optional[bool] = None) -> None:
        """Removes a single calculation from the database by wrapping the batch method."""
        self.remove_calculations_batch([calculation_id], dry_run_override)

    def remove_calculations_batch(self, calculation_ids: List[int], dry_run_override: Optional[bool] = None) -> None:
        """Removes a batch of calculations from the database.

        Args:
            calculation_ids: A list of calculation IDs to remove.
            dry_run_override: If provided, overrides the instance's dry_run setting for this operation.
        """
        is_dry_run = self.dry_run if dry_run_override is None else dry_run_override

        if not calculation_ids:
            print("[WARN] remove_calculations_batch called with an empty list. No action taken.")
            return

        if is_dry_run:
            print(f"[DRY RUN] Would attempt to remove {len(calculation_ids)} calculations: {calculation_ids}")
            return

        if not self.conn:
            print("[ERROR] Database connection is not available. Cannot remove calculations.")
            return
            
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM calculations WHERE calculation_id = ANY(%s) RETURNING calculation_id;",
                    (calculation_ids,)
                )
                deleted_ids = [row[0] for row in cur.fetchall()]
                if deleted_ids:
                    print(f"[INFO] Successfully removed {len(deleted_ids)} calculations: {deleted_ids}")
                else:
                    print("[INFO] No calculations were removed (possibly none of the provided IDs existed).")
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR] Error during batch removal of calculations {calculation_ids}: {e}")
            raise

    def add_mlip_model(
        self,
        model_generation: int,
        model_file_path: str,
        mlip_type: Optional[str] = None,
        train_structure_ids: Optional[List[int]] = None,
        validation_structure_ids: Optional[List[int]] = None,
        test_structure_ids: Optional[List[int]] = None,
        model_parameters: Optional[Dict] = None,
        wandb_link: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Adds a record for a trained MLIP model."""
        if self.dry_run:
             print(f"[DRY RUN] Would add MLIP model record for gen {model_generation}")
             fake_id = self._fake_id_counter
             self._fake_id_counter += 1
             return fake_id

        if self.conn is None:
            raise ConnectionError("Database is not connected.")

        if model_generation is None: # Check explicitly as it's NOT NULL
             raise ValueError("model_generation is required.")
        if not model_file_path:
             raise ValueError("model_file_path is required.")


        # Optional: Add validation to check if structure IDs actually exist? Might be slow.

        with self.conn.cursor() as cur:
             cur.execute("""
                INSERT INTO mlip_models (
                    model_generation, mlip_type,
                    train_structure_ids, validation_structure_ids, test_structure_ids,
                    model_parameters, wandb_link, model_file_path, notes
                 )
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                 RETURNING mlip_model_id
             """, (
                 model_generation,
                 mlip_type,
                 train_structure_ids if train_structure_ids else None, # Pass lists directly for INTEGER[]
                 validation_structure_ids if validation_structure_ids else None,
                 test_structure_ids if test_structure_ids else None,
                 Json(fix_numpy(model_parameters)) if model_parameters else None,
                 wandb_link,
                 model_file_path,
                 notes,
             ))
             mlip_model_id = cur.fetchone()[0]
        self.conn.commit()
        return mlip_model_id

    def close_connection(self):
         """Closes the database connection if it is open."""
         if self.conn is not None:
              try:
                   self.conn.close()
                   self.conn = None
                   if self.debug: print("[DEBUG] Database connection closed.")
              except psycopg2.Error as e:
                   print(f"[ERROR] Failed to close database connection: {e}")

    def __del__(self):
         """Ensures the database connection is closed when the object is deleted."""
         self.close_connection()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        # Can add logic here if needed, e.g., ensure connection is active
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object, ensuring connection closure."""
        self.close_connection()
        # Return False to propagate exceptions, True to suppress them
        return False
