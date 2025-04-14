# Core database interface (core/database.py)
from typing import Dict, List, Optional, Union, Tuple
import psycopg2
from psycopg2.extras import Json
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
    
    def add_structure(self, atoms: Atoms, source_type: str = 'vasp',
                     parent_id: Optional[int] = None, metadata: Optional[Dict] = None) -> int:
        """Add structure to database with thorough numpy fixing."""
        if self.dry_run:
            # Simulate adding structure and return fake ID
            fake_id = self._fake_id_counter
            self._fake_id_counter += 1
            print(f"[DRY RUN] Would add structure to database with ID: {fake_id}")
            return fake_id
            
        if parent_id is not None:
            parent_id = int(parent_id)

        # Generate formula string if not already in metadata
        formula_str = atoms.get_chemical_formula() 

        # Combine provided metadata with atoms.info and generated formula
        combined_metadata = atoms.info.copy() # Start with atoms.info
        if metadata:
            combined_metadata.update(metadata) # Add/overwrite with provided metadata
        combined_metadata['formula_string'] = formula_str # Add formula string

        # Ensure parent_id from atoms.info is captured if not explicitly passed
        if parent_id is None and 'parent_id' in combined_metadata:
             try:
                  parent_id = int(combined_metadata['parent_id'])
             except (ValueError, TypeError):
                  print(f"[WARN] Could not parse parent_id {combined_metadata.get('parent_id')} from metadata.")
                  parent_id = None # Ensure it's None if parsing fails


        # Calculate composition and number of atoms
        symbols = atoms.get_chemical_symbols()
        if not symbols:
             raise ValueError("Cannot add structure with no atoms.")
        total_atoms = len(symbols)
        composition = {}

        for symbol in set(symbols):
            count = symbols.count(symbol)
            composition[symbol] = {
                "at_frac": count / total_atoms,
                "num_atoms": count
            }

        # Compute composition hash
        composition_hash = _compute_composition_hash(composition, decimal=4)

        # Fix all data before JSON serialization
        safe_data = {
            'formula': atoms.get_chemical_formula(), # Keep full formula for table column
            'composition': fix_numpy(composition),
            'positions': fix_numpy(atoms.positions),
            'cell': fix_numpy(atoms.cell.tolist()),
            'pbc': fix_numpy(atoms.pbc.tolist()),
            'metadata': fix_numpy(combined_metadata) # Use combined, fixed metadata
        }

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO structures (
                    formula, composition, positions, cell, pbc,
                    parent_structure_id, source_type, metadata, composition_hash
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING structure_id
                """,
                (
                    safe_data['formula'],
                    Json(safe_data['composition']),
                    Json(safe_data['positions']),
                    Json(safe_data['cell']),
                    safe_data['pbc'],
                    parent_id, # Use validated parent_id
                    source_type,
                    Json(safe_data['metadata']) if safe_data['metadata'] else None,
                    composition_hash
                )
            )
            structure_id = cur.fetchone()[0]
        self.conn.commit()
        return structure_id
    
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
            Atoms: ASE Atoms object
        """
        if self.dry_run:
            # Return a dummy structure in dry run mode
            from ase.build import bulk
            atoms = bulk('Ti', 'bcc', a=3.3)
            atoms.info['structure_id'] = structure_id
            atoms.info['structure_name'] = f'test_struct_{structure_id}'
            return atoms
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT positions, cell, pbc, formula, metadata
                FROM structures
                WHERE structure_id = %s
            """, (structure_id,))
            
            positions, cell, pbc, formula, metadata = cur.fetchone()
            
            atoms = Atoms(
                symbols=formula,
                positions=np.array(positions),
                cell=np.array(cell),
                pbc=pbc
            )

            if metadata:
                atoms.info.update(metadata)
            
            return atoms

    def add_calculation(self, structure_id: int, calc_data: Dict) -> int:
        """Add calculation results to database (new schema)."""
        if self.dry_run:
             print(f"[DRY RUN] Would add calculation for structure {structure_id}")
             fake_id = self._fake_id_counter
             self._fake_id_counter += 1
             return fake_id

        # Ensure structure_id exists
        with self.conn.cursor() as cur:
             cur.execute("SELECT 1 FROM structures WHERE structure_id = %s", (structure_id,))
             if cur.fetchone() is None:
                  raise ValueError(f"Cannot add calculation: Structure with ID {structure_id} does not exist.")

        safe_data = fix_numpy(calc_data)

        # Extract fields for the NEW calculations schema
        # Use 'calculator' key preferentially, fall back to 'model_type' for compatibility
        calculator = safe_data.get('calculator', safe_data.get('model_type', 'vasp'))
        # Use 'calculation_source_path' preferentially, fall back to 'model_path'
        calculation_source_path = safe_data.get('calculation_source_path', safe_data.get('model_path'))
        energy_val = safe_data.get('energy')
        forces_val = safe_data.get('forces')
        stress_val = safe_data.get('stress')

        # Handle single energy value if provided (assuming REAL column type)
        db_energy = None
        if isinstance(energy_val, (int, float)):
            db_energy = float(energy_val)
        elif isinstance(energy_val, list) and len(energy_val) == 1 and isinstance(energy_val[0], (int, float)):
             db_energy = float(energy_val[0])
        elif energy_val is not None:
             # Check if it's a numpy array/scalar that fix_numpy handled
             try:
                  db_energy = float(energy_val)
             except (ValueError, TypeError):
                  print(f"[WARN] Unexpected energy format for structure {structure_id}: {energy_val}. Storing NULL.")


        # Metadata: Collect everything not explicitly mapped to a column
        # Define keys mapped to specific columns in the *new* schema
        column_keys = {'structure_id', 'calculator', 'calculation_source_path',
                       'energy', 'forces', 'stress'}
        # Include old names to ensure they don't leak into metadata if passed
        old_column_keys = {'model_type', 'model_path', 'model_generation', 'ensemble_variance'}
        # Also explicitly exclude the 'metadata' key itself from the initial collection
        exclude_keys = column_keys.union(old_column_keys).union({'metadata'})

        # Start metadata_dict with any top-level keys from safe_data not explicitly excluded
        metadata_dict = {k: v for k, v in safe_data.items() if k not in exclude_keys}

        # If the incoming calc_data had its own 'metadata' key (e.g., from VaspParser), merge its contents
        if 'metadata' in safe_data and isinstance(safe_data['metadata'], dict):
             nested_meta = safe_data['metadata']
             # Update the dict with keys/values from the nested metadata
             # This correctly unpacks the parser's metadata into the top level
             metadata_dict.update(nested_meta)


        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO calculations (
                    structure_id, calculator, calculation_source_path,
                    energy, forces, stress, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING calculation_id
            """, (
                structure_id,
                calculator,
                calculation_source_path, # Allows NULL
                db_energy,               # Processed REAL value or None
                Json(forces_val) if forces_val is not None else None,
                Json(stress_val) if stress_val is not None else None,
                Json(metadata_dict) if metadata_dict else None # Store the correctly structured metadata
            ))
            calc_id = cur.fetchone()[0]
        self.conn.commit()
        return calc_id

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
    
    def find_structures_without_calculation(self, model_type: Optional[str] = None, status: Optional[str] = None) -> List[int]:
        """
        Return a list of structure_ids that do NOT have a calculation
        with the given model_type (and optionally status).
        If model_type and status are None, finds structures with no calculations at all.
        """
        if model_type is None:
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
            model_pattern = model_type.replace('*', '%')
            query = """
                SELECT s.structure_id
                FROM structures s
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM calculations c
                    WHERE c.structure_id = s.structure_id
                    AND c.model_type LIKE %s
                    AND (%s IS NULL OR c.metadata->>'status' = %s)
                )
            """
            params = [model_pattern, status, status]

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [r[0] for r in rows]

    def add_structures_from_xyz(
        self,
        xyz_file: str,
        skip_duplicates: bool = True,
        default_calculator: str = "vasp" # Changed default
    ) -> List[int]:
        """
        Reads frames from a .xyz file, adds structures, and optionally adds calculations
        if energy/forces/stress are present in the XYZ frame's info/arrays.

        Args:
            xyz_file: Path to the .xyz file containing frames.
            skip_duplicates: Whether to skip duplicates by position and composition hash.
            default_calculator: The calculator name to use if adding calculation data.

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
        Check if 'atoms' is a duplicate of any structure in the DB by:
        1) Checking if composition_hash already exists
        2) Checking positions only for those structures that share the same composition_hash
        """
        # Compute composition dictionary
        symbols = atoms.get_chemical_symbols()
        if not symbols: # Handle empty atoms object
             return False
        total_atoms = len(symbols)
        comp_dict = {}
        for symbol in set(symbols):
            count = symbols.count(symbol)
            comp_dict[symbol] = {
                "at_frac": count / total_atoms,
                "num_atoms": count
            }

        # Compute hash
        composition_hash = _compute_composition_hash(comp_dict, decimal=decimal)

        # Find potential duplicates by composition_hash (should be faster with index)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT structure_id, positions
                FROM structures
                WHERE composition_hash = %s
            """, (composition_hash,))
            potential_duplicates = cur.fetchall()

        if not potential_duplicates:
            # No structures with the same composition hash, definitely not a duplicate
            return False

        # Check positions only for these structures
        # Round positions before comparison
        new_positions = np.round(atoms.positions, decimals=int(abs(np.log10(position_tol))))

        for struct_id, db_positions_json in potential_duplicates:
            db_positions = np.array(db_positions_json)
            db_positions_rounded = np.round(db_positions, decimals=int(abs(np.log10(position_tol))))

            # Compare shape and values
            if db_positions_rounded.shape == new_positions.shape and \
               np.allclose(db_positions_rounded, new_positions, atol=position_tol):
                # Found a match
                # print(f"[DEBUG] Duplicate found: New structure matches existing structure {struct_id}") # Optional debug
                return True

        # No exact match found
        return False

    def remove_duplicate_structures(self,
                                    decimal: int = 5,
                                    position_tol: float = 1e-5) -> Dict[int, List[int]]:
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
        """Update structure metadata in the database. WARNING: This will overwrite the existing metadata.
        
        Args:
            structure_id: Structure ID
            metadata: New metadata dictionary
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
        Get an Atoms object with the latest calculation results attached.

        Args:
            structure_id: ID of structure to retrieve.
            calculator: Name of calculator results to attach (e.g., 'vasp'). Defaults to 'vasp'.

        Returns:
            Atoms object with calculation results in .info/.arrays, or None if structure/calc not found.
        """
        try:
            atoms = self.get_structure(structure_id)
        except Exception as e:
            print(f"[WARN] Failed to retrieve structure {structure_id}: {e}")
            return None

        calcs = self.get_calculations(structure_id, calculator=calculator) # Gets newest first

        if calcs:
            calc_data = calcs[0] # Use the latest matching calculation

            # Store energy and stress in atoms.info
            if calc_data.get('energy') is not None:
                # Ensure energy is float, handle potential list artifact from older schema if necessary
                energy_val = calc_data['energy']
                atoms.info['energy'] = float(energy_val) if isinstance(energy_val, (int, float)) else None
            else:
                atoms.info.pop('energy', None) # Remove if None

            if calc_data.get('stress') is not None:
                # Stress should already be numpy array from get_calculations
                atoms.info['stress'] = calc_data['stress']
            else:
                 atoms.info.pop('stress', None) # Remove if None


            # Store forces in atoms.arrays
            atoms.arrays.pop('forces', None) # Clear previous forces first
            if calc_data.get('forces') is not None:
                forces = calc_data['forces'] # Should be numpy array
                if forces.shape == (len(atoms), 3):
                    atoms.arrays['forces'] = forces
                else:
                     print(f"[WARN] Mismatched forces shape for structure {structure_id}, calc {calc_data.get('calculation_id')}. Expected {(len(atoms), 3)}, got {forces.shape}. Forces not attached.")


            # Add calculation metadata to atoms.info under a specific key
            atoms.info['calculation_info'] = {
                 'calculation_id': calc_data.get('calculation_id'),
                 'calculator': calc_data.get('calculator'),
                 'calculation_source_path': calc_data.get('calculation_source_path'),
                 **(calc_data.get('metadata', {})) # Unpack calc metadata here
             }

        else:
             # Clear previous calc info if no matching calculation found
             atoms.info.pop('energy', None)
             atoms.info.pop('stress', None)
             atoms.arrays.pop('forces', None)
             atoms.info.pop('calculation_info', None)


        return atoms

    def get_atoms_with_calculations(self, structure_ids: Union[int, List[int]], calculator: Optional[str] = 'vasp') -> List[Atoms]:
        """
        Get multiple Atoms objects with calculation results attached.

        Args:
            structure_ids: Single ID or list of structure IDs to retrieve.
            calculator: Name of calculator results to attach (e.g., 'vasp').

        Returns:
            List of Atoms objects with calculation results.
        """
        if isinstance(structure_ids, int):
            structure_ids = [structure_ids]

        atoms_list = []
        for struct_id in structure_ids:
            atoms = self.get_atoms_with_calculation(struct_id, calculator=calculator)
            if atoms:
                atoms.info['structure_id'] = struct_id # Ensure structure_id is in info
                atoms_list.append(atoms)
            # Warning printed by get_atoms_with_calculation if retrieval failed

        return atoms_list

    def find_structures_by_metadata(self, 
                                 metadata_filters: Dict[str, any],
                                 operator: str = 'exact',
                                 debug: bool = False) -> List[int]:
        """
        Search for structures by metadata fields.
        
        Args:
            metadata_filters: Dictionary of {metadata_key: value} to search for
            operator: How to match values - 'exact' (default), 'contains', '>', '<', '>=', '<='
                     String values only support 'exact' and 'contains'
            debug: If True, print debug information about the query
            
        Returns:
            List of structure IDs matching the criteria
        """
        if self.dry_run:
            print(f"[DRY RUN] Would search for structures with metadata: {metadata_filters}")
            return [1, 2, 3]  # Return dummy IDs
        
        query = "SELECT structure_id FROM structures WHERE 1=1"
        params = []
        
        for key, value in metadata_filters.items():
            # Handle nested keys with -> operator in PostgreSQL
            if '.' in key:
                # Convert Python-style dot notation to PostgreSQL JSON path
                parts = key.split('.')
                json_path = '->'.join([f"'{part}'" for part in parts[:-1]]) + "->>'" + parts[-1] + "'"
                db_key = f"metadata->{json_path}"
            else:
                db_key = f"metadata->>'{key}'"
            
            # Build query based on operator and value type
            if operator == 'exact':
                if value is None:
                    query += f" AND {db_key} IS NULL"
                else:
                    query += f" AND {db_key} = %s"
                    params.append(str(value))  # Convert to string for JSON key comparison
            elif operator == 'contains' and isinstance(value, str):
                query += f" AND {db_key} LIKE %s"
                params.append(f'%{value}%')
            elif operator in ('>', '<', '>=', '<=') and isinstance(value, (int, float)):
                query += f" AND ({db_key})::float {operator} %s"
                params.append(value)
            else:
                raise ValueError(f"Unsupported operator '{operator}' for value type {type(value)}")
        
        if debug:
            print(f"[DEBUG] Executing query: {query}")
            print(f"[DEBUG] With params: {params}")
        
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            results = [row[0] for row in cur.fetchall()]
        
        if debug:
            print(f"[DEBUG] Query returned {len(results)} matching structures")
        
        return results

    def remove_structure(self, structure_id: int, dry_run_override: Optional[bool] = None) -> None:
        """
        Remove a structure and all its associated calculations from the database.

        Args:
            structure_id: The ID of the structure to remove.
            dry_run_override: Optionally override the instance's dry_run setting for this operation.
                              If None, uses the instance's self.dry_run setting.
        """
        # Determine if this operation should be a dry run
        is_dry_run = self.dry_run if dry_run_override is None else dry_run_override

        if is_dry_run:
            print(f"[DRY RUN] Preparing to remove structure ID: {structure_id}")
            # Simulate finding associated calculations
            try:
                with self.conn.cursor() as cur: # Need connection even for dry run reads
                    cur.execute(
                        "SELECT calculation_id FROM calculations WHERE structure_id = %s",
                        (structure_id,)
                    )
                    calc_ids = [row[0] for row in cur.fetchall()]
                if calc_ids:
                    print(f"[DRY RUN] Would remove associated calculation IDs: {calc_ids}")
                else:
                    print("[DRY RUN] No associated calculations found.")
                print(f"[DRY RUN] Would remove structure ID: {structure_id}")
            except AttributeError: # Handle case where self.conn is None in dry_run init
                 print("[DRY RUN] Cannot query calculations as no database connection exists in dry run mode.")
                 print(f"[DRY RUN] Would attempt to remove structure ID: {structure_id} and any associated calculations.")
            return # Stop here for dry run

        # --- Actual Deletion ---
        if self.conn is None:
             print("[ERROR] Cannot remove structure: Database connection is not initialized.")
             return

        print(f"[INFO] Attempting to remove structure ID: {structure_id} and its calculations...")
        try:
            with self.conn.cursor() as cur:
                # Step 1: Remove associated calculations first due to foreign key constraints
                cur.execute(
                    """
                    DELETE FROM calculations
                    WHERE structure_id = %s
                    RETURNING calculation_id
                    """,
                    (structure_id,)
                )
                deleted_calc_ids = [row[0] for row in cur.fetchall()]
                if deleted_calc_ids:
                    print(f"[INFO] Removed associated calculation IDs: {deleted_calc_ids}")
                else:
                    print("[INFO] No associated calculations found to remove.")

                # Step 2: Remove the structure itself
                cur.execute(
                    """
                    DELETE FROM structures
                    WHERE structure_id = %s
                    RETURNING structure_id
                    """,
                    (structure_id,)
                )
                deleted_struct_id = cur.fetchone()

                if deleted_struct_id:
                    print(f"[INFO] Successfully removed structure ID: {deleted_struct_id[0]}")
                else:
                    print(f"[WARN] Structure ID {structure_id} not found or already removed.")

            # Step 3: Commit the transaction
            self.conn.commit()
            print(f"[INFO] Removal of structure {structure_id} committed.")

        except Exception as e:
            print(f"[ERROR] Failed to remove structure {structure_id}: {e}")
            # Rollback in case of error during the transaction
            if self.conn:
                self.conn.rollback()
            print("[INFO] Transaction rolled back.")

    def add_mlip_model(self,
                       model_generation: int,
                       model_file_path: str, # S3 URI or local path
                       mlip_type: Optional[str] = None,
                       train_structure_ids: Optional[List[int]] = None,
                       validation_structure_ids: Optional[List[int]] = None,
                       test_structure_ids: Optional[List[int]] = None,
                       model_parameters: Optional[Dict] = None,
                       wandb_link: Optional[str] = None,
                       notes: Optional[str] = None) -> int:
        """Adds a record for a trained MLIP model."""
        if self.dry_run:
             print(f"[DRY RUN] Would add MLIP model record for generation {model_generation}")
             fake_id = self._fake_id_counter
             self._fake_id_counter += 1
             return fake_id

        if not model_file_path:
             raise ValueError("model_file_path is required.")
        if model_generation is None: # Check explicitly as it's NOT NULL
             raise ValueError("model_generation is required.")


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
                 notes
             ))
             mlip_model_id = cur.fetchone()[0]
        self.conn.commit()
        return mlip_model_id
