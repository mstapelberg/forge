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
    
    def __init__(self, config_path: Union[str, Path] = None, config_dict: Dict = None, dry_run: bool = False):
        """
        Initialize database connection using configuration.
        
        Args:
            config_path: Path to database configuration YAML
            config_dict: Dictionary containing database configuration
            dry_run: If True, simulate database operations without actually writing
            
        Note: If both config_path and config_dict are provided, config_dict takes precedence
        """
        self.dry_run = dry_run
        if not dry_run:
            self.config = self._load_config(config_path, config_dict)
            self.conn = self._initialize_connection()
            self._initialize_tables()
        else:
            self.conn = None
            self.cursor = None
            self._fake_id_counter = 1  # For generating fake IDs in dry run mode
    
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
        """Create database tables if they don't exist."""
        with self.conn.cursor() as cur:
            # Structures table
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
            
            # Calculations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS calculations (
                    calculation_id SERIAL PRIMARY KEY,
                    structure_id INTEGER REFERENCES structures(structure_id),
                    model_type TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    model_generation INTEGER,
                    energy JSONB,
                    forces JSONB,
                    stress JSONB,
                    ensemble_variance JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.conn.commit()
    
    def add_structure(self, atoms: Atoms, source_type: str = 'original',
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

        if metadata is None:
            metadata = {}
        else:
            metadata = fix_numpy(metadata.copy())
        
        if 'structure_id' in atoms.info:
            metadata['parent_id'] = fix_numpy(atoms.info['structure_id'])
        
        # Deep copy and fix the atoms.info dictionary
        info_dict = dict(atoms.info)  # Make a copy
        metadata.update(info_dict)  # Include ALL atoms.info in metadata

        # Calculate composition and number of atoms
        symbols = atoms.get_chemical_symbols()
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
            'formula': fix_numpy(atoms.get_chemical_formula()),
            'composition': fix_numpy(composition),
            'positions': fix_numpy(atoms.positions),
            'cell': fix_numpy(atoms.cell.tolist()),
            'pbc': fix_numpy(atoms.pbc.tolist()),
            'metadata': fix_numpy(metadata)
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
                    parent_id,
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
        """Add calculation results to database."""
        safe_data = fix_numpy(calc_data)
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO calculations (
                    structure_id, model_type, model_path, model_generation,
                    energy, forces, stress, ensemble_variance,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING calculation_id
            """, (
                structure_id,
                safe_data['model_type'],
                safe_data.get('model_path', ''),
                safe_data.get('model_generation', None),
                Json(safe_data.get('energies', [])) if 'energies' in safe_data else Json([safe_data.get('energy')]),  # right
                Json(safe_data.get('forces', []) if 'forces' in safe_data else []),
                Json(safe_data.get('stress', []) if 'stress' in safe_data else []),
                Json(safe_data.get('variance', [])),  # Wrap variance in Json()
                Json({
                    'runtime_seconds': safe_data.get('runtime_seconds'),
                    'gpu_count': safe_data.get('gpu_count'),
                    'gpu_memory_mb': safe_data.get('gpu_memory_mb'),
                    'date_started': safe_data.get('date_started'),
                    'date_completed': safe_data.get('date_completed'),
                    'status': safe_data.get('status', 'completed'),
                    'error': safe_data.get('error'),
                    'model_version': safe_data.get('model_version'),
                    'parameters': safe_data.get('parameters', {})
                })
            ))
            calc_id = cur.fetchone()[0]
        self.conn.commit()
        return calc_id

    def get_calculations(self, structure_id: int, model_type: str = None,
                        status: str = None, order_by: str = None) -> List[Dict]:
        """Retrieve calculations for structure.
        
        Args:
            structure_id: ID of structure to get calculations for
            model_type: Filter by model type. Can use '*' as wildcard (e.g. 'vasp*')
            status: Filter by status in metadata
            order_by: Sort by this metadata field
        """
        query = """
            SELECT calculation_id, model_type, energy, 
                forces, stress, ensemble_variance, metadata
            FROM calculations 
            WHERE structure_id = %s
        """
        params = [structure_id]
        
        if model_type:
            # Convert python-style wildcards to SQL LIKE pattern
            if '*' in model_type:
                like_pattern = model_type.replace('*', '%')
                query += " AND model_type LIKE %s"
                params.append(like_pattern)
            else:
                query += " AND model_type = %s"
                params.append(model_type)
        
        if status:
            query += " AND metadata->>'status' = %s"
            params.append(status)
            
        if order_by:
            query += f" ORDER BY metadata->>'{order_by}'"
            
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            
        return [{
            'id': row[0],
            'model_type': row[1],
            'energy': row[2],
            'forces': np.array(row[3]) if row[3] else None,
            'stress': np.array(row[4]) if row[4] else None,
            'variance': row[5],
            **row[6]  # Unpack metadata
        } for row in rows]

    
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
        default_model_type: str = "vasp-static"
    ) -> None:
        """
        Reads frames from a .xyz file, and optionally checks for duplicates
        prior to adding. If skip_duplicates=True, it uses self.check_duplicate_structure.

        Args:
            xyz_file: Path to the .xyz file containing frames.
            skip_duplicates: Whether to skip duplicates by position and composition hash.
            default_model_type: The model_type to use if `atoms.calc` is a SinglePointCalculator.
        """

        if not os.path.isfile(xyz_file):
            print(f"[WARN] XYZ file not found: {xyz_file}")
            return

        frames = read(xyz_file, index=":")
        print(f"[INFO] Found {len(frames)} frames in {xyz_file}")

        for i, atoms in enumerate(frames):
            print(f"[INFO] Processing frame {i} from {xyz_file}...")

            # If skipping duplicates, do a quick check
            if skip_duplicates:
                if self.check_duplicate_structure(atoms, 
                                                  decimal=4, # composition rounding
                                                  position_tol=1e-5):
                    print("[INFO] Skipping duplicate structure.")
                    continue

            # Add the structure
            struct_id = self.add_structure(atoms)
            print(f"[INFO] Added structure, ID={struct_id}")

            # If there's a SinglePointCalculator in atoms.calc, parse it
            if atoms.calc and isinstance(atoms.calc, SinglePointCalculator):
                calc_data = {}
                if hasattr(atoms.calc, "results"):
                    results = atoms.calc.results
                    calc_data["energy"] = float(results.get("energy", np.nan))
                    calc_data["forces"] = results.get("forces", None)
                    calc_data["stress"] = results.get("stress", None)
                calc_data["model_type"] = default_model_type
                calc_data["status"] = "completed"
                
                # Insert calculation row
                calc_id = self.add_calculation(struct_id, calc_data)
                print(f"[INFO] Added calculation, ID={calc_id}")

        print("[INFO] Finished adding frames from xyz file!")

    def check_duplicate_structure(self, atoms: Atoms, 
                             decimal: int = 4, position_tol: float = 1e-5) -> bool:
        """
        Check if 'atoms' is a duplicate of any structure in the DB by:
        1) Checking if composition_hash already exists
        2) Checking positions only for those structures that share the same composition_hash
        """
        # Compute composition dictionary
        symbols = atoms.get_chemical_symbols()
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
        
        # Find existing structures by composition_hash
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
        # Round positions to 'position_tol' to see if they match
        new_positions = np.round(atoms.positions, decimals=int(abs(np.log10(position_tol))))

        for row in potential_duplicates:
            db_struct_id = row[0]
            db_positions = np.array(row[1])  # loaded from JSON
            db_positions_rounded = np.round(db_positions, decimals=int(abs(np.log10(position_tol))))
            
            # Compare shape and values
            if db_positions_rounded.shape == new_positions.shape and \
               np.allclose(db_positions_rounded, new_positions, atol=position_tol):
                # Found a match
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
        """Update structure metadata in the database.
        
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

    def get_atoms_with_calculation(self, structure_id: int, model_type: Optional[str] = None) -> Atoms:
        """
        Get an Atoms object with calculation results attached.
        
        Args:
            structure_id: ID of structure to retrieve
            model_type: Type of calculation to attach (e.g., 'vasp-static', 'mace')
                       If None, attaches the first available calculation
        
        Returns:
            Atoms object with calculation results
        """
        # Get structure
        atoms = self.get_structure(structure_id)
        
        # Get calculations
        calcs = self.get_calculations(structure_id, model_type=model_type)
        
        if calcs:
            # Use the first matching calculation
            calc_data = calcs[0]
            
            # Store energy and stress in atoms.info
            if calc_data.get('energy') is not None:
                atoms.info['energy'] = float(calc_data['energy'][0])
            
            if calc_data.get('stress') is not None:
                atoms.info['stress'] = np.array(calc_data['stress'])
            
            # Store forces in atoms.arrays
            if calc_data.get('forces') is not None:
                forces = np.array(calc_data['forces'])
                if forces.shape == (len(atoms), 3):  # Verify correct shape
                    atoms.arrays['forces'] = forces
            
            # Add calculation metadata to atoms.info
            atoms.info['calculation_id'] = calc_data.get('id')
            atoms.info['model_type'] = calc_data.get('model_type')

            # check if atoms.info has forces
            if 'forces' in atoms.info:
                # remove forces from atoms.info
                atoms.info.pop('forces')

        return atoms
        
    def get_atoms_with_calculations(self, structure_ids: Union[int, List[int]], model_type: Optional[str] = None) -> List[Atoms]:
        """
        Get multiple Atoms objects with calculation results attached.
        
        Args:
            structure_ids: Single ID or list of structure IDs to retrieve
            model_type: Type of calculation to attach (e.g., 'vasp-static', 'mace')
                       If None, attaches the first available calculation for each structure
        
        Returns:
            List of Atoms objects with calculation results
        """
        # Handle single ID case
        if isinstance(structure_ids, int):
            structure_ids = [structure_ids]
            
        atoms_list = []
        for struct_id in structure_ids:
            try:
                atoms = self.get_atoms_with_calculation(struct_id, model_type=model_type)
                atoms.info['structure_id'] = struct_id
                atoms_list.append(atoms)
            except Exception as e:
                print(f"[WARNING] Failed to retrieve structure {struct_id} with calculation: {e}")
                
        return atoms_list
