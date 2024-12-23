# Core database interface (core/database.py)
from typing import Dict, List, Optional, Union
import psycopg2
from psycopg2.extras import Json
from ase import Atoms
import numpy as np
import yaml
from pathlib import Path

def fix_numpy(obj):
        """Recursively convert np.ndarray into lists so psycopg2 can handle them."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: fix_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix_numpy(x) for x in obj]
        return obj

class DatabaseManager:
    """PostgreSQL database manager for FORGE."""
    
    def __init__(self, config_path: Union[str, Path] = None):
        """
        Initialize database connection using configuration.
        
        Args:
            config_path: Path to database configuration YAML
        """
        self.config = self._load_config(config_path)
        self.conn = self._initialize_connection()
        self._initialize_tables()
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load database configuration from YAML file."""
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
        """
        Add structure to database.
        
        Args:
            atoms: ASE Atoms object
            source_type: Type of structure ('original', 'adversarial', 'md')
            parent_id: ID of parent structure if derived
            metadata: Additional structure metadata
            
        Returns:
            int: Structure ID
        """
        if metadata is None:
            metadata = {}
        
        #merge the atoms.info with metadata
        metadata.update(atoms.info)
        # store the parent_id in metadata so it shows in the retrieved.info 
        if parent_id is not None:
            metadata['parent_id'] = parent_id

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO structures (
                    formula, composition, positions, cell, pbc,
                    parent_structure_id, source_type, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING structure_id
            """, (
                atoms.get_chemical_formula(),
                Json(self._get_composition_dict(atoms)),
                Json(atoms.positions.tolist()),
                Json(atoms.cell.tolist()),
                atoms.pbc.tolist(),
                parent_id,
                source_type,
                Json(metadata)
            ))
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
        """Retrieve calculations for structure."""
        query = """
            SELECT calculation_id, model_type, energy, 
                forces, stress, ensemble_variance, metadata
            FROM calculations 
            WHERE structure_id = %s
        """
        params = [structure_id]
        
        if model_type:
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
                max_lattice_parameter: float = None) -> List[int]:
        """Search for structures matching criteria."""
        query = "SELECT structure_id FROM structures WHERE 1=1"
        params = []
        
        if elements:
            # Convert elements to array if it's not already
            elements_array = '{' + ','.join(elements) + '}'
            query += " AND composition ?| %s"
            params.append(elements_array)
            
        if structure_type:
            query += " AND metadata->>'structure_type' = %s"
            params.append(structure_type)
            
        if min_lattice_parameter or max_lattice_parameter:
            # Approximate lattice parameter from cell volume
            query += """ 
                AND (
                    SELECT POWER(ABS(cell->0->0 * (cell->1->1 * cell->2->2 - 
                    cell->1->2 * cell->2->1)), 1.0/3.0)
                )
            """
            if min_lattice_parameter:
                query += " >= %s"
                params.append(min_lattice_parameter)
            if max_lattice_parameter:
                query += " <= %s"
                params.append(max_lattice_parameter)
        
        print(f"Executing query: {query}")  # Debug
        print(f"With params: {params}")  # Debug
                
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            results = [row[0] for row in cur.fetchall()]
            print(f"Found {len(results)} matching structures")  # Debug
            return results