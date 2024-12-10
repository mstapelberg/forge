# Core database interface (core/database.py)
from typing import Dict, List, Optional, Union
import psycopg2
from psycopg2.extras import Json
from ase import Atoms
import numpy as np
import yaml
from pathlib import Path

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
            # Structures table (keeps current schema)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS structures (
                    structure_id SERIAL PRIMARY KEY,
                    formula TEXT NOT NULL,
                    composition JSONB NOT NULL,
                    positions JSONB NOT NULL,
                    cell JSONB NOT NULL,
                    pbc BOOLEAN[] NOT NULL,
                    vasp_energy FLOAT,
                    vasp_forces JSONB,
                    vasp_stress JSONB,
                    parent_structure_id INTEGER REFERENCES structures(structure_id),
                    source_type TEXT,
                    metadata JSONB,
                    predicted_energy FLOAT,
                    predicted_forces JSONB,
                    predicted_stress JSONB,
                    model_type TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Calculations table (for ensemble predictions)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS calculations (
                    calculation_id SERIAL PRIMARY KEY,
                    structure_id INTEGER REFERENCES structures(structure_id),
                    model_type TEXT NOT NULL,
                    model_version TEXT,
                    ensemble_energies JSONB,
                    ensemble_forces JSONB,
                    energy_variance FLOAT,
                    forces_variance JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        self.conn.commit()
    
    def _get_composition_dict(self, atoms: Atoms) -> Dict[str, float]:
        """Convert Atoms object to composition dictionary."""
        symbols = atoms.get_chemical_symbols()
        unique_symbols = set(symbols)
        total_atoms = len(symbols)
        
        return {
            symbol: symbols.count(symbol) / total_atoms
            for symbol in unique_symbols
        }
    
    # In database.py
    def add_structure(self, structure, parent_id=None):
        """Add structure to database with proper metadata handling"""
        with self.conn.cursor() as cur:
            # Required fields
            formula = structure.get_chemical_formula()
            symbols = structure.get_chemical_symbols()
            composition = {sym: symbols.count(sym) for sym in set(symbols)}
            
            # Convert numpy arrays to lists
            positions = structure.positions.tolist()
            cell = structure.cell.tolist()
            pbc = [bool(x) for x in structure.pbc]
            
            # Get metadata from structure.info
            info = structure.info.copy() if hasattr(structure, 'info') else {}
            
            # Create metadata object
            metadata = {
                'source': info.get('source', 'unknown'),
                'mp_id': info.get('mp_id'),
                'formation_energy': info.get('formation_energy'),
                'parent_id': parent_id,  # Track lineage in metadata
                **info  # Keep any other metadata
            }

            cur.execute("""
                INSERT INTO structures (
                    formula,
                    composition,
                    positions,
                    cell,
                    pbc,
                    parent_structure_id,
                    source_type,
                    metadata,
                    vasp_energy,
                    vasp_forces,
                    vasp_stress,
                    predicted_energy,
                    predicted_forces,
                    predicted_stress,
                    model_type,
                    model_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING structure_id
            """, (
                formula,
                Json(composition),
                Json(positions),
                Json(cell),
                pbc,
                parent_id,
                info.get('source', 'unknown'),
                Json(metadata),
                info.get('vasp_energy'),
                Json(info.get('vasp_forces')),
                Json(info.get('vasp_stress')),
                info.get('predicted_energy'),
                Json(info.get('predicted_forces')),
                Json(info.get('predicted_stress')),
                info.get('model_type'),
                info.get('model_version')
            ))
            structure_id = cur.fetchone()[0]
            self.conn.commit()
            return structure_id

    def get_structure(self, structure_id: int) -> Atoms:
        """Retrieve structure from database."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT positions, cell, pbc, metadata, formula
                FROM structures WHERE structure_id = %s
            """, (structure_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"No structure found with id {structure_id}")
                
            positions, cell, pbc, metadata, formula = row
            
            # Create new Atoms object
            atoms = Atoms(
                symbols=formula,  # Use formula to recreate symbols
                positions=positions,
                cell=cell,
                pbc=pbc
            )
            
            # Restore metadata
            atoms.info.update(metadata)
            
            return atoms

    def add_calculation(self, structure_id: int, calc_data: Dict) -> int:
        """Add calculation results with proper numpy handling"""
        with self.conn.cursor() as cur:
            # Convert numpy arrays to lists
            forces = calc_data.get('forces', [])
            if isinstance(forces, np.ndarray):
                forces = forces.tolist()
            
            energies = calc_data.get('energies', [])
            if isinstance(energies, np.ndarray):
                energies = energies.tolist()

            variance = calc_data.get('variance', [])
            if isinstance(variance, np.ndarray):
                variance = variance.tolist()

            cur.execute("""
                INSERT INTO calculations (
                    structure_id, model_type, energies, forces, variance
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING calculation_id
            """, (
                structure_id,
                calc_data['model_type'],
                Json(energies),
                Json(forces),
                Json(variance)
            ))
            calc_id = cur.fetchone()[0]
            self.conn.commit()
            return calc_id

    def get_calculations(self, structure_id: int, model_type: str = None,
                        status: str = None, order_by: str = None) -> List[Dict]:
        """Retrieve calculations for structure."""
        query = """
            SELECT calculation_id, model_type, predicted_energy, 
                predicted_forces, ensemble_variance, metadata
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
            'variance': row[4],
            **row[5]  # Unpack metadata
        } for row in rows]

    def find_structures(self, elements: List[str] = None, 
                   structure_type: str = None,
                   min_lattice_parameter: float = None,
                   max_lattice_parameter: float = None) -> List[int]:
        """Search for structures matching criteria."""
        query = "SELECT structure_id FROM structures WHERE 1=1"
        params = []
        
        if elements:
            # Look for exact composition match
            query += " AND composition = %s"
            # Create composition dict with single element
            comp = {elements[0]: 1.0}
            params.append(Json(comp))
            
        if structure_type:
            query += " AND metadata->>'structure_type' = %s"
            params.append(structure_type)
            
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]