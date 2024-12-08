# Core database interface (core/database.py)
from typing import Dict, List, Optional, Union
import psycopg2
from psycopg2.extras import Json
from ase import Atoms
import numpy as np
import yaml
from pathlib import Path

class ForgeDatabase:
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
                    vasp_energy FLOAT,
                    vasp_forces JSONB,
                    vasp_stress JSONB,
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
                    predicted_energy FLOAT,
                    predicted_forces JSONB,
                    ensemble_variance FLOAT,
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
                Json(metadata or {})
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
                SELECT positions, cell, pbc, formula
                FROM structures
                WHERE structure_id = %s
            """, (structure_id,))
            
            positions, cell, pbc, formula = cur.fetchone()
            
            return Atoms(
                symbols=formula,
                positions=np.array(positions),
                cell=np.array(cell),
                pbc=pbc
            )