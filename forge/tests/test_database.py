# forge/tests/test_database.py
import pytest
import numpy as np
from ase.io import read
from ase.build import bulk
import yaml
import tempfile
from pathlib import Path
from forge.core.database import DatabaseManager

class TestDatabaseManager:
    @pytest.fixture(autouse=True)
    def db_config(self):
        """Create test database configuration"""
        return {
            'database': {
                'dbname': 'test_db',
                'user': 'postgres',
                'password': 'postgres',
                'host': 'localhost',
                'port': 5432
            }
        }

    @pytest.fixture
    def db_manager(self, db_config):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.yaml') as tmp:
            config_path = Path(tmp.name)
            with open(config_path, 'w') as f:
                yaml.dump(db_config, f)
            
            db = DatabaseManager(config_path=config_path)
            yield db
            
    @pytest.fixture
    def test_structure(self):
        """Load test structure from resources"""
        test_file = Path(__file__).parent / 'resources' / 'structures' / 'neb_structures.xyz'
        return read(test_file)

    def test_structure_handling(self, db_manager, test_structure):
        """Test structure storage and retrieval with metadata and PBC"""
        # Add metadata
        test_structure.info.update({
            'source': 'MaterialsProject',
            'mp_id': 'mp-123',
            'formation_energy': -2.5
        })
        
        # Set PBC and cell
        test_structure.pbc = [True, True, False]
        test_structure.cell = [[3.01, 0, 0], [0, 3.01, 0], [0, 0, 10.0]]
        
        # Add and retrieve
        structure_id = db_manager.add_structure(test_structure)
        retrieved = db_manager.get_structure(structure_id)
        
        # Verify structure data
        assert len(retrieved) == len(test_structure)
        assert np.allclose(retrieved.positions, test_structure.positions)
        assert np.array_equal(retrieved.pbc, test_structure.pbc)
        assert np.allclose(retrieved.cell, test_structure.cell)
        
        # Verify metadata
        assert retrieved.info['source'] == 'MaterialsProject'
        assert retrieved.info['mp_id'] == 'mp-123'
        assert retrieved.info['formation_energy'] == -2.5

    def test_structure_lineage(self, db_manager, test_structure):
        """Test parent-child relationship tracking"""
        parent_id = db_manager.add_structure(test_structure)
        child = test_structure.copy()
        child.positions += 0.1
        child_id = db_manager.add_structure(child, parent_id=parent_id)
        retrieved = db_manager.get_structure(child_id)
        assert retrieved.info['parent_id'] == parent_id

    def test_calculation_management(self, db_manager, test_structure):
        """Test comprehensive calculation handling"""
        structure_id = db_manager.add_structure(test_structure)
        
        # Test single calculation with metadata
        calc_v1 = {
            'energy': -100.0,
            'forces': np.random.random((len(test_structure), 3)),
            'model_type': 'mace',
            'model_version': '0.1.0',
            'date_calculated': '2024-03-19',
            'parameters': {'cutoff': 5.0},
            'runtime_seconds': 120.5,
            'gpu_count': 2,
            'status': 'completed'
        }
        calc_id = db_manager.add_calculation(structure_id, calc_v1)
        assert calc_id > 0
        
        # Add VASP calculation
        vasp_calc = {
            'energy': -99.5,
            'forces': np.random.random((len(test_structure), 3)),
            'stress': np.random.random(6),
            'model_type': 'vasp',
            'status': 'failed',
            'error': 'VASP calculation diverged'
        }
        db_manager.add_calculation(structure_id, vasp_calc)
        
        # Test querying
        all_calcs = db_manager.get_calculations(structure_id)
        assert len(all_calcs) == 2
        
        mace_calcs = db_manager.get_calculations(structure_id, model_type='mace')
        assert len(mace_calcs) == 1
        assert mace_calcs[0]['model_version'] == '0.1.0'
        
        failed_calcs = db_manager.get_calculations(structure_id, status='failed')
        assert len(failed_calcs) == 1
        assert failed_calcs[0]['model_type'] == 'vasp'

    def test_ensemble_calculations(self, db_manager, test_structure):
        """Test storing ensemble calculation results"""
        structure_id = db_manager.add_structure(test_structure)
        ensemble_data = {
            'energies': np.random.random(5),
            'forces': np.random.random((5, len(test_structure), 3)),
            'model_type': 'mace_ensemble',
            'variance': np.random.random(len(test_structure))
        }
        calc_id = db_manager.add_calculation(structure_id, ensemble_data)
        calcs = db_manager.get_calculations(structure_id)
        assert len(calcs) == 1
        assert calcs[0]['model_type'] == 'mace_ensemble'
        assert 'variance' in calcs[0]

    def test_structure_search(self, db_manager):
        """Test structure search functionality"""
        structures = [
            bulk('V', 'bcc', a=3.01),
            bulk('Cr', 'bcc', a=2.8),
            bulk('W', 'bcc', a=3.16)
        ]
        for struct in structures:
            db_manager.add_structure(struct)
            
        v_structs = db_manager.find_structures(elements=['V'])
        assert len(v_structs) == 1
        
        bcc_structs = db_manager.find_structures(structure_type='bcc')
        assert len(bcc_structs) == 3