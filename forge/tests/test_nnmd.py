# test_nnmd.py
import pytest
import numpy as np
from pathlib import Path
from ase import Atoms
from mace.calculators import MACECalculator
from forge.core.database import DatabaseManager
from forge.core.nnmd import NNMDSimulator, CompositionAnalyzer

class TestNNMD:
    @pytest.fixture
    def db_manager(self):
        """Create test database connection"""
        config = {
            'database': {
                'dbname': 'test_db',
                'user': 'postgres',
                'password': 'postgres',
                'host': 'localhost',
                'port': 5432
            }
        }
        return DatabaseManager(config)

    @pytest.fixture
    def mace_calculator(self):
        """Load test MACE model"""
        model_path = Path(__file__).parent / 'resources' / 'models' / 'mace_model.pt'
        return MACECalculator(model_path=str(model_path))

    def test_composition_workflow(self, db_manager, mace_calculator):
        """Test full NNMD workflow"""
        # 1. Query compositions with VASP forces
        query = """
            SELECT DISTINCT composition
            FROM structures 
            WHERE vasp_forces IS NOT NULL
        """
        with db_manager.conn.cursor() as cur:
            cur.execute(query)
            compositions = cur.fetchall()
        
        assert len(compositions) > 0

        # 2. Run clustering
        analyzer = CompositionAnalyzer(n_components=2)
        new_compositions = analyzer.select_compositions(compositions, n_select=5)
        assert len(new_compositions) == 5

        # 3. Calculate force variances
        variances = []
        structures = []
        for comp in new_compositions:
            structure = db_manager.find_structures(elements=list(comp.keys()))[0]
            atoms = db_manager.get_structure(structure)
            atoms.calc = mace_calculator
            
            # Get force variance
            forces = atoms.get_forces()
            var = np.var(forces)
            variances.append(var)
            structures.append(atoms)

        # 4. Run MD on highest variance structure
        max_var_idx = np.argmax(variances)
        test_atoms = structures[max_var_idx]
        
        simulator = NNMDSimulator(
            calculator=mace_calculator,
            temp=300,
            timestep=1.0,
            friction=0.002
        )

        # Run 1ps simulation (1000 steps of 1fs)
        simulator.run_md(test_atoms, steps=1000)
        
        # 5. Save frames at specified intervals
        frames = [250, 500, 750, 1000]
        for frame in frames:
            frame_atoms = test_atoms.copy()
            frame_atoms.info['md_frame'] = frame
            structure_id = db_manager.add_structure(frame_atoms)
            assert structure_id is not None