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

    def _drop_tables(self, db_manager):
        """Drop all tables in the database"""
        # Ensure connection exists before dropping
        if db_manager.conn:
            with db_manager.conn.cursor() as cur:
                # Drop mlip_models first if it exists due to potential dependencies (though not explicit FKs in schema provided)
                cur.execute("DROP TABLE IF EXISTS mlip_models CASCADE;")
                cur.execute("DROP TABLE IF EXISTS calculations CASCADE;")
                cur.execute("DROP TABLE IF EXISTS structures CASCADE;")
            db_manager.conn.commit()

    @pytest.fixture
    def db_manager(self, db_config):
        """Create temporary database for testing"""
        db = None # Initialize db to None
        try:
            # No need for tempfile, DatabaseManager handles config dict directly
            db = DatabaseManager(config_dict=db_config, debug=False) # Set debug=False for less output

            # Drop tables if they exist
            self._drop_tables(db)

            # Reinitialize tables
            db._initialize_tables()

            yield db

        finally:
            # Ensure connection is closed even if setup fails
            if db and db.conn:
                 # Optional: Clean up tables after tests run
                 # self._drop_tables(db)
                 db.close_connection()

    @pytest.fixture
    def test_structure(self):
        """Load test structure from resources"""
        # Using a simple bulk structure for basic tests
        atoms = bulk('Si', 'diamond', a=5.43)
        atoms.info['test_info'] = 'original_info'
        return atoms

    def test_structure_handling(self, db_manager, test_structure):
        """Test structure storage and retrieval with metadata and PBC"""
        # Add metadata
        test_structure.info.update({
            'source': 'TestFunction',
            'structure_id': '123',
            'value': 10.5
        })

        # Set PBC and cell (already set by bulk, but verify)
        assert all(test_structure.pbc)

        # Add and retrieve
        structure_id = db_manager.add_structure(test_structure, source_type='test_add')
        retrieved = db_manager.get_structure(structure_id)

        # Verify structure data
        assert len(retrieved) == len(test_structure)
        assert np.allclose(retrieved.positions, test_structure.positions)
        assert np.array_equal(retrieved.pbc, test_structure.pbc)
        assert np.allclose(retrieved.cell, test_structure.cell)

        # Verify metadata (retrieved metadata includes structure_id)
        assert retrieved.info['source'] == 'TestFunction'
        assert retrieved.info['structure_id'] == '123'
        assert retrieved.info['value'] == 10.5
        assert retrieved.info['source_type'] == 'test_add' # check source_type from add_structure
        assert retrieved.info['structure_id'] == structure_id
        assert 'test_info' in retrieved.info # Check original info is preserved

    def test_structure_lineage(self, db_manager, test_structure):
        """Test parent-child relationship tracking"""
        parent_id = db_manager.add_structure(test_structure)
        child = test_structure.copy()
        child.positions += 0.1
        # Add parent_id explicitly and also in metadata to test precedence
        child.info['parent_id'] = 999 # Incorrect parent in info
        child_id = db_manager.add_structure(child, parent_id=parent_id, metadata={'extra_meta': True})

        # Retrieve using get_structure
        retrieved = db_manager.get_structure(child_id)
        # Check metadata stored in DB (which includes parent_id if passed)
        retrieved_meta = db_manager.get_structure_metadata(child_id)

        assert retrieved_meta is not None
        assert retrieved_meta.get('parent_id') == parent_id
        assert retrieved.info.get('parent_id') == parent_id # Should be updated by get_structure
        assert retrieved_meta.get('extra_meta') is True

    def test_calculation_management(self, db_manager, test_structure):
        """Test comprehensive calculation handling (add, get, filter)."""
        structure_id = db_manager.add_structure(test_structure)

        # Test single calculation with metadata
        calc_mace = {
            'energy': -100.0,
            'forces': np.random.random((len(test_structure), 3)).tolist(), # Use list for JSON
            'calculator': 'mace', # Use 'calculator' key
            'calculator_version': '0.1.0', # Changed key name
            'date_calculated': '2024-03-19',
            'parameters': {'cutoff': 5.0},
            'runtime_seconds': 120.5,
            'status': 'completed' # Status goes into metadata
        }
        calc_id_mace = db_manager.add_calculation(structure_id, calc_mace)
        assert calc_id_mace > 0

        # Add VASP calculation
        calc_vasp = {
            'energy': -99.5,
            'forces': np.random.random((len(test_structure), 3)).tolist(),
            'stress': np.random.random(6).tolist(),
            'calculator': 'vasp', # Use 'calculator' key
            'status': 'failed', # Status goes into metadata
            'error': 'VASP calculation diverged'
        }
        calc_id_vasp = db_manager.add_calculation(structure_id, calc_vasp)
        assert calc_id_vasp > 0

        # Test querying all calculations
        all_calcs = db_manager.get_calculations(structure_id)
        # Should be ordered newest first (VASP then MACE)
        assert len(all_calcs) == 2
        assert all_calcs[0]['calculation_id'] == calc_id_vasp
        assert all_calcs[1]['calculation_id'] == calc_id_mace

        # Test querying by calculator
        mace_calcs = db_manager.get_calculations(structure_id, calculator='mace')
        assert len(mace_calcs) == 1
        assert mace_calcs[0]['calculation_id'] == calc_id_mace
        # Check metadata was stored correctly
        assert mace_calcs[0]['metadata']['calculator_version'] == '0.1.0'
        assert mace_calcs[0]['metadata']['status'] == 'completed'
        assert 'parameters' in mace_calcs[0]['metadata']

        vasp_calcs = db_manager.get_calculations(structure_id, calculator='vasp')
        assert len(vasp_calcs) == 1
        assert vasp_calcs[0]['calculation_id'] == calc_id_vasp
        # Check metadata was stored correctly
        assert vasp_calcs[0]['metadata']['status'] == 'failed'
        assert vasp_calcs[0]['metadata']['error'] == 'VASP calculation diverged'
        # Check numpy arrays are restored correctly
        assert isinstance(vasp_calcs[0]['forces'], np.ndarray)
        assert isinstance(vasp_calcs[0]['stress'], np.ndarray)

    def test_batch_operations(self, db_manager):
        """Test batch retrieval of structures and calculations."""
        struct1 = bulk('Cu', 'fcc', a=3.6)
        struct2 = bulk('Au', 'fcc', a=4.0)
        struct3 = bulk('Ag', 'fcc', a=4.1) # No calculation for this one

        id1 = db_manager.add_structure(struct1)
        id2 = db_manager.add_structure(struct2)
        id3 = db_manager.add_structure(struct3)

        calc1_data = {'calculator': 'vasp', 'energy': -1.0, 'status': 'ok'}
        calc2_data = {'calculator': 'vasp', 'energy': -2.0, 'status': 'ok'}
        calc1b_data = {'calculator': 'mace', 'energy': -1.1, 'status': 'better'} # Add second calc for id1

        db_manager.add_calculation(id1, calc1_data)
        db_manager.add_calculation(id2, calc2_data)
        db_manager.add_calculation(id1, calc1b_data) # Add newer MACE calc for id1

        # --- Test get_structures_batch ---
        ids_to_get = [id1, id3, 999] # Include non-existent ID
        atoms_map = db_manager.get_structures_batch(ids_to_get)
        assert len(atoms_map) == 2 # Should only return found structures
        assert id1 in atoms_map
        assert id3 in atoms_map
        assert id2 not in atoms_map # Not requested
        assert 999 not in atoms_map # Not found
        assert atoms_map[id1].get_chemical_formula() == 'Cu'
        assert atoms_map[id3].get_chemical_formula() == 'Ag'

        # --- Test get_calculations_batch ---
        # Get latest VASP calculation
        calc_map_vasp = db_manager.get_calculations_batch([id1, id2, id3], calculator='vasp')
        assert len(calc_map_vasp) == 2 # Only id1 and id2 have VASP calcs
        assert id1 in calc_map_vasp
        assert id2 in calc_map_vasp
        assert calc_map_vasp[id1]['energy'] == -1.0
        assert calc_map_vasp[id2]['energy'] == -2.0

        # Get latest calculation regardless of calculator
        calc_map_latest = db_manager.get_calculations_batch([id1, id2, id3])
        assert len(calc_map_latest) == 2 # id3 still has no calcs
        assert id1 in calc_map_latest
        assert id2 in calc_map_latest
        assert calc_map_latest[id1]['calculator'] == 'mace' # MACE is newer for id1
        assert calc_map_latest[id1]['energy'] == -1.1
        assert calc_map_latest[id2]['calculator'] == 'vasp'

        # --- Test get_batch_atoms_with_calculation ---
        atoms_with_calc = db_manager.get_batch_atoms_with_calculation([id1, id2, id3], calculator='vasp')
        assert len(atoms_with_calc) == 3 # Returns all found structures
        # Check id1 (has VASP calc)
        assert atoms_with_calc[0].info['structure_id'] == id1
        assert atoms_with_calc[0].info['energy'] == -1.0
        assert 'calculation_info' in atoms_with_calc[0].info
        assert atoms_with_calc[0].info['calculation_info']['calculator'] == 'vasp'
        # Check id2 (has VASP calc)
        assert atoms_with_calc[1].info['structure_id'] == id2
        assert atoms_with_calc[1].info['energy'] == -2.0
        assert 'calculation_info' in atoms_with_calc[1].info
        # Check id3 (no VASP calc)
        assert atoms_with_calc[2].info['structure_id'] == id3
        assert 'energy' not in atoms_with_calc[2].info
        assert 'calculation_info' not in atoms_with_calc[2].info

        # Get with latest calc (MACE for id1)
        atoms_with_latest = db_manager.get_batch_atoms_with_calculation([id1, id2, id3], calculator=None)
        assert len(atoms_with_latest) == 3
        assert atoms_with_latest[0].info['energy'] == -1.1 # MACE energy
        assert atoms_with_latest[0].info['calculation_info']['calculator'] == 'mace'
        assert atoms_with_latest[1].info['energy'] == -2.0 # VASP energy
        assert atoms_with_latest[1].info['calculation_info']['calculator'] == 'vasp'
        assert 'energy' not in atoms_with_latest[2].info # No calc for id3

    def test_structure_search(self, db_manager):
        """Test structure search functionality (find_structures)."""
        structures = [
            bulk('V', 'bcc', a=3.01),
            bulk('Cr', 'bcc', a=2.8),
            bulk('W', 'bcc', a=3.16)
        ]

        ids = []
        for i, struct in enumerate(structures):
            struct.info['structure_type'] = 'bcc'
            struct.info['index'] = i
            ids.append(db_manager.add_structure(struct, metadata={'custom_tag': f'tag_{i}'}))

        # Search by element
        v_structs = db_manager.find_structures(elements=['V'])
        assert len(v_structs) == 1
        assert v_structs[0] == ids[0]

        # Search by metadata structure_type
        bcc_structs = db_manager.find_structures(structure_type='bcc')
        assert len(bcc_structs) == 3
        assert set(bcc_structs) == set(ids)

        # Search by non-existent element
        fe_structs = db_manager.find_structures(elements=['Fe'])
        assert len(fe_structs) == 0

        # Search by lattice parameter
        # Note: This uses a simplified volume calculation; may be inaccurate for complex cells
        lp_structs = db_manager.find_structures(min_lattice_parameter=3.0, max_lattice_parameter=3.1)
        assert len(lp_structs) == 1 # Only V (3.01)
        assert lp_structs[0] == ids[0]

    def test_metadata_search(self, db_manager):
        """Test searching by metadata fields."""
        struct1 = bulk('Al', 'fcc', a=4.05)
        struct2 = bulk('Ni', 'fcc', a=3.52)
        struct1.info = {'source': 'icsd', 'id': 101, 'quality': 'high', 'nested': {'value': 5}}
        struct2.info = {'source': 'oqmd', 'id': 202, 'quality': 'medium', 'nested': {'value': 10}}

        id1 = db_manager.add_structure(struct1)
        id2 = db_manager.add_structure(struct2)

        # Exact match
        res_icsd = db_manager.find_structures_by_metadata({'source': 'icsd'})
        assert res_icsd == [id1]

        # Numeric comparison
        res_qual = db_manager.find_structures_by_metadata({'quality': 'high'})
        assert res_qual == [id1]

        # Nested exact match
        res_nested = db_manager.find_structures_by_metadata({'nested.value': 5})
        assert res_nested == [id1]

        # Nested numeric comparison
        res_nested_gt = db_manager.find_structures_by_metadata({'nested.value': 7}, operator='>')
        assert res_nested_gt == [id2]

        # Contains (string)
        res_contains = db_manager.find_structures_by_metadata({'quality': 'diu'}, operator='contains')
        assert res_contains == [id2]

        # Non-existent key
        res_nokey = db_manager.find_structures_by_metadata({'non_existent_key': 'foo'})
        assert res_nokey == []

    def test_add_from_xyz(self, db_manager, tmp_path):
        """Test adding structures and calculations from an XYZ file."""
        # Create a dummy XYZ file
        xyz_content = """2
Properties=species:S:1:pos:R:3 energy=-1.0 REF_energy=-1.0 stress="1.0 0.0 0.0 0.0 0.0 0.0" REF_stress="1.0 0.0 0.0 0.0 0.0 0.0" Lattice="3.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 3.0" pbc="T T T" forces=REF_force="0.1 0.0 0.0 -0.1 0.0 0.0"
Si 0.0 0.0 0.0
Si 0.5 0.5 0.5
2
Properties=species:S:1:pos:R:3 energy=-2.0 REF_energy=-2.0 Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" pbc="T T T"
Ge 0.0 0.0 0.0
Ge 0.5 0.5 0.5
"""
        xyz_file = tmp_path / "test.xyz"
        xyz_file.write_text(xyz_content)

        added_ids = db_manager.add_structures_from_xyz(xyz_file)
        assert len(added_ids) == 2

        # Verify first structure and its calculation
        atoms1 = db_manager.get_batch_atoms_with_calculation([added_ids[0]])[0]
        assert atoms1.get_chemical_formula() == 'Si2'
        assert 'energy' in atoms1.info
        assert atoms1.info['energy'] == -1.0
        assert 'stress' in atoms1.info
        assert np.allclose(atoms1.info['stress'], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert 'forces' in atoms1.arrays
        assert np.allclose(atoms1.arrays['forces'], [[0.1, 0, 0], [-0.1, 0, 0]])
        assert atoms1.info['calculation_info']['calculator'] == 'vasp' # Default

        # Verify second structure (no forces/stress in XYZ, so no calc data attached by default)
        # Let's retrieve calc separately to see if it was added
        atoms2 = db_manager.get_structure(added_ids[1])
        calcs2 = db_manager.get_calculations(added_ids[1])

        assert atoms2.get_chemical_formula() == 'Ge2'
        assert len(calcs2) == 1 # Calculation should still be added
        assert calcs2[0]['energy'] == -2.0
        assert calcs2[0]['forces'] is None # Forces weren't in XYZ frame
        assert calcs2[0]['stress'] is None # Stress wasn't in XYZ frame

        # Test get_batch_atoms_with_calculation for the second structure
        atoms2_batch = db_manager.get_batch_atoms_with_calculation([added_ids[1]])[0]
        assert atoms2_batch.get_chemical_formula() == 'Ge2'
        assert 'energy' in atoms2_batch.info # Energy should be attached
        assert atoms2_batch.info['energy'] == -2.0
        assert 'forces' not in atoms2_batch.arrays
        assert 'stress' not in atoms2_batch.info

    def test_duplicate_handling(self, db_manager):
        """Test detection and removal of duplicate structures."""
        struct1 = bulk('Fe', 'bcc', a=2.87)
        struct2 = struct1.copy() # Identical structure
        struct3 = struct1.copy()
        struct3.positions[0] += 0.000001 # Tiny difference, below default tol
        struct4 = bulk('Fe', 'bcc', a=2.88) # Different lattice param

        id1 = db_manager.add_structure(struct1)
        id2 = db_manager.add_structure(struct2) # Duplicate of id1
        id3 = db_manager.add_structure(struct3) # Duplicate of id1 (within tol)
        id4 = db_manager.add_structure(struct4) # Not a duplicate

        # Check duplicates
        assert db_manager.check_duplicate_structure(struct2) is True
        assert db_manager.check_duplicate_structure(struct3) is True
        assert db_manager.check_duplicate_structure(struct4) is False

        # Remove duplicates (should keep id1, remove id2, id3)
        duplicates_map = db_manager.remove_duplicate_structures(position_tol=1e-5)

        assert len(duplicates_map) == 1 # One group of duplicates found
        assert id1 in duplicates_map # id1 was kept
        assert set(duplicates_map[id1]) == {id2, id3} # id2 and id3 were removed

        # Verify remaining structures
        remaining_ids = db_manager.find_structures(elements=['Fe'])
        assert set(remaining_ids) == {id1, id4}