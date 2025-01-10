import pytest
import tempfile
from pathlib import Path
import yaml
import numpy as np
from forge.workflows.db_to_vasp import prepare_vasp_job
from forge.workflows.vasp_to_db import parse_vasp_output, add_vasp_results_to_db
from forge.core.database import DatabaseManager
from ase.build import bulk
from ase import Atoms
import os
import shutil

@pytest.fixture
def test_structure():
    """Create a 2x2x2 BCC supercell with specific composition."""
    # Create BCC V as base structure
    atoms = bulk('V', 'bcc', a=3.03, cubic=True)
    # Create 2x2x2 supercell
    supercell = atoms * (2, 2, 2)
    # Total 8 atoms in the supercell
    
    # Set composition: 3 V, 1 Cr, 1 Ti, 1 W, and 1 Zr
    symbols = ['V'] * 3 + ['Cr', 'Ti', 'W', 'Zr', 'V']
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(symbols)
    supercell.symbols = symbols
    
    # Add small random displacements
    rng = np.random.default_rng(42)
    displacements = rng.uniform(-0.02, 0.02, size=(len(supercell), 3))
    supercell.positions += displacements
    
    return supercell

@pytest.fixture
def db_config():
    """Create test database configuration."""
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
def db_manager(db_config):
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.yaml') as tmp:
        config_path = Path(tmp.name)
        with open(config_path, 'w') as f:
            yaml.dump(db_config, f)
        
        db = DatabaseManager(config_path=config_path)
        
        # Drop old tables if needed
        with db.conn.cursor() as cur:
            cur.execute("""
                DROP TABLE IF EXISTS calculations CASCADE;
                DROP TABLE IF EXISTS structures CASCADE;
            """)
        db.conn.commit()
        
        # Reinitialize tables
        db._initialize_tables()
        
        yield db

@pytest.fixture
def test_dir():
    """Create and clean up a temporary test directory."""
    test_dir = Path("test_vasp_workflow")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    yield test_dir
    shutil.rmtree(test_dir)

def test_full_workflow(db_manager, test_structure, test_dir):
    """Test the complete workflow from DB to VASP and back."""
    # 1. Add structure to database
    structure_id = db_manager.add_structure(test_structure)
    assert structure_id is not None
    
    # 2. Prepare VASP job
    job_dir = test_dir / f"job_{structure_id}"
    prepare_vasp_job(
        db_manager=db_manager,
        structure_id=structure_id,
        vasp_profile_path="forge/workflows/vasp_settings/static.json",
        hpc_profile_path="forge/workflows/hpc_profiles/Perlmutter-CPU.json",
        output_dir=str(job_dir),
        auto_kpoints=True
    )
    
    # Verify VASP input files
    assert (job_dir / "POSCAR").exists()
    assert (job_dir / "INCAR").exists()
    assert (job_dir / "KPOINTS").exists()
    assert (job_dir / "POTCAR").exists()
    assert (job_dir / "submit.sh").exists()
    
    # 3. Copy example OUTCAR for testing
    example_outcar = Path("tests/resources/vasp-out/OUTCAR")
    if example_outcar.exists():
        shutil.copy(example_outcar, job_dir / "OUTCAR")
        
        # 4. Parse VASP output and add to database
        calc_data = parse_vasp_output(str(job_dir))
        assert calc_data is not None
        assert "energy" in calc_data
        assert "forces" in calc_data
        assert "stress" in calc_data
        
        # 5. Add calculation to database
        add_vasp_results_to_db(db_manager, structure_id, str(job_dir))
        
        # 6. Verify calculation was added
        calcs = db_manager.get_calculations(structure_id)
        assert len(calcs) == 1
        assert calcs[0]["status"] == "completed"
        
        # 7. Verify structure metadata was updated
        metadata = db_manager.get_structure_metadata(structure_id)
        assert "jobs" in metadata
        for profile_info in metadata["jobs"].values():
            assert profile_info["status"] == "completed"

def test_structure_composition(test_structure):
    """Verify the test structure has the correct composition."""
    symbols = test_structure.get_chemical_symbols()
    composition = {sym: symbols.count(sym) for sym in set(symbols)}
    
    assert composition['V'] == 4
    assert composition['Cr'] == 1
    assert composition['Ti'] == 1
    assert composition['W'] == 1
    assert composition['Zr'] == 1
    assert len(test_structure) == 8

def test_vasp_output_parsing(test_dir):
    """Test parsing of VASP output files."""
    # Copy example OUTCAR to test directory
    example_outcar = Path("tests/resources/vasp-out/OUTCAR")
    if not example_outcar.exists():
        pytest.skip("Example OUTCAR not available")
    
    job_dir = test_dir / "test_parsing"
    job_dir.mkdir()
    shutil.copy(example_outcar, job_dir / "OUTCAR")
    
    # Parse output
    calc_data = parse_vasp_output(str(job_dir))
    
    # Verify parsed data
    assert calc_data is not None
    assert isinstance(calc_data["energy"], float)
    assert isinstance(calc_data["forces"], list)
    assert isinstance(calc_data["stress"], list)
    assert calc_data["status"] == "completed"
    assert calc_data["parser"] in ["ase", "pymatgen"]

def test_prepare_vasp_job(db_manager, test_structure, test_dir):
    """Test preparation of VASP job files."""
    # Add structure to database
    structure_id = db_manager.add_structure(test_structure)
    
    # Prepare job
    job_dir = test_dir / f"job_{structure_id}"
    prepare_vasp_job(
        db_manager=db_manager,
        structure_id=structure_id,
        vasp_profile_path="forge/workflows/vasp_settings/static.json",
        hpc_profile_path="forge/workflows/hpc_profiles/Perlmutter-CPU.json",
        output_dir=str(job_dir),
        auto_kpoints=True
    )
    
    # Verify file contents
    with open(job_dir / "POSCAR", 'r') as f:
        poscar = f.read()
        assert all(element in poscar for element in ['V', 'Cr', 'Ti', 'W', 'Zr'])
    
    with open(job_dir / "KPOINTS", 'r') as f:
        kpoints = f.read()
        assert "KPOINTS generated by db_to_vasp" in kpoints
    
    with open(job_dir / "submit.sh", 'r') as f:
        submit = f.read()
        assert "#SBATCH" in submit
        assert "cd " in submit