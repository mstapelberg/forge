#python:forge/tests/test_md_simulator.py
import pytest
from pathlib import Path
import yaml
import tempfile
import numpy as np
from ase.build import bulk
from forge.core.database import DatabaseManager
from forge.workflows.md import MDSimulator
from mace.calculators.mace import MACECalculator

@pytest.fixture
def db_config():
    """Create test database configuration for MD simulator tests."""
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
    """Create a temporary database (and tear down) for MD tests."""
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
        
        # Reinit tables
        db._initialize_tables()
        yield db

def test_md_simulator(db_manager):
    """
    Test the MDSimulator by:
    1. Creating a small BCC metal Atoms object
    2. Attaching a simple EMT calculator
    3. Running a short (1000-step) MD, with frames saved every 200 steps
    4. Changing the temperature and continuing the simulation
    5. Storing frames in the database
    """

    # 1. Build an example structure
    atoms = bulk("V", "bcc", a=3.01).repeat((3,3,3))  # 16 atoms -> 2x2x2 = 16
    # For a real test with 100+ atoms you can do bigger repeats
    # but let's keep it small for speed.

    # 2. Attach calculator
    # Define the base directory (where the tests are located)
    base_dir = Path(__file__).resolve().parent  # This gets the directory of the current file

    # Construct the absolute path to the model file
    model_path = base_dir / 'resources' / 'potentials' / 'mace' / 'gen_5_model_0-11-28_stagetwo.model'

    # Use the model path in your MACECalculator
    atoms.calc = MACECalculator(model_paths=[str(model_path)],
                                device="cpu",
                                default_dtype="float32")

    # 3. Initialize MDSimulator
    simulator = MDSimulator(
        calculator=atoms.calc,
        temp=300.0,
        timestep=1.0,
        friction=0.5,
        trajectory_file=None  # We'll store frames in DB instead of a .traj
    )

    # We add the initial structure to DB
    parent_id = db_manager.add_structure(atoms, source_type="md_test")

    # We define a small function to store frames in DB
    # This can be called every sample_interval steps.
    def store_frame_in_db(step, atoms_snapshot):
        # Add structure, referencing parent_id
        # if you want each frame as child of parent or chain them
        frame_id = db_manager.add_structure(
            atoms_snapshot,
            source_type="md_frame",
            parent_id=parent_id
        )
        # Optionally add a "calculation" that stores potential energy
        # or forces if you want.
        e = atoms_snapshot.get_potential_energy()
        f = atoms_snapshot.get_forces()
        s = atoms_snapshot.get_stress(voigt=False)
        db_manager.add_calculation(frame_id, {
            'model_type': 'MACE',
            'model_path': "./resources/potentials/mace/gen_5_model_0-11-28_stagetwo.model",
            'model_generation': 5,
            'energy': e,
            'forces': f,
            'stress': s,
            'status': 'completed'
        })

    # We modify the run_md method or attach a callback to do the storing 
    # every N steps. One approach: we create a custom loop:
    steps = 1000
    sample_interval = 200
   
    # We can reuse the internal Langevin from MDSimulator:
    # We'll do a short run in chunks to illustrate temperature change later.
    chunk_size = 500

    # We run the first chunk:
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    
    dyn = Langevin(atoms, simulator.timestep, simulator.temp, simulator.friction)
    MaxwellBoltzmannDistribution(atoms, temperature_K=simulator.temp)
    
    # Attach a callback to dyn
    def store_callback():
        store_frame_in_db(dyn.get_number_of_steps(), atoms)

    dyn.attach(store_callback, interval=sample_interval)
    dyn.run(chunk_size)

    # 4. Now let's change the temperature to 600K and continue
    simulator.temp = 600.0
    # re-init velocities if desired
    MaxwellBoltzmannDistribution(atoms, temperature_K=simulator.temp)
    dyn.set_temperature(simulator.temp)
    dyn.run(chunk_size)  # second chunk

    # One final store to capture the end state
    store_frame_in_db(steps, atoms)

    # 5. Check how many frames got stored
    with db_manager.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM structures WHERE source_type='md_frame'")
        row = cur.fetchone()
        frame_count = row[0]
    
    # Because we attach a callback every 200 steps plus the final store, 
    # we expect ~ ( chunk_size*(2)/sample_interval ) + 1 calls
    # For 2 chunks of 500 steps: we'd get 500/200=2.5 => 2 calls in the first chunk
    # + 2 calls in the second chunk = 4 calls plus one final store => 5 total
    # But your exact count can vary depending on the final step alignment. 
    assert frame_count >= 4, f"Expected at least 4 frames stored, got {frame_count}"

    print(f"MD test completed. Stored {frame_count} frames in the DB.")

