from nequip.ase import NequIPCalculator
from forge.core.database import DatabaseManager
from forge.workflows.md import MDSimulator
import glob
from ase.io import write
from ase import Atoms

db = DatabaseManager()
ens_calc = []
compiled_paths = glob.glob('../data/potentials/compiled_models/*.pt2')
for compiled_path in compiled_paths:
    ens_calc.append(NequIPCalculator.from_compiled_model(compile_path=compiled_path, device='cuda'))


print(ens_calc[0])
start_structure = db.get_batch_atoms_with_calculation(structure_ids=[1617])

_atoms = start_structure[0]
start_atoms = Atoms(
    symbols=_atoms.get_chemical_symbols(),
    positions=_atoms.get_positions(),
    cell=_atoms.get_cell(),
    pbc=_atoms.get_pbc(),
    constraint=_atoms.constraints
)

nequip_calc = NequIPCalculator.from_compiled_model(compile_path=compiled_paths[0], device='cuda')

T = 1000
ts = 1
friction = 0.02
num_steps = 1000
md_sim = MDSimulator(nequip_calc, T, ts, friction=friction, trajectory_file='structure_1617_traj.xyz')


md_sim.run_md(start_atoms, steps=num_steps)







