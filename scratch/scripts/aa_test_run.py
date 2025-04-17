from forge.core.database import DatabaseManager
from forge.workflows.adversarial_attack import run_adversarial_attacks
import random
import numpy as np 
from ase.io import read, write


db_manager = DatabaseManager()

dimer_ids = db_manager.find_structures_by_metadata(metadata_filters={'config_type' : 'dimer'})
gen_7_ids = db_manager.find_structures_by_metadata(metadata_filters={'generation' : '7'})

# combine the two lists without duplicates
all_ids = list(set(dimer_ids + gen_7_ids))
print(len(all_ids))

calcs = db_manager.get_calculations_batch(all_ids)

model_paths = ['../potentials/mace_gen_6_ensemble/gen_7_model_0-2025-02-12_stagetwo_compiled.model', '../potentials/mace_gen_6_ensemble/gen_7_model_1-2025-02-12_stagetwo_compiled.model', '../potentials/mace_gen_6_ensemble/gen_7_model_2-2025-02-12_stagetwo_compiled.model']

random.seed(42)

trajectories = run_adversarial_attacks(
    db_manager = db_manager,
    model_paths = model_paths,
    structure_ids = random.sample(gen_7_ids, 2500), # select 1000 at random from list
    generation = 8,
    n_iterations=100,
    learning_rate=0.01,
    temperature=1000,
    include_probability=False,
    min_distance=1.2,
    use_energy_per_atom=True,
    device='cpu',
    debug=True,
    top_n=10,
)

print(f"Number of trajectories: {len(trajectories)}")
#write('test_aa_traj.xyz', trajectories, format='extxyz')
print(trajectories[0])

print(f"All done")
