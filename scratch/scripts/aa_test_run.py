from forge.core.database import DatabaseManager
from forge.workflows.adversarial_attack import run_adversarial_attacks
import random
import numpy as np 
from ase.io import read, write


db_manager = DatabaseManager()

#dimer_ids = db_manager.find_structures_by_metadata(metadata_filters={'config_type' : 'dimer'})
gen_7_ids = db_manager.find_structures_by_metadata(metadata_filters={'generation' : '7'})
gen_6_ids = db_manager.find_structures_by_metadata(metadata_filters={'generation' : '6'})

# combine the two lists without duplicates
all_ids = list(set(gen_6_ids + gen_7_ids))
print(len(all_ids))

calcs = db_manager.get_calculations_batch(all_ids)

#model_paths = ['../potentials/mace_gen_6_ensemble/gen_7_model_0-2025-02-12_stagetwo_compiled.model', '../potentials/mace_gen_6_ensemble/gen_7_model_1-2025-02-12_stagetwo_compiled.model', '../potentials/mace_gen_6_ensemble/gen_7_model_2-2025-02-12_stagetwo_compiled.model']
model_paths = ['../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model',
              '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_1_pr_stagetwo.model',
              '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_2_pr_stagetwo.model',
              '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_3_pr_stagetwo.model',
              '../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_4_pr_stagetwo.model',
              ]

random.seed(42)

trajectories = run_adversarial_attacks(
    db_manager = db_manager,
    model_paths = model_paths,
    structure_ids = all_ids, # select 1000 at random from list
    generation = 8,
    n_iterations=200,
    learning_rate=0.01,
    temperature=1000,
    include_probability=False,
    min_distance=1.0,
    use_energy_per_atom=True,
    device='cuda',
    debug=False,
    top_n=100,
    save_output=True,
    output_dir='../data/adversarial_attacks/gen_8'
)

# --- Handle None return value when saving --- 
if trajectories is not None:
    print(f"Number of trajectories returned: {len(trajectories)}")
else:
    print("Trajectories were saved to files.")

print(f"All done")
