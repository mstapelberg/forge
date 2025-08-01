# Defect-Aware Adversarial Attacks

This module extends the standard adversarial attack workflow to handle defect structures by identifying and fixing the positions of interstitial atoms during optimization.

## Overview

The key innovation is that during adversarial attacks on defect structures, we want to preserve the defect core (interstitial atoms) while allowing the surrounding lattice to respond to perturbations. This prevents the optimization from "destroying" the interstitials while still generating challenging structures for MLIP training.

## Key Features

1. **Automatic Interstitial Detection**: Identifies interstitial atoms based on motif type and position
2. **Position Constraints**: Uses ASE's `FixAtoms` constraint to prevent interstitial movement
3. **Trajectory Selection**: Can select N evenly-spaced structures from each optimization trajectory
4. **Flexible Input**: Works with both database structures and newly generated defect structures
5. **Defect-Specific Handling**: Different strategies for different defect types (SIA, vacancy, surfaces)

## Usage

### Basic Usage with Generated Defect Structures

```python
from scratch.scripts.defect_adversarial_attack import run_defect_adversarial_attacks

# Define compositions
compositions = [
    {'V': 0.75, 'Cr': 0.25},
    {'V': 0.50, 'Cr': 0.30, 'Ti': 0.20},
]

# Run adversarial attacks
trajectories = run_defect_adversarial_attacks(
    compositions=compositions,
    model_paths=['path/to/model1.model', 'path/to/model2.model'],
    top_n=10,
    generation=9,
    select_n_from_trajectory=20,  # Select 20 structures from each trajectory
    include_motifs=['sia', 'di-sia'],  # Focus on interstitial defects
    save_output=True,
    output_dir='defect_aa_output'
)
```

### Usage with Database Structures

```python
from forge.core.database import DatabaseManager
from scratch.scripts.defect_adversarial_attack import run_defect_adversarial_attacks

db_manager = DatabaseManager()
structure_ids = [1234, 5678, 9012]  # Your structure IDs

trajectories = run_defect_adversarial_attacks(
    db_manager=db_manager,
    structure_ids=structure_ids,
    model_paths=['path/to/model1.model', 'path/to/model2.model'],
    top_n=5,
    generation=9,
    select_n_from_trajectory=15
)
```

### Command Line Usage

```bash
# Generate new defect structures and run attacks
python defect_adversarial_attack.py \
    --compositions compositions.json \
    --model-paths model1.model model2.model \
    --top-n 10 \
    --generation 9 \
    --select-n-from-trajectory 20 \
    --include-motifs sia di-sia \
    --output-dir defect_aa_output \
    --save-output

# Use existing database structures
python defect_adversarial_attack.py \
    --structure-ids 1234 5678 9012 \
    --model-paths model1.model model2.model \
    --top-n 5 \
    --generation 9 \
    --select-n-from-trajectory 15
```

## Interstitial Detection Strategies

### Self-Interstitial Atoms (SIA)

For SIA structures, the algorithm:
1. Calculates the base lattice parameter from the cell
2. Identifies atoms that are not at regular lattice sites
3. Uses a distance tolerance to determine if an atom is interstitial

### Surface Structures

For surface structures, the algorithm:
1. Fixes atoms in the bottom 20% of the structure (configurable)
2. Allows the surface layers to respond to perturbations

### Vacancy Structures

For vacancy structures, no atoms are fixed since vacancies are defined by missing atoms, not extra ones.

## Parameters

### Core Parameters

- `compositions`: List of composition dictionaries for generating new structures
- `structure_ids`: List of structure IDs from database (alternative to compositions)
- `model_paths`: Paths to MACE model files for the ensemble
- `top_n`: Number of top structures to select for attack
- `generation`: Generation tag for new structures
- `select_n_from_trajectory`: Number of structures to select from each trajectory

### Optimization Parameters

- `n_iterations`: Number of optimization steps (default: 200)
- `learning_rate`: Optimizer learning rate (default: 0.01)
- `temperature`: Temperature for Boltzmann weighting (default: 1000)
- `patience`: Patience parameter for optimizer (default: 25)
- `shake`: Whether to apply random shake when patience is reached (default: False)

### Defect-Specific Parameters

- `interstitial_tolerance`: Distance tolerance for identifying interstitials (default: 0.5 Å)
- `include_motifs`: List of motif types to include
- `exclude_motifs`: List of motif types to exclude
- `custom_motif_path`: Path to custom motif templates

### Output Parameters

- `output_dir`: Directory to save output files
- `save_output`: Whether to save trajectories to files
- `debug`: Enable debug output

## Output Structure

When `save_output=True`, the script creates:

```
output_dir/
├── structure_gen_9_0.xyz      # Trajectory for first structure
├── structure_gen_9_1.xyz      # Trajectory for second structure
├── plots/                     # Optimization plots
│   ├── structure_gen_9_0_combined_plot.png
│   ├── structure_gen_9_0_energy_plot.png
│   └── structure_gen_9_0_loss_plot.png
└── rmse_distribution.png      # RMSE distribution (if applicable)
```

Each `.xyz` file contains the full trajectory, including the initial structure and all selected intermediate structures.

## Example Workflow

1. **Generate defect structures** with interstitials using `generate_defect_motifs.py`
2. **Run adversarial attacks** with this script, fixing interstitial positions
3. **Select N structures** from each trajectory for training data
4. **Use the selected structures** for MLIP training or further analysis

## Integration with Existing Workflows

This module integrates seamlessly with existing Forge workflows:

- **Database integration**: Can use structures from the database
- **Defect generation**: Works with the existing defect motif generation
- **MLIP training**: Generated structures can be used for training
- **Analysis**: Results can be analyzed using existing analysis modules

## Confidence Score

**Confidence: 8/10**

This implementation provides a robust solution for defect-aware adversarial attacks. The main uncertainties are:
1. The interstitial detection algorithm may need tuning for specific crystal structures
2. The surface fixing strategy is simplified and may need customization for different surface types
3. The integration with the existing adversarial attack workflow has been tested but may need refinement based on specific use cases.

## Future Improvements

1. **Enhanced interstitial detection**: More sophisticated algorithms for different crystal structures
2. **Customizable constraints**: Allow users to specify which atoms to fix
3. **Defect-specific strategies**: Tailored approaches for different defect types
4. **Performance optimization**: Parallel processing for multiple structures
5. **Integration with relaxation**: Combine with structure relaxation workflows 