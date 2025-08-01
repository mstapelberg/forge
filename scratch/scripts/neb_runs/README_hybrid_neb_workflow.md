# Hybrid NEB Workflow

A comprehensive workflow that combines composition generation, hybrid MD/MC optimization, and NEB calculations for vacancy diffusion studies in alloy systems.

## Overview

This workflow integrates three main components:

1. **Composition Generation**: Uses the `CompositionAnalyzer` to generate new alloy compositions based on existing data
2. **MCMC Optimization**: Optimizes atomic positions using standard MCMC with Allegro calculator
3. **NEB Calculations**: Performs vacancy diffusion barrier calculations using Allegro calculator

## Features

- **Flexible Composition Generation**: Generate new compositions with customizable constraints
- **Allegro Integration**: Uses Allegro calculator for both MCMC optimization and NEB calculations
- **Comprehensive NEB Analysis**: Full vacancy diffusion workflow with statistical analysis
- **Visualization**: Automatic generation of plots for barriers and compositions
- **Modular Design**: Use individual components or run the complete workflow
- **Reproducible**: All random seeds are controlled for reproducibility

## Requirements

### Core Dependencies
- `numpy`
- `torch`
- `ase` (Atomic Simulation Environment)
- `nequip` (Allegro calculator)
- `matplotlib`
- `scikit-learn`

### Optional Dependencies
- `umap-learn` (for composition analysis)

### Model Requirements
- A trained Allegro model file (`.nequip.zip` format)

## Installation

1. Ensure you have the core dependencies installed:
```bash
pip install numpy torch ase nequip matplotlib scikit-learn
```

2. For advanced composition analysis, install UMAP:
```bash
pip install umap-learn
```

## Usage

### Basic Usage

```python
from run_hybrid_neb_workflow import HybridNEBWorkflow

# Initialize workflow
workflow = HybridNEBWorkflow(
    model_path="path/to/your/allegro_model.nequip.zip",
    device="cuda",  # or "cpu"
    seed=42,
    output_dir="results"
)

# Run complete workflow
results = workflow.run_full_workflow(
    existing_compositions=[
        {'V': 0.85, 'Cr': 0.05, 'Ti': 0.05, 'W': 0.03, 'Zr': 0.02},
        {'V': 0.80, 'Cr': 0.08, 'Ti': 0.06, 'W': 0.04, 'Zr': 0.02}
    ],
    n_new_compositions=3,
    elements=['V', 'Cr', 'Ti', 'W', 'Zr'],
    crystal_type='bcc',
    dimensions=[6, 6, 6],
    temperature=873.15,  # 600°C
    n_steps=5000
)
```

### Step-by-Step Usage

You can also run individual components:

```python
# 1. Generate compositions
new_compositions = workflow.generate_compositions(
    existing_compositions=existing_compositions,
    n_new_compositions=5,
    constraints={
        'V': (0.5, 0.9),
        'Cr': (0.05, 0.3),
        'Ti': (0.02, 0.25),
        'W': (0.01, 0.1),
        'Zr': (0.001, 0.05)
    }
)

# 2. Create initial structures
initial_structures = workflow.create_initial_structures(
    compositions=new_compositions,
    crystal_type='bcc',
    dimensions=[8, 8, 8],
    lattice_constant=3.01
)

# 3. Optimize structures
optimized_structures = workflow.optimize_with_mcmc(
    structures=initial_structures,
    temperature=873.15,
    n_steps=10000,
    convergence_window=1000,
    energy_threshold=0.0002
)

# 4. Run NEB calculations
neb_results = workflow.run_neb_calculations(
    structures=optimized_structures,
    n_nearest=3,
    n_next_nearest=3,
    num_images=5
)

# 5. Analyze results
analysis_results = workflow.analyze_results(save_plots=True)
```

## Configuration Options

### Composition Generation

- `n_new_compositions`: Number of new compositions to generate
- `elements`: List of elements to include
- `constraints`: Dictionary mapping elements to (min, max) fraction ranges
- `balance_element`: Element used to balance the composition

### Structure Creation

- `crystal_type`: Crystal structure type ('bcc', 'fcc', 'hcp', etc.)
- `dimensions`: Supercell dimensions [nx, ny, nz]
- `lattice_constant`: Lattice parameter in Angstrom
- `balance_element`: Base element for structure creation

### Optimization

- `temperature`: Simulation temperature in Kelvin
- `n_steps`: Number of optimization steps
- `convergence_window`: Steps to check for convergence
- `energy_threshold`: Energy change threshold for convergence

### NEB Calculations

- `n_nearest`: Number of nearest neighbors to sample per vacancy
- `n_next_nearest`: Number of next-nearest neighbors to sample per vacancy
- `num_images`: Number of NEB images
- `neb_method`: NEB method ("dyneb" or "neb")
- `climb`: Whether to use climbing image
- `relax_fmax`: Force tolerance for endpoint relaxation
- `neb_fmax`: Force tolerance for NEB calculation

## Output Structure

The workflow creates the following directory structure:

```
output_dir/
├── initial_structure_0_formula.xyz
├── initial_structure_1_formula.xyz
├── optimized_structure_0_formula.xyz
├── optimized_structure_1_formula.xyz
├── neb_structure_0/
│   ├── neb_results.json
│   ├── structure_initial.xyz
│   └── structure_final.xyz
├── neb_structure_1/
│   ├── neb_results.json
│   └── ...
├── barrier_distributions.png
├── barriers_by_element.png
├── composition_analysis.png
├── analysis_results.json
└── workflow_summary.json
```

## Examples

See `example_hybrid_neb_usage.py` for detailed examples:

1. **Basic Workflow**: Complete workflow with minimal settings
2. **Composition Generation**: Focus on generating new compositions
3. **Optimization Only**: Structure optimization at different temperatures
4. **NEB Only**: NEB calculations on existing structures
5. **Custom Workflow**: Workflow with specific requirements

## Performance Considerations

### Computational Resources

- **GPU**: Recommended for MACE calculations and torch-sim optimization
- **Memory**: Large supercells may require significant memory
- **Time**: NEB calculations are the most time-consuming component

### Optimization Tips

1. **Start Small**: Use small supercells (4x4x4 or 6x6x6) for testing
2. **Reduce Steps**: Use fewer optimization and NEB steps for quick testing
3. **Sample Sparsely**: Reduce n_nearest and n_next_nearest for faster NEB
4. **Use GPU**: Allegro calculations are faster on GPU

### Scaling

- **Compositions**: Linear scaling with number of compositions
- **Optimization**: Linear scaling with number of structures and steps
- **NEB**: Quadratic scaling with number of vacancy sites and neighbors

## Troubleshooting

### Common Issues

1. **NequIP Import Error**: Install nequip to use Allegro models
2. **Memory Issues**: Reduce supercell size or use CPU
3. **NEB Convergence**: Increase relax_steps or reduce relax_fmax
4. **Model Loading**: Ensure model path is correct and model is compatible

### Debug Mode

Enable verbose output for debugging:

```python
neb_results = workflow.run_neb_calculations(
    structures=structures,
    verbose=2  # Detailed output
)
```

## Advanced Features

### Custom Constraints

Define specific composition constraints:

```python
constraints = {
    'V': (0.6, 0.9),      # V must be 60-90%
    'Cr': (0.05, 0.25),   # Cr must be 5-25%
    'Ti': (0.02, 0.15),   # Ti must be 2-15%
    'W': (0.01, 0.08),    # W must be 1-8%
    'Zr': (0.001, 0.02)   # Zr must be 0.1-2%
}
```

### Temperature Ramping

Optimize at multiple temperatures:

```python
temperatures = [300, 600, 900]  # K
for temp in temperatures:
    optimized = workflow.optimize_with_hybrid_md_mc(
        structures=structures,
        temperature=temp + 273.15,
        n_steps=2000
    )
```

### Custom Analysis

Extend the analysis with custom plots:

```python
# Get NEB analyzer
neb_analyzer = NEBAnalyzer()
for result in neb_results:
    if result.get('success', False):
        neb_analyzer.add_calculation(result)

# Custom filtering
filtered_results = neb_analyzer.filter_calculations(
    vacancy_element='V',
    target_element='Cr',
    min_barrier=0.1
)
```

## Contributing

To extend the workflow:

1. Add new optimization methods to `_optimize_with_*` methods
2. Extend analysis capabilities in `analyze_results`
3. Add new visualization functions
4. Implement additional composition generation strategies

## References

- Allegro: Equivariant atomic neural network
- NequIP: Neural Equivariant Interatomic Potentials
- ASE: Atomic Simulation Environment
- NEB: Nudged Elastic Band method for transition state calculations 