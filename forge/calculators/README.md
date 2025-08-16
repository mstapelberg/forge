# Forge Calculator Factory

This module provides a unified interface for creating calculator backends for machine learning interatomic potentials (MLIPs), with automatic optimization for single vs. multiple model scenarios.

## Overview

The `create_ensemble_calculator` function automatically detects the optimal calculator type based on the number of models provided:

- **Single Model**: Returns native ASE calculator (e.g., `NequIPCalculator`, `MACECalculator`) for optimal performance
- **Multiple Models**: Returns ensemble calculator for uncertainty estimation and adversarial attacks

## Supported Backends

### MACE
- **File Format**: `.model` files
- **Native Calculator**: `MACECalculator`
- **Ensemble Calculator**: `MACEBackend`

### Allegro/NequIP
- **File Formats**: 
  - `.pt2` or `.nequip.pt2` (compiled models)
  - `.zip` or `.nequip.zip` (packaged models)
- **Native Calculator**: `NequIPCalculator`
- **Ensemble Calculator**: `AllegroBackend`

## Usage Examples

### Single Model (High Performance)
```python
from forge.calculators.factory import create_ensemble_calculator

# Returns native NequIPCalculator for optimal performance
calculator = create_ensemble_calculator(
    model_paths="path/to/model.nequip.zip",
    backend='allegro',
    device='cuda',
    chemical_symbols={'V': 'V', 'Cr': 'Cr', 'Ti': 'Ti', 'W': 'W', 'Zr': 'Zr'}
)

# Use like any ASE calculator
energy = calculator.get_potential_energy(atoms)
forces = calculator.get_forces(atoms)
```

### Multiple Models (Ensemble Uncertainty)
```python
# Returns AllegroBackend for ensemble calculations
calculator = create_ensemble_calculator(
    model_paths=[
        "path/to/model1.nequip.zip",
        "path/to/model2.nequip.zip",
        "path/to/model3.nequip.zip"
    ],
    backend='allegro',
    device='cuda',
    chemical_symbols={'V': 'V', 'Cr': 'Cr', 'Ti': 'Ti', 'W': 'W', 'Zr': 'Zr'}
)

# Standard ASE interface (returns mean values)
energy = calculator.get_potential_energy(atoms)
forces = calculator.get_forces(atoms)

# Ensemble interface (returns all model predictions)
all_energies = calculator.energies_all(atoms)  # Shape: (n_models,)
all_forces = calculator.forces_all(atoms)      # Shape: (n_models, n_atoms, 3)

# Calculate uncertainty
force_variance = all_forces.var(axis=0)  # Shape: (n_atoms, 3)
energy_std = all_energies.std()
```

### Auto-Detection
```python
# Backend automatically detected from file extension
calculator = create_ensemble_calculator(
    model_paths="path/to/model.model",  # Auto-detects MACE
    device='cuda'
)
```

## Performance Characteristics

### Single Model Performance
- **Native Performance**: Matches direct use of `NequIPCalculator` or `MACECalculator`
- **Use Cases**: NEB calculations, structural optimization, hybrid MCMC-MD
- **Overhead**: Minimal (just factory function call)

### Multiple Model Performance
- **Ensemble Overhead**: Some overhead due to wrapper and data conversion
- **Use Cases**: Adversarial attacks, uncertainty quantification
- **Features**: Access to individual model predictions and uncertainty metrics

## Workflow Recommendations

### For NEB and Structural Optimization
```python
# Use single model for best performance
calculator = create_ensemble_calculator(
    model_paths="best_model.nequip.zip",
    backend='allegro'
)
# Use with ASE optimizers, NEB, etc.
```

### For Adversarial Attacks
```python
# Use multiple models for uncertainty estimation
calculator = create_ensemble_calculator(
    model_paths=["model1.zip", "model2.zip", "model3.zip"],
    backend='allegro'
)
# Access ensemble methods for uncertainty quantification
```

## Configuration

### Species Mapping
For Allegro/NequIP models, provide species mapping as a dictionary:
```python
chemical_symbols = {
    'V': 'V',   # Vanadium is type 0
    'Cr': 'Cr',  # Chromium is type 1
    'Ti': 'Ti',  # Titanium is type 2
    'W': 'W',   # Tungsten is type 3
    'Zr': 'Zr'   # Zirconium is type 4
}
```

The factory automatically converts this to the list format expected by NequIP calculators.

### Device Selection
```python
# GPU acceleration (recommended)
calculator = create_ensemble_calculator(
    model_paths="model.zip",
    device='cuda'
)

# CPU fallback
calculator = create_ensemble_calculator(
    model_paths="model.zip",
    device='cpu'
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure required packages are installed
   - MACE: `pip install mace-torch`
   - NequIP: `pip install nequip`

2. **Model Format**: Check file extensions match expected formats
   - MACE: `.model`
   - NequIP: `.pt2` (compiled) or `.zip` (packaged)

3. **Species Mapping**: Ensure species mapping matches model training
   - Check model metadata for correct element ordering
   - Verify all elements in your system are included

### Performance Issues

- **Single Model Slow**: Should match native performance. Check if using correct file format
- **Multiple Models Slow**: Expected overhead for ensemble calculations
- **Memory Issues**: Consider using compiled models (`.pt2`) for large systems 