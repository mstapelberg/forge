# Calculator Interface Migration Guide

## Overview

The `forge.workflows.calculator_interface` module has been **deprecated** and will be removed in a future version. Please migrate to the new `forge.calculators.factory` module, which provides better backend support and improved functionality.

## Migration Summary

| Old Import | New Import |
|------------|------------|
| `from forge.workflows.calculator_interface import create_calculator` | `from forge.calculators.factory import create_ensemble_calculator` |
| `from forge.workflows.calculator_interface import check_calculator_availability` | `from forge.calculators.factory import get_supported_backends` |
| `from forge.workflows.calculator_interface import UnifiedCalculator` | Use `create_ensemble_calculator` directly |

## Detailed Migration Guide

### 1. Basic Calculator Creation

**Old Code:**
```python
from forge.workflows.calculator_interface import create_calculator

# Create MACE calculator
calc = create_calculator(
    model_path='path/to/model.model',
    calculator_type='mace',
    device='cuda'
)

# Create Allegro calculator
calc = create_calculator(
    model_path='path/to/model.nequip.zip',
    calculator_type='allegro',
    device='cuda',
    chemical_symbols={'Ti': 'Ti', 'V': 'V'}
)
```

**New Code:**
```python
from forge.calculators.factory import create_ensemble_calculator

# Create MACE calculator
calc = create_ensemble_calculator(
    model_paths='path/to/model.model',
    backend='mace',
    device='cuda'
)

# Create Allegro calculator
calc = create_ensemble_calculator(
    model_paths='path/to/model.nequip.zip',
    backend='allegro',
    device='cuda',
    chemical_symbols={'Ti': 'Ti', 'V': 'V'}
)
```

### 2. Availability Checking

**Old Code:**
```python
from forge.workflows.calculator_interface import check_calculator_availability

available = check_calculator_availability()
if available['mace']:
    print("MACE is available")
```

**New Code:**
```python
from forge.calculators.factory import get_supported_backends

available_backends = get_supported_backends()
if 'mace' in available_backends:
    print("MACE is available")
```

### 3. Ensemble Calculators

**Old Code:**
```python
# Old interface didn't support ensembles well
calc = create_calculator(
    model_path=['model1.model', 'model2.model'],
    calculator_type='mace'
)
```

**New Code:**
```python
# New factory supports ensembles natively
calc = create_ensemble_calculator(
    model_paths=['model1.model', 'model2.model'],
    backend='mace'
)

# Use ensemble methods
forces = calc.forces_all(atoms)  # Shape: (n_models, n_atoms, 3)
energies = calc.energies_all(atoms)  # Shape: (n_models,)
```

### 4. Adversarial Attack Workflow

**Old Code:**
```python
from forge.core.adversarial_attack import GradientAdversarialOptimizer

optimizer = GradientAdversarialOptimizer(
    model_paths=['model1.model', 'model2.model'],
    calculator_type='mace'
)
```

**New Code:**
```python
from forge.core.adversarial_attack import GradientAdversarialOptimizer

optimizer = GradientAdversarialOptimizer(
    model_paths=['model1.model', 'model2.model'],
    backend='mace'  # Changed from calculator_type
)
```

### 5. Auto-detection

**Old Code:**
```python
# Auto-detection was limited
calc = create_calculator(
    model_path='model.model',
    calculator_type=None  # Auto-detect
)
```

**New Code:**
```python
# Better auto-detection based on file extensions
calc = create_ensemble_calculator(
    model_paths='model.model',
    backend='auto'  # Auto-detect from file extension
)
```

## Key Improvements in the New Factory

1. **Better Ensemble Support**: Native support for multiple models
2. **Improved Auto-detection**: Based on file extensions (.model, .nequip.zip, .pt2)
3. **Unified Interface**: Both MACE and Allegro use the same API
4. **Better Error Handling**: More informative error messages
5. **Performance**: Optimized for ensemble calculations
6. **Extensibility**: Easy to add new backends

## Backward Compatibility

The old interface will continue to work but will show deprecation warnings. We recommend migrating as soon as possible to avoid future compatibility issues.

## Testing Your Migration

After migrating, test your code with:

```python
# Test basic functionality
from forge.calculators.factory import create_ensemble_calculator, get_supported_backends

print(f"Available backends: {get_supported_backends()}")

# Test calculator creation
calc = create_ensemble_calculator(
    model_paths='path/to/your/model',
    backend='auto'
)

# Test ensemble methods
atoms = your_atoms_object
forces = calc.forces_all(atoms)
energies = calc.energies_all(atoms)
```

## Support

If you encounter issues during migration, please:

1. Check this migration guide
2. Look at the test examples in `scratch/scripts/creating_defect_motifs/`
3. Review the new factory documentation in `forge/calculators/factory.py`
4. Open an issue with specific error messages and code examples

## Timeline

- **Current**: Old interface is deprecated with warnings
- **Next Release**: Old interface will be removed
- **Future**: Only new factory interface will be supported 