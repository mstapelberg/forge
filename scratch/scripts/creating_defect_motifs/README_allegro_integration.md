# Allegro Integration with Adversarial Attacks

This document explains how to use Allegro models with the adversarial attack functionality in FORGE.

## Overview

The adversarial attack module has been updated to support both MACE and Allegro models. Allegro models can now be used with autograd optimization, providing the same capabilities as MACE models.

## Key Features

- **Autograd Support**: Allegro models support autograd optimization for efficient adversarial attack generation
- **Unified Interface**: Same API for both MACE and Allegro models
- **Fallback Support**: Automatic fallback to finite difference optimization if autograd is not available
- **Species Mapping**: Support for custom species-to-type mappings required by Allegro

## Requirements

- `nequip` package installed
- Allegro model files in `.nequip.zip` format (packaged models)
- Note: Compiled models (`.pt2`) are not supported for autograd

## Usage

### Command Line Interface

```bash
python defect_adversarial_attack.py \
    --compositions compositions.json \
    --model-paths path/to/allegro_model1.nequip.zip path/to/allegro_model2.nequip.zip \
    --backend allegro \
    --species-mapping '{"Ti": "Ti", "V": "V", "Cr": "Cr", "Zr": "Zr", "W": "W"}' \
    --top-n 10 \
    --generation 10 \
    --include-motifs sia di-sia \
    --output-dir output_directory \
    --save-output \
    --require-structure-id  # Optional: enforce structure_id requirement
```

### Python API

```python
from forge.core.adversarial_attack import GradientAdversarialOptimizer

# Initialize optimizer with Allegro models
optimizer = GradientAdversarialOptimizer(
    model_paths=['model1.nequip.zip', 'model2.nequip.zip'],
    device='cuda',
    learning_rate=0.01,
    temperature=1000,
    backend='allegro',
    species_to_type_name={'Ti': 'Ti', 'V': 'V', 'Cr': 'Cr', 'Zr': 'Zr', 'W': 'W'}
)

# Run optimization
trajectory = optimizer.optimize(
    atoms=atoms,
    generation=1,
    n_iterations=100,
    min_distance=1.5
)
```

### Species Mapping

Allegro models require a chemical symbols mapping that defines how chemical symbols map to model type names. This is typically provided as a dictionary:

```python
species_to_type_name = {
    'Ti': 'Ti',  # Titanium maps to type 'Ti'
    'V': 'V',    # Vanadium maps to type 'V'
    'Cr': 'Cr',  # Chromium maps to type 'Cr'
    'Zr': 'Zr',  # Zirconium maps to type 'Zr'
    'W': 'W'     # Tungsten maps to type 'W'
}
```

**Note**: For most Allegro models, the chemical symbols map to themselves (identity mapping). This is different from the numeric type indices used in some other frameworks.

You can provide this mapping in several ways:

1. **JSON string**: `'{"Ti": 0, "V": 1, "Cr": 2, "Zr": 3, "W": 4}'`
2. **JSON file**: Path to a file containing the mapping
3. **Python dict**: Direct dictionary object

## Model File Formats

### Supported Formats

- **Packaged models** (`.nequip.zip`): ✅ Full support with autograd
- **Compiled models** (`.pt2`): ❌ Not supported for autograd (use finite difference)

### Model Loading

The system automatically detects the model format and uses the appropriate loading method:

```python
# For packaged models (.nequip.zip)
calculator = NequIPCalculator._from_packaged_model(
    package_path='model.nequip.zip',
    chemical_symbols=species_mapping,
    device='cuda'
)

# For compiled models (.pt2) - limited functionality
calculator = NequIPCalculator.from_compiled_model(
    compile_path='model.pt2',
    chemical_symbols=species_mapping,
    device='cuda'
)
```

## Optimization Methods

### Autograd Optimization (Recommended)

When using packaged Allegro models, the system automatically uses autograd optimization:

```python
# This will use autograd if the model supports it
optimizer = GradientAdversarialOptimizer(
    model_paths=['model.nequip.zip'],
    backend='allegro',
    # ... other parameters
)
```

### Finite Difference Optimization (Fallback)

If autograd is not available (e.g., compiled models), the system falls back to finite difference optimization:

```python
# This will use finite difference optimization
optimizer = GradientAdversarialOptimizer(
    model_paths=['model.pt2'],  # Compiled model
    backend='allegro',
    # ... other parameters
)
```

## Example Scripts

### Basic Example

See `example_defect_aa.py` for a complete example using Allegro models.

### Testing

Use `test_allegro_integration.py` to test the Allegro integration:

```bash
python test_allegro_integration.py
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `nequip` is installed
   ```bash
   pip install nequip
   ```

2. **Species Mapping Error**: Ensure the species mapping matches your model's expected types
   ```python
   # Check what types your model expects
   from nequip.ase import NequIPCalculator
   calc = NequIPCalculator._from_packaged_model('model.nequip.zip')
   print(calc.model.metadata['type_names'])
   ```

3. **Model Format Error**: Use packaged models (`.nequip.zip`) for full autograd support
   ```bash
   # Convert compiled model to packaged format if needed
   nequip-deploy build --model model.pt2 --output model.nequip.zip
   ```

### Debug Mode

Enable debug mode to see detailed information about the optimization process:

```python
optimizer = GradientAdversarialOptimizer(
    # ... other parameters
    debug=True
)
```

## Performance Considerations

- **Autograd**: Faster and more efficient for packaged models
- **Finite Difference**: Slower but works with compiled models
- **Memory**: Allegro models may use more memory than MACE models
- **GPU**: Use GPU for better performance with large models

## Structure ID Handling

The adversarial attack system handles structure IDs in the following ways:

### Default Behavior (Recommended)
- If `structure_id` is missing from the atoms info, uses reserved ID `99999999`
- This allows processing of new structures without database dependencies
- The reserved ID is an integer and won't conflict with normal database IDs

### Strict Mode
- Use `--require-structure-id` flag to enforce strict structure_id requirements
- This will raise an error if `structure_id` is missing
- Useful for database-dependent workflows

### Usage Examples
```bash
# Default: use reserved ID for new structures
python defect_adversarial_attack.py --model-paths model.nequip.zip ...

# Strict: require structure_id in atoms info
python defect_adversarial_attack.py --model-paths model.nequip.zip --require-structure-id ...
```

## Limitations

1. **Compiled Models**: No autograd support, only finite difference
2. **Single Model**: Allegro ensemble support is limited compared to MACE
3. **Memory Usage**: Allegro models may require more memory
4. **Species Mapping**: Must be provided for all models
5. **Structure ID**: Uses reserved ID (99999999) for new structures by default

## Future Improvements

- Better ensemble support for Allegro models
- Automatic species mapping detection
- Improved memory efficiency
- Support for more model formats 