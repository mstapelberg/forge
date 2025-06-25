# Migration Guide: V1/V2 to V3

This guide helps you migrate from older implementations of the allegro_utils module to the current V3 implementation.

## What Changed

### V1/V2 Issues
- **V1**: Attempted to override `train_dataloader()` but faced config object issues
- **V2**: Added distributed wrapper but still saw only partial dataset per GPU
- **Both**: Created samplers after dataset distribution, causing "1 batch" issue

### V3 Solution
- Precomputes all indices before any dataset distribution
- Reads training file directly to count structures
- Works correctly with PyTorch Lightning's distributed training

## Migration Steps

### 1. Update Your Imports

**Old (V1/V2):**
```python
from forge.workflows.allegro_utils.data import CustomSamplingASEDataModule
# or
from forge.workflows.allegro_utils.data_v2 import CustomSamplingASEDataModuleV2
```

**New (V3):**
```python
from forge.workflows.allegro_utils import CustomSamplingASEDataModule  # Alias for V3
# or explicitly
from forge.workflows.allegro_utils.data_v3 import CustomSamplingASEDataModuleV3
```

### 2. Remove Implementation Parameters

**Old:**
```python
prepare_allegro_job(
    db_manager=db,
    job_name="experiment",
    sampler="rare_weighted",
    sampler_params={...},
    sampler_implementation="v2",  # Remove this
    ...
)
```

**New:**
```python
prepare_allegro_job(
    db_manager=db,
    job_name="experiment",
    sampler="rare_weighted",
    sampler_params={...},
    # No implementation parameter needed
    ...
)
```

### 3. Update Config Files (if manually editing)

**Old:**
```yaml
data:
  _target_: forge.workflows.allegro_utils.data_v2.CustomSamplingASEDataModuleV2
  # or data.CustomSamplingASEDataModule
```

**New:**
```yaml
data:
  _target_: forge.workflows.allegro_utils.data_v3.CustomSamplingASEDataModuleV3
```

## Feature Compatibility

All features from V1/V2 are supported in V3:

| Feature | V1/V2 | V3 | Notes |
|---------|-------|-----|-------|
| Rare oversampling | ✓ | ✓ | Works correctly in distributed mode |
| Force weighting | ✓ | ✓ | Currently uniform for efficiency |
| Custom samplers | ✓ | ✓ | Better distributed support |
| Loss functions | ✓ | ✓ | No changes needed |
| Metrics | ✓ | ✓ | No changes needed |

## Troubleshooting

### Still Getting "1 Batch" Error

1. Verify you're using V3:
   ```bash
   grep "_target_.*CustomSampling" config.yaml
   # Should show: data_v3.CustomSamplingASEDataModuleV3
   ```

2. Check logs for precomputation:
   ```
   "Precomputing indices for X structures from data/..."
   "Precomputed Y total indices with Z rare structures replicated Nx"
   ```

### Import Errors

If you get import errors after updating:
```python
# Old imports that no longer work
from forge.workflows.allegro_utils.data import CustomSamplingASEDataModule  # V1
from forge.workflows.allegro_utils.data_v2 import CustomSamplingASEDataModuleV2  # V2

# Use the new standard import
from forge.workflows.allegro_utils import CustomSamplingASEDataModule  # V3 alias
```

### Performance Differences

V3 may have slightly different performance characteristics:
- **Initialization**: Slightly slower (reads file to count structures)
- **Training**: Same or better (proper distribution across GPUs)
- **Memory**: Similar (indices are lightweight)

## Example Migration

### Before (V2):
```python
# run_experiment.py
from forge.workflows.db_to_allegro import prepare_allegro_job

prepare_allegro_job(
    db_manager=db,
    job_name="rare_sampling_v2",
    sampler="rare_weighted",
    sampler_params={"replica": 5, "rare_idx": rare_indices},
    sampler_implementation="v2",
    ...
)
```

### After (V3):
```python
# run_experiment.py
from forge.workflows.db_to_allegro import prepare_allegro_job

prepare_allegro_job(
    db_manager=db,
    job_name="rare_sampling",  # No need for version suffix
    sampler="rare_weighted",
    sampler_params={"replica": 5, "rare_idx": rare_indices},
    # sampler_implementation removed
    ...
)
```

## Cleanup

After migration, you can safely remove old files:
- `forge/workflows/allegro_utils/data.py` (V1)
- `forge/workflows/allegro_utils/data_v2.py` (V2)

These are no longer needed as V3 (`data_v3.py`) is now the standard implementation. 