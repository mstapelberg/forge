# Allegro Utils Module

This module provides custom utilities for enhancing Allegro/NequIP training workflows, particularly focused on handling rare sample emphasis in distributed training environments.

## Overview

The module includes:
- **Custom Data Modules**: For integrating custom sampling strategies with NequIP's data pipeline
- **Samplers**: For oversampling rare configurations
- **Loss Functions**: Custom loss implementations including focal loss and weighted loss
- **Metrics**: Additional validation metrics like tail MSE
- **Callbacks**: Training callbacks for curriculum learning and gradient monitoring

## Units policy for stress metrics/losses

Many datasets store virial stress in eV/Å^3, while physics-based thresholds and reporting are often in GPa. To ensure numerical stability and fair comparisons to legacy baselines:

- Internally, config-aware stress metrics compute in the GPa domain.
- By default, the returned per-sample loss values are converted back to match the input units (eV/Å^3 if inputs are eV/Å^3). This keeps your existing stress loss coefficients meaningful.
- You can control this behavior via `loss_return_units` in `ConfigAwareStressHuber`:
  - `match_inputs` (default): return loss in the same units as inputs
  - `eVa3`: force return in eV/Å^3
  - `GPa`: force return in GPa

Additionally, `PressureMAE` and `VonMisesMAE` expose differentiable per-batch values (`last_batch_value`) so they can optionally be included in the optimized loss by setting a non-zero `coeff` in the metrics manager.

## Key Components

### 1. Data Module (`data_v3.py`)

The `CustomSamplingASEDataModuleV3` is our primary data module that handles custom sampling in distributed training:

```python
from forge.workflows.allegro_utils.data_v3 import CustomSamplingASEDataModuleV3

# In your config generation:
config['data']['_target_'] = "forge.workflows.allegro_utils.data_v3.CustomSamplingASEDataModuleV3"
config['data']['sampler_config'] = {
    '_target_': 'forge.workflows.allegro_utils.samplers.RareWeightedSampler',
    'replica': 5,  # Replicate rare samples 5x
    'alpha': 0.25,  # Force-based weighting factor
    'rare_idx': [...]  # List of rare sample indices
}
```

**Key Features:**
- Precomputes sampling indices before dataset distribution
- Works correctly with PyTorch Lightning's distributed training
- Handles both oversampling and force-based weighting

### 2. Samplers (`samplers.py`)

The `RareWeightedSampler` implements:
- Oversampling of specified rare configurations
- Optional force-based weighting (currently uniform in V3 for efficiency)

### 3. Loss Functions

#### Focal Loss (`custom_losses.py`)
```python
loss_function: "focal"
loss_params: {"gamma": 2.0, "alpha": 0.25}
```

#### Weighted Loss (`weighted_loss.py`)
Alternative to oversampling - applies sample weights during loss computation:
```python
loss_function: "weighted_mse"
loss_params: {
    "rare_indices": [...],
    "rare_weight": 5.0,
    "force_weight_alpha": 0.25
}
```

### 4. Custom Metrics (`custom_metrics.py`)

Additional validation metrics like `TailMSE` for evaluating performance on high-error samples.

## Usage Examples

### Basic Rare Sampling
```python
prepare_allegro_job(
    db_manager=db,
    job_name="experiment_rare_sampling",
    sampler="rare_weighted",
    sampler_params={
        "replica": 5,
        "alpha": 0.25,
        "rare_idx": rare_structure_indices
    },
    # Other parameters...
)
```

### Combined with Custom Loss
```python
prepare_allegro_job(
    db_manager=db,
    job_name="experiment_focal_rare",
    sampler="rare_weighted",
    sampler_params={"replica": 3, "rare_idx": rare_indices},
    loss_function="focal",
    loss_params={"gamma": 2.0},
    # Other parameters...
)
```

### Using Weighted Loss (No Oversampling)
```python
prepare_allegro_job(
    db_manager=db,
    job_name="experiment_weighted",
    # No sampler configuration
    loss_function="weighted_mse",
    loss_params={
        "rare_indices": rare_indices,
        "rare_weight": 5.0
    },
    # Other parameters...
)
```

## Implementation Details

### Why V3?

The V3 implementation (`data_v3.py`) solves a critical issue with distributed training:
- V1/V2 attempted to create samplers after dataset distribution, seeing only 1 sample per GPU
- V3 precomputes all indices before any distribution happens
- This ensures each GPU gets the correct portion of the oversampled dataset

### Distributed Training Flow

1. **Initialization**: Datamodule reads training file and precomputes indices with replication
2. **Distribution**: PyTorch Lightning distributes the precomputed indices across GPUs
3. **Training**: Each GPU processes its portion of the replicated dataset

## Extending the Framework

### Adding New Samplers

1. Create a new sampler class inheriting from `torch.utils.data.Sampler`
2. Implement `__init__`, `__iter__`, and `__len__` methods
3. Register in `db_to_allegro.py`:

```python
if sampler == 'your_new_sampler':
    sampler_config = {
        "_target_": "forge.workflows.allegro_utils.samplers.YourNewSampler",
        **sampler_params
    }
```

### Adding New Loss Functions

1. Implement in `custom_losses.py` or create a new file
2. Add to the loss function map in `db_to_allegro.py`:

```python
loss_function_map = {
    # ... existing mappings ...
    "your_loss": "forge.workflows.allegro_utils.custom_losses.YourLoss",
}
```

### Adding New Metrics

1. Implement in `custom_metrics.py`
2. Add to validation metrics in `db_to_allegro.py`:

```python
if extra_val_metrics and 'your_metric' in extra_val_metrics:
    val_metrics.append({
        "name": f"forces_your_metric",
        "field": "forces",
        "metric": {"_target_": "forge.workflows.allegro_utils.custom_metrics.YourMetric"}
    })
```

## Troubleshooting

### Only Getting 1 Batch
- Ensure you're using V3 implementation (check config for `CustomSamplingASEDataModuleV3`)
- Check logs for "Precomputed X total indices" message
- Verify distributed training is working (4 processes should start)

### Memory Issues
- Reduce `replica` parameter if dataset becomes too large
- Consider using weighted loss instead of oversampling for very large datasets

### Custom Sampler Not Applied
- Check that `sampler_config` is in the data section of your config
- Verify the sampler target path is correct
- Look for "Creating dataloader with precomputed X indices" in logs

## Future Enhancements

1. **Dynamic Replication**: Adjust replication factor based on rarity score
2. **Force-aware Precomputation**: Include force norms in V3 precomputation
3. **Multi-criteria Sampling**: Sample based on multiple properties (energy, forces, stress)
4. **Adaptive Sampling**: Change sampling strategy during training based on loss 