# Allegro Utils Design Document

## Architecture Overview

The `allegro_utils` module follows a modular, extensible design that integrates seamlessly with NequIP/Allegro's Hydra-based configuration system. Each component is designed to be independently usable and composable.

## Core Design Principles

1. **Hydra Integration**: All components use `_target_` for instantiation via Hydra
2. **Distributed Compatibility**: V3 implementation ensures proper function in multi-GPU setups
3. **Modularity**: Each component (sampler, loss, metric) can be used independently
4. **Extensibility**: New components can be added without modifying existing code

## Component Architecture

### 1. Data Module Layer

The `CustomSamplingASEDataModuleV3` acts as the integration point between custom sampling logic and NequIP's data pipeline:

```
┌─────────────────────────────────────┐
│   CustomSamplingASEDataModuleV3     │
├─────────────────────────────────────┤
│ - Precomputes indices at init       │
│ - Handles distributed training      │
│ - Integrates custom samplers        │
└───────────────┬─────────────────────┘
                │
                ├── Reads XYZ file directly
                ├── Counts structures
                └── Creates replicated indices
```

### 2. Sampler Layer

Samplers implement PyTorch's `Sampler` interface:

```
┌─────────────────────────────────────┐
│      RareWeightedSampler            │
├─────────────────────────────────────┤
│ - Oversamples rare configurations   │
│ - Optional force-based weighting    │
│ - Returns sampling indices          │
└─────────────────────────────────────┘
```

### 3. Loss Function Layer

Loss functions extend NequIP's base metrics:

```
┌─────────────────────────────────────┐
│         Loss Functions              │
├─────────────────────────────────────┤
│ FocalMSELoss                        │
│ - Down-weights easy samples         │
│                                     │
│ WeightedMSELoss                     │
│ - Applies per-sample weights        │
│ - Alternative to oversampling       │
└─────────────────────────────────────┘
```

### 4. Metrics Layer

Custom validation metrics for model evaluation:

```
┌─────────────────────────────────────┐
│           Metrics                   │
├─────────────────────────────────────┤
│ TailMSE                             │
│ - Evaluates high-error samples      │
│ - Configurable percentile           │
└─────────────────────────────────────┘
```

## Integration Flow

1. **Configuration Generation** (`db_to_allegro.py`):
   ```python
   config['data']['_target_'] = "forge.workflows.allegro_utils.data_v3.CustomSamplingASEDataModuleV3"
   config['data']['sampler_config'] = {...}
   ```

2. **Hydra Instantiation**:
   - Hydra reads `config.yaml`
   - Instantiates components using `_target_`
   - Passes parameters from config

3. **Training Loop**:
   ```
   DataModule → Custom Sampler → DataLoader → Model → Loss Function
                                                ↓
                                          Metrics → Validation
   ```

## Adding New Components

### New Sampler Example

```python
# 1. Create sampler in samplers.py
class StratifiedSampler(Sampler):
    def __init__(self, dataset_size, strata_indices, **kwargs):
        self.strata_indices = strata_indices
        # Implementation...
    
    def __iter__(self):
        # Return stratified indices
    
    def __len__(self):
        # Return total samples

# 2. Register in db_to_allegro.py
if sampler == 'stratified':
    sampler_config = {
        "_target_": "forge.workflows.allegro_utils.samplers.StratifiedSampler",
        **sampler_params
    }
```

### New Loss Function Example

```python
# 1. Create loss in custom_losses.py
class AdaptiveLoss(torch.nn.Module):
    def __init__(self, adaptation_rate=0.1):
        super().__init__()
        self.adaptation_rate = adaptation_rate
    
    def forward(self, pred, target):
        # Adaptive loss computation

# 2. Add to loss_function_map in db_to_allegro.py
loss_function_map = {
    # ... existing mappings ...
    "adaptive": "forge.workflows.allegro_utils.custom_losses.AdaptiveLoss",
}
```

## V3 Implementation Details

The V3 implementation solves distributed training issues by:

1. **Precomputation**: Reads training file and creates indices before any distribution
2. **Index Replication**: Handles rare sample replication at the index level
3. **Distributed Compatibility**: Works with PyTorch Lightning's DistributedSampler

```python
def __init__(self, ...):
    # Count structures from file
    with open(self.train_file_path, 'r') as f:
        n_structures = sum(1 for line in f if line.strip() == 'Properties=...')
    
    # Precompute all indices with replication
    self.precomputed_indices = self._precompute_indices(n_structures)

def train_dataloader(self):
    # Use precomputed indices
    sampler = PrecomputedIndicesSampler(self.precomputed_indices)
    return DataLoader(..., sampler=sampler)
```

## Best Practices

1. **Configuration**: Always use `_target_` for Hydra compatibility
2. **Logging**: Use module-specific loggers for debugging
3. **Type Hints**: Include type hints for all public methods
4. **Documentation**: Document expected parameters and behavior
5. **Testing**: Test components individually and in integration

## Future Enhancements

1. **Dynamic Sampling**: Adjust sampling strategy during training
2. **Multi-criteria Sampling**: Sample based on multiple properties
3. **Curriculum Learning**: Progressive sampling strategies
4. **Adaptive Losses**: Loss functions that adapt based on training progress 