# Forge Analysis Training Module

A comprehensive toolkit for analyzing machine learning interatomic potential (MLIP) errors, identifying problematic structures, and facilitating active learning workflows.

## Overview

This module helps identify structures where MLIPs perform poorly by:
- Computing comprehensive error metrics (RMSE, Kurtosis, Gini coefficient)
- Analyzing spatial error patterns (Moran's I, DBSCAN clustering)
- Calculating ensemble uncertainty and disagreement
- Scoring structure difficulty for targeted active learning
- Providing visualization and export capabilities

## Key Features

###  Error Analysis
- **Statistical Metrics**: RMSE, MAE, quantiles, heavy-tail statistics
- **Custom Loss Functions**: Tail MSE, Focal MSE, Force angle errors
- **Spatial Analysis**: Detect error clustering and hotspots
- **Ensemble Analysis**: Quantify model uncertainty and disagreement

###  Difficulty Scoring
- Composite difficulty scores combining multiple error characteristics
- Automatic categorization of error causes (physics vs. geometry issues)
- Ranking of structures for active learning prioritization

###  Extensibility
- Plugin-style metric registry for custom metrics
- Support for any ASE-compatible calculator
- Flexible batch processing with configurable parameters

###  Visualization & Export
- Interactive 3D structure visualization with error coloring
- Statistical plots (histograms, Q-Q plots, correlations)
- Export to multiple formats (XYZ, CSV, JSON, Parquet)
- Automated report generation

## Installation

```bash
# Core dependencies
pip install numpy pandas scipy scikit-learn ase tqdm
pip install -e /path/to/forge/root

# Optional dependencies for full functionality
pip install matplotlib nglview pysal  # For plotting and spatial analysis
pip install jupyter notebook  # For example notebooks
```

## Quick Start

```python
from forge.core.database import DatabaseManager
from forge.analysis.training import ErrorAnalyser

# Initialize with database and calculators
db = DatabaseManager()
analyser = ErrorAnalyser(db, calculators=[calc1, calc2])

# Run analysis on structures
results = analyser.run(
    structure_ids=[1, 2, 3, 4, 5],
    batch_size=32,
    metrics=["force_stats", "tail_mse", "focal_mse"]
)

# Get most difficult structures
difficult_ids = results.get_difficult_structures(top_n=10)
print(f"Most difficult structures: {difficult_ids}")

# Generate report
report = results.get_structure_report(difficult_ids[0])
print(report)

# Save results
analyser.save("analysis_output/")

# Export for visualization
from forge.analysis.training.utils import export_structures_to_extxyz
atoms_list = [results.results_cache[sid]['atoms'] for sid in difficult_ids]
export_structures_to_extxyz(atoms_list, results, "difficult_structures.extxyz")
```

## Module Structure

```
analysis/training/
├── core/
│   ├── analyser.py      # Main ErrorAnalyser class
│   ├── evaluator.py     # Calculator evaluation
│   └── results.py       # AnalysisResults storage
├── metrics/
│   ├── base.py          # Metric protocol/interface
│   ├── statistical.py   # Standard error metrics
│   ├── losses.py        # Custom loss functions
│   ├── spatial.py       # Spatial analysis (Moran's I, DBSCAN)
│   └── registry.py      # Metric registration system
├── difficulty/
│   ├── scoring.py       # Difficulty score calculation
│   └── ensemble.py      # Ensemble variance metrics
└── utils/
    ├── geometry.py      # Structure sanity checks
    ├── io.py           # Import/export utilities
    └── plotting.py      # Visualization functions
```

## Custom Metrics

Register custom metrics for your specific analysis needs:

```python
from forge.analysis.training import register_metric
import numpy as np

@register_metric("my_custom_metric")
def custom_error_metric(pred, ref, threshold=0.1):
    """Example custom metric."""
    errors = np.abs(pred - ref)
    return {
        "above_threshold_fraction": np.mean(errors > threshold),
        "median_error": np.median(errors)
    }

# Use in analysis
results = analyser.run(
    structure_ids=[1, 2, 3],
    metrics=["force_stats", "my_custom_metric"]
)
```

## Spatial Analysis

Detect spatial clustering of errors:

```python
from forge.analysis.training.metrics import analyze_spatial_patterns

# Get force errors for a structure
errors = results.results_cache[structure_id]['force_error_magnitudes']
atoms = results.results_cache[structure_id]['atoms']

# Analyze spatial patterns
spatial_results = analyze_spatial_patterns(
    atoms, errors,
    k=12,  # neighbors for Moran's I
    eps=2.5,  # DBSCAN radius
    min_samples=3
)

print(f"Moran's I: {spatial_results['morans_i_global_metric']:.3f}")
print(f"Error clusters found: {spatial_results['n_error_clusters_metric']}")
```

## Integration with Training Pipelines

Use analysis results to select structures for retraining:

```python
# Get structures with specific characteristics
high_kurtosis = results.filter_by_score('force_kurtosis_metric', 5.0)
clustered_errors = results.filter_by_score('n_error_clusters_metric', 1)

# Combine criteria
difficult_physics = set(high_kurtosis) & set(clustered_errors)

# Export for MLIP training
from forge.workflows.db_to_allegro import prepare_allegro_job
prepare_allegro_job(
    db, "retrain_job",
    structure_ids=list(difficult_physics),
    # ... other parameters
)
```

## Configuration Options

### Analysis Parameters
- `batch_size`: Number of structures to process at once (default: 32)
- `spatial_k`: Neighbors for Moran's I calculation (default: 12)
- `dbscan_eps`: Clustering distance threshold (default: 2.5 Å)
- `check_geometry_sanity`: Validate structure geometry (default: True)

### Difficulty Weights
Customize importance of different error characteristics:

```python
difficulty_weights = {
    'w_heavy_tail': 2.0,  # Emphasize outliers
    'w_spatial': 1.0,     # Spatial clustering
    'w_ensemble': 1.5     # Model uncertainty
}

results = analyser.run(
    structure_ids=ids,
    difficulty_weights=difficulty_weights
)
```

## Output Formats

### AnalysisResults Object
- `structure_metrics`: DataFrame with per-structure metrics
- `atom_metrics`: DataFrame with per-atom error data
- `results_cache`: Dictionary of intermediate results for plotting

### Export Options
- **Extended XYZ**: Structure files with error data for Ovito visualization
- **CSV/Parquet**: Tabular data for further analysis
- **JSON**: Complete results including metadata
- **Markdown**: Human-readable analysis reports

## Performance Considerations

- Use appropriate `batch_size` based on available memory
- Spatial analysis scales as O(n²) for n atoms per structure
- Consider disabling geometry checks for pre-validated datasets
- PyTorch calculators should use appropriate device placement

## Troubleshooting

### Missing Dependencies
- Install `pysal` for spatial statistics: `pip install pysal`
- Install `matplotlib` for plotting: `pip install matplotlib scipy`
- Install `nglview` for 3D visualization: `pip install nglview`

### Memory Issues
- Reduce `batch_size` parameter
- Process structures in smaller chunks
- Use `save()` and `load()` for checkpoint/resume

### Calculator Compatibility
- Ensure calculators implement ASE interface
- Check calculator device placement matches system
- Verify reference calculations exist in database

## Citation

If you use this module in your research, please cite the Forge framework:
```
[Citation information to be added]
```

## Contributing

Contributions are welcome! Please:
1. Add tests for new features
2. Follow NumPy docstring conventions
3. Run linting with `ruff`
4. Update documentation as needed 