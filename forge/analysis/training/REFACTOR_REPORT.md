# Analysis Training Module Refactor Report

## Overview

This document summarizes the comprehensive refactoring of the `forge/analysis/training` module. The refactor aimed to modularize the training analysis code, implement auto-scoring of structures, and integrate with the forge database and export workflows.

## Architecture Changes

### Previous Structure
The original code consisted of several standalone scripts with mixed responsibilities:
- `difficulty.py` - Monolithic script with all difficulty analysis
- `geometry_sanity.py` - CLI script for geometry checks
- `identify_defect_hotspot.py` - Mixed analysis and visualization
- `evaluator.py` - Basic calculator wrapper
- `metrics.py`, `spatial.py`, `plotting.py` - Utility modules

### New Modular Structure
```
analysis/training/
├── core/                    # Core components
│   ├── analyser.py         # Main ErrorAnalyser facade
│   ├── evaluator.py        # Enhanced calculator evaluation
│   └── results.py          # Structured results storage
├── metrics/                 # Metrics system
│   ├── base.py             # Metric protocol/interface
│   ├── registry.py         # Plugin-style metric registry
│   ├── statistical.py      # Standard error metrics
│   ├── losses.py           # Custom loss functions
│   └── spatial.py          # Spatial analysis metrics
├── difficulty/              # Difficulty analysis
│   ├── scoring.py          # Composite difficulty scoring
│   └── ensemble.py         # Ensemble variance/agreement
├── utils/                   # Utilities
│   ├── geometry.py         # Geometry sanity checks
│   ├── io.py              # Import/export functions
│   └── plotting.py         # Visualization tools
└── examples/               # Example workflows
    └── run_full_workflow.py
```

## Key Improvements

### 1. **Modular Design**
- **Separation of Concerns**: Each module has a single, clear responsibility
- **Reusable Components**: All functionality extracted into importable functions/classes
- **Clean API**: Simple imports from `forge.analysis.training`

### 2. **Metric Registry System**
- **Plugin Architecture**: Register custom metrics dynamically
- **Consistent Interface**: All metrics follow the same protocol
- **Built-in Metrics**: Comprehensive set of default metrics
- **Extensibility**: Easy to add new metrics without modifying core code

### 3. **Enhanced Error Analysis**
- **ErrorAnalyser Class**: Single entry point for all analysis workflows
- **Batch Processing**: Efficient processing of large structure sets
- **Ensemble Support**: Built-in support for multiple calculators
- **Comprehensive Metrics**: Statistical, spatial, and custom loss metrics

### 4. **Structured Results**
- **AnalysisResults Class**: Organized storage with convenient access methods
- **DataFrame Integration**: Structure and atom-level metrics in pandas DataFrames
- **Filtering/Ranking**: Easy identification of problematic structures
- **Export Options**: Multiple formats for different use cases

### 5. **Database Integration**
- **Native Support**: Works directly with forge DatabaseManager
- **Batch Operations**: Efficient loading of structures and calculations
- **Metadata Preservation**: Maintains structure IDs and properties

## New Features

### 1. **Composite Difficulty Scoring**
```python
# Combines multiple error characteristics
difficulty_score = w1 * heavy_tail + w2 * spatial + w3 * ensemble
```
- Heavy-tail component (kurtosis, extreme quantiles)
- Spatial clustering component (Moran's I, DBSCAN)
- Ensemble uncertainty component

### 2. **Spatial Error Analysis**
- **Moran's I**: Global spatial autocorrelation
- **Local Moran's I**: Per-atom spatial correlation
- **DBSCAN Clustering**: Identify error hotspots
- **Integrated Analysis**: Single function for all spatial metrics

### 3. **Custom Loss Metrics**
- **Tail MSE**: Focus on outliers
- **Focal MSE**: Adaptive weighting of errors
- **Force Angle Loss**: Directional error analysis
- **Tail Huber**: Robust loss for outliers

### 4. **Categorization System**
Automatic classification of difficulty causes:
- Heavy-tailed errors (outliers)
- Spatial clustering (error hotspots)
- High ensemble disagreement
- Geometry issues
- Multiple simultaneous issues

### 5. **Visualization & Export**
- **Extended XYZ**: Structure files with error data for Ovito
- **Interactive 3D**: NGLView integration for Jupyter
- **Statistical Plots**: Histograms, Q-Q plots, correlations
- **Markdown Reports**: Human-readable summaries

## API Examples

### Basic Usage
```python
from forge.core.database import DatabaseManager
from forge.analysis.training import ErrorAnalyser

# Initialize
db = DatabaseManager()
analyser = ErrorAnalyser(db, calculators=[calc1, calc2])

# Run analysis
results = analyser.run(structure_ids=[1, 2, 3, 4, 5])

# Get difficult structures
difficult_ids = results.get_difficult_structures(top_n=10)

# Save results
analyser.save("analysis_output/")
```

### Custom Metrics
```python
from forge.analysis.training import register_metric

@register_metric("my_metric")
def custom_metric(pred, ref, **kwargs):
    return {"my_value": calculate_something(pred, ref)}

# Use in analysis
results = analyser.run(structure_ids, metrics=["force_stats", "my_metric"])
```

### Integration with Training
```python
# Filter structures for retraining
high_error = results.filter_by_score('force_rmse_metric', 0.1)
clustered = results.filter_by_score('n_error_clusters_metric', 1)

# Export for training
from forge.workflows.db_to_allegro import prepare_allegro_job
prepare_allegro_job(db, "retrain", structure_ids=high_error)
```

## Testing

Comprehensive test suite added:
- **Unit Tests**: Each component tested in isolation
- **Integration Tests**: Full workflow testing
- **Mock Database**: Tests run without real database
- **Fixtures**: Reusable test data
- **Coverage**: All major functionality tested

Test files:
- `test_metrics.py`: Metric calculations and registry
- `test_core.py`: ErrorAnalyser, Evaluator, Results
- `test_difficulty.py`: Difficulty scoring and categorization
- `test_utils.py`: Geometry checks and I/O functions

## Documentation

- **Module Docstrings**: Clear purpose for each module
- **NumPy Style**: Comprehensive parameter documentation
- **Type Hints**: Full typing for better IDE support
- **README.md**: Complete usage guide with examples
- **Example Notebook**: Step-by-step workflow demonstration

## Performance Considerations

- **Batch Processing**: Configurable batch size for memory management
- **Lazy Evaluation**: Results cached to avoid recomputation
- **Parallel Capability**: Structure processing can be parallelized
- **Efficient Storage**: Pickle format for large results

## Migration Guide

### From Old Scripts
```python
# Old way (CLI script)
python identify_defect_hotspot.py --structure-ids 1,2,3

# New way (Python API)
from forge.analysis.training import ErrorAnalyser
analyser = ErrorAnalyser(db, calculators)
results = analyser.run([1, 2, 3])
```

### From Individual Functions
```python
# Old way
from difficulty import rank_structures_by_difficulty_multi_loss
from identify_defect_hotspot import identify_defect_prone_structures

# New way
from forge.analysis.training import ErrorAnalyser
# All functionality integrated in ErrorAnalyser.run()
```

## Future Enhancements

1. **Parallel Processing**: Add multiprocessing support for large datasets
2. **GPU Acceleration**: Spatial analysis on GPU for large structures
3. **Active Learning**: Automated selection strategies
4. **Real-time Analysis**: Streaming analysis during training
5. **Web Dashboard**: Interactive visualization of results

## Conclusion

The refactored module provides a clean, extensible, and well-tested framework for analyzing MLIP errors and identifying problematic structures. The modular design allows easy customization while the high-level API simplifies common workflows. Integration with the forge ecosystem enables seamless active learning pipelines. 