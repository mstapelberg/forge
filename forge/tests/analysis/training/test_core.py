"""Tests for the core analysis components."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from ase import Atoms
from ase.build import molecule, bulk
from ase.calculators.emt import EMT

from forge.analysis.training.core import (
    Evaluator, AnalysisResults, ErrorAnalyser
)


class TestEvaluator:
    """Test the Evaluator class."""
    
    def test_single_calculator(self):
        """Test evaluation with single calculator."""
        calc = EMT()
        evaluator = Evaluator(calc)
        
        assert evaluator.n_calculators == 1
        assert len(evaluator.calculator_names) == 1
        
        # Test evaluation
        atoms = molecule('H2O')
        results = evaluator.evaluate([atoms])
        
        assert len(results) == 1
        assert 'calc_0' in results[0]
        assert 'energy' in results[0]['calc_0']
        assert 'forces' in results[0]['calc_0']
    
    def test_ensemble_calculators(self):
        """Test evaluation with multiple calculators."""
        calcs = [EMT(), EMT(), EMT()]
        evaluator = Evaluator(calcs)
        
        assert evaluator.n_calculators == 3
        assert len(evaluator.calculator_names) == 3
        
        atoms = molecule('NH3')
        results = evaluator.evaluate([atoms])
        
        assert len(results[0]) == 3
        for i in range(3):
            assert f'calc_{i}' in results[0]
    
    def test_calculator_validation(self):
        """Test calculator validation."""
        # Mock calculator without required methods
        bad_calc = Mock(spec=[])
        
        with pytest.raises(AttributeError):
            Evaluator(bad_calc)
    
    def test_evaluation_error_handling(self):
        """Test handling of evaluation errors."""
        # Mock calculator that raises error
        calc = Mock()
        calc.get_potential_energy = Mock(side_effect=RuntimeError("Calc failed"))
        calc.get_forces = Mock(return_value=np.zeros((3, 3)))
        calc.get_stress = Mock(return_value=np.zeros(6))
        
        evaluator = Evaluator(calc)
        atoms = molecule('H2O')
        
        results = evaluator.evaluate([atoms])
        assert results[0]['calc_0'] is None  # Should return None on error


class TestAnalysisResults:
    """Test the AnalysisResults class."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample analysis results."""
        # Structure metrics
        struct_data = {
            'structure_id': [1, 2, 3],
            'difficulty_metric': [0.5, 0.8, 0.3],
            'force_rmse_metric': [0.1, 0.3, 0.05],
            'force_kurtosis_metric': [3.5, 8.2, 2.9],
            'n_atoms': [10, 20, 15]
        }
        
        # Atom metrics
        atom_data = {
            'structure_id': [1, 1, 2, 2, 3],
            'atom_index': [0, 1, 0, 1, 0],
            'force_error_mag': [0.1, 0.2, 0.3, 0.4, 0.05]
        }
        
        return AnalysisResults(
            structure_metrics=pd.DataFrame(struct_data),
            atom_metrics=pd.DataFrame(atom_data),
            metadata={'test': True},
            results_cache={}
        )
    
    def test_get_difficult_structures(self, sample_results):
        """Test getting difficult structures."""
        difficult = sample_results.get_difficult_structures(top_n=2)
        
        assert len(difficult) == 2
        assert difficult[0] == 2  # Highest difficulty
        assert difficult[1] == 1  # Second highest
    
    def test_filter_by_score(self, sample_results):
        """Test filtering by metric score."""
        high_rmse = sample_results.filter_by_score('force_rmse_metric', 0.2)
        
        assert len(high_rmse) == 1
        assert high_rmse[0] == 2
    
    def test_summary_statistics(self, sample_results):
        """Test summary statistics calculation."""
        summary = sample_results.summary_statistics()
        
        assert 'n_structures' in summary
        assert summary['n_structures'] == 3
        assert 'n_atoms_total' in summary
        assert summary['n_atoms_total'] == 45
        
        assert 'metrics' in summary
        assert 'difficulty_metric' in summary['metrics']
        assert 'mean' in summary['metrics']['difficulty_metric']
    
    def test_get_structure_report(self, sample_results):
        """Test individual structure report."""
        report = sample_results.get_structure_report(1)
        
        assert 'structure_metrics' in report
        assert report['structure_metrics']['structure_id'] == 1
        
        assert 'atom_summary' in report
        assert report['atom_summary']['n_atoms'] == 2
        assert report['atom_summary']['mean_force_error'] == 0.15
    
    def test_merge_with(self, sample_results):
        """Test merging with custom data."""
        custom_df = pd.DataFrame({
            'structure_id': [1, 2, 3],
            'custom_metric': [10, 20, 30]
        })
        
        merged = sample_results.merge_with(custom_df)
        
        assert 'custom_metric' in merged.columns
        assert len(merged) == 3


class TestErrorAnalyser:
    """Test the ErrorAnalyser class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database manager."""
        db = Mock()
        
        # Create test atoms with proper structure
        atoms1 = bulk('Cu', 'fcc', a=3.6)
        atoms1.info = {
            'structure_id': 1,
            'energy': -10.0,
            'config_type': 'bulk'
        }
        atoms1.new_array('forces', np.random.randn(len(atoms1), 3) * 0.1)
        
        atoms2 = molecule('H2O')
        atoms2.info = {
            'structure_id': 2,
            'energy': -5.0,
            'config_type': 'molecule'
        }
        atoms2.new_array('forces', np.random.randn(len(atoms2), 3) * 0.1)
        
        # Mock database methods
        db.get_batch_atoms_with_calculation = Mock(
            return_value=[atoms1, atoms2]
        )
        
        return db
    
    def test_initialization(self, mock_db):
        """Test analyser initialization."""
        calc = EMT()
        analyser = ErrorAnalyser(mock_db, calc)
        
        assert analyser.db is mock_db
        assert analyser.evaluator is not None
        assert analyser.evaluator.n_calculators == 1
    
    def test_register_custom_metric(self, mock_db):
        """Test registering custom metrics."""
        analyser = ErrorAnalyser(mock_db, None)
        
        def custom_metric(pred, ref):
            return {"custom_value": 42}
        
        analyser.register_metric("custom", custom_metric)
        
        # Check it's registered
        assert "custom" in analyser.metric_registry.list_metrics()
    
    def test_run_analysis_basic(self, mock_db):
        """Test basic analysis run."""
        calc = EMT()
        analyser = ErrorAnalyser(mock_db, calc)
        
        results = analyser.run(
            structure_ids=[1, 2],
            batch_size=2,
            metrics=["force_stats"],
            check_geometry_sanity=False
        )
        
        assert isinstance(results, AnalysisResults)
        assert len(results.structure_metrics) > 0
        assert 'structure_id' in results.structure_metrics.columns
        assert 'difficulty_metric' in results.structure_metrics.columns
    
    def test_run_without_calculator(self, mock_db):
        """Test running without calculator (reference only)."""
        analyser = ErrorAnalyser(mock_db, None)
        
        # Should handle gracefully
        results = analyser.run(
            structure_ids=[1, 2],
            check_geometry_sanity=False
        )
        
        # Should still produce some results
        assert isinstance(results, AnalysisResults)
    
    def test_save_and_load(self, mock_db, tmp_path):
        """Test saving and loading results."""
        calc = EMT()
        analyser = ErrorAnalyser(mock_db, calc)
        
        # Run analysis
        results = analyser.run([1, 2], check_geometry_sanity=False)
        
        # Save
        save_dir = tmp_path / "test_save"
        analyser.save(save_dir)
        
        assert (save_dir / "structure_metrics.csv").exists()
        assert (save_dir / "atom_metrics.csv").exists()
        
        # Load
        loaded_analyser = ErrorAnalyser.load(save_dir, mock_db, calc)
        assert loaded_analyser.results is not None
        assert len(loaded_analyser.results.structure_metrics) == len(results.structure_metrics)


class TestIntegration:
    """Integration tests with real calculators."""
    
    def test_full_workflow(self, mock_db):
        """Test complete analysis workflow."""
        # Initialize
        calc = EMT()
        analyser = ErrorAnalyser(mock_db, [calc, calc])  # Ensemble
        
        # Register custom metric
        @analyser.register_metric("test_metric")
        def test_metric(pred, ref):
            return {"test_value": np.mean(np.abs(pred - ref))}
        
        # Run analysis
        results = analyser.run(
            structure_ids=[1, 2],
            metrics=["force_stats", "test_metric"],
            spatial_k=4,
            dbscan_eps=3.0,
            check_geometry_sanity=True
        )
        
        # Verify results
        assert len(results.structure_metrics) > 0
        assert "test_value" in results.structure_metrics.columns
        assert "geometry_valid" in results.structure_metrics.columns
        assert "ensemble_force_std_metric" in results.structure_metrics.columns
        
        # Get difficult structures
        difficult = results.get_difficult_structures(top_n=1)
        assert len(difficult) == 1
        
        # Get report
        report = results.get_structure_report(difficult[0])
        assert 'structure_metrics' in report
        assert 'atom_summary' in report 