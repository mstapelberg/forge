"""Tests for the metrics module."""
import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk

from forge.analysis.training.metrics import (
    MetricRegistry, register_metric, get_metric, calculate_metric,
    ForceErrorStats, EnergyErrorStats, StressErrorStats,
    gini_coefficient, gini_metric, kurtosis_metric,
    TailMSE, TailHuberLoss, FocalMSELoss, ForceAngleLoss,
    MoransI, ErrorClustering, analyze_spatial_patterns
)


class TestMetricRegistry:
    """Test the metric registry system."""
    
    def test_register_and_get_metric(self):
        """Test registering and retrieving metrics."""
        registry = MetricRegistry()
        
        # Register a simple metric
        @register_metric("test_metric", registry=registry)
        def test_metric(pred, ref):
            return {"diff": np.mean(pred - ref)}
        
        # Retrieve metric
        metric = registry.get("test_metric")
        assert metric is not None
        
        # Calculate
        result = registry.calculate("test_metric", np.array([1, 2]), np.array([0, 1]))
        assert "diff" in result
        assert result["diff"] == 1.0
    
    def test_list_metrics(self):
        """Test listing registered metrics."""
        registry = MetricRegistry()
        registry.register("metric1", lambda p, r: {"val": 1})
        registry.register("metric2", lambda p, r: {"val": 2})
        
        metrics = registry.list_metrics()
        assert "metric1" in metrics
        assert "metric2" in metrics


class TestStatisticalMetrics:
    """Test statistical metric calculations."""
    
    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        # Perfect equality
        equal = np.ones(100)
        assert abs(gini_coefficient(equal)) < 1e-6
        
        # Perfect inequality
        unequal = np.zeros(100)
        unequal[0] = 100
        assert gini_coefficient(unequal) > 0.9
        
        # Normal case
        normal = np.random.randn(100)**2
        gini = gini_coefficient(normal)
        assert 0 < gini < 1
    
    def test_force_error_stats(self):
        """Test force error statistics."""
        metric = ForceErrorStats()
        
        # Create dummy forces
        pred = np.random.randn(10, 3)
        ref = pred + 0.1 * np.random.randn(10, 3)
        
        result = metric.calculate(pred, ref)
        
        # Check all expected keys
        assert "force_rmse_metric" in result
        assert "force_mae_metric" in result
        assert "force_max_metric" in result
        assert "force_kurtosis_metric" in result
        assert "force_gini_metric" in result
        
        # Sanity checks
        assert result["force_rmse_metric"] >= 0
        assert result["force_mae_metric"] >= 0
        assert result["force_max_metric"] >= result["force_mae_metric"]
    
    def test_energy_error_stats(self):
        """Test energy error statistics."""
        metric = EnergyErrorStats()
        
        pred = 10.5
        ref = 10.0
        n_atoms = 20
        
        result = metric.calculate(pred, ref, n_atoms=n_atoms)
        
        assert "energy_error_metric" in result
        assert "energy_error_per_atom_metric" in result
        assert abs(result["energy_error_metric"] - 0.5) < 1e-6
        assert abs(result["energy_error_per_atom_metric"] - 0.025) < 1e-6


class TestLossMetrics:
    """Test custom loss function metrics."""
    
    def test_tail_mse(self):
        """Test tail MSE calculation."""
        metric = TailMSE()
        
        # Create data with outliers
        pred = np.zeros(100)
        ref = np.zeros(100)
        ref[-10:] = 1.0  # Last 10 are outliers
        
        result = metric.calculate(pred, ref, quantile=0.9)
        
        assert "tail_mse_metric" in result
        assert "tail_fraction_metric" in result
        assert result["tail_mse_metric"] > 0
        assert abs(result["tail_fraction_metric"] - 0.1) < 0.01
    
    def test_focal_mse_loss(self):
        """Test focal MSE loss."""
        metric = FocalMSELoss()
        
        pred = np.array([0, 0, 0, 0])
        ref = np.array([0.1, 0.5, 1.0, 2.0])
        
        result = metric.calculate(pred, ref, alpha=1.0, gamma=2.0)
        
        assert "focal_mse_metric" in result
        assert "weighted_fraction_metric" in result
        assert result["focal_mse_metric"] > 0
    
    def test_force_angle_loss(self):
        """Test force angle loss."""
        metric = ForceAngleLoss()
        
        # Parallel forces (low error)
        pred1 = np.array([[1, 0, 0], [0, 1, 0]])
        ref1 = np.array([[2, 0, 0], [0, 2, 0]])
        
        result1 = metric.calculate(pred1, ref1)
        assert result1["mean_angle_error_metric"] < 0.1
        
        # Perpendicular forces (high error)
        pred2 = np.array([[1, 0, 0], [0, 1, 0]])
        ref2 = np.array([[0, 1, 0], [1, 0, 0]])
        
        result2 = metric.calculate(pred2, ref2)
        assert result2["mean_angle_error_metric"] > 1.0


class TestSpatialMetrics:
    """Test spatial analysis metrics."""
    
    @pytest.fixture
    def test_atoms(self):
        """Create test atoms structure."""
        atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
        atoms = atoms * (2, 2, 2)  # 32 atoms
        return atoms
    
    def test_error_clustering(self, test_atoms):
        """Test DBSCAN error clustering."""
        metric = ErrorClustering(test_atoms, eps=3.0, min_samples=3)
        
        # Create clustered errors
        errors = np.zeros(len(test_atoms))
        errors[:8] = 1.0  # First 8 atoms have high error
        
        result = metric.calculate(errors, errors)  # ref not used
        
        assert "n_error_clusters_metric" in result
        assert "largest_cluster_size_metric" in result
        assert "fraction_clustered_metric" in result
        assert result["n_error_clusters_metric"] >= 1
    
    @pytest.mark.skipif(
        True,  # Skip by default as pysal is optional
        reason="PySAL not installed"
    )
    def test_morans_i(self, test_atoms):
        """Test Moran's I calculation."""
        metric = MoransI(test_atoms, k=12)
        
        # Random errors (no spatial correlation)
        errors = np.random.randn(len(test_atoms))
        
        result = metric.calculate(errors, errors)
        
        assert "morans_i_global_metric" in result
        assert "morans_i_pvalue_metric" in result
        assert -1 <= result["morans_i_global_metric"] <= 1
    
    def test_analyze_spatial_patterns(self, test_atoms):
        """Test integrated spatial analysis."""
        errors = np.random.exponential(0.1, len(test_atoms))
        errors[:4] = 1.0  # Create a hotspot
        
        result = analyze_spatial_patterns(
            test_atoms, errors,
            k=6, eps=3.0, min_samples=2
        )
        
        assert "morans_i_global_metric" in result
        assert "n_error_clusters_metric" in result
        assert "cluster_labels" in result
        assert len(result["cluster_labels"]) == len(test_atoms)


class TestMetricIntegration:
    """Test metric integration scenarios."""
    
    def test_custom_metric_with_kwargs(self):
        """Test custom metric with additional parameters."""
        registry = MetricRegistry()
        
        def custom_threshold_metric(pred, ref, threshold=0.1, power=2):
            errors = np.abs(pred - ref)
            above = np.mean(errors > threshold)
            powered = np.mean(errors**power)
            return {
                "above_threshold": above,
                "powered_error": powered
            }
        
        registry.register(
            "threshold_metric",
            custom_threshold_metric,
            params={"threshold": 0.2, "power": 3}
        )
        
        pred = np.array([0, 0.1, 0.3])
        ref = np.array([0, 0, 0])
        
        result = registry.calculate("threshold_metric", pred, ref)
        assert "above_threshold" in result
        assert result["above_threshold"] == 1/3  # Only 0.3 > 0.2 