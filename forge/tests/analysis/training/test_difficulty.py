"""Tests for the difficulty analysis module."""
import pytest
import numpy as np

from forge.analysis.training.difficulty import (
    calculate_difficulty_score,
    rank_structures_by_difficulty,
    categorize_difficulty_causes,
    calculate_ensemble_variance,
    calculate_ensemble_agreement,
    identify_uncertain_atoms
)


class TestDifficultyScoring:
    """Test difficulty score calculations."""
    
    def test_calculate_difficulty_score_basic(self):
        """Test basic difficulty score calculation."""
        # Setup test data
        force_errors = np.array([0.1, 0.1, 0.1, 0.1, 1.0])  # One outlier
        morans_i = 0.5  # Moderate spatial correlation
        cluster_labels = np.array([-1, -1, -1, -1, 0])  # One in cluster
        ensemble_variance = 0.2
        n_atoms = 5
        
        result = calculate_difficulty_score(
            force_errors, morans_i, cluster_labels,
            ensemble_variance, n_atoms
        )
        
        assert 'difficulty_metric' in result
        assert 'heavy_tail_component' in result
        assert 'spatial_component' in result
        assert 'ensemble_component' in result
        
        # Check ranges
        assert 0 <= result['difficulty_metric'] <= 1
        assert result['heavy_tail_component'] > 0  # Has outlier
        assert result['spatial_component'] > 0  # Has clustering
    
    def test_difficulty_score_with_weights(self):
        """Test difficulty score with custom weights."""
        force_errors = np.ones(10) * 0.1
        morans_i = 0.0
        cluster_labels = np.full(10, -1)
        ensemble_variance = 0.5
        n_atoms = 10
        
        # Custom weights emphasizing ensemble
        weights = {
            'w_heavy_tail': 0.0,
            'w_spatial': 0.0,
            'w_ensemble': 1.0
        }
        
        result = calculate_difficulty_score(
            force_errors, morans_i, cluster_labels,
            ensemble_variance, n_atoms, weights_dict=weights
        )
        
        # Should be dominated by ensemble component
        assert result['ensemble_component'] > 0
        assert abs(result['difficulty_metric'] - result['ensemble_component']) < 0.01
    
    def test_rank_structures_by_difficulty(self):
        """Test ranking structures by difficulty."""
        scores_dict = {
            1: {'difficulty_metric': 0.8},
            2: {'difficulty_metric': 0.3},
            3: {'difficulty_metric': 0.9},
            4: {'difficulty_metric': 0.1}
        }
        
        ranked = rank_structures_by_difficulty(scores_dict, top_n=3)
        
        assert len(ranked) == 3
        assert ranked[0] == 3  # Highest difficulty
        assert ranked[1] == 1
        assert ranked[2] == 2


class TestDifficultyCategorization:
    """Test difficulty cause categorization."""
    
    def test_categorize_high_kurtosis(self):
        """Test categorization of high kurtosis structures."""
        metrics = {
            'force_kurtosis_metric': 10.0,  # Very high
            'morans_i_global_metric': 0.1,
            'n_error_clusters_metric': 0,
            'ensemble_force_std_metric': 0.01,
            'geometry_valid': True
        }
        
        categories = categorize_difficulty_causes(metrics)
        
        assert categories['heavy_tailed_errors'] is True
        assert categories['spatial_clustering'] is False
        assert categories['high_ensemble_disagreement'] is False
    
    def test_categorize_spatial_clustering(self):
        """Test categorization of spatial clustering."""
        metrics = {
            'force_kurtosis_metric': 3.0,
            'morans_i_global_metric': 0.8,  # High correlation
            'n_error_clusters_metric': 3,    # Multiple clusters
            'ensemble_force_std_metric': 0.01,
            'geometry_valid': True
        }
        
        categories = categorize_difficulty_causes(metrics)
        
        assert categories['heavy_tailed_errors'] is False
        assert categories['spatial_clustering'] is True
        assert categories['strong_spatial_correlation'] is True
    
    def test_categorize_geometry_issues(self):
        """Test categorization of geometry issues."""
        metrics = {
            'force_kurtosis_metric': 3.0,
            'geometry_valid': False,
            'has_close_atoms': True
        }
        
        categories = categorize_difficulty_causes(metrics)
        
        assert categories['geometry_issues'] is True
    
    def test_categorize_multiple_issues(self):
        """Test structures with multiple issues."""
        metrics = {
            'force_kurtosis_metric': 8.0,     # High
            'morans_i_global_metric': 0.7,    # High
            'n_error_clusters_metric': 2,     # Clusters
            'ensemble_force_std_metric': 0.3, # High disagreement
            'geometry_valid': True
        }
        
        categories = categorize_difficulty_causes(metrics)
        
        # Should have multiple issues
        assert sum(categories.values()) >= 3


class TestEnsembleAnalysis:
    """Test ensemble variance calculations."""
    
    def test_calculate_ensemble_variance_basic(self):
        """Test basic ensemble variance calculation."""
        # Create predictions from 3 models
        pred1 = {
            'energy': -10.0,
            'forces': np.ones((5, 3)) * 0.1
        }
        pred2 = {
            'energy': -10.2,
            'forces': np.ones((5, 3)) * 0.15
        }
        pred3 = {
            'energy': -9.8,
            'forces': np.ones((5, 3)) * 0.12
        }
        
        ensemble_preds = [pred1, pred2, pred3]
        n_atoms = 5
        
        result = calculate_ensemble_variance(ensemble_preds, n_atoms)
        
        assert 'energy_variance_metric' in result
        assert 'force_variance_metric' in result
        assert 'ensemble_force_std_metric' in result
        
        # Check values are positive
        assert result['energy_variance_metric'] > 0
        assert result['force_variance_metric'] > 0
    
    def test_ensemble_variance_single_model(self):
        """Test with single model (no variance)."""
        pred = {
            'energy': -10.0,
            'forces': np.ones((5, 3)) * 0.1
        }
        
        result = calculate_ensemble_variance([pred], 5)
        
        # Should have zero variance
        assert result['energy_variance_metric'] == 0
        assert result['force_variance_metric'] == 0
    
    def test_calculate_ensemble_agreement(self):
        """Test ensemble agreement calculation."""
        # High agreement case
        forces1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        forces2 = np.array([[0.9, 0.1, 0], [0.1, 0.9, 0], [0, 0.1, 0.9]])
        
        preds_high = [
            {'forces': forces1},
            {'forces': forces2}
        ]
        
        agreement_high = calculate_ensemble_agreement(preds_high)
        
        # Low agreement case  
        forces3 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        preds_low = [
            {'forces': forces1},
            {'forces': forces3}
        ]
        
        agreement_low = calculate_ensemble_agreement(preds_low)
        
        assert 'mean_cosine_similarity' in agreement_high
        assert 'min_cosine_similarity' in agreement_low
        assert agreement_high['mean_cosine_similarity'] > agreement_low['mean_cosine_similarity']
    
    def test_identify_uncertain_atoms(self):
        """Test identification of uncertain atoms."""
        # Create predictions with varying uncertainty
        pred1 = {'forces': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}
        pred2 = {'forces': np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])}  # Disagree on atom 1
        pred3 = {'forces': np.array([[1, 0, 0], [0, -0.5, 0.5], [0, 0, 1]])}
        
        uncertain_atoms = identify_uncertain_atoms(
            [pred1, pred2, pred3],
            threshold_std=0.5
        )
        
        assert len(uncertain_atoms) > 0
        assert 1 in uncertain_atoms  # Atom 1 should be uncertain
        
        # Test with percentile threshold
        uncertain_atoms_pct = identify_uncertain_atoms(
            [pred1, pred2, pred3],
            threshold_percentile=50
        )
        
        assert len(uncertain_atoms_pct) >= 1  # At least half should be selected 