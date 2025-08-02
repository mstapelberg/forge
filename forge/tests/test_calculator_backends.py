"""Tests for calculator backends."""

import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from unittest.mock import Mock, patch, MagicMock

from forge.calculators.interface import BaseEnsembleCalculator

# Conditional imports for backends (to check availability)
try:
    from forge.calculators.mace_backend import MACEBackend, MACE_AVAILABLE
except ImportError:
    MACEBackend = None
    MACE_AVAILABLE = False

try:
    from forge.calculators.allegro_backend import AllegroBackend, NEQUIP_AVAILABLE
except ImportError:
    AllegroBackend = None
    NEQUIP_AVAILABLE = False

# Import torch if available for tensor tests
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestBaseEnsembleCalculator:
    """Test the abstract interface."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that the abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseEnsembleCalculator()


@pytest.mark.skipif(not MACE_AVAILABLE, reason="MACE not available in this environment")
class TestMACEBackend:
    """Test the MACE backend implementation."""
    
    @pytest.fixture
    def mock_mace_calculator(self):
        """Create a mock MACECalculator for testing."""
        mock_calc = Mock()
        
        # Mock models
        mock_model1 = Mock()
        mock_model1.r_max = torch.tensor(5.0)
        mock_model2 = Mock()
        mock_model2.r_max = torch.tensor(5.0)
        
        mock_calc.models = [mock_model1, mock_model2]
        mock_calc.z_table = {1: 0, 6: 1}  # H, C mapping
        
        return mock_calc
    
    @pytest.fixture
    def sample_atoms(self):
        """Create sample atoms for testing."""
        return bulk('Al', 'fcc', a=4.0, cubic=True)  # 4 atoms
    
    @pytest.fixture
    def mace_backend(self, mock_mace_calculator):
        """Create MACEBackend with mocked calculator."""
        with patch('forge.calculators.mace_backend.MACECalculator') as mock_mace_class:
            mock_mace_class.return_value = mock_mace_calculator
            backend = MACEBackend(
                model_paths=['model1.model', 'model2.model'],
                device='cpu'
            )
            return backend
    
    def test_init_single_model_path(self):
        """Test initialization with single model path."""
        with patch('forge.calculators.mace_backend.MACECalculator') as mock_mace_class:
            backend = MACEBackend('single_model.model', device='cpu')
            assert backend.model_paths == ['single_model.model']
            assert backend.device == 'cpu'
    
    def test_init_multiple_model_paths(self):
        """Test initialization with multiple model paths."""
        model_paths = ['model1.model', 'model2.model']
        with patch('forge.calculators.mace_backend.MACECalculator') as mock_mace_class:
            backend = MACEBackend(model_paths, device='cuda')
            assert backend.model_paths == model_paths
            assert backend.device == 'cuda'
    
    def test_forces_all_shape(self, mace_backend, sample_atoms):
        """Test that forces_all returns correct shape."""
        # Use fixed arrays to avoid randomness issues
        mock_forces1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        mock_forces2 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
        
        call_count = 0
        def mock_get_forces():
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return mock_forces1
            else:
                return mock_forces2
        
        sample_atoms.get_potential_energy = Mock(return_value=0.0)
        sample_atoms.get_forces = mock_get_forces
        
        forces = mace_backend.forces_all(sample_atoms)
        
        # Should have shape (n_models, n_atoms, 3)
        assert forces.shape == (2, 4, 3)
        np.testing.assert_array_equal(forces[0], mock_forces1)
        np.testing.assert_array_equal(forces[1], mock_forces2)
    
    def test_energies_all_shape(self, mace_backend, sample_atoms):
        """Test that energies_all returns correct shape."""
        mock_energy1 = -15.5
        mock_energy2 = -15.3
        
        def mock_get_energy():
            if sample_atoms.calc == mace_backend._calculator.models[0]:
                return mock_energy1
            else:
                return mock_energy2
        
        sample_atoms.get_potential_energy = mock_get_energy
        
        energies = mace_backend.energies_all(sample_atoms)
        
        # Should have shape (n_models,)
        assert energies.shape == (2,)
        assert energies[0] == mock_energy1
        assert energies[1] == mock_energy2
    
    def test_mean_forces_agreement(self, mace_backend, sample_atoms):
        """Test that mean forces agree with manual calculation."""
        mock_forces1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
        mock_forces2 = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=float)
        expected_mean = (mock_forces1 + mock_forces2) / 2
        
        def mock_get_forces():
            if sample_atoms.calc == mace_backend._calculator.models[0]:
                return mock_forces1
            else:
                return mock_forces2
        
        sample_atoms.get_potential_energy = Mock(return_value=0.0)
        sample_atoms.get_forces = mock_get_forces
        
        mean_forces = mace_backend.get_mean_forces(sample_atoms)
        
        np.testing.assert_array_almost_equal(mean_forces, expected_mean)
    
    def test_properties(self, mace_backend):
        """Test that properties return expected values."""
        assert mace_backend.device == 'cpu'
        assert len(mace_backend.models) == 2
        assert mace_backend.z_table == {1: 0, 6: 1}
        assert mace_backend.r_max == 5.0
    
    def test_legacy_method_compatibility(self, mace_backend, sample_atoms):
        """Test that legacy calculate_forces method works."""
        mock_forces1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        mock_forces2 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
        
        call_count = 0
        def mock_get_forces():
            nonlocal call_count
            result = mock_forces1 if call_count % 2 == 0 else mock_forces2
            call_count += 1
            return result
        
        sample_atoms.get_potential_energy = Mock(return_value=0.0)
        sample_atoms.get_forces = mock_get_forces
        
        # Legacy method should return same as forces_all
        legacy_forces = mace_backend.calculate_forces(sample_atoms)
        new_forces = mace_backend.forces_all(sample_atoms)
        
        np.testing.assert_array_equal(legacy_forces, new_forces)
    
    def test_normalized_force_variance(self, mace_backend):
        """Test the normalized force variance calculation."""
        # Create test data with known variance
        n_models, n_atoms = 3, 2
        forces = np.array([
            [[1, 0, 0], [0, 1, 0]],  # Model 1
            [[0, 1, 0], [1, 0, 0]],  # Model 2  
            [[0, 0, 1], [0, 0, 1]]   # Model 3
        ])
        
        variance = mace_backend.calculate_normalized_force_variance(forces)
        
        # Should return array of shape (n_atoms,)
        assert variance.shape == (n_atoms,)
        assert all(v >= 0 for v in variance)  # Variance should be non-negative
    
    def test_force_calculation_error_handling(self, mace_backend, sample_atoms):
        """Test error handling when force calculation fails."""
        sample_atoms.get_potential_energy = Mock(side_effect=Exception("Calculation failed"))
        
        forces = mace_backend.forces_all(sample_atoms)
        
        # Should return zeros when calculation fails
        expected_shape = (len(mace_backend.models), len(sample_atoms), 3)
        assert forces.shape == expected_shape
        np.testing.assert_array_equal(forces, np.zeros(expected_shape))


@pytest.mark.skipif(not NEQUIP_AVAILABLE, reason="NequIP/Allegro not available in this environment")
class TestAllegroBackend:
    """Test the Allegro backend implementation."""
    
    @pytest.fixture
    def mock_nequip_calculator(self):
        """Create a mock NequIPCalculator for testing."""
        mock_calc = Mock()
        
        # Mock underlying model
        mock_model = Mock()
        mock_model.r_max = torch.tensor(6.0)
        mock_model.chemical_symbols = ['Al', 'Cu']
        mock_calc.model = mock_model
        
        return mock_calc
    
    @pytest.fixture
    def sample_atoms(self):
        """Create sample atoms for testing."""
        return bulk('Al', 'fcc', a=4.0, cubic=True)  # 4 atoms
    
    @pytest.fixture
    def allegro_backend(self, mock_nequip_calculator):
        """Create AllegroBackend with mocked calculator."""
        with patch('forge.calculators.allegro_backend.NequIPCalculator') as mock_nequip_class:
            mock_nequip_class.from_compiled_model.return_value = mock_nequip_calculator
            with patch('pathlib.Path.exists', return_value=True):
                backend = AllegroBackend(
                    model_paths=['model1.pt2', 'model2.pt2'],
                    device='cpu'
                )
                return backend
    
    def test_init_with_model_paths(self):
        """Test initialization with model paths."""
        model_paths = ['model1.pt2', 'model2.pt2']
        
        with patch('forge.calculators.allegro_backend.NequIPCalculator') as mock_nequip_class:
            with patch('pathlib.Path.exists', return_value=True):
                backend = AllegroBackend(model_paths, device='cuda')
                assert backend.model_paths == model_paths
                assert backend.device == 'cuda'
                assert len(backend._calculators) == 2
    
    def test_file_not_found_error(self):
        """Test error handling when model file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                AllegroBackend(['nonexistent_model.pt2'], device='cpu')
    
    def test_forces_all_shape(self, allegro_backend, sample_atoms):
        """Test that forces_all returns correct shape."""
        # Use fixed arrays to avoid randomness issues
        mock_forces1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        mock_forces2 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
        
        call_count = 0
        def mock_get_forces():
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return mock_forces1
            else:
                return mock_forces2
        
        sample_atoms.get_potential_energy = Mock(return_value=0.0)
        sample_atoms.get_forces = mock_get_forces
        
        forces = allegro_backend.forces_all(sample_atoms)
        
        # Should have shape (n_models, n_atoms, 3)
        assert forces.shape == (2, 4, 3)
        np.testing.assert_array_equal(forces[0], mock_forces1)
        np.testing.assert_array_equal(forces[1], mock_forces2)
    
    def test_properties(self, allegro_backend):
        """Test that properties return expected values."""
        assert allegro_backend.device == 'cpu'
        assert len(allegro_backend.models) == 2
        assert allegro_backend.r_max == 6.0
        # z_table might be None for NequIP models depending on implementation
        z_table = allegro_backend.z_table
        assert z_table is None or isinstance(z_table, (list, dict))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTorchIntegration:
    """Test torch-specific functionality."""
    
    @pytest.mark.skipif(not MACE_AVAILABLE, reason="MACE not available in this environment")
    def test_mace_tensor_r_max_handling(self):
        """Test handling of tensor r_max values in MACE backend."""
        with patch('forge.calculators.mace_backend.MACECalculator') as mock_mace_class:
            mock_calc = Mock()
            mock_model = Mock()
            mock_model.r_max = torch.tensor(5.5)
            mock_calc.models = [mock_model]
            mock_calc.z_table = {}
            mock_mace_class.return_value = mock_calc
            
            backend = MACEBackend('model.model')
            assert backend.r_max == 5.5
    
    @pytest.mark.skipif(not NEQUIP_AVAILABLE, reason="NequIP/Allegro not available in this environment")
    def test_allegro_tensor_r_max_handling(self):
        """Test handling of tensor r_max values in Allegro backend."""
        with patch('forge.calculators.allegro_backend.NequIPCalculator') as mock_nequip_class:
            with patch('pathlib.Path.exists', return_value=True):
                mock_calc = Mock()
                mock_model = Mock()
                mock_model.r_max = torch.tensor(6.5)
                mock_calc.model = mock_model
                mock_nequip_class.from_compiled_model.return_value = mock_calc
                
                backend = AllegroBackend('model.pt2')
                assert backend.r_max == 6.5 