# forge/tests/test_mace_calculator.py
import unittest
import numpy as np
from pathlib import Path
import torch
from ase.build import bulk
from ase.io import read, write
import tempfile
import os
from forge.potentials.mace import MACECalculator

import pytest

# At the top of the test file
@pytest.mark.filterwarnings("ignore::FutureWarning")

class TestMACECalculator(unittest.TestCase):
    """Test cases for MACE calculator implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources."""
        # Create a simple test structure
        cls.test_atoms = bulk('V', 'bcc', a=3.01, cubic=False)
        cls.test_atoms *= (4, 4, 4)  # 4x4x4 supercell
        
        # Define test device
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_dir = Path(__file__).parent / "resources/potentials/mace"
        cls.model_paths = sorted(str(p) for p in model_dir.glob("gen_5_model_*-11-28_stagetwo.model"))
        
        # Create single calculator
        cls.single_calc = MACECalculator(
            model_paths=cls.model_paths[0],
            device=cls.device
        )
        
        # Create ensemble calculator
        cls.ensemble_calc = MACECalculator(
            model_paths=cls.model_paths,
            device=cls.device
        )

    def test_single_model_forces(self):
        """Test force prediction with single model."""
        forces = self.single_calc.calculate_forces(self.test_atoms)
        
        # Check shape
        self.assertEqual(forces.shape[0], 1)  # Single model
        self.assertEqual(forces.shape[1], len(self.test_atoms))  # Number of atoms
        self.assertEqual(forces.shape[2], 3)  # 3D forces
        
        # Check forces are not all zero
        self.assertFalse(np.allclose(forces, 0))
        
        # Check forces are finite
        self.assertTrue(np.all(np.isfinite(forces)))

    def test_single_model_energy(self):
        """Test energy prediction with single model."""
        energy = self.single_calc.calculate_energy(self.test_atoms)
        
        # Check shape
        self.assertEqual(energy.shape, (1,))
        
        # Check energy is finite
        self.assertTrue(np.isfinite(energy))
        
        # Check energy is reasonable (should be negative for bulk Cu)
        self.assertLess(energy[0], 0)

    def test_ensemble_forces(self):
        """Test force prediction with ensemble."""
        forces = self.ensemble_calc.calculate_forces(self.test_atoms)
        
        # Check shape
        self.assertEqual(forces.shape[0], len(self.model_paths))  # Number of models
        self.assertEqual(forces.shape[1], len(self.test_atoms))   # Number of atoms
        self.assertEqual(forces.shape[2], 3)  # 3D forces
        
        # Check forces vary between models
        force_std = np.std(forces, axis=0)
        self.assertFalse(np.allclose(force_std, 0))

    def test_ensemble_energy(self):
        """Test energy prediction with ensemble."""
        energies = self.ensemble_calc.calculate_energy(self.test_atoms)
        
        # Check shape
        self.assertEqual(energies.shape, (len(self.model_paths),))
        
        # Check energies vary between models
        self.assertGreater(np.std(energies), 0)
        
        # Check all energies are reasonable
        self.assertTrue(np.all(energies < 0))

    def test_uncertainty_calculation(self):
        """Test uncertainty calculation with ensemble."""
        uncertainty = self.ensemble_calc.calculate_structure_uncertainty(
            self.test_atoms,
            calculate_forces=True,
            calculate_energy=True
        )
        
        # Check all expected keys are present
        expected_keys = {
            'energy_std', 
            'force_variance',
            'mean_force_variance',
            'max_force_variance'
        }
        self.assertEqual(set(uncertainty.keys()), expected_keys)
        
        # Check shapes and values
        self.assertGreater(uncertainty['energy_std'], 0)
        self.assertEqual(
            uncertainty['force_variance'].shape,
            (len(self.test_atoms),)
        )
        self.assertGreater(uncertainty['mean_force_variance'], 0)
        self.assertGreaterEqual(
            uncertainty['max_force_variance'],
            uncertainty['mean_force_variance']
        )

    def test_force_variance(self):
        """Test force variance calculation."""
        forces = self.ensemble_calc.calculate_forces(self.test_atoms)
        variances = self.ensemble_calc.calculate_force_variance(forces)
        
        # Check shape
        self.assertEqual(variances.shape, (len(self.test_atoms),))
        
        # Check variances are positive
        self.assertTrue(np.all(variances >= 0))
        
        # Check variances are not all identical
        self.assertGreater(np.std(variances), 0)

    def test_model_versions(self):
        """Test model version retrieval."""
        versions = self.ensemble_calc.get_model_versions()
        
        # Check number of versions matches number of models
        self.assertEqual(len(versions), len(self.model_paths))
        
        # Check all versions are strings
        self.assertTrue(all(isinstance(v, str) for v in versions))

    def test_device_handling(self):
        """Test calculator behavior with different devices."""
        # Test CPU calculator
        cpu_calc = MACECalculator(
            model_paths=self.model_paths[0],
            device='cpu'
        )
        forces_cpu = cpu_calc.calculate_forces(self.test_atoms)
        
        # Test CUDA calculator if available
        if torch.cuda.is_available():
            cuda_calc = MACECalculator(
                model_paths=self.model_paths[0],
                device='cuda'
            )
            forces_cuda = cuda_calc.calculate_forces(self.test_atoms)
            
            # Results should be the same regardless of device
            self.assertTrue(np.allclose(forces_cpu, forces_cuda))

if __name__ == '__main__':
    unittest.main()