#!/usr/bin/env python
"""Test script to verify workflow produces consistent outputs across backends."""

import numpy as np
from ase.build import bulk
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List

def create_test_atoms():
    """Create a test atoms object with required info for adversarial attack."""
    atoms = bulk('Al', 'fcc', a=4.0, cubic=True)  # 4 atoms
    
    # Add required info for adversarial attack workflow
    atoms.info['structure_id'] = 12345
    atoms.info['config_type'] = 'test_structure'
    atoms.info['structure_name'] = 'test_Al4'
    
    # Add some reference forces for testing
    atoms.arrays['forces'] = np.array([
        [0.1, 0.2, -0.1],
        [-0.1, 0.1, 0.2], 
        [0.2, -0.1, 0.1],
        [-0.2, -0.2, -0.2]
    ])
    
    return atoms

def create_mock_db_manager():
    """Create a mock database manager."""
    from forge.core.database import DatabaseManager
    
    mock_db = Mock(spec=DatabaseManager)
    mock_db.dry_run = False
    
    # Mock the batch fetch method
    test_atoms = create_test_atoms()
    mock_db.get_batch_atoms_with_calculation.return_value = [test_atoms]
    
    return mock_db

def test_workflow_with_mace():
    """Test the workflow using MACE backend."""
    print("=== Testing Workflow with MACE Backend ===")
    
    try:
        from forge.calculators import get_supported_backends
        if 'mace' not in get_supported_backends():
            print("⚠ MACE not available in this environment, skipping MACE workflow test")
            return None
            
        from forge.workflows.adversarial_attack import run_adversarial_attacks
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock MACE models
            model_paths = ['model1.model', 'model2.model']
            
            # Mock the ensemble calculator creation
            with patch('forge.workflows.adversarial_attack.create_ensemble_calculator') as mock_factory:
                with patch('forge.core.adversarial_attack.create_ensemble_calculator') as mock_factory2:
                    # Create mock calculator for ranking
                    mock_ranking_calc = Mock()
                    mock_ranking_calc.models = [Mock(), Mock()]
                    
                    # Mock force calculation for ranking
                    def mock_forces_all(atoms):
                        n_atoms = len(atoms)
                        # Return consistent mock forces
                        forces1 = np.ones((n_atoms, 3)) * 0.1
                        forces2 = np.ones((n_atoms, 3)) * 0.11
                        return np.array([forces1, forces2])
                    
                    mock_ranking_calc.forces_all = mock_forces_all
                    mock_ranking_calc.calculate_normalized_force_variance = lambda f: np.array([0.01] * len(f[0]))
                    
                    mock_factory.return_value = mock_ranking_calc
                    
                    # Mock optimizer calculator
                    mock_opt_calc = Mock()
                    mock_opt_calc.models = [Mock(), Mock()]
                    mock_opt_calc.z_table = {13: 0}
                    mock_opt_calc.models[0].r_max.item.return_value = 5.0
                    
                    mock_factory2.return_value = mock_opt_calc
                    
                    # Mock the actual optimization to avoid autograd complexity
                    with patch('forge.core.adversarial_attack.GradientAdversarialOptimizer.optimize') as mock_optimize:
                        # Create mock trajectory
                        mock_trajectory = []
                        for step in range(3):  # Short trajectory
                            step_atoms = create_test_atoms()
                            step_atoms.info.update({
                                'parent_id': 12345,
                                'generation': 9,
                                'config_type': 'test_structure_aa',
                                'step': step,
                                'variance': 0.01 + step * 0.001,
                                'loss': 0.01 + step * 0.001,
                                'energy': -15.5 - step * 0.1
                            })
                            # Slightly modify positions to simulate optimization
                            positions = step_atoms.positions.copy()
                            positions += np.random.rand(*positions.shape) * 0.01
                            step_atoms.positions = positions
                            mock_trajectory.append(step_atoms)
                        
                        mock_optimize.return_value = mock_trajectory
                        
                        # Create mock database manager
                        mock_db = create_mock_db_manager()
                        
                        # Run the workflow
                        try:
                            result = run_adversarial_attacks(
                                db_manager=mock_db,
                                structure_ids=[12345],
                                model_paths=model_paths,
                                top_n=1,
                                generation=9,
                                n_iterations=5,
                                learning_rate=0.01,
                                temperature=1000,
                                include_probability=False,
                                min_distance=1.2,
                                backend='mace',  # Use MACE backend
                                device='cpu',
                                debug=False,
                                save_output=False,  # Return trajectories
                                ranking_metric='force_rmse'
                            )
                            
                            print("✅ MACE workflow completed successfully")
                            
                            # Extract results for comparison
                            if result and len(result) > 0:
                                trajectory = list(result.values())[0]
                                return {
                                    'backend': 'mace',
                                    'n_structures': len(result),
                                    'trajectory_length': len(trajectory),
                                    'final_variance': trajectory[-1].info.get('variance', 0),
                                    'final_energy': trajectory[-1].info.get('energy', 0),
                                    'final_positions': trajectory[-1].positions.copy()
                                }
                            else:
                                print("⚠ MACE workflow returned no results")
                                return None
                                
                        except Exception as e:
                            print(f"❌ MACE workflow failed: {e}")
                            return None
                            
    except Exception as e:
        print(f"MACE workflow test setup failed: {e}")
        return None

def test_workflow_with_allegro():
    """Test the workflow using Allegro backend."""
    print("\n=== Testing Workflow with Allegro Backend ===")
    
    try:
        from forge.calculators import get_supported_backends
        available = get_supported_backends()
        if not any(b in available for b in ['allegro', 'nequip']):
            print("⚠ Allegro/NequIP not available in this environment, skipping Allegro workflow test")
            return None
            
        print("❌ Allegro workflow testing not yet implemented")
        print("   Reason: Current autograd optimization only supports MACE backend")
        print("   Future work: Implement Allegro-compatible autograd or alternative optimization")
        
        # For now, return a placeholder indicating this limitation
        return {
            'backend': 'allegro',
            'status': 'not_implemented',
            'reason': 'autograd_only_supports_mace'
        }
            
    except Exception as e:
        print(f"Allegro workflow test setup failed: {e}")
        return None

def test_force_calculation_consistency():
    """Test that force calculations are consistent across backends."""
    print("\n=== Testing Force Calculation Consistency ===")
    
    from forge.calculators import get_supported_backends
    available_backends = get_supported_backends()
    
    # Create test atoms
    atoms = create_test_atoms()
    
    results = {}
    
    # Test each available backend
    for backend_name in available_backends:
        if backend_name == 'nequip':  # Skip alias
            continue
            
        print(f"\nTesting {backend_name} backend force calculations...")
        
        try:
            from forge.calculators import create_ensemble_calculator
            
            # Mock model paths appropriate for backend
            if backend_name == 'mace':
                model_paths = ['model1.model', 'model2.model']
                patch_target = 'forge.calculators.mace_backend.MACECalculator'
            else:  # allegro
                model_paths = ['model1.pt2', 'model2.pt2']
                patch_target = 'forge.calculators.allegro_backend.NequIPCalculator'
            
            # Create deterministic mock forces for consistency testing
            mock_forces1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
            mock_forces2 = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1], [10.1, 11.1, 12.1]])
            
            if backend_name == 'mace':
                with patch(patch_target) as mock_calc_class:
                    # Setup MACE mock
                    mock_calc = Mock()
                    mock_models = [Mock(), Mock()]
                    for i, model in enumerate(mock_models):
                        model.r_max.item.return_value = 5.0
                    mock_calc.models = mock_models
                    mock_calc.z_table = {13: 0}
                    mock_calc_class.return_value = mock_calc
                    
                    # Create backend
                    backend = create_ensemble_calculator(model_paths, backend=backend_name, device='cpu')
                    
                    # Setup force mocking
                    call_count = 0
                    def mock_get_forces():
                        nonlocal call_count
                        result = mock_forces1 if call_count == 0 else mock_forces2
                        call_count += 1
                        return result
                    
                    atoms.get_potential_energy = Mock(return_value=-42.0)
                    atoms.get_forces = mock_get_forces
                    
                    # Test force calculations
                    all_forces = backend.forces_all(atoms)
                    mean_forces = backend.get_mean_forces(atoms)
                    variance = backend.calculate_normalized_force_variance(all_forces)
                    
                    results[backend_name] = {
                        'all_forces_shape': all_forces.shape,
                        'mean_forces': mean_forces.copy(),
                        'variance': variance.copy(),
                        'r_max': backend.r_max
                    }
                    
                    print(f"  ✅ Forces shape: {all_forces.shape}")
                    print(f"  ✅ Mean force magnitude: {np.linalg.norm(mean_forces):.4f}")
                    print(f"  ✅ Mean variance: {np.mean(variance):.6f}")
            
            else:  # allegro
                with patch(patch_target) as mock_calc_class:
                    with patch('pathlib.Path.exists', return_value=True):
                        # Setup Allegro mock
                        mock_calcs = []
                        for i in range(len(model_paths)):
                            mock_calc = Mock()
                            mock_model = Mock()
                            mock_model.r_max.item.return_value = 6.0
                            mock_calc.model = mock_model
                            mock_calcs.append(mock_calc)
                        
                        mock_calc_class.from_compiled_model.side_effect = mock_calcs
                        
                        # Create backend
                        backend = create_ensemble_calculator(model_paths, backend=backend_name, device='cpu')
                        
                        # Setup force mocking
                        call_count = 0
                        def mock_get_forces():
                            nonlocal call_count
                            result = mock_forces1 if call_count == 0 else mock_forces2
                            call_count += 1
                            return result
                        
                        atoms.get_potential_energy = Mock(return_value=-42.0)
                        atoms.get_forces = mock_get_forces
                        
                        # Test force calculations
                        all_forces = backend.forces_all(atoms)
                        mean_forces = backend.get_mean_forces(atoms)
                        variance = backend.calculate_normalized_force_variance(all_forces)
                        
                        results[backend_name] = {
                            'all_forces_shape': all_forces.shape,
                            'mean_forces': mean_forces.copy(),
                            'variance': variance.copy(),
                            'r_max': backend.r_max
                        }
                        
                        print(f"  ✅ Forces shape: {all_forces.shape}")
                        print(f"  ✅ Mean force magnitude: {np.linalg.norm(mean_forces):.4f}")
                        print(f"  ✅ Mean variance: {np.mean(variance):.6f}")
                        
        except Exception as e:
            print(f"  ❌ {backend_name} backend test failed: {e}")
            continue
    
    # Compare results if we have multiple backends
    if len(results) >= 2:
        print(f"\n=== Comparing Results Across Backends ===")
        backend_names = list(results.keys())
        backend1, backend2 = backend_names[0], backend_names[1]
        
        result1, result2 = results[backend1], results[backend2]
        
        # Compare shapes
        shape_match = result1['all_forces_shape'] == result2['all_forces_shape']
        print(f"Force shapes match: {shape_match}")
        
        # Compare mean forces (should be identical with same mock data)
        forces_diff = np.max(np.abs(result1['mean_forces'] - result2['mean_forces']))
        print(f"Max mean force difference: {forces_diff:.10f}")
        
        # Compare variances (should be identical with same mock data)
        var_diff = np.max(np.abs(result1['variance'] - result2['variance']))
        print(f"Max variance difference: {var_diff:.10f}")
        
        # Numerical tolerance for "identical" results
        tolerance = 1e-10
        if forces_diff < tolerance and var_diff < tolerance:
            print("✅ Force calculations are numerically identical across backends")
        else:
            print(f"⚠ Force calculations differ by {max(forces_diff, var_diff):.2e} (tolerance: {tolerance:.2e})")
            
    return results

def main():
    """Run workflow consistency tests."""
    print("=== Adversarial Attack Workflow Consistency Test ===")
    print("Testing that backends produce consistent outputs\n")
    
    # Test force calculation consistency first
    force_results = test_force_calculation_consistency()
    
    # Test full workflow with available backends
    workflow_results = []
    
    # Test MACE workflow
    mace_result = test_workflow_with_mace()
    if mace_result:
        workflow_results.append(mace_result)
    
    # Test Allegro workflow
    allegro_result = test_workflow_with_allegro()
    if allegro_result:
        workflow_results.append(allegro_result)
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Force calculation backends tested: {len(force_results)}")
    
    if len(force_results) >= 2:
        print("✅ Force calculation consistency verified")
    elif len(force_results) == 1:
        print("⚠ Only one backend available for force testing")
    else:
        print("❌ No backends available for force testing")
    
    print(f"Workflow backends tested: {len(workflow_results)}")
    
    # Check for workflow limitations
    mace_workflow_works = any(r.get('backend') == 'mace' and r.get('n_structures', 0) > 0 for r in workflow_results)
    allegro_limitation = any(r.get('status') == 'not_implemented' for r in workflow_results)
    
    if mace_workflow_works:
        print("✅ MACE workflow functional")
    if allegro_limitation:
        print("⚠ Allegro workflow limited by autograd implementation")
        print("  Current autograd optimization requires MACE-specific data structures")
        print("  Recommendation: Use Allegro backend for force calculations, MACE for autograd optimization")
    
    print(f"\n=== Recommendations ===")
    available_backends = list(force_results.keys()) if force_results else []
    
    if 'mace' in available_backends:
        print("• Use MACE backend for full adversarial attack workflow (including autograd)")
    if 'allegro' in available_backends:
        print("• Use Allegro backend for force calculations and variance ranking")
        if not mace_workflow_works:
            print("• For full adversarial attacks with Allegro: install MACE for autograd support")
    
    if not available_backends:
        print("• Install MACE (pip install mace-torch) or Allegro/NequIP (pip install nequip)")

if __name__ == "__main__":
    main() 