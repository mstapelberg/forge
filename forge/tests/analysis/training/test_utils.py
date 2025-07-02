"""Tests for the utils module."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
from ase import Atoms
from ase.build import bulk, molecule, fcc111

from forge.analysis.training.utils import (
    check_geometry, batch_check_geometry, get_interatomic_distances,
    identify_surface_atoms, analyze_bond_lengths,
    save_analysis_results, load_analysis_results,
    export_structures_to_extxyz, generate_markdown_report,
    export_for_visualization
)
from forge.analysis.training.core import AnalysisResults


class TestGeometryUtils:
    """Test geometry utility functions."""
    
    def test_check_geometry_valid(self):
        """Test geometry check on valid structure."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        result = check_geometry(atoms)
        
        assert result['is_valid'] is True
        assert result['has_duplicates'] is False
        assert result['has_close_atoms'] is False
        assert result['min_distance'] > 2.0  # Cu-Cu distance
    
    def test_check_geometry_duplicates(self):
        """Test detection of duplicate atoms."""
        atoms = molecule('H2O')
        # Add duplicate atom at same position
        atoms.append(Atoms('H', positions=[atoms.positions[0]]))
        
        result = check_geometry(atoms, cutoff=0.1)
        
        assert result['is_valid'] is False
        assert result['has_duplicates'] is True
        assert len(result['duplicate_indices']) > 0
    
    def test_check_geometry_close_atoms(self):
        """Test detection of atoms too close."""
        atoms = Atoms('H2', positions=[[0, 0, 0], [0.5, 0, 0]])  # Very close
        
        result = check_geometry(atoms, cutoff=1.0)
        
        assert result['is_valid'] is False
        assert result['has_close_atoms'] is True
        assert len(result['close_pairs']) > 0
        assert result['min_distance'] < 1.0
    
    def test_batch_check_geometry(self):
        """Test batch geometry checking."""
        atoms_list = [
            bulk('Cu', 'fcc'),
            molecule('H2O'),
            molecule('NH3')
        ]
        
        results = batch_check_geometry(atoms_list)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results.values())
        assert all(results[i]['is_valid'] for i in range(3))
    
    def test_get_interatomic_distances(self):
        """Test interatomic distance calculation."""
        atoms = molecule('H2O')
        
        result = get_interatomic_distances(atoms)
        
        assert 'min_distance' in result
        assert 'max_distance' in result
        assert 'mean_distance' in result
        assert 'distance_percentiles' in result
        assert result['n_pairs'] == 3  # 3 pairs in H2O
    
    def test_identify_surface_atoms(self):
        """Test surface atom identification."""
        # Create slab with clear surface
        atoms = fcc111('Cu', size=(3, 3, 4), vacuum=10.0)
        
        surface_indices = identify_surface_atoms(atoms)
        
        assert len(surface_indices) > 0
        assert len(surface_indices) < len(atoms)  # Not all atoms are surface
        
        # Check that top layer atoms are identified
        z_positions = atoms.positions[:, 2]
        max_z = np.max(z_positions)
        top_atoms = np.where(z_positions > max_z - 3.0)[0]
        
        # Most top atoms should be in surface_indices
        overlap = set(surface_indices) & set(top_atoms)
        assert len(overlap) > len(top_atoms) * 0.5
    
    def test_analyze_bond_lengths(self):
        """Test bond length analysis."""
        # Create structure with known bond lengths
        atoms = Atoms('H2O',
                     positions=[[0, 0, 0],
                               [1, 0, 0],
                               [0, 1, 0]])
        
        result = analyze_bond_lengths(atoms, species_pairs=[('H', 'O')])
        
        assert 'H-O' in result or 'O-H' in result
        bond_key = 'H-O' if 'H-O' in result else 'O-H'
        
        assert result[bond_key]['count'] == 2  # Two H-O bonds
        assert result[bond_key]['mean'] > 0
        assert result[bond_key]['min'] <= result[bond_key]['mean']
        assert result[bond_key]['max'] >= result[bond_key]['mean']


class TestIOUtils:
    """Test I/O utility functions."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample AnalysisResults for testing."""
        struct_df = pd.DataFrame({
            'structure_id': [1, 2],
            'difficulty_metric': [0.5, 0.8],
            'force_rmse_metric': [0.1, 0.2]
        })
        
        atom_df = pd.DataFrame({
            'structure_id': [1, 1, 2],
            'atom_index': [0, 1, 0],
            'force_error_mag': [0.1, 0.2, 0.3]
        })
        
        metadata = {'test': True, 'version': '1.0'}
        cache = {1: {'test_data': 'value'}}
        
        return AnalysisResults(struct_df, atom_df, metadata, cache)
    
    def test_save_and_load_results(self, sample_results, tmp_path):
        """Test saving and loading analysis results."""
        save_dir = tmp_path / "test_save"
        
        # Save
        save_analysis_results(
            sample_results, save_dir,
            save_pickle=True, save_csv=True, save_json=True
        )
        
        # Check files exist
        assert (save_dir / "structure_metrics.csv").exists()
        assert (save_dir / "atom_metrics.csv").exists()
        assert (save_dir / "structure_metrics.pkl").exists()
        assert (save_dir / "summary.json").exists()
        
        # Load back
        loaded = load_analysis_results(save_dir)
        
        assert 'structure_metrics' in loaded
        assert 'atom_metrics' in loaded
        assert len(loaded['structure_metrics']) == 2
        assert loaded['metadata']['test'] is True
    
    def test_export_structures_to_extxyz(self, sample_results, tmp_path):
        """Test exporting structures to extended XYZ."""
        # Create test atoms
        atoms1 = molecule('H2O')
        atoms1.info['structure_id'] = 1
        
        atoms2 = molecule('NH3')
        atoms2.info['structure_id'] = 2
        
        # Add to results cache
        sample_results.results_cache[1] = {
            'atoms': atoms1,
            'force_error_magnitudes': np.array([0.1, 0.2, 0.3])
        }
        sample_results.results_cache[2] = {
            'atoms': atoms2,
            'force_error_magnitudes': np.array([0.3, 0.2, 0.1, 0.15])
        }
        
        # Export
        output_path = tmp_path / "structures.extxyz"
        export_structures_to_extxyz(
            [atoms1, atoms2], sample_results, output_path
        )
        
        assert output_path.exists()
        
        # Read back and check
        from ase.io import read
        loaded_atoms = read(output_path, index=':')
        
        assert len(loaded_atoms) == 2
        assert 'difficulty_metric' in loaded_atoms[0].info
        assert 'force_error_mag' in loaded_atoms[0].arrays
    
    def test_generate_markdown_report(self, sample_results, tmp_path):
        """Test markdown report generation."""
        report = generate_markdown_report(sample_results, top_n=2)
        
        assert "# Force Field Analysis Report" in report
        assert "Summary Statistics" in report
        assert "Most Difficult Structures" in report
        assert "Structure ID" in report
        
        # Test saving
        report_path = tmp_path / "report.md"
        generate_markdown_report(sample_results, output_path=report_path)
        assert report_path.exists()
    
    def test_export_for_visualization_json(self, sample_results, tmp_path):
        """Test JSON export for visualization."""
        # Export as dict
        export_dict = export_for_visualization(
            sample_results, output_format="json"
        )
        
        assert 'structures' in export_dict
        assert 'atoms' in export_dict
        assert 'metadata' in export_dict
        assert len(export_dict['structures']) == 2
        
        # Export to file
        json_path = tmp_path / "viz_data.json"
        export_for_visualization(
            sample_results,
            output_format="json",
            output_path=json_path
        )
        
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded['structures'][0]['structure_id'] == 1
    
    def test_export_for_visualization_csv(self, sample_results, tmp_path):
        """Test CSV export for visualization."""
        base_path = tmp_path / "viz_data"
        
        paths = export_for_visualization(
            sample_results,
            output_format="csv",
            output_path=base_path
        )
        
        assert "structures.csv" in paths
        assert "atoms.csv" in paths
        
        # Check files exist
        assert (tmp_path / "viz_data_structures.csv").exists()
        assert (tmp_path / "viz_data_atoms.csv").exists()
    
    def test_export_filtered_structures(self, sample_results, tmp_path):
        """Test exporting filtered subset of structures."""
        # Export only structure 1
        export_dict = export_for_visualization(
            sample_results,
            structure_ids=[1],
            output_format="json"
        )
        
        assert len(export_dict['structures']) == 1
        assert export_dict['structures'][0]['structure_id'] == 1
        assert len(export_dict['atoms']) == 2  # Two atoms for structure 1 