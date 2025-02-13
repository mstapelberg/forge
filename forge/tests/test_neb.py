import pytest
from pathlib import Path
import numpy as np
from ase.io import read
from mace.calculators.mace import MACECalculator
from ase.build import bulk
from forge.analysis.composition import CompositionAnalyzer

from forge.workflows.neb import (
    NEBCalculation,
    VacancyDiffusion,
    NEBAnalyzer,
    NEBMethod,
    NEBResult,
    NeighborInfo
)

# Path to test resources
RESOURCE_PATH = Path(__file__).parent / "resources"

def get_test_calculator():
    """Get MACE calculator for testing."""
    model_path = str(RESOURCE_PATH / "potentials/mace/gen_5_model_0-11-28_stagetwo.model")
    return MACECalculator(
        model_paths=[model_path],
        device="cpu",
        default_dtype="float32"
    )

@pytest.fixture
def test_structure():
    """Load test structure."""
    return read(RESOURCE_PATH / "structures/final_Cr132Ti177V3853W221Zr11.xyz")

@pytest.fixture
def vacancy_workflow(test_structure):
    """Create VacancyDiffusion workflow with test structure."""
    calc = get_test_calculator()
    return VacancyDiffusion(
        atoms=test_structure,
        calculator=calc,
        nn_cutoff=2.8,
        nnn_cutoff=3.2,
        seed=42
    )

@pytest.fixture
def small_test_structure():
    """Create a small V-Cr test structure for basic testing."""
    # Create 4x4x4 BCC V supercell
    vanadium = bulk('V', 'bcc', a=3.03, cubic=True)
    supercell = vanadium * (4, 4, 4)
    
    # Convert half of V atoms to Cr randomly with fixed seed
    rng = np.random.default_rng(42)
    n_atoms = len(supercell)
    n_cr = n_atoms // 2
    cr_indices = rng.choice(n_atoms, size=n_cr, replace=False)
    
    symbols = supercell.get_chemical_symbols()
    for idx in cr_indices:
        symbols[idx] = 'Cr'
    supercell.set_chemical_symbols(symbols)
    
    return supercell

@pytest.fixture
def small_vacancy_workflow(small_test_structure):
    """Create VacancyDiffusion workflow with small test structure."""
    calc = get_test_calculator()
    return VacancyDiffusion(
        atoms=small_test_structure,
        calculator=calc,
        nn_cutoff=2.8,
        nnn_cutoff=3.2,
        seed=42
    )

def test_neb_calculation_basic():
    """Test basic NEBCalculation functionality."""
    # Create simple start and end configurations
    start = bulk('V', 'bcc', a=3.03, cubic=True) * (2, 2, 2)
    end = start.copy()
    end.positions[0] += [0.1, 0, 0]  # Small displacement
    
    calc = get_test_calculator()
    
    neb = NEBCalculation(
        start_atoms=start,
        end_atoms=end,
        calculator=calc,
        n_images=3,
        method=NEBMethod.REGULAR,
        climbing=True,
        steps=10,  # Small number for testing
        seed=42
    )
    
    result = neb.run()
    assert isinstance(result, NEBResult)
    assert len(result.energies) == 5  # 3 images + 2 endpoints
    assert result.barrier >= 0

def test_neighbor_info():
    """Test NeighborInfo dataclass."""
    info = NeighborInfo(
        nn_indices=np.array([1, 2, 3]),
        nn_distances=np.array([2.5, 2.6, 2.7]),
        nnn_indices=np.array([4, 5, 6]),
        nnn_distances=np.array([3.1, 3.2, 3.3]),
        center_element="V",
        neighbor_elements={"Cr": [1, 4], "V": [2, 3, 5, 6]}
    )
    
    assert len(info.nn_indices) == 3
    assert len(info.nnn_indices) == 3
    assert "Cr" in info.neighbor_elements
    assert "V" in info.neighbor_elements

def test_vacancy_diffusion_init(small_test_structure):
    """Test VacancyDiffusion initialization."""
    calc = get_test_calculator()
    vd = VacancyDiffusion(
        atoms=small_test_structure,
        calculator=calc,
        nn_cutoff=2.8,
        nnn_cutoff=3.2,
        seed=42
    )
    
    assert vd.nn_cutoff == 2.8
    assert vd.nnn_cutoff == 3.2
    assert vd.seed == 42
    assert len(vd.atoms) == len(small_test_structure)

def test_vacancy_diffusion_neighbors(small_vacancy_workflow):
    """Test neighbor finding in VacancyDiffusion."""
    # Get a V atom
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V']
    test_idx = v_indices[0]
    
    neighbors = small_vacancy_workflow.get_neighbors(test_idx)
    
    # Test NeighborInfo properties
    assert isinstance(neighbors, NeighborInfo)
    assert len(neighbors.nn_indices) > 0
    assert len(neighbors.nnn_indices) > 0
    assert neighbors.center_element == 'V'
    
    # Test distance sorting
    assert np.all(np.diff(neighbors.nn_distances) >= 0)
    assert np.all(np.diff(neighbors.nnn_distances) >= 0)
    
    # Test neighbor cache
    cached_neighbors = small_vacancy_workflow.get_neighbors(test_idx)
    assert id(neighbors) == id(cached_neighbors)  # Should return cached result

def test_vacancy_diffusion_endpoints(small_vacancy_workflow):
    """Test endpoint creation in VacancyDiffusion."""
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V']
    test_idx = v_indices[0]
    
    neighbors = small_vacancy_workflow.get_neighbors(test_idx)
    target_idx = neighbors.nn_indices[0]
    
    start, end, metadata = small_vacancy_workflow.create_endpoints(
        vacancy_index=test_idx,
        target_index=target_idx,
        relax=False
    )
    
    # Test basic properties
    assert len(start) == len(small_vacancy_workflow.atoms) - 1
    assert len(end) == len(small_vacancy_workflow.atoms) - 1
    assert metadata["vacancy_element"] == "V"
    
    # Test atom positions
    vacancy_pos = small_vacancy_workflow.atoms.positions[test_idx]
    target_moved = np.allclose(end.positions[target_idx], vacancy_pos)
    assert target_moved

def test_neighbor_finding(small_vacancy_workflow):
    """Test neighbor finding functionality."""
    # Get neighbors for a V atom (most common element)
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V']
    test_idx = v_indices[0]
    
    neighbors = small_vacancy_workflow.get_neighbors(test_idx)
    
    # Basic checks
    assert len(neighbors.nn_indices) > 0, "Should find nearest neighbors"
    assert len(neighbors.nnn_indices) > 0, "Should find next-nearest neighbors"
    assert neighbors.center_element == 'V'
    
    # Check distances are sorted
    assert np.all(np.diff(neighbors.nn_distances) >= 0), "NN distances should be sorted"
    assert np.all(np.diff(neighbors.nnn_distances) >= 0), "NNN distances should be sorted"

def test_endpoint_creation(small_vacancy_workflow):
    """Test creation of NEB endpoints."""
    # Get a V atom and one of its neighbors
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V']
    test_idx = v_indices[0]
    
    neighbors = small_vacancy_workflow.get_neighbors(test_idx)
    target_idx = neighbors.nn_indices[0]
    
    # Create endpoints without relaxation for speed
    start, end, metadata = small_vacancy_workflow.create_endpoints(
        vacancy_index=test_idx,
        target_index=target_idx,
        relax=False
    )
    
    # Check basic properties
    assert len(start) == len(small_vacancy_workflow.atoms) - 1
    assert len(end) == len(small_vacancy_workflow.atoms) - 1
    assert metadata["vacancy_element"] == "V"
    
    # Check that atom IDs are preserved (except for removed atom)
    original_positions = small_vacancy_workflow.atoms.positions
    start_positions = start.positions
    end_positions = end.positions
    
    # The vacancy position should be the original position of test_idx
    vacancy_pos = original_positions[test_idx]
    
    # In the end configuration, target atom should be at vacancy position
    target_moved = np.allclose(end_positions[target_idx], vacancy_pos)
    assert target_moved, "Target atom should move to vacancy position"

def test_neighbor_sampling(small_vacancy_workflow):
    """Test random sampling of neighbors."""
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V'][:3]  # Test with first 3 V atoms
    
    # Sample with fixed seed
    pairs = small_vacancy_workflow.sample_neighbors(
        vacancy_indices=v_indices,
        n_nearest=2,
        n_next_nearest=1,
        rng_seed=42
    )
    
    # Check we get expected number of pairs
    expected_pairs = len(v_indices) * (2 + 1)  # 2 NN + 1 NNN per vacancy
    assert len(pairs) == expected_pairs
    
    # Check reproducibility
    pairs2 = small_vacancy_workflow.sample_neighbors(
        vacancy_indices=v_indices,
        n_nearest=2,
        n_next_nearest=1,
        rng_seed=42
    )
    assert pairs == pairs2, "Same seed should give same results"

def test_single_neb(small_vacancy_workflow):
    """Test running a single NEB calculation."""
    # Get a V atom and neighbor for testing
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V']
    test_idx = v_indices[0]
    
    neighbors = small_vacancy_workflow.get_neighbors(test_idx)
    target_idx = neighbors.nn_indices[0]
    
    # Run NEB with minimal steps for testing
    result = small_vacancy_workflow.run_single(
        vacancy_index=test_idx,
        target_index=target_idx,
        num_images=3,  # Minimal images for testing
        save_xyz=True,
        output_dir=RESOURCE_PATH / "test_output"
    )
    
    assert result["success"]
    assert result["barrier"] > 0
    assert len(result["energies"]) == 5  # 3 images + 2 endpoints
    
    # Check output files
    output_dir = RESOURCE_PATH / "test_output"
    assert output_dir.exists()
    
    formula = small_vacancy_workflow.atoms.get_chemical_formula()
    base_name = f"{formula}_V_to_{result['target_element']}_site{test_idx}_to_{target_idx}"
    
    assert (output_dir / f"{base_name}_initial.xyz").exists()
    assert (output_dir / f"{base_name}_final.xyz").exists()

def test_neb_analyzer_basic():
    """Test basic NEBAnalyzer functionality."""
    analyzer = NEBAnalyzer()
    
    # Add some test calculations
    test_calcs = [
        {
            "success": True,
            "vacancy_element": "V",
            "target_element": "Cr",
            "barrier": 0.5,
            "is_nearest_neighbor": True
        },
        {
            "success": True,
            "vacancy_element": "V",
            "target_element": "Cr",
            "barrier": 0.6,
            "is_nearest_neighbor": False
        }
    ]
    
    for calc in test_calcs:
        analyzer.add_calculation(calc)
    
    # Test filtering
    nn_calcs = analyzer.filter_calculations(neighbor_type="nn")
    assert len(nn_calcs) == 1
    
    v_calcs = analyzer.filter_calculations(vacancy_element="V")
    assert len(v_calcs) == 2
    
    # Test statistics
    stats = analyzer.calculate_statistics()
    assert "nearest_neighbor" in stats
    assert "next_nearest_neighbor" in stats
    assert "overall" in stats

def test_neb_analyzer_io(tmp_path):
    """Test NEBAnalyzer save/load functionality."""
    analyzer = NEBAnalyzer()
    
    # Add test calculation
    test_calc = {
        "success": True,
        "vacancy_element": "V",
        "target_element": "Cr",
        "barrier": 0.5,
        "is_nearest_neighbor": True
    }
    analyzer.add_calculation(test_calc)
    
    # Save results
    save_path = tmp_path / "test_results.json"
    analyzer.save_results(save_path, composition={"V": 0.8, "Cr": 0.2})
    
    # Load results
    loaded = NEBAnalyzer.load_results(save_path)
    assert len(loaded.calculations) == len(analyzer.calculations)
    assert loaded.calculations[0]["barrier"] == test_calc["barrier"]

def test_analyzer(small_vacancy_workflow):
    """Test NEBAnalyzer functionality."""
    # Run a few NEB calculations
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V'][:2]
    
    results = small_vacancy_workflow.run_multiple(
        vacancy_indices=v_indices,
        n_nearest=1,
        n_next_nearest=1,
        save_xyz=False
    )
    
    # Test filtering
    filtered = small_vacancy_workflow.analyzer.filter_calculations(
        vacancy_element='V',
        neighbor_type='nn'
    )
    assert len(filtered) > 0
    
    # Test statistics
    stats = small_vacancy_workflow.analyzer.calculate_statistics()
    assert "nearest_neighbor" in stats
    assert "next_nearest_neighbor" in stats
    assert "overall" in stats
    
    # Test saving and loading
    save_path = RESOURCE_PATH / "test_output" / "test_results.json"
    small_vacancy_workflow.analyzer.save_results(
        save_path,
        composition={'V': 0.8, 'Cr': 0.2}
    )
    
    loaded = NEBAnalyzer.load_results(save_path)
    assert len(loaded.calculations) == len(small_vacancy_workflow.analyzer.calculations)

def test_small_structure_neb(small_vacancy_workflow):
    """Test NEB workflow on small test structure."""
    # Get a V atom and its Cr neighbor
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V']
    
    # Find a V atom with a Cr neighbor
    for v_idx in v_indices:
        neighbors = small_vacancy_workflow.get_neighbors(v_idx)
        cr_neighbors = neighbors.neighbor_elements.get('Cr', [])
        if cr_neighbors:
            test_idx = v_idx
            target_idx = cr_neighbors[0]
            break
    else:
        pytest.skip("Could not find V atom with Cr neighbor")
    
    # Run single NEB
    result = small_vacancy_workflow.run_single(
        vacancy_index=test_idx,
        target_index=target_idx,
        num_images=3,
        save_xyz=True,
        output_dir=RESOURCE_PATH / "test_output"
    )
    
    assert result["success"]
    assert result["barrier"] > 0
    assert result["vacancy_element"] == "V"
    assert result["target_element"] == "Cr"

def test_small_structure_multiple(small_vacancy_workflow):
    """Test multiple NEB calculations on small structure."""
    # Get first 2 V atoms
    v_indices = [i for i, atom in enumerate(small_vacancy_workflow.atoms) 
                if atom.symbol == 'V'][:2]
    
    results = small_vacancy_workflow.run_multiple(
        vacancy_indices=v_indices,
        n_nearest=2,  # Sample 2 NN
        n_next_nearest=1,  # Sample 1 NNN
        save_xyz=False,
        rng_seed=42
    )
    
    # Should have 6 calculations (3 per V atom)
    assert len(results) == 6
    
    # Check reproducibility
    results2 = small_vacancy_workflow.run_multiple(
        vacancy_indices=v_indices,
        n_nearest=2,
        n_next_nearest=1,
        save_xyz=False,
        rng_seed=42
    )
    
    # Same seed should give same vacancy-target pairs
    for r1, r2 in zip(results, results2):
        assert r1["vacancy_index"] == r2["vacancy_index"]
        assert r1["target_index"] == r2["target_index"] 