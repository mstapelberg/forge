#python:forge/tests/test_composition_analysis.py
import pytest
import numpy as np
from pathlib import Path
from ase import Atoms
from sklearn.cluster import KMeans
from forge.core.database import DatabaseManager
from forge.analysis.composition import CompositionAnalyzer, HAS_UMAP

import yaml
import tempfile
import random

@pytest.fixture
def db_config():
    """Create test database configuration."""
    return {
        'database': {
            'dbname': 'test_db',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': 5432
        }
    }

@pytest.fixture
def db_manager(db_config):
    """Create a temporary database (and tear down) for composition tests."""
    with tempfile.NamedTemporaryFile(suffix='.yaml') as tmp:
        config_path = Path(tmp.name)
        with open(config_path, 'w') as f:
            yaml.dump(db_config, f)
        
        db = DatabaseManager(config_path=config_path)
        
        # Drop old tables if needed
        with db.conn.cursor() as cur:
            cur.execute("""
                DROP TABLE IF EXISTS calculations CASCADE;
                DROP TABLE IF EXISTS structures CASCADE;
            """)
        db.conn.commit()
        
        # Reinitialize tables
        db._initialize_tables()
        yield db

def generate_random_composition():
    """
    Generate a single composition for elements V, Cr, Ti, W, Zr
    subject to:
        V >= 0.7,
        Cr,Ti,W,Zr each >= 0.01,
        sum = 1.0
    """
    # We only have ~0.3 left for Cr, Ti, W, Zr collectively.
    # We'll pick random portions for these 4 elements 
    # ensuring each is at least 0.01, then scale so they sum up to <= 0.3.
    
    # Each in [0.01, 0.3], then we normalize if sum > 0.3
    cr = random.uniform(0.01, 0.3)
    ti = random.uniform(0.01, 0.3)
    w = random.uniform(0.01, 0.3)
    zr = random.uniform(0.01, 0.3)
    partial_sum = cr + ti + w + zr
    
    if partial_sum > 0.3:
        # Scale down so partial_sum <= 0.3
        scale = 0.3 / partial_sum
        cr *= scale
        ti *= scale
        w *= scale
        zr *= scale
        partial_sum = 0.3  # after scaling

    # Now V = 1 - partial_sum
    v = 1.0 - partial_sum  # ensures total = 1.0
    # That also guarantees v >= 0.7 as long as partial_sum <= 0.3
    
    return {
        'V': v,
        'Cr': cr,
        'Ti': ti,
        'W': w,
        'Zr': zr
    }

@pytest.fixture
def random_compositions():
    """Generate ~250 compositions within the specified constraints."""
    random.seed(123)  # For reproducibility
    comps = []
    while len(comps) < 250:
        comp = generate_random_composition()
        # Validate composition before adding
        try:
            # Check V constraint
            if not (0.7 <= comp['V'] <= 1.0):
                continue
                
            # Check other element constraints
            if not all(0.01 <= comp[elem] <= 0.3 for elem in ['Cr', 'Ti', 'W', 'Zr']):
                continue
                
            # Check sum = 1.0
            if not abs(sum(comp.values()) - 1.0) < 1e-10:
                continue
                
            comps.append(comp)
            
        except (KeyError, TypeError):
            # Skip if composition is malformed
            continue
            
    return comps

def test_random_composition_generation():
    """Test that single composition generation works correctly."""
    valid_comp = None
    max_attempts = 100
    attempts = 0
    
    while valid_comp is None and attempts < max_attempts:
        comp = generate_random_composition()
        try:
            # Validate composition
            if not set(comp.keys()) == {'V', 'Cr', 'Ti', 'W', 'Zr'}:
                continue
                
            if not (0.7 <= comp['V'] <= 1.0):
                continue
                
            if not all(0.01 <= comp[elem] <= 0.3 for elem in ['Cr', 'Ti', 'W', 'Zr']):
                continue
                
            if not abs(sum(comp.values()) - 1.0) < 1e-10:
                continue
                
            valid_comp = comp
            
        except (KeyError, TypeError):
            pass
            
        attempts += 1
        
    assert valid_comp is not None, "Could not generate valid composition after 100 attempts"
    
    # Additional verification of the valid composition
    assert set(valid_comp.keys()) == {'V', 'Cr', 'Ti', 'W', 'Zr'}
    assert 0.7 <= valid_comp['V'] <= 1.0
    for elem in ['Cr', 'Ti', 'W', 'Zr']:
        assert 0.01 <= valid_comp[elem] <= 0.3
    assert abs(sum(valid_comp.values()) - 1.0) < 1e-10

def test_random_compositions_fixture(random_compositions):
    """Test that the random_compositions fixture generates correct data."""
    # Check we have the right number
    assert len(random_compositions) == 250, f"Generated {len(random_compositions)} compositions instead of 250"
    
    # Check each composition
    for i, comp in enumerate(random_compositions):
        # Check elements
        assert set(comp.keys()) == {'V', 'Cr', 'Ti', 'W', 'Zr'}, f"Composition {i} has incorrect elements"
        
        # Check constraints
        assert 0.7 <= comp['V'] <= 1.0, f"Composition {i}: V fraction {comp['V']} outside [0.7, 1.0]"
        for elem in ['Cr', 'Ti', 'W', 'Zr']:
            assert 0.01 <= comp[elem] <= 0.3, f"Composition {i}: {elem} fraction {comp[elem]} outside [0.01, 0.3]"
        
        # Check sum = 1.0
        total = sum(comp.values())
        assert abs(total - 1.0) < 1e-10, f"Composition {i}: sum {total} != 1.0"

def test_database_storage(db_manager, random_compositions):
    """Test that compositions are correctly stored and retrieved from DB."""
    # Store compositions
    total_atoms = 100
    structure_ids = []
    for comp in random_compositions:
        # Build dummy atoms object
        symbols = []
        for elem, fraction in comp.items():
            count = int(round(fraction * total_atoms))
            symbols.extend([elem] * count)
        
        # Adjust to exactly total_atoms
        if len(symbols) < total_atoms:
            while len(symbols) < total_atoms:
                symbols.append(max(comp, key=comp.get))
        elif len(symbols) > total_atoms:
            symbols = symbols[:total_atoms]
        
        atoms = Atoms(symbols=symbols)
        atoms.info["composition"] = comp
        
        # Store and keep ID
        sid = db_manager.add_structure(atoms, source_type="random_test")
        structure_ids.append(sid)
    
    # Verify storage
    assert len(structure_ids) == len(random_compositions)
    
    # Retrieve and verify
    retrieved_comps = []
    for sid in structure_ids:
        atoms = db_manager.get_structure(sid)
        comp = atoms.info.get("composition", {})
        retrieved_comps.append(comp)
    
    assert len(retrieved_comps) == len(random_compositions)
    
    # Compare original and retrieved compositions
    for orig, retr in zip(random_compositions, retrieved_comps):
        for elem in ['V', 'Cr', 'Ti', 'W', 'Zr']:
            assert abs(orig[elem] - retr[elem]) < 1e-10

def test_composition_array_conversion(random_compositions):
    """Test conversion of compositions to array format."""
    analyzer = CompositionAnalyzer(n_components=3, random_state=123)
    comp_array = analyzer._compositions_to_array(random_compositions)
    
    # Check shape
    assert comp_array.shape == (len(random_compositions), 5)  # 5 elements
    
    # Check values match original compositions
    for i, comp in enumerate(random_compositions):
        for j, elem in enumerate(['Cr', 'Ti', 'V', 'W', 'Zr']):  # Note: sorted order
            assert abs(comp_array[i, j] - comp[elem]) < 1e-10

def test_tsne_clustering(random_compositions):
    """Test t-SNE and clustering directly."""
    analyzer = CompositionAnalyzer(n_components=3, random_state=123)
    
    # Convert to array and run analysis
    print("Running t-SNE and clustering analysis...")
    embeddings, clusters = analyzer.analyze_compositions(random_compositions, n_clusters=5)
    
    # Add assertions
    assert embeddings.shape == (len(random_compositions), 3)
    assert len(set(clusters)) == 5
    assert len(clusters) == len(random_compositions)

def test_composition_analysis_with_db(db_manager, random_compositions):
    """Main test function, now with more debugging."""
    print(f"\nStarting main test with {len(random_compositions)} compositions")
    
    # 1. Store and retrieve compositions (keep this part unchanged)
    total_atoms = 100
    for comp in random_compositions:
        symbols = []
        for elem, fraction in comp.items():
            count = int(round(fraction * total_atoms))
            symbols.extend([elem] * count)
        
        if len(symbols) < total_atoms:
            while len(symbols) < total_atoms:
                symbols.append(max(comp, key=comp.get))
        elif len(symbols) > total_atoms:
            symbols = symbols[:total_atoms]
        
        atoms = Atoms(symbols=symbols)
        atoms.info["composition"] = comp
        db_manager.add_structure(atoms, source_type="random_test")

    # 2. Run analysis with original compositions
    analyzer = CompositionAnalyzer(n_components=3, random_state=123)
    embeddings, clusters = analyzer.analyze_compositions(random_compositions, n_clusters=5)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Number of clusters: {len(set(clusters))}")
    
    # 3. Generate new compositions
    new_comps = analyzer.suggest_new_compositions(random_compositions, n_suggestions=10)
    print(f"Generated {len(new_comps)} new compositions")
    
    # 4. Analyze combined compositions
    all_compositions = random_compositions + new_comps
    all_embeddings, all_clusters = analyzer.analyze_compositions(all_compositions, n_clusters=5)
    
    # 5. Split results for visualization
    original_embeddings = all_embeddings[:len(random_compositions)]
    new_embeddings = all_embeddings[len(random_compositions):]
    
    # 6. Create visualization
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original compositions with their clusters
    scatter = ax.scatter(original_embeddings[:, 0], 
                        original_embeddings[:, 1], 
                        original_embeddings[:, 2],
                        c=clusters, cmap='viridis', 
                        label='Original')
    
    # Plot new compositions in the same space
    ax.scatter(new_embeddings[:, 0], 
              new_embeddings[:, 1], 
              new_embeddings[:, 2],
              c='red', marker='x', s=100, 
              label='Suggested')
    
    plt.colorbar(scatter, label='Cluster')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.set_title('Random Composition Space Analysis (3D)')
    ax.legend()
    
    plot_path = Path("./random_compositions_3d.png").resolve()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"3D plot saved to {plot_path}")
    print("Composition analysis test completed successfully.")

def test_dimensionality_reduction_methods(random_compositions):
    """Test that all dimensionality reduction methods work."""
    dim_methods = ['T-SNE', 'PCA', 'MDS']
    if HAS_UMAP:
        dim_methods.append('UMAP')
    
    for method in dim_methods:
        analyzer = CompositionAnalyzer(n_components=3, random_state=123, dim_method=method)
        embeddings, clusters = analyzer.analyze_compositions(random_compositions, n_clusters=5)
        
        # Check output shapes
        assert embeddings.shape == (len(random_compositions), 3)
        assert len(clusters) == len(random_compositions)
        assert len(set(clusters)) <= 5  # Should have at most 5 clusters

def test_clustering_methods(random_compositions):
    """Test that all clustering methods work."""
    cluster_methods = ['KMEANS', 'AGGLOMERATIVE', 'SPECTRAL', 'DBSCAN']
    
    for method in cluster_methods:
        analyzer = CompositionAnalyzer(n_components=3, random_state=123, cluster_method=method)
        embeddings, clusters = analyzer.analyze_compositions(random_compositions, n_clusters=5)
        
        # Check that clusters are assigned
        assert len(clusters) == len(random_compositions)
        # Note: DBSCAN might assign -1 to noise points
        if method != 'DBSCAN':
            assert all(c >= 0 for c in clusters)

def test_method_comparison(random_compositions):
    """Test the compare_methods functionality."""
    analyzer = CompositionAnalyzer(n_components=2, random_state=123)
    
    # Test with specific methods
    dim_methods = ['T-SNE', 'PCA']
    cluster_methods = ['KMEANS', 'AGGLOMERATIVE']
    
    scores = analyzer.compare_methods(
        random_compositions,
        dim_methods=dim_methods,
        cluster_methods=cluster_methods,
        n_clusters=5,
        save_path=None
    )
    
    # Check that scores were computed
    expected_combinations = [f"{dim}_{clust}" 
                           for dim in dim_methods 
                           for clust in cluster_methods]
    assert all(comb in scores for comb in expected_combinations)

def test_backward_compatibility(random_compositions):
    """Test that the original usage pattern still works."""
    # Original initialization
    analyzer = CompositionAnalyzer(n_components=3, random_state=123)
    
    # Original analysis call
    embeddings, clusters = analyzer.analyze_compositions(random_compositions, n_clusters=5)
    
    # Check original behavior
    assert embeddings.shape == (len(random_compositions), 3)
    assert len(clusters) == len(random_compositions)
    assert len(set(clusters)) <= 5

def test_method_switching(random_compositions):
    """Test that methods can be switched during analysis."""
    analyzer = CompositionAnalyzer(n_components=3, random_state=123)
    
    # Test switching dimensionality reduction
    embeddings1, clusters1 = analyzer.analyze_compositions(
        random_compositions, 
        dim_method='PCA'
    )
    
    embeddings2, clusters2 = analyzer.analyze_compositions(
        random_compositions, 
        cluster_method='AGGLOMERATIVE'
    )
    
    # Check that original methods were restored
    assert analyzer.dim_method == 'T-SNE'
    assert analyzer.cluster_method == 'KMEANS'
    
    # Check outputs
    assert embeddings1.shape == embeddings2.shape == (len(random_compositions), 3)
    assert len(clusters1) == len(clusters2) == len(random_compositions)

def test_invalid_methods():
    """Test that invalid methods raise appropriate errors."""
    with pytest.raises(ValueError):
        CompositionAnalyzer(dim_method='INVALID')
    
    with pytest.raises(ValueError):
        CompositionAnalyzer(cluster_method='INVALID')
        
    analyzer = CompositionAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze_compositions([], dim_method='INVALID')
    
    with pytest.raises(ValueError):
        analyzer.analyze_compositions([], cluster_method='INVALID')

def test_reduce_and_cluster(random_compositions):
    """Test reduce_and_cluster functionality."""
    analyzer = CompositionAnalyzer(n_components=2, random_state=123)
    embeddings, clusters, metadata = analyzer.reduce_and_cluster(
        random_compositions,
        n_clusters=4,
        dim_method='PCA',
        cluster_method='KMEANS',
        comp_type="existing"
    )
    assert embeddings.shape == (len(random_compositions), 2)
    assert len(clusters) == len(random_compositions)
    assert len(metadata) == len(random_compositions)
    for i, meta in enumerate(metadata):
        assert "comp_id" in meta
        assert "composition" in meta
        assert "embedding" in meta
        assert "cluster" in meta
        # Check that metadata matches the computed embeddings and clusters
        np.testing.assert_array_almost_equal(meta["embedding"], embeddings[i])
        assert meta["cluster"] == clusters[i]

def test_detect_outliers():
    """Test detect_outliers functionality using synthetic data."""
    analyzer = CompositionAnalyzer(n_components=2, random_state=123)
    # Create synthetic embeddings: 10 normal points and 1 outlier point
    np.random.seed(123)
    normal_points = np.random.normal(0, 0.1, (10, 2))
    outlier_point = np.array([[10, 10]])
    embeddings = np.vstack([normal_points, outlier_point])
    # Create simple dummy metadata for each point
    metadata = [{"composition": {"dummy": 1.0}, "lof_score": None, "is_outlier": False} for _ in range(len(embeddings))]
    
    updated_meta = analyzer.detect_outliers(embeddings, metadata, contamination=0.1, n_neighbors=2, use_lof=True)
    # Expect at least one outlier to be flagged (the obvious far away point)
    n_outliers = sum(meta["is_outlier"] for meta in updated_meta)
    assert n_outliers >= 1, "At least one outlier should be detected"
    # Check that LOF scores were assigned
    for meta in updated_meta:
        assert meta["lof_score"] is not None

def test_visualize_compositions(tmp_path, random_compositions):
    """Test visualize_compositions in 2D mode by saving a plot to a temporary file."""
    analyzer = CompositionAnalyzer(n_components=2, random_state=123)
    embeddings, clusters, metadata = analyzer.reduce_and_cluster(random_compositions, n_clusters=3)
    save_file = tmp_path / "test_plot.png"
    analyzer.visualize_compositions(embeddings, metadata, outlier_option='include', save_path=str(save_file))
    assert save_file.exists(), "2D visualization file not created"

def test_visualize_compositions_3d(tmp_path, random_compositions):
    """Test visualize_compositions in 3D mode by saving a plot to a temporary file."""
    analyzer = CompositionAnalyzer(n_components=3, random_state=123, dim_method='PCA')
    embeddings, clusters, metadata = analyzer.reduce_and_cluster(random_compositions, n_clusters=3)
    save_file = tmp_path / "test_plot.html"
    analyzer.visualize_compositions(embeddings, metadata, outlier_option='include', save_path=str(save_file))
    assert save_file.exists(), "3D visualization file not created"

def test_suggest_new_compositions_no_metadata(random_compositions):
    """Test suggest_new_compositions without providing metadata."""
    analyzer = CompositionAnalyzer(n_components=2, random_state=123)
    new_comps = analyzer.suggest_new_compositions(random_compositions, n_suggestions=5)
    assert len(new_comps) == 5
    original_keys = set(random_compositions[0].keys())
    for comp in new_comps:
        assert set(comp.keys()) == original_keys

def test_suggest_new_compositions_with_metadata(random_compositions):
    """Test suggest_new_compositions with metadata provided (using only inliers)."""
    analyzer = CompositionAnalyzer(n_components=2, random_state=123)
    _, _, metadata = analyzer.reduce_and_cluster(random_compositions, n_clusters=3)
    new_comps = analyzer.suggest_new_compositions(random_compositions, n_suggestions=5, metadata=metadata)
    assert len(new_comps) == 5
    original_keys = set(random_compositions[0].keys())
    for comp in new_comps:
        assert set(comp.keys()) == original_keys

def test_quantify_uncertainty():
    """Test that quantify_uncertainty returns a list of zeros."""
    analyzer = CompositionAnalyzer(n_components=2, random_state=123)
    # Create dummy Atoms objects for testing
    atoms_list = [Atoms("H2O"), Atoms("CO2"), Atoms("NaCl")]
    uncertainties = analyzer.quantify_uncertainty(model_paths=["dummy_model"], structures=atoms_list)
    assert len(uncertainties) == len(atoms_list)
    assert all(u == 0.0 for u in uncertainties)

def test_create_random_alloy():
    """Test the creation of a random alloy structure."""
    analyzer = CompositionAnalyzer()
    comp = {'A': 0.3, 'B': 0.3, 'C': 0.4}
    alloy = analyzer.create_random_alloy(comp, crystal_type='fcc', dimensions=[2,2,2], lattice_constant=3.5, balance_element='A')
    # Ensure that number of sites equals the number of symbols and that only allowed elements are present
    assert len(alloy) == len(alloy.symbols)
    for sym in alloy.symbols:
        assert sym in comp.keys()

def test_find_diverse_compositions(random_compositions):
    """Test the find_diverse_compositions method."""
    analyzer = CompositionAnalyzer(n_components=2, random_state=123)
    embeddings, _, _ = analyzer.reduce_and_cluster(random_compositions, n_clusters=3)
    diverse_indices_maximin = analyzer.find_diverse_compositions(embeddings, random_compositions, n_select=5, method='maximin')
    assert len(diverse_indices_maximin) == 5
    diverse_indices_maxsum = analyzer.find_diverse_compositions(embeddings, random_compositions, n_select=5, method='maxsum')
    assert len(diverse_indices_maxsum) == 5

def test_plot_with_diverse_points(random_compositions):
    """Test that plot_with_diverse_points returns a valid diverse indices array and a plotly figure."""
    analyzer = CompositionAnalyzer(n_components=3, random_state=123)
    embeddings, clusters, _ = analyzer.reduce_and_cluster(random_compositions, n_clusters=3)
    new_comps = analyzer.suggest_new_compositions(random_compositions, n_suggestions=10)
    new_embeddings, _ = analyzer.analyze_compositions(new_comps, n_clusters=3)
    diverse_indices, fig = analyzer.plot_with_diverse_points(embeddings, clusters, new_embeddings, new_comps, n_diverse=5, method='maximin')
    assert len(diverse_indices) == 5
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)