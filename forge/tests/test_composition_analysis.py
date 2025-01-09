#python:forge/tests/test_composition_analysis.py
import pytest
import numpy as np
from pathlib import Path
from ase import Atoms
from sklearn.cluster import KMeans
from forge.core.database import DatabaseManager
from forge.analysis.composition import CompositionAnalyzer

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
    
    # Convert to array
    comp_array = analyzer._compositions_to_array(random_compositions)
    print(f"Composition array shape: {comp_array.shape}")
    
    # Try t-SNE directly
    print("Running t-SNE...")
    embeddings = analyzer.tsne.fit_transform(comp_array)
    print(f"t-SNE embeddings shape: {embeddings.shape}")
    
    # Try clustering
    print("Running clustering...")
    analyzer.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = analyzer.kmeans.fit_predict(comp_array)
    print(f"Number of clusters: {len(set(clusters))}")
    
    # Add assertions instead of returning values
    assert embeddings.shape == (len(random_compositions), 3)
    assert len(set(clusters)) == 5
    assert len(clusters) == len(random_compositions)

def test_composition_analysis_with_db(db_manager, random_compositions):
    """Main test function, now with more debugging."""
    print(f"\nStarting main test with {len(random_compositions)} compositions")
    
    # 1. Store in DB (reusing test_database_storage logic)
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

    # 2. Retrieve from DB
    with db_manager.conn.cursor() as cur:
        cur.execute("SELECT structure_id FROM structures")
        all_ids = [row[0] for row in cur.fetchall()]
        db_manager.conn.commit()
    
    print(f"Retrieved {len(all_ids)} structure IDs from database")
    
    compositions = []
    for sid in all_ids:
        atoms = db_manager.get_structure(sid)
        comp_dict = atoms.info.get("composition", {})
        compositions.append(comp_dict)
    
    print(f"Retrieved {len(compositions)} compositions from database")
    
    # 3. Run analysis
    analyzer = CompositionAnalyzer(n_components=3, random_state=123)
    
    # Debug the composition array
    comp_array = analyzer._compositions_to_array(compositions)
    print(f"Composition array shape before t-SNE: {comp_array.shape}")
    
    # Run t-SNE and clustering
    embeddings, clusters = analyzer.analyze_compositions(compositions, n_clusters=5)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Number of clusters: {len(set(clusters))}")
    
    # Generate new compositions
    new_comps = analyzer.suggest_new_compositions(compositions, n_suggestions=10)
    print(f"Generated {len(new_comps)} new compositions")
    
    # Combine original and new compositions for t-SNE
    all_compositions = compositions + new_comps
    print(f"Total compositions for t-SNE: {len(all_compositions)}")
    
    # Convert all compositions to array format
    all_comp_array = analyzer._compositions_to_array(all_compositions)
    print(f"Combined composition array shape: {all_comp_array.shape}")
    
    # Run t-SNE on all compositions together
    all_embeddings = analyzer.tsne.fit_transform(all_comp_array)
    print(f"Combined embeddings shape: {all_embeddings.shape}")
    
    # Split embeddings back into original and new
    original_embeddings = all_embeddings[:len(compositions)]
    new_embeddings = all_embeddings[len(compositions):]
    
    # Create visualization
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