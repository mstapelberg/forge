# forge/tests/test_nnmd.py
import pytest
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from forge.workflows.nnmd.nnmd import CompositionAnalyzer

def test_analyze_compositions():
    compositions = [
        {'H': 0.5, 'O': 0.5},
        {'H': 0.6, 'O': 0.4},
        {'H': 0.4, 'O': 0.6},
        {'H': 0.7, 'O': 0.3},
        {'H': 0.3, 'O': 0.7}
    ]
    analyzer = CompositionAnalyzer()
    embeddings, clusters = analyzer.analyze_compositions(compositions, n_clusters=2)
    
    assert embeddings.shape[0] == len(compositions)
    assert len(clusters) == len(compositions)
    assert len(set(clusters)) == 2

def test_compositions_to_array():
    compositions = [
        {'H': 0.5, 'O': 0.5},
        {'H': 0.6, 'O': 0.4},
        {'H': 0.4, 'O': 0.6}
    ]
    analyzer = CompositionAnalyzer()
    comp_array = analyzer._compositions_to_array(compositions)
    
    expected_array = np.array([
        [0.5, 0.5],
        [0.6, 0.4],
        [0.4, 0.6]
    ])
    
    np.testing.assert_array_almost_equal(comp_array, expected_array)

def test_suggest_new_compositions():
    compositions = [
        {'H': 0.5, 'O': 0.5},
        {'H': 0.6, 'O': 0.4},
        {'H': 0.4, 'O': 0.6},
        {'H': 0.7, 'O': 0.3},
        {'H': 0.3, 'O': 0.7}
    ]
    analyzer = CompositionAnalyzer()
    analyzer.analyze_compositions(compositions, n_clusters=2)
    new_compositions = analyzer.suggest_new_compositions(compositions, n_suggestions=3)
    
    assert len(new_compositions) == 3
    for comp in new_compositions:
        assert 'H' in comp and 'O' in comp
        assert 0 <= comp['H'] <= 1
        assert 0 <= comp['O'] <= 1

def test_plot_analysis(tmp_path):
    compositions = [
        {'H': 0.5, 'O': 0.5},
        {'H': 0.6, 'O': 0.4},
        {'H': 0.4, 'O': 0.6},
        {'H': 0.7, 'O': 0.3},
        {'H': 0.3, 'O': 0.7}
    ]
    analyzer = CompositionAnalyzer()
    embeddings, clusters = analyzer.analyze_compositions(compositions, n_clusters=2)
    
    save_path = tmp_path / "plot.png"
    analyzer.plot_analysis(embeddings, clusters, save_path=str(save_path))
    
    assert save_path.exists()