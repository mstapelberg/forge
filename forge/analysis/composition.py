from typing import List, Dict, Optional, Tuple
import numpy as np
from ase import Atoms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import json

class CompositionAnalyzer:
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.tsne = TSNE(n_components=n_components, random_state=random_state)
        self.kmeans = None
        
    def analyze_compositions(self, compositions: List[Dict[str, float]], 
                           n_clusters: int = 5):
        """Analyze composition space using t-SNE and clustering"""
        # Convert compositions to array
        comp_array = self._compositions_to_array(compositions)
        
        # Perform t-SNE
        embeddings = self.tsne.fit_transform(comp_array)
        
        # Perform clustering with explicit n_init
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(comp_array)
        
        return embeddings, clusters
        
    def _compositions_to_array(self, compositions: List[Dict[str, float]]) -> np.ndarray:
        """Convert composition dictionaries to numpy array"""
        elements = sorted(set().union(*compositions))
        return np.array([[comp.get(elem, 0.0) for elem in elements]
                        for comp in compositions])
                        
    def plot_analysis(self, embeddings: np.ndarray, clusters: np.ndarray,
                     save_path: Optional[str] = None):
        """Create visualization of composition analysis"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                            c=clusters, cmap='viridis')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Composition Space Analysis')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def suggest_new_compositions(self, compositions: List[Dict[str, float]],
                               n_suggestions: int = 10,
                               constraints: Optional[Dict[str, Tuple[float, float]]] = None
                               ) -> List[Dict[str, float]]:
        """Suggest new compositions based on clustering analysis"""
        if self.kmeans is None:
            raise ValueError("Must run analyze_compositions first")
            
        # Find cluster centers
        centers = self.kmeans.cluster_centers_
        
        # Generate new compositions near cluster boundaries
        new_compositions = []
        max_attempts = n_suggestions * 10  # Maximum attempts to find valid compositions
        attempts = 0
        
        while len(new_compositions) < n_suggestions and attempts < max_attempts:
            # Select random pair of clusters
            c1, c2 = np.random.choice(len(centers), 2, replace=False)
            
            # Generate composition between clusters
            mix = np.random.random()
            new_comp = centers[c1] * mix + centers[c2] * (1 - mix)
            
            # Convert to dictionary
            elements = sorted(set().union(*compositions))
            comp_dict = dict(zip(elements, new_comp))
            
            # Check constraints if provided
            if constraints:
                valid = True
                for element, (min_frac, max_frac) in constraints.items():
                    if element in comp_dict:
                        if not (min_frac <= comp_dict[element] <= max_frac):
                            valid = False
                            break
                
                if not valid:
                    attempts += 1
                    continue
            
            new_compositions.append(comp_dict)
            attempts += 1
            
        return new_compositions