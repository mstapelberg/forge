from typing import List, Dict, Optional, Tuple
import numpy as np
from ase import Atoms
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import json
from ase.build import bulk
import random
from scipy.spatial.distance import pdist, squareform

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed. Please install umap-learn to use it, using t-SNE instead")


class CompositionAnalyzer:
    def __init__(self, n_components: int = 2, random_state: int = 42, 
                 dim_method: str = 't-SNE', cluster_method: str = 'KMeans'):
        """
        Initialize the CompositionAnalyzer for analyzing chemical composition spaces.

        This class provides tools for dimensionality reduction and clustering of chemical
        compositions, supporting multiple methods for both tasks. It can visualize the
        reduced composition space and identify clusters of similar compositions.

        Args:
            n_components (int, optional): Number of dimensions for reduced space. 
                Typically 2 or 3 for visualization. Defaults to 2.
            random_state (int, optional): Seed for reproducible results. Defaults to 42.
            dim_method (str, optional): Dimensionality reduction method to use. Options:
                - 't-SNE': t-Distributed Stochastic Neighbor Embedding (preserves local structure)
                - 'PCA': Principal Component Analysis (preserves global variance)
                - 'MDS': Multidimensional Scaling (preserves distances)
                - 'UMAP': Uniform Manifold Approximation and Projection (if installed)
                Defaults to 't-SNE'.
            cluster_method (str, optional): Clustering algorithm to use. Options:
                - 'KMeans': K-means clustering (assumes spherical clusters)
                - 'AGGLOMERATIVE': Hierarchical clustering
                - 'SPECTRAL': Spectral clustering (good for non-spherical clusters)
                - 'DBSCAN': Density-based clustering (handles noise, variable density)
                Defaults to 'KMeans'.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.dim_method = dim_method.upper()  # Store in uppercase for case-insensitive comparison
        self.cluster_method = cluster_method.upper()
        
        # Initialize methods
        self.reducer = self.initialize_reducer()
        self.clusterer = self.initialize_cluster_method()  # Renamed from kmeans for clarity

    def initialize_reducer(self):
        """Initialize dimensionality reduction method based on self.dim_method."""
        method = self.dim_method.upper()  # Normalize case
        
        if method in ['T-SNE', 'TSNE']:  # Allow both forms
            return TSNE(n_components=self.n_components, random_state=self.random_state)
        elif method == 'PCA':
            return PCA(n_components=self.n_components, random_state=self.random_state)
        elif method == 'MDS':
            return MDS(n_components=self.n_components, random_state=self.random_state, n_init=4, normalized_stress='auto')
        elif method == 'UMAP':
            if not HAS_UMAP:
                print("UMAP not installed, falling back to t-SNE")
                self.dim_method = 'T-SNE'
                return TSNE(n_components=self.n_components, random_state=self.random_state)
            return umap.UMAP(n_components=self.n_components, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {self.dim_method}. "
                           f"Supported methods are: 't-SNE', 'PCA', 'MDS', 'UMAP'")

    def initialize_cluster_method(self):
        """Initialize clustering method based on self.cluster_method."""
        if self.cluster_method == 'KMEANS':
            return KMeans(random_state=self.random_state, n_init=10)
        elif self.cluster_method == 'AGGLOMERATIVE':
            from sklearn.cluster import AgglomerativeClustering
            return AgglomerativeClustering()
        elif self.cluster_method == 'DBSCAN':
            from sklearn.cluster import DBSCAN
            return DBSCAN(min_samples=5)
        elif self.cluster_method == 'SPECTRAL':
            from sklearn.cluster import SpectralClustering
            return SpectralClustering(random_state=self.random_state, n_init=10)
        else:
            raise ValueError(f"Unsupported clustering method: {self.cluster_method}")

    def compare_methods(self, compositions: List[Dict[str, float]], 
                       dim_methods: Optional[List[str]] = None,
                       cluster_methods: Optional[List[str]] = None,
                       n_clusters: int = 5,
                       save_path: Optional[str] = None):
        """
        Compare different dimensionality reduction and clustering methods.
        
        Args:
            compositions: List of composition dictionaries
            dim_methods: List of dimensionality reduction methods to try
            cluster_methods: List of clustering methods to try
            n_clusters: Number of clusters for applicable methods
            save_path: Path to save comparison plot
        """
        if dim_methods is None:
            dim_methods = ['T-SNE', 'PCA', 'MDS']
            if HAS_UMAP:
                dim_methods.append('UMAP')
            
        if cluster_methods is None:
            cluster_methods = ['KMEANS', 'AGGLOMERATIVE', 'DBSCAN', 'SPECTRAL']

        # Convert compositions to array once
        comp_array = self._compositions_to_array(compositions)
        
        # Setup subplot grid
        n_rows = len(dim_methods)
        n_cols = len(cluster_methods)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        
        # Store scores for comparison
        scores = {}
        
        for i, dim_method in enumerate(dim_methods):
            # Initialize and fit reducer
            self.dim_method = dim_method
            reducer = self.initialize_reducer()
            embeddings = reducer.fit_transform(comp_array)
            
            for j, clust_method in enumerate(cluster_methods):
                # Initialize and fit clusterer
                self.cluster_method = clust_method
                clusterer = self.initialize_cluster_method()
                
                # Set n_clusters if applicable
                if hasattr(clusterer, 'n_clusters'):
                    clusterer.n_clusters = n_clusters
                
                # Fit clusterer and get labels
                clusters = clusterer.fit_predict(comp_array)
                
                # Calculate silhouette score if applicable
                from sklearn.metrics import silhouette_score
                try:
                    score = silhouette_score(comp_array, clusters)
                    scores[f"{dim_method}_{clust_method}"] = score
                except:
                    scores[f"{dim_method}_{clust_method}"] = None
                
                # Plot results
                ax = axes[i, j] if n_rows > 1 and n_cols > 1 else axes[j] if n_rows == 1 else axes[i]
                scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                                   c=clusters, cmap='viridis')
                ax.set_title(f"{dim_method} + {clust_method}\nScore: {scores[f'{dim_method}_{clust_method}']:.3f}"
                            if scores[f"{dim_method}_{clust_method}"] is not None 
                            else f"{dim_method} + {clust_method}")
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
        return scores

    def analyze_compositions(self, compositions: List[Dict[str, float]], 
                            n_clusters: int = 5,
                            dim_method: Optional[str] = None,
                            cluster_method: Optional[str] = None):
        """
        Analyze composition space using specified dimensionality reduction and clustering.
        
        Args:
            compositions: List of composition dictionaries
            n_clusters: Number of clusters (for applicable methods)
            dim_method: Override current dimensionality reduction method
            cluster_method: Override current clustering method
        """
        # Set temporary methods if provided
        orig_dim = self.dim_method
        orig_clust = self.cluster_method
        
        if dim_method:
            self.dim_method = dim_method.upper()
        if cluster_method:
            self.cluster_method = cluster_method.upper()
        
        try:
            # Convert compositions to array
            comp_array = self._compositions_to_array(compositions)
            
            # Perform dimensionality reduction
            reducer = self.initialize_reducer()
            embeddings = reducer.fit_transform(comp_array)
            
            # Perform clustering
            clusterer = self.initialize_cluster_method()
            if hasattr(clusterer, 'n_clusters'):
                clusterer.n_clusters = n_clusters
            clusters = clusterer.fit_predict(comp_array)
            
            return embeddings, clusters
        
        finally:
            # Restore original methods
            self.dim_method = orig_dim
            self.cluster_method = orig_clust

    def _compositions_to_array(self, compositions: List[Dict[str, float]]) -> np.ndarray:
        """Convert composition dictionaries to numpy array"""
        elements = sorted(set().union(*compositions))
        return np.array([[comp.get(elem, 0.0) for elem in elements] for comp in compositions])

    def plot_analysis(self, embeddings: np.ndarray, clusters: np.ndarray, save_path: Optional[str] = None):
        """Create visualization of composition analysis"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters, cmap='viridis')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Composition Space Analysis')

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def suggest_new_compositions(self, compositions: List[Dict[str, float]], 
                               n_suggestions: int = 10, 
                               constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict[str, float]]:
        """
        Suggest new compositions based on clustering analysis.
        
        For methods without explicit cluster centers (like DBSCAN), uses centroids
        of points in each cluster instead.
        """
        if self.clusterer is None:
            raise ValueError("Must run analyze_compositions first")

        # Get cluster assignments for the data
        comp_array = self._compositions_to_array(compositions)
        clusters = self.clusterer.fit_predict(comp_array)
        
        # Get cluster centers (either from clusterer or compute centroids)
        if hasattr(self.clusterer, 'cluster_centers_'):
            centers = self.clusterer.cluster_centers_
        else:
            # Compute centroids manually for each cluster
            unique_clusters = np.unique(clusters)
            centers = np.array([
                comp_array[clusters == i].mean(axis=0)
                for i in unique_clusters if i != -1  # Skip noise points (-1)
            ])

        # Generate new compositions near cluster boundaries
        new_compositions = []
        max_attempts = n_suggestions * 10
        attempts = 0

        while len(new_compositions) < n_suggestions and attempts < max_attempts:
            if len(centers) < 2:
                # If we have fewer than 2 centers, interpolate between random points
                idx1, idx2 = np.random.choice(len(comp_array), 2, replace=False)
                point1, point2 = comp_array[idx1], comp_array[idx2]
            else:
                # Select random pair of centers
                c1, c2 = np.random.choice(len(centers), 2, replace=False)
                point1, point2 = centers[c1], centers[c2]

            # Generate composition between points
            mix = np.random.random()
            new_comp = point1 * mix + point2 * (1 - mix)

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

    def create_random_alloy(self, composition: Dict[str, float], crystal_type: str, dimensions: List[int], lattice_constant: float, balance_element: str, cubic: bool = True) -> Atoms:
        """
        Create a random alloy with specified composition.

        Args:
            composition: Dictionary of element symbols and their fractions
            crystal_type: Crystal structure ('bcc', 'fcc', etc.)
            dimensions: Supercell dimensions [nx, ny, nz]
            lattice_constant: Lattice constant in Å
            balance_element: Element to use as balance (usually majority element)
            cubic: Whether to create a cubic structure (default True)
        Returns:
            ASE Atoms object with randomized atomic positions
        """
        # Validate composition sums to 1 (within tolerance)
        total = sum(composition.values())
        if not np.isclose(total, 1.0, rtol=1e-3):
            raise ValueError(f"Composition fractions must sum to 1.0 (got {total})")

        # Create base structure and supercell
        base_atoms = bulk(balance_element, crystal_type, a=lattice_constant, cubic=cubic)
        atoms = base_atoms * dimensions
        total_sites = len(atoms)

        # Calculate number of atoms for each element
        atom_counts = {}
        remaining_sites = total_sites

        # Handle all elements except balance element
        for element, fraction in composition.items():
            if element != balance_element:
                # Ceil the count for non-balance elements
                count = int(np.ceil(fraction * total_sites))
                atom_counts[element] = count
                remaining_sites -= count

        # Assign remaining sites to balance element
        atom_counts[balance_element] = remaining_sites

        # Verify we haven't created an impossible situation
        if remaining_sites < 0:
            raise ValueError("Composition resulted in negative sites for balance element")

        # Create list of atomic symbols
        symbols = []
        for element, count in atom_counts.items():
            symbols.extend([element] * count)

        # Randomly shuffle the symbols
        random.shuffle(symbols)

        # Assign shuffled symbols to atomic positions
        atoms.symbols = symbols

        return atoms

    def find_diverse_compositions(self, embeddings: np.ndarray, compositions: List[Dict[str, float]], 
                               n_select: int = 5, method: str = 'maximin',
                               constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        """
        Find diverse compositions based on t-SNE embeddings, with optional composition constraints.

        Args:
            embeddings: Array of t-SNE embeddings
            compositions: List of composition dictionaries corresponding to embeddings
            n_select: Number of diverse points to select
            method: Either 'maximin' or 'maxsum'
            constraints: Dictionary mapping element symbols to (min, max) fraction tuples

        Returns:
            Indices of selected diverse points
        """
        if method not in ['maximin', 'maxsum']:
            raise ValueError("Method must be either 'maximin' or 'maxsum'")

        # Apply composition constraints if provided
        valid_indices = np.arange(len(embeddings))
        if constraints:
            valid_mask = np.ones(len(embeddings), dtype=bool)
            for element, (min_frac, max_frac) in constraints.items():
                element_fractions = np.array([comp.get(element, 0.0) for comp in compositions])
                valid_mask &= (element_fractions >= min_frac) & (element_fractions <= max_frac)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) < n_select:
                raise ValueError(f"Only {len(valid_indices)} compositions meet constraints, but {n_select} requested")
            
            # Filter embeddings to only valid ones
            embeddings = embeddings[valid_indices]

        # Calculate pairwise distances for valid embeddings
        distances = squareform(pdist(embeddings))
        
        if method == 'maximin':
            # Maximin approach: Maximize minimum distance between selected points
            selected = []
            
            # Start with the two points that are furthest apart
            initial_pair = np.unravel_index(np.argmax(distances), distances.shape)
            selected.extend(initial_pair)
            
            # Iteratively add points that maximize the minimum distance to selected points
            while len(selected) < n_select:
                min_distances = np.min(distances[selected][:, ~np.isin(range(len(embeddings)), selected)], axis=0)
                next_point = np.where(~np.isin(range(len(embeddings)), selected))[0][np.argmax(min_distances)]
                selected.append(next_point)
            
            # Map back to original indices if constraints were applied
            return valid_indices[np.array(selected)] if constraints else np.array(selected)
        
        else:  # maxsum approach
            # Initialize with random point
            selected = [np.random.randint(len(embeddings))]
            
            # Iteratively add points that maximize sum of distances to selected points
            while len(selected) < n_select:
                sum_distances = np.sum(distances[selected][:, ~np.isin(range(len(embeddings)), selected)], axis=0)
                next_point = np.where(~np.isin(range(len(embeddings)), selected))[0][np.argmax(sum_distances)]
                selected.append(next_point)
            
            # Map back to original indices if constraints were applied
            return valid_indices[np.array(selected)] if constraints else np.array(selected)

    def plot_with_diverse_points(self, embeddings: np.ndarray, clusters: np.ndarray,
                               new_embeddings: np.ndarray, new_compositions: List[Dict[str, float]],
                               n_diverse: int = 5, method: str = 'maximin',
                               constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[np.ndarray, dict]:
        """
        Create an interactive plot highlighting diverse compositions.

        Args:
            embeddings: Original embeddings from dimensionality reduction
            clusters: Cluster assignments for original embeddings
            new_embeddings: Embeddings for new compositions
            new_compositions: List of new composition dictionaries
            n_diverse: Number of diverse points to select
            method: Either 'maximin' or 'maxsum'
            constraints: Optional composition constraints

        Returns:
            Tuple of (indices of diverse points, plotly figure dict)
        """
        import plotly.graph_objects as go

        # Find diverse points
        diverse_indices = self.find_diverse_compositions(
            new_embeddings, new_compositions, n_diverse, method, constraints
        )

        # Calculate statistics for outlier detection
        mean = np.mean(np.vstack([embeddings, new_embeddings]), axis=0)
        std = np.std(np.vstack([embeddings, new_embeddings]), axis=0)
        n_std = 2

        # Create masks
        mask_original = np.all(np.abs(embeddings - mean) < n_std * std, axis=1)
        mask_new = np.all(np.abs(new_embeddings - mean) < n_std * std, axis=1)

        # Create the figure
        fig = go.Figure()

        # Plot original compositions
        fig.add_trace(go.Scatter3d(
            x=embeddings[mask_original, 0],
            y=embeddings[mask_original, 1],
            z=embeddings[mask_original, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=clusters[mask_original],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Cluster')
            ),
            name='Existing'
        ))

        # Plot new compositions (non-diverse)
        non_diverse_mask = mask_new & ~np.isin(np.arange(len(new_embeddings)), diverse_indices)
        fig.add_trace(go.Scatter3d(
            x=new_embeddings[non_diverse_mask, 0],
            y=new_embeddings[non_diverse_mask, 1],
            z=new_embeddings[non_diverse_mask, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond'
            ),
            name='Suggested'
        ))

        # Plot diverse points
        fig.add_trace(go.Scatter3d(
            x=new_embeddings[diverse_indices, 0],
            y=new_embeddings[diverse_indices, 1],
            z=new_embeddings[diverse_indices, 2],
            mode='markers',
            marker=dict(
                size=12,
                color='black',
                symbol='diamond'
            ),
            name=f'Most Diverse ({method})'
        ))

        # Update layout with current dimensionality reduction method
        fig.update_layout(
            title=f'Composition Space Analysis with {n_diverse} Most Diverse Points ({method}) using {self.dim_method}',
            scene=dict(
                xaxis_title=f'{self.dim_method} 1',
                yaxis_title=f'{self.dim_method} 2',
                zaxis_title=f'{self.dim_method} 3'
            ),
            width=1000,
            height=800,
            showlegend=True
        )

        return diverse_indices, fig