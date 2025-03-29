from typing import List, Dict, Optional, Tuple, Union
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

from sklearn.neighbors import LocalOutlierFactor  # for outlier detection

class CompositionAnalyzer:
    def __init__(self, n_components: int = 2, random_state: int = 42, 
                 dim_method: str = 't-SNE', cluster_method: str = 'KMeans'):
        """
        Initialize the CompositionAnalyzer for analyzing chemical composition spaces.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.dim_method = dim_method.upper()  # Normalize to uppercase
        self.cluster_method = cluster_method.upper()
        
        # Initialize methods
        self.reducer = self.initialize_reducer()
        self.clusterer = self.initialize_cluster_method()

    def initialize_reducer(self):
        """Initialize dimensionality reduction method based on self.dim_method."""
        method = self.dim_method.upper()
        if method in ['T-SNE', 'TSNE']:
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
            raise ValueError(f"Unsupported dimensionality reduction method: {self.dim_method}. Supported methods are: 't-SNE', 'PCA', 'MDS', 'UMAP'")

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
        """
        if dim_methods is None:
            dim_methods = ['T-SNE', 'PCA', 'MDS']
            if HAS_UMAP:
                dim_methods.append('UMAP')
            
        if cluster_methods is None:
            cluster_methods = ['KMEANS', 'AGGLOMERATIVE', 'DBSCAN', 'SPECTRAL']

        comp_array = self._compositions_to_array(compositions)
        n_rows = len(dim_methods)
        n_cols = len(cluster_methods)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        scores = {}
        
        for i, dim_method in enumerate(dim_methods):
            self.dim_method = dim_method
            reducer = self.initialize_reducer()
            embeddings = reducer.fit_transform(comp_array)
            
            for j, clust_method in enumerate(cluster_methods):
                self.cluster_method = clust_method
                clusterer = self.initialize_cluster_method()
                if hasattr(clusterer, 'n_clusters'):
                    clusterer.n_clusters = n_clusters
                clusters = clusterer.fit_predict(comp_array)
                
                from sklearn.metrics import silhouette_score
                try:
                    score = silhouette_score(comp_array, clusters)
                    scores[f"{dim_method}_{clust_method}"] = score
                except Exception:
                    scores[f"{dim_method}_{clust_method}"] = None
                
                ax = axes[i, j] if n_rows > 1 and n_cols > 1 else axes[j] if n_rows == 1 else axes[i]
                scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters, cmap='viridis')
                score_text = f"Score: {scores[f'{dim_method}_{clust_method}']:.3f}" if scores[f"{dim_method}_{clust_method}"] is not None else ""
                ax.set_title(f"{dim_method} + {clust_method}\n{score_text}")
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        return scores

    def analyze_compositions(self, compositions: List[Dict[str, float]], 
                             n_clusters: int = 5,
                             dim_method: Optional[str] = None,
                             cluster_method: Optional[str] = None,
                             seed: int = 42):
        """
        Legacy method: Analyze composition space using specified dimensionality reduction and clustering.
        """
        orig_dim = self.dim_method
        orig_clust = self.cluster_method
        
        if dim_method:
            self.dim_method = dim_method.upper()
        if cluster_method:
            self.cluster_method = cluster_method.upper()
        
        try:
            comp_array = self._compositions_to_array(compositions)
            reducer = self.initialize_reducer()
            embeddings = reducer.fit_transform(comp_array)
            
            clusterer = self.initialize_cluster_method()
            if hasattr(clusterer, 'n_clusters'):
                clusterer.n_clusters = n_clusters
            clusters = clusterer.fit_predict(comp_array)
            
            return embeddings, clusters
        finally:
            self.dim_method = orig_dim
            self.cluster_method = orig_clust

    def _compositions_to_array(self, compositions: List[Dict[str, float]]) -> np.ndarray:
        """Convert composition dictionaries to numpy array."""
        elements = sorted(set().union(*compositions))
        return np.array([[comp.get(elem, 0.0) for elem in elements] for comp in compositions])
    
    def _create_metadata(self, compositions: List[Dict[str, float]], comp_type: str = 'existing') -> List[Dict]:
        """
        Create a metadata dictionary for each composition.
        Each entry will have a unique comp_id, the composition dictionary, type, and placeholders for embedding, cluster, and outlier info.
        """
        metadata = []
        for idx, comp in enumerate(compositions):
            metadata.append({
                "comp_id": idx,
                "composition": comp,
                "type": comp_type,
                "embedding": None,
                "cluster": None,
                "lof_score": None,
                "is_outlier": False
            })
        return metadata

    def reduce_and_cluster(self, compositions: List[Dict[str, float]], 
                           n_clusters: int = 5, comp_type: str = 'existing',
                           dim_method: Optional[str] = None, cluster_method: Optional[str] = None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Perform dimensionality reduction and clustering while generating metadata for each composition.
        
        Returns:
            embeddings: Reduced space coordinates.
            clusters: Cluster labels.
            metadata: List of dictionaries with composition metadata.
        """
        # Optionally override methods
        orig_dim = self.dim_method
        orig_clust = self.cluster_method
        if dim_method:
            self.dim_method = dim_method.upper()
        if cluster_method:
            self.cluster_method = cluster_method.upper()
        
        metadata = self._create_metadata(compositions, comp_type)
        comp_array = self._compositions_to_array(compositions)
        
        # Dimensionality reduction
        reducer = self.initialize_reducer()
        embeddings = reducer.fit_transform(comp_array)
        
        # Clustering
        clusterer = self.initialize_cluster_method()
        if hasattr(clusterer, 'n_clusters'):
            clusterer.n_clusters = n_clusters
        clusters = clusterer.fit_predict(comp_array)
        
        # Update metadata with embedding and cluster information
        for i, meta in enumerate(metadata):
            meta["embedding"] = embeddings[i]
            meta["cluster"] = clusters[i]
        
        # Restore original methods
        self.dim_method = orig_dim
        self.cluster_method = orig_clust
        
        return embeddings, clusters, metadata

    def detect_outliers(self, embeddings: np.ndarray, metadata: List[Dict], 
                        contamination: float = 0.05, n_neighbors: int = 5, use_lof: bool = True, seed: int = 42) -> List[Dict]:
        """
        Flag outliers using LocalOutlierFactor. Updates metadata entries with 'lof_score' and 'is_outlier'.
        
        Args:
            embeddings: Reduced dimensionality embeddings.
            metadata: Metadata list from reduce_and_cluster.
            contamination: Proportion of expected outliers.
            n_neighbors: Number of neighbors for LOF.
            use_lof: Toggle LOF-based detection.
        Returns:
            Updated metadata.
        """
        if use_lof:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            # Note: fit_predict returns -1 for outliers and 1 for inliers
            labels = lof.fit_predict(embeddings)
            lof_scores = -lof.negative_outlier_factor_
            for i, meta in enumerate(metadata):
                meta["lof_score"] = lof_scores[i]
                meta["is_outlier"] = (labels[i] == -1)
        return metadata

    def visualize_compositions(self, embeddings: np.ndarray, metadata: List[Dict],
                               outlier_option: str = 'include',  # 'include', 'exclude', or 'only'
                               save_path: Optional[str] = None):
        """
        Visualize the composition space with flexibility to include/exclude outliers.
        
        The visualization is 2D or 3D based on self.n_components.
        
        Args:
            embeddings: Reduced dimensionality coordinates.
            metadata: Metadata list with composition info.
            outlier_option: How to handle outliers: 'include' (all), 'exclude' (omit outliers), or 'only' (only outliers).
            save_path: Optional path to save the figure.
        """
        # Filter metadata indices based on outlier_option
        indices = []
        for i, meta in enumerate(metadata):
            if outlier_option == 'exclude' and meta["is_outlier"]:
                continue
            if outlier_option == 'only' and not meta["is_outlier"]:
                continue
            indices.append(i)
        
        filtered_embeddings = embeddings[indices]
        filtered_meta = [metadata[i] for i in indices]
        
        # Prepare arrays for plotting markers by type
        existing = [meta for meta in filtered_meta if meta["type"] == 'existing']
        new = [meta for meta in filtered_meta if meta["type"] == 'new']
        
        # Extract coordinates for plotting
        def get_coords(meta_list):
            return np.array([m["embedding"] for m in meta_list])
        
        existing_coords = get_coords(existing) if existing else np.empty((0, self.n_components))
        new_coords = get_coords(new) if new else np.empty((0, self.n_components))
        
        if self.n_components == 3:
            # 3D visualization using plotly
            import plotly.graph_objects as go
            fig = go.Figure()
            
            if existing_coords.size:
                fig.add_trace(go.Scatter3d(
                    x=existing_coords[:, 0],
                    y=existing_coords[:, 1],
                    z=existing_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='blue',
                        opacity=0.7
                    ),
                    name='Existing'
                ))
            if new_coords.size:
                fig.add_trace(go.Scatter3d(
                    x=new_coords[:, 0],
                    y=new_coords[:, 1],
                    z=new_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='diamond'
                    ),
                    name='New'
                ))
            
            fig.update_layout(
                title=f'Composition Space ({self.dim_method})',
                scene=dict(
                    xaxis_title=f'{self.dim_method} 1',
                    yaxis_title=f'{self.dim_method} 2',
                    zaxis_title=f'{self.dim_method} 3'
                ),
                width=1000,
                height=800
            )
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            # 2D visualization using matplotlib
            plt.figure(figsize=(10, 8))
            if existing_coords.size:
                plt.scatter(existing_coords[:, 0], existing_coords[:, 1], c='blue', label='Existing', alpha=0.7, s=50)
            if new_coords.size:
                plt.scatter(new_coords[:, 0], new_coords[:, 1], c='red', label='New', marker='D', s=70)
            plt.xlabel(f'{self.dim_method} 1')
            plt.ylabel(f'{self.dim_method} 2')
            plt.title(f'Composition Space ({self.dim_method})')
            plt.legend()
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

    def suggest_new_compositions(self, compositions: List[Dict[str, float]], 
                                 n_suggestions: int = 10, 
                                 constraints: Optional[Dict[str, Tuple[float, float]]] = None,
                                 metadata: Optional[List[Dict]] = None,
                                 seed: Optional[int] = 42) -> List[Dict[str, float]]:
        """
        Suggest new compositions based on clustering analysis.
        
        If metadata is provided, suggestions are generated only from inlier compositions.
        """
        if seed:
            random.seed(seed)

        # Use only inlier compositions if metadata is provided
        if metadata:
            valid_indices = [i for i, m in enumerate(metadata) if not m["is_outlier"]]
            if not valid_indices:
                raise ValueError("No inlier compositions available for suggestions.")
            comp_array = self._compositions_to_array([metadata[i]["composition"] for i in valid_indices])
            # Use clustering on the filtered array
            clusterer = self.initialize_cluster_method()
            if hasattr(clusterer, 'n_clusters'):
                clusterer.n_clusters = n_suggestions
            clusters = clusterer.fit_predict(comp_array)
            # Compute centroids manually
            unique_clusters = np.unique(clusters)
            centers = []
            for cl in unique_clusters:
                if cl == -1:  # Skip noise
                    continue
                pts = comp_array[clusters == cl]
                centers.append(pts.mean(axis=0))
            centers = np.array(centers)
        else:
            comp_array = self._compositions_to_array(compositions)
            clusterer = self.initialize_cluster_method()
            if hasattr(clusterer, 'n_clusters'):
                clusterer.n_clusters = n_suggestions
            clusters = clusterer.fit_predict(comp_array)
            if hasattr(clusterer, 'cluster_centers_'):
                centers = clusterer.cluster_centers_
            else:
                unique_clusters = np.unique(clusters)
                centers = np.array([comp_array[clusters == i].mean(axis=0) for i in unique_clusters if i != -1])
        
        new_compositions = []
        max_attempts = n_suggestions * 100
        attempts = 0
        elements = sorted(set().union(*compositions))
        while len(new_compositions) < n_suggestions and attempts < max_attempts:
            if len(centers) < 2:
                idx1, idx2 = np.random.choice(len(comp_array), 2, replace=False)
                point1, point2 = comp_array[idx1], comp_array[idx2]
            else:
                c1, c2 = np.random.choice(len(centers), 2, replace=False)
                point1, point2 = centers[c1], centers[c2]
            mix = np.random.random()
            new_comp = point1 * mix + point2 * (1 - mix)
            comp_dict = dict(zip(elements, new_comp))
            valid = True
            if constraints:
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

    def quantify_uncertainty(self, model_paths: List[str], structures: List[Atoms]) -> List[float]:
        """
        Placeholder method to quantify uncertainty.
        Given paths to ML potential models and a list of 128-atom Atoms objects,
        compute the force variance across the ensemble for each structure.
        
        Args:
            model_paths: List of file paths to ML potential models.
            structures: List of ASE Atoms objects (each with 128 atoms).
            
        Returns:
            List of force variance values (one per structure).
            
        To be implemented.
        """
        # Placeholder: return a list of zeros.
        uncertainties = [0.0 for _ in structures]
        return uncertainties

    def create_random_alloy(self, composition: Dict[str, float], crystal_type: str, dimensions: List[int], lattice_constant: float, balance_element: str, cubic: bool = True) -> Atoms:
        """
        Create a random alloy with specified composition.
        """
        total = sum(composition.values())
        if not np.isclose(total, 1.0, rtol=1e-3):
            raise ValueError(f"Composition fractions must sum to 1.0 (got {total})")
        base_atoms = bulk(balance_element, crystal_type, a=lattice_constant, cubic=cubic)
        atoms = base_atoms * dimensions
        total_sites = len(atoms)
        atom_counts = {}
        remaining_sites = total_sites
        for element, fraction in composition.items():
            if element != balance_element:
                count = int(np.ceil(fraction * total_sites))
                atom_counts[element] = count
                remaining_sites -= count
        atom_counts[balance_element] = remaining_sites
        if remaining_sites < 0:
            raise ValueError("Composition resulted in negative sites for balance element")
        symbols = []
        for element, count in atom_counts.items():
            symbols.extend([element] * count)
        random.shuffle(symbols)
        atoms.symbols = symbols
        return atoms


    

def analyze_composition_distribution(analyzer,
                                     existing_compositions,
                                     new_compositions,
                                     n_clusters: int = 5,
                                     n_neighbors: int = 5,
                                     top_n: int = 5,
                                     weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
                                     clustering_params: Optional[Dict] = None,
                                     constraint_dict: Optional[Dict[str, Tuple[float, float]]] = None,
                                     grid_resolution: int = 10,
                                     constraint_tol: float = 0.05,
                                     random_state: int = 42,
                                     manual_compositions: Optional[List[Dict[str, float]]] = None):
    """
    Comprehensive analysis of new compositions' distribution in composition space,
    highlighting the most novel compositions based on a combined metric.
    
    Args:
        analyzer: Initialized CompositionAnalyzer.
        existing_compositions: List of existing composition dictionaries.
        new_compositions: List of new composition dictionaries.
        n_clusters: Number of clusters for visualization.
        n_neighbors: Number of neighbors for LOF/density estimation.
        top_n: Number of top compositions to select automatically by novelty score.
        weights: Tuple of (distance_weight, lof_weight, diversity_weight) for scoring.
        clustering_params: Dictionary of additional parameters for the clustering algorithm.
        constraint_dict: Optional dictionary defining the composition constraints.
        grid_resolution: Number of points along each element axis for grid search.
        constraint_tol: Tolerance for sum-to-one constraint on compositions.
        random_state: Seed for random number generation.
        manual_compositions: Optional list of composition dictionaries to include in the analysis
                           regardless of their novelty score. These will be marked as 'Suggested': False.
    Returns:
        A dictionary containing:
          - embeddings for existing and new compositions,
          - distance metrics and LOF scores,
          - top compositions with novelty scores,
          - dictionary of generated figures.
    """
    import numpy as np
    import random
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.neighbors import LocalOutlierFactor
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    from copy import deepcopy

    # Ensure reproducibility
    np.random.seed(random_state)
    random.seed(random_state)
    
    # --- Step 1: Add 'Suggested' flags and deep copy to avoid modifying the originals ---
    existing_copy = []
    for comp in existing_compositions:
        comp_copy = deepcopy(comp)
        comp_copy['Suggested'] = True  # Not used but for consistency
        existing_copy.append(comp_copy)
    
    new_copy = []
    for comp in new_compositions:
        comp_copy = deepcopy(comp)
        comp_copy['Suggested'] = True
        new_copy.append(comp_copy)
    
    manual_copy = []
    if manual_compositions:
        for comp in manual_compositions:
            comp_copy = deepcopy(comp)
            comp_copy['Suggested'] = False
            manual_copy.append(comp_copy)
    
    # --- Step 2: Combine compositions for analysis ---
    # We'll keep existing separate for reference data (centroid, outlier detection)
    # But combine new and manual for novelty assessment
    combined_new = new_copy + manual_copy
    
    # --- Step 3: Compute embeddings for all compositions ---
    # We need to remove the 'Suggested' flag before converting to array
    existing_for_embedding = [
        {k: v for k, v in comp.items() if k != 'Suggested'} 
        for comp in existing_copy
    ]
    combined_for_embedding = [
        {k: v for k, v in comp.items() if k != 'Suggested'} 
        for comp in combined_new
    ]
    
    all_for_embedding = existing_for_embedding + combined_for_embedding
    all_embeddings, clusters = analyzer.analyze_compositions(
        all_for_embedding, n_clusters=n_clusters, seed=random_state
    )
    
    # Split back into our categories
    embeddings_existing = all_embeddings[:len(existing_for_embedding)]
    embeddings_combined = all_embeddings[len(existing_for_embedding):]
    
    # --- Apply custom clustering parameters if provided ---
    if clustering_params and isinstance(clustering_params, dict):
        original_clusterer = analyzer.clusterer
        if analyzer.cluster_method == 'KMEANS':
            from sklearn.cluster import KMeans
            analyzer.clusterer = KMeans(n_clusters=n_clusters, random_state=analyzer.random_state, **clustering_params)
        elif analyzer.cluster_method == 'DBSCAN':
            from sklearn.cluster import DBSCAN
            analyzer.clusterer = DBSCAN(**clustering_params)
        elif analyzer.cluster_method == 'AGGLOMERATIVE':
            from sklearn.cluster import AgglomerativeClustering
            analyzer.clusterer = AgglomerativeClustering(n_clusters=n_clusters, **clustering_params)
        elif analyzer.cluster_method == 'SPECTRAL':
            from sklearn.cluster import SpectralClustering
            analyzer.clusterer = SpectralClustering(n_clusters=n_clusters, random_state=analyzer.random_state, **clustering_params)

    # --- Step 4: Create metadata and update outlier detection ---
    meta_existing = analyzer._create_metadata(existing_for_embedding, comp_type='existing')
    meta_combined = analyzer._create_metadata(combined_for_embedding, comp_type='new')
    meta_existing = analyzer.detect_outliers(
        embeddings_existing, meta_existing, 
        contamination=0.05, n_neighbors=n_neighbors, seed=random_state
    )
    meta_combined = analyzer.detect_outliers(
        embeddings_combined, meta_combined, 
        contamination=0.05, n_neighbors=n_neighbors, seed=random_state
    )

    # --- Step 5: Compute centroid and distances (based on existing compositions) ---
    centroid = np.mean(embeddings_existing, axis=0)
    distances_existing = np.linalg.norm(embeddings_existing - centroid, axis=1)
    distances_combined = np.linalg.norm(embeddings_combined - centroid, axis=1)

    # --- Step 6: Compute LOF scores ---
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(embeddings_existing)
    combined_lof_scores = -lof.score_samples(embeddings_combined)
    existing_lof_scores = -lof.score_samples(embeddings_existing)

    # --- Step 7: Compute diversity scores for combined compositions ---
    diversity_scores = np.zeros(len(combined_new))
    if len(combined_new) > 1:
        pairwise_dists = squareform(pdist(embeddings_combined))
        for i in range(len(combined_new)):
            pairwise_dists[i, i] = np.inf
            diversity_scores[i] = np.min(pairwise_dists[i])

    # --- Step 8: Normalize metrics to [0,1] ---
    norm_distances = (distances_combined - np.min(distances_combined)) / (np.max(distances_combined) - np.min(distances_combined) + 1e-10)
    norm_lof = (combined_lof_scores - np.min(combined_lof_scores)) / (np.max(combined_lof_scores) - np.min(combined_lof_scores) + 1e-10)
    if len(combined_new) > 1:
        norm_diversity = (diversity_scores - np.min(diversity_scores)) / (np.max(diversity_scores) - np.min(diversity_scores) + 1e-10)
    else:
        norm_diversity = np.zeros(1)

    # --- Step 9: Compute combined novelty score ---
    distance_weight, lof_weight, diversity_weight = weights
    novelty_scores = (distance_weight * norm_distances +
                      lof_weight * norm_lof +
                      diversity_weight * norm_diversity)
    
    # --- Step 10: Select top compositions from suggested only, but keep all manual ---
    # First, split indices into suggested and manual
    suggested_indices = [i for i, comp in enumerate(combined_new) if comp['Suggested']]
    manual_indices = [i for i, comp in enumerate(combined_new) if not comp['Suggested']]
    
    # Get top N from suggested
    top_suggested = sorted(
        [(i, novelty_scores[i]) for i in suggested_indices],
        key=lambda x: x[1], reverse=True
    )[:top_n]
    top_suggested_indices = [i for i, _ in top_suggested]
    
    # Combine with all manual indices
    all_top_indices = top_suggested_indices + manual_indices
    
    # --- Step 11: Prepare readable string representations ---
    string_compositions = []
    for comp in combined_new:
        elements = sorted([k for k in comp.keys() if k != 'Suggested'])
        string_comp = "-".join([f"{elem}{comp[elem]:.3f}" for elem in elements])
        string_compositions.append(string_comp)

    # --- Step 12: Prepare detailed info for top compositions ---
    top_compositions = []
    for rank, idx in enumerate(all_top_indices):
        comp = combined_new[idx]
        top_compositions.append({
            'rank': rank + 1,
            'index': idx,
            'composition': {k: v for k, v in comp.items() if k != 'Suggested'},
            'composition_string': string_compositions[idx],
            'metrics': {
                'distance': distances_combined[idx],
                'distance_percentile': stats.percentileofscore(distances_existing, distances_combined[idx]),
                'lof_score': combined_lof_scores[idx],
                'diversity': diversity_scores[idx] if len(combined_new) > 1 else 0,
            },
            'normalized_metrics': {
                'distance': norm_distances[idx],
                'lof_score': norm_lof[idx],
                'diversity': norm_diversity[idx] if len(combined_new) > 1 else 0,
            },
            'novelty_score': novelty_scores[idx],
            'embedding': embeddings_combined[idx].tolist(),
            'suggested': comp['Suggested']
        })
    
    # Sort by novelty score (highest first)
    top_compositions.sort(key=lambda x: x['novelty_score'], reverse=True)
    
    # Reassign ranks after sorting
    for i, comp_info in enumerate(top_compositions):
        comp_info['rank'] = i + 1

    # --- Step 13: Create Plots ---
    figs = {}

    # --- 2D Plot with improved label placement ---
    fig2d, ax = plt.subplots(figsize=(12, 8))
    
    # Plot existing compositions with a lighter gray color
    scatter = ax.scatter(
        embeddings_existing[:, 0],
        embeddings_existing[:, 1],
        color='#D3D3D3',  # Light gray color
        s=50,
        alpha=0.6,        # Slightly higher alpha so they are visible but still subdued
        label='Existing'
    )
    
    # Plot top compositions with an increased size and black edge
    top_novelty = np.array([comp_info['novelty_score'] for comp_info in top_compositions])
    top_embeddings = np.array([embeddings_combined[comp_info['index']] for comp_info in top_compositions])
    if len(top_compositions) > 0:
        new_scatter = ax.scatter(
            top_embeddings[:, 0],
            top_embeddings[:, 1],
            c=top_novelty,
            cmap='plasma',
            s=200,
            marker='D',
            edgecolors='black'
        )
        cbar2 = plt.colorbar(new_scatter, label='Novelty Score')
        
        # Improve label placement to avoid overlap with points
        texts = []
        for comp_info in top_compositions:
            idx = comp_info['index']
            x, y = embeddings_combined[idx, 0], embeddings_combined[idx, 1]
            label_color = 'brown' if not comp_info.get('suggested', True) else 'black'
            txt = ax.text(
                x, y,
                f"#{comp_info['rank']}",
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                color=label_color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=label_color, alpha=0.8)
            )
            texts.append(txt)
        
        # Attempt to import and use adjustText
        try:
            from adjustText import adjust_text
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                expand_text=(1.2, 1.2),
                expand_points=(1.5, 1.5),
                force_text=(0.5, 0.5),
                force_points=(0.5, 0.5)
            )
        except ImportError:
            # Fallback if adjustText is not available
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            for i, comp_info in enumerate(top_compositions):
                idx = comp_info['index']
                texts[i].remove()  # Remove original text
                x, y = embeddings_combined[idx, 0], embeddings_combined[idx, 1]
                offset = 0.05 * (x_max - x_min)  # Basic offset
                angle = 2 * np.pi * i / max(1, len(top_compositions))
                dx = np.cos(angle) * offset
                dy = np.sin(angle) * offset
                ax.annotate(
                    f"#{comp_info['rank']}",
                    xy=(x, y),
                    xytext=(x + dx, y + dy),
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="black",
                        shrinkA=15,
                        shrinkB=5,
                        lw=1.5
                    )
                )

    # Plot centroid with a star
    ax.scatter(centroid[0], centroid[1], c='black', marker='*', s=300, label='Centroid')
    
    # ===== Overlay constraint manifold if provided =====
    if constraint_dict is not None:
        # Use ONLY the elements explicitly defined in the constraint dictionary
        all_elements = sorted(constraint_dict.keys())
        
        # First, verify that all suggested compositions satisfy the constraints
        for i, comp in enumerate(combined_new):
            if comp['Suggested']:
                comp_clean = {k: v for k, v in comp.items() if k != 'Suggested'}
                sum_comp = sum(comp_clean.values())
                if abs(sum_comp - 1.0) > constraint_tol:
                    print(f"Warning: Suggested composition #{i} sum = {sum_comp}, not 1.0 within tolerance")
                
                for elem in comp_clean:
                    if elem in constraint_dict:
                        min_val, max_val = constraint_dict[elem]
                        if comp_clean[elem] < min_val or comp_clean[elem] > max_val:
                            print(f"Warning: Suggested composition #{i} has {elem}={comp_clean[elem]}, outside range ({min_val}, {max_val})")
        
        missing = [e for e in all_elements if e not in constraint_dict]
        if missing:
            print(f"Warning: The following elements are missing constraints and will not be sampled: {missing}")
        else:
            grid_axes = []
            for elem in all_elements:
                lo, hi = constraint_dict[elem]
                grid_axes.append(np.linspace(lo, hi, grid_resolution))
            mesh = np.meshgrid(*grid_axes)
            grid_points = np.vstack([m.flatten() for m in mesh]).T  # shape (N, d)
            valid_indices = np.where(np.abs(grid_points.sum(axis=1) - 1) < constraint_tol)[0]
            if valid_indices.size:
                valid_points = grid_points[valid_indices]
                try:
                    # Transform the valid points into embedding space
                    constraint_proj = analyzer.reducer.transform(valid_points)
                except Exception as e:
                    # If using UMAP, we may need to refit the reducer
                    if analyzer.dim_method.upper() == "UMAP":
                        # Use the original data that the reducer was fit on
                        original_data = existing_for_embedding + combined_for_embedding  
                        comp_array = analyzer._compositions_to_array(original_data)
                        analyzer.reducer.fit(comp_array)
                        constraint_proj = analyzer.reducer.transform(valid_points)
                    else:
                        print("Error during transforming constraint grid points; ensure your reducer supports 'transform':", e)
                        constraint_proj = None
                if constraint_proj is not None and constraint_proj.shape[1] >= 2:
                    try:
                        hull = ConvexHull(constraint_proj[:, :2])
                        hull_points = constraint_proj[hull.vertices, :2]
                        patch = Polygon(hull_points, closed=True, facecolor='green', edgecolor='none',
                                        alpha=0.05, label="Constraint Manifold")
                        ax.add_patch(patch)
                    except Exception as e:
                        print("Could not compute convex hull for constraint manifold:", e)
            else:
                print("No grid points satisfied the sum-to-one constraint within tolerance.")
    
    ax.set_xlabel(f'{analyzer.dim_method} 1')
    ax.set_ylabel(f'{analyzer.dim_method} 2')
    ax.set_title(f'Composition Space Analysis - Top {len(top_compositions)} Novel Compositions')

    # Add legend entries for top compositions
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', label='Suggested', markerfacecolor='black', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w', label='Manual', markerfacecolor='brown', markersize=10, markeredgecolor='brown')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    figs['2d'] = fig2d
    
    # --- Distribution Plot (distance boxplot + KDE) ---
    fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.boxplot([distances_existing, distances_combined], labels=['Existing', 'New'])
    ax1.set_title('Distance Distribution')
    ax1.set_ylabel('Distance from Centroid')
    
    # Mark top compositions on the boxplot
    for comp_info in top_compositions:
        idx = comp_info['index']
        ax1.plot(2, distances_combined[idx], 'ro', markersize=8, alpha=0.7)
        ax1.annotate(
            f"#{comp_info['rank']}",
            (2, distances_combined[idx]),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )
    
    # KDE plot of existing distances + vertical lines for new top compositions
    sns.kdeplot(distances_existing, ax=ax2, label='Existing')
    ymin_lim, ymax_lim = ax2.get_ylim()
    spacing = ymax_lim * 0.7 / len(top_compositions) if top_compositions else 0
    start_y = ymax_lim * 0.85
    for j, comp_info in enumerate(top_compositions):
        idx = comp_info['index']
        dist = distances_combined[idx]
        ax2.axvline(x=dist, color='red', linestyle='--', alpha=0.7, linewidth=2)
        y_pos = start_y - (j * spacing)
        ax2.annotate(
            f"#{comp_info['rank']} {comp_info['composition_string']}",
            xy=(dist, y_pos),
            xytext=(max(distances_existing) * 1.6, y_pos),
            ha='left',
            va='center',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, boxstyle="round,pad=0.3"),
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='bar,fraction=0',
                color='red',
                alpha=0.7,
                linewidth=2
            )
        )
    x_max = max(max(distances_existing), max(distances_combined)) * 2.5
    ax2.set_xlim(0, x_max)
    ax2.set_title('Distance Density')
    ax2.set_xlabel('Distance from Centroid')
    ax2.legend()
    figs['dist'] = fig_dist
    
    # --- Print distance statistics ---
    print("\nDistance Statistics:")
    print("Existing Compositions:")
    print(f"  Mean: {np.mean(distances_existing):.3f}, Median: {np.median(distances_existing):.3f}, Std: {np.std(distances_existing):.3f}")
    print("\nTop Novel Compositions:")
    for comp_info in top_compositions:
        print(f"\nRank #{comp_info['rank']}: {comp_info['composition_string']}")
        print(f"  {'Suggested' if comp_info['suggested'] else 'Manual'}")
        print(f"  Novelty Score: {comp_info['novelty_score']:.3f}")
        print(f"  Distance: {comp_info['metrics']['distance']:.3f} (percentile: {comp_info['metrics']['distance_percentile']:.1f}%)")
        print(f"  LOF Score: {comp_info['metrics']['lof_score']:.3f}")
        print(f"  Diversity: {comp_info['metrics']['diversity']:.3f}")
        print(f"  Composition: {', '.join([f'{k}: {v:.3f}' for k, v in comp_info['composition'].items()])}")
    
    return {
        'embeddings_existing': embeddings_existing,
        'embeddings_combined': embeddings_combined,
        'distances': {'existing': distances_existing, 'combined': distances_combined},
        'lof_scores': {'existing': existing_lof_scores, 'combined': combined_lof_scores},
        'top_compositions': top_compositions,
        'figures': figs
    }