from typing import List, Dict, Optional, Tuple
import numpy as np
from ase import Atoms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import json
from ase.build import bulk
import random
from scipy.spatial.distance import pdist, squareform


class CompositionAnalyzer:
    def __init__(self, n_components: int = 2, random_state: int = 42):
        """
        Initialize the CompositionAnalyzer with t-SNE and KMeans clustering.

        Args:
            n_components (int): Number of dimensions for t-SNE embedding.
            random_state (int): Random seed for reproducibility.
        """
        self.tsne = TSNE(n_components=n_components, random_state=random_state)
        self.kmeans = None

    def analyze_compositions(self, compositions: List[Dict[str, float]], n_clusters: int = 5):
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

    def suggest_new_compositions(self, compositions: List[Dict[str, float]], n_suggestions: int = 10, constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict[str, float]]:
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
            embeddings: Original t-SNE embeddings
            clusters: Cluster assignments for original embeddings
            new_embeddings: t-SNE embeddings for new compositions
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
        
        # Update layout
        fig.update_layout(
            title=f'Composition Space Analysis with {n_diverse} Most Diverse Points ({method})',
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3'
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        return diverse_indices, fig