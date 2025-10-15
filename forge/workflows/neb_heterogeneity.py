"""
Enhanced NEB analysis module focused on quantifying heterogeneity in the potential energy landscape.

This module extends the NEB workflow to capture and analyze local chemical environments
around vacancy sites and their correlation with diffusion barriers.

Key features:
1. Local environment characterization (1st, 2nd, 3rd neighbor shells)
2. Per-vacancy-site statistics and heterogeneity metrics
3. Environment-barrier correlation analysis
4. Dimensionality reduction and clustering of local environments
5. Rich dataset export for downstream analysis/ML

Confidence: 9/10
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from ase import Atoms
from ase.neighborlist import NeighborList

# Try to import optional dependencies
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class LocalEnvironment:
    """
    Characterization of the local chemical environment around a vacancy site.
    
    Attributes:
        vacancy_index: Index of the vacancy site
        vacancy_element: Element at the vacancy site
        shell_compositions: List of composition dicts for each neighbor shell
        shell_distances: Mean distances for each shell
        nn_elements: List of elements for each NN (ordered by distance)
        nn_distances: List of distances to each NN
        coordination_numbers: Number of neighbors in each shell
    """
    vacancy_index: int
    vacancy_element: str
    shell_compositions: List[Dict[str, float]]  # [{element: fraction}, ...] for each shell
    shell_distances: List[float]  # Mean distance for each shell
    nn_elements: List[str]  # Elements of nearest neighbors
    nn_distances: List[float]  # Distances to nearest neighbors
    coordination_numbers: List[int]  # Number of atoms in each shell


@dataclass
class VacancySiteData:
    """
    Complete dataset for a single vacancy site including local environment and barrier statistics.
    
    Attributes:
        vacancy_index: Index of the vacancy site
        local_env: LocalEnvironment characterization
        nn_barriers: List of barriers for NN jumps
        nn_target_elements: List of target elements for each NN jump
        barrier_statistics: Statistics (mean, std, min, max, range) of barriers from this site
        converged_fraction: Fraction of NEB calculations that converged
        structure_info: Metadata about the parent structure
    """
    vacancy_index: int
    local_env: LocalEnvironment
    nn_barriers: List[float]
    nn_target_elements: List[str]
    barrier_statistics: Dict[str, float]
    converged_fraction: float
    structure_info: Dict[str, Any]


class LocalEnvironmentAnalyzer:
    """
    Analyzes local chemical environments around vacancy sites and their correlation
    with diffusion barriers.
    """
    
    def __init__(
        self,
        atoms: Atoms,
        shell_cutoffs: List[float] = [2.8, 3.5, 4.0]  # 1st, 2nd, 3rd shell cutoffs
    ):
        """
        Initialize the local environment analyzer.
        
        Args:
            atoms: ASE Atoms object (the perfect/optimized structure)
            shell_cutoffs: Cutoff distances for neighbor shells in Angstroms
        """
        self.atoms = atoms.copy()
        self.shell_cutoffs = shell_cutoffs
        self._neighbor_cache: Dict[int, LocalEnvironment] = {}
        
    def characterize_local_environment(self, index: int) -> LocalEnvironment:
        """
        Characterize the local chemical environment around an atom.
        
        Args:
            index: Index of the atom to characterize
            
        Returns:
            LocalEnvironment dataclass with full characterization
        """
        if index in self._neighbor_cache:
            return self._neighbor_cache[index]
        
        # Get element at this site
        element = self.atoms[index].symbol
        
        # Create neighbor list with largest cutoff
        max_cutoff = max(self.shell_cutoffs)
        nl = NeighborList(
            [max_cutoff/2] * len(self.atoms),
            skin=0.0,
            self_interaction=False,
            bothways=True
        )
        nl.update(self.atoms)
        
        # Get all neighbors and distances
        indices, offsets = nl.get_neighbors(index)
        positions = self.atoms.positions
        cell = self.atoms.get_cell()
        
        distances = []
        for i, offset in zip(indices, offsets):
            pos_i = positions[i] + np.dot(offset, cell)
            dist = np.linalg.norm(pos_i - positions[index])
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Organize into shells
        shell_compositions = []
        shell_distances = []
        coordination_numbers = []
        
        prev_cutoff = 0.0
        for cutoff in self.shell_cutoffs:
            # Get atoms in this shell
            mask = (distances > prev_cutoff) & (distances <= cutoff)
            shell_indices = indices[mask]
            shell_dists = distances[mask]
            
            if len(shell_indices) > 0:
                # Calculate composition
                elements = [self.atoms[i].symbol for i in shell_indices]
                composition = {}
                for elem in set(elements):
                    composition[elem] = elements.count(elem) / len(elements)
                
                shell_compositions.append(composition)
                shell_distances.append(float(np.mean(shell_dists)))
                coordination_numbers.append(len(shell_indices))
            else:
                shell_compositions.append({})
                shell_distances.append(cutoff)
                coordination_numbers.append(0)
            
            prev_cutoff = cutoff
        
        # Get first shell details (NN)
        nn_mask = distances <= self.shell_cutoffs[0]
        nn_indices = indices[nn_mask]
        nn_dists = distances[nn_mask]
        
        # Sort by distance
        sort_idx = np.argsort(nn_dists)
        nn_elements = [self.atoms[i].symbol for i in nn_indices[sort_idx]]
        nn_distances = nn_dists[sort_idx].tolist()
        
        local_env = LocalEnvironment(
            vacancy_index=index,
            vacancy_element=element,
            shell_compositions=shell_compositions,
            shell_distances=shell_distances,
            nn_elements=nn_elements,
            nn_distances=nn_distances,
            coordination_numbers=coordination_numbers
        )
        
        self._neighbor_cache[index] = local_env
        return local_env


class NEBHeterogeneityAnalyzer:
    """
    Analyzes heterogeneity in the potential energy landscape using NEB results
    and local environment characterization.
    """
    
    def __init__(self):
        """Initialize the heterogeneity analyzer."""
        self.vacancy_sites: List[VacancySiteData] = []
        self.structure_id = 0
        
    def add_structure_results(
        self,
        atoms: Atoms,
        neb_results: List[Dict],
        structure_info: Optional[Dict] = None,
        shell_cutoffs: List[float] = [2.8, 3.5, 4.0]
    ):
        """
        Process NEB results for a structure and extract local environment + barrier data.
        
        Args:
            atoms: The optimized structure used for NEB calculations
            neb_results: List of NEB calculation results from VacancyDiffusion.run_multiple()
            structure_info: Optional metadata about the structure (composition, etc.)
            shell_cutoffs: Cutoff distances for neighbor shells
        """
        if structure_info is None:
            structure_info = {
                'structure_id': self.structure_id,
                'formula': atoms.get_chemical_formula()
            }
        else:
            structure_info['structure_id'] = self.structure_id
        
        self.structure_id += 1
        
        # Initialize environment analyzer
        env_analyzer = LocalEnvironmentAnalyzer(atoms, shell_cutoffs)
        
        # Group results by vacancy site (only NN jumps)
        vacancy_groups = defaultdict(list)
        for result in neb_results:
            if result.get('success') and result.get('is_nearest_neighbor', True):
                vac_idx = int(result['vacancy_index'])
                vacancy_groups[vac_idx].append(result)
        
        # Process each vacancy site
        for vac_idx, site_results in vacancy_groups.items():
            # Characterize local environment
            local_env = env_analyzer.characterize_local_environment(vac_idx)
            
            # Extract barriers and target elements
            nn_barriers = []
            nn_target_elements = []
            converged_count = 0
            
            for result in site_results:
                if result.get('barrier') is not None:
                    nn_barriers.append(result['barrier'])
                    nn_target_elements.append(result['target_element'])
                    if result.get('converged', False):
                        converged_count += 1
            
            if not nn_barriers:
                continue
            
            # Calculate barrier statistics
            barrier_stats = {
                'mean': float(np.mean(nn_barriers)),
                'std': float(np.std(nn_barriers)),
                'min': float(np.min(nn_barriers)),
                'max': float(np.max(nn_barriers)),
                'range': float(np.max(nn_barriers) - np.min(nn_barriers)),
                'median': float(np.median(nn_barriers)),
                'n_barriers': len(nn_barriers)
            }
            
            # Create vacancy site data
            site_data = VacancySiteData(
                vacancy_index=vac_idx,
                local_env=local_env,
                nn_barriers=nn_barriers,
                nn_target_elements=nn_target_elements,
                barrier_statistics=barrier_stats,
                converged_fraction=converged_count / len(nn_barriers) if nn_barriers else 0.0,
                structure_info=structure_info
            )
            
            self.vacancy_sites.append(site_data)
    
    def calculate_global_heterogeneity(self) -> Dict[str, float]:
        """
        Calculate global heterogeneity metrics across all vacancy sites.
        
        Returns:
            Dictionary with various heterogeneity metrics
        """
        if not self.vacancy_sites:
            return {}
        
        # Collect mean barriers from each site
        site_mean_barriers = [site.barrier_statistics['mean'] for site in self.vacancy_sites]
        
        # Collect barrier ranges from each site
        site_ranges = [site.barrier_statistics['range'] for site in self.vacancy_sites]
        
        # Collect all individual barriers
        all_barriers = []
        for site in self.vacancy_sites:
            all_barriers.extend(site.nn_barriers)
        
        # Calculate heterogeneity metrics
        metrics = {
            # Site-to-site heterogeneity
            'site_mean_barriers_std': float(np.std(site_mean_barriers)),
            'site_mean_barriers_range': float(np.max(site_mean_barriers) - np.min(site_mean_barriers)),
            'site_mean_barriers_cv': float(np.std(site_mean_barriers) / np.mean(site_mean_barriers)),
            
            # Within-site heterogeneity (average)
            'avg_within_site_range': float(np.mean(site_ranges)),
            'avg_within_site_std': float(np.mean([site.barrier_statistics['std'] for site in self.vacancy_sites])),
            
            # Overall statistics
            'global_mean_barrier': float(np.mean(all_barriers)),
            'global_std_barrier': float(np.std(all_barriers)),
            'global_min_barrier': float(np.min(all_barriers)),
            'global_max_barrier': float(np.max(all_barriers)),
            
            # Sample statistics
            'n_vacancy_sites': len(self.vacancy_sites),
            'n_total_barriers': len(all_barriers),
            'avg_barriers_per_site': float(np.mean([len(site.nn_barriers) for site in self.vacancy_sites]))
        }
        
        return metrics
    
    def export_dataset(self, filepath: Path) -> None:
        """
        Export the complete dataset to JSON for downstream analysis.
        
        Args:
            filepath: Path to save the JSON file
        """
        dataset = {
            'metadata': {
                'n_structures': len(set(site.structure_info['structure_id'] for site in self.vacancy_sites)),
                'n_vacancy_sites': len(self.vacancy_sites),
                'n_total_barriers': sum(len(site.nn_barriers) for site in self.vacancy_sites)
            },
            'global_heterogeneity': self.calculate_global_heterogeneity(),
            'vacancy_sites': []
        }
        
        for site in self.vacancy_sites:
            site_dict = {
                'vacancy_index': site.vacancy_index,
                'structure_info': site.structure_info,
                'local_environment': {
                    'vacancy_element': site.local_env.vacancy_element,
                    'shell_compositions': site.local_env.shell_compositions,
                    'shell_distances': site.local_env.shell_distances,
                    'coordination_numbers': site.local_env.coordination_numbers,
                    'nn_elements': site.local_env.nn_elements,
                    'nn_distances': site.local_env.nn_distances
                },
                'barriers': {
                    'values': site.nn_barriers,
                    'target_elements': site.nn_target_elements,
                    'statistics': site.barrier_statistics,
                    'converged_fraction': site.converged_fraction
                }
            }
            dataset['vacancy_sites'].append(site_dict)
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
    
    def export_to_dataframe(self) -> Optional['pd.DataFrame']:
        """
        Export flattened dataset to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with one row per vacancy site, or None if pandas not available
        """
        if not HAS_PANDAS:
            print("Warning: pandas not available. Install with: pip install pandas")
            return None
        
        rows = []
        for site in self.vacancy_sites:
            row = {
                'vacancy_index': site.vacancy_index,
                'vacancy_element': site.local_env.vacancy_element,
                'structure_id': site.structure_info['structure_id'],
                'structure_formula': site.structure_info.get('formula', ''),
                
                # Barrier statistics
                'barrier_mean': site.barrier_statistics['mean'],
                'barrier_std': site.barrier_statistics['std'],
                'barrier_min': site.barrier_statistics['min'],
                'barrier_max': site.barrier_statistics['max'],
                'barrier_range': site.barrier_statistics['range'],
                'n_barriers': site.barrier_statistics['n_barriers'],
                'converged_fraction': site.converged_fraction,
            }
            
            # Add shell-wise composition features
            for shell_idx, shell_comp in enumerate(site.local_env.shell_compositions):
                for elem, frac in shell_comp.items():
                    row[f'shell{shell_idx+1}_{elem}'] = frac
                row[f'shell{shell_idx+1}_mean_dist'] = site.local_env.shell_distances[shell_idx]
                row[f'shell{shell_idx+1}_coord'] = site.local_env.coordination_numbers[shell_idx]
            
            # Add global composition if available
            if 'composition' in site.structure_info:
                for elem, frac in site.structure_info['composition'].items():
                    row[f'global_{elem}'] = frac
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_heterogeneity_overview(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Create comprehensive visualization of heterogeneity in the PEL.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
        """
        if not self.vacancy_sites:
            print("No vacancy site data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Potential Energy Landscape Heterogeneity Analysis', fontsize=16, y=1.00)
        
        # 1. Distribution of mean barriers per site
        ax = axes[0, 0]
        site_means = [site.barrier_statistics['mean'] for site in self.vacancy_sites]
        ax.hist(site_means, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(site_means), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(site_means):.3f} eV')
        ax.set_xlabel('Mean Barrier per Site (eV)')
        ax.set_ylabel('Count')
        ax.set_title('Site-to-Site Heterogeneity')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Distribution of within-site ranges
        ax = axes[0, 1]
        site_ranges = [site.barrier_statistics['range'] for site in self.vacancy_sites]
        ax.hist(site_ranges, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(np.mean(site_ranges), color='red', linestyle='--',
                   label=f'Mean: {np.mean(site_ranges):.3f} eV')
        ax.set_xlabel('Barrier Range per Site (eV)')
        ax.set_ylabel('Count')
        ax.set_title('Within-Site Heterogeneity')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. All barriers distribution
        ax = axes[0, 2]
        all_barriers = []
        for site in self.vacancy_sites:
            all_barriers.extend(site.nn_barriers)
        ax.hist(all_barriers, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(np.mean(all_barriers), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_barriers):.3f} eV')
        ax.set_xlabel('Barrier (eV)')
        ax.set_ylabel('Count')
        ax.set_title('Overall Barrier Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Mean barrier vs std (site characterization)
        ax = axes[1, 0]
        means = [site.barrier_statistics['mean'] for site in self.vacancy_sites]
        stds = [site.barrier_statistics['std'] for site in self.vacancy_sites]
        scatter = ax.scatter(means, stds, alpha=0.6, s=50)
        ax.set_xlabel('Mean Barrier (eV)')
        ax.set_ylabel('Std Deviation (eV)')
        ax.set_title('Site Characterization: Mean vs Spread')
        ax.grid(alpha=0.3)
        
        # 5. Barrier range by vacancy element
        ax = axes[1, 1]
        element_data = defaultdict(list)
        for site in self.vacancy_sites:
            element_data[site.local_env.vacancy_element].append(site.barrier_statistics['mean'])
        
        elements = sorted(element_data.keys())
        data_to_plot = [element_data[elem] for elem in elements]
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=elements, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_xlabel('Vacancy Element')
            ax.set_ylabel('Mean Barrier (eV)')
            ax.set_title('Barriers by Vacancy Element Type')
            ax.grid(alpha=0.3, axis='y')
        
        # 6. Convergence statistics
        ax = axes[1, 2]
        conv_fracs = [site.converged_fraction for site in self.vacancy_sites]
        ax.hist(conv_fracs, bins=20, alpha=0.7, edgecolor='black', color='purple')
        ax.axvline(np.mean(conv_fracs), color='red', linestyle='--',
                   label=f'Mean: {np.mean(conv_fracs):.2%}')
        ax.set_xlabel('Convergence Fraction')
        ax.set_ylabel('Count')
        ax.set_title('NEB Convergence Quality')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def analyze_environment_barrier_correlation(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Analyze correlation between local environment features and barriers.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not HAS_PANDAS:
            print("Warning: pandas required for correlation analysis")
            return
        
        df = self.export_to_dataframe()
        if df is None or len(df) < 5:
            print("Insufficient data for correlation analysis")
            return
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['vacancy_index', 'structure_id']]
        
        if not feature_cols:
            print("No numeric features found for correlation analysis")
            return
        
        # Focus on barrier statistics
        barrier_cols = ['barrier_mean', 'barrier_std', 'barrier_range']
        
        # Compute correlations
        correlation_matrix = df[feature_cols].corr()
        
        # Extract correlations with barrier statistics
        barrier_correlations = {}
        for bcol in barrier_cols:
            if bcol in correlation_matrix.columns:
                corrs = correlation_matrix[bcol].drop(barrier_cols, errors='ignore')
                # Sort by absolute correlation
                corrs_sorted = corrs.abs().sort_values(ascending=False)
                barrier_correlations[bcol] = corrs.loc[corrs_sorted.index]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Local Environment - Barrier Correlation Analysis', fontsize=16)
        
        # 1. Top correlations with mean barrier
        ax = axes[0, 0]
        if 'barrier_mean' in barrier_correlations:
            top_corrs = barrier_correlations['barrier_mean'].head(10)
            colors = ['red' if x < 0 else 'blue' for x in top_corrs.values]
            ax.barh(range(len(top_corrs)), top_corrs.values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_corrs)))
            ax.set_yticklabels(top_corrs.index, fontsize=8)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_title('Top 10 Correlations with Mean Barrier')
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(alpha=0.3, axis='x')
        
        # 2. Top correlations with barrier range (heterogeneity)
        ax = axes[0, 1]
        if 'barrier_range' in barrier_correlations:
            top_corrs = barrier_correlations['barrier_range'].head(10)
            colors = ['red' if x < 0 else 'green' for x in top_corrs.values]
            ax.barh(range(len(top_corrs)), top_corrs.values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_corrs)))
            ax.set_yticklabels(top_corrs.index, fontsize=8)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_title('Top 10 Correlations with Barrier Range')
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(alpha=0.3, axis='x')
        
        # 3. Shell composition effect (if available)
        ax = axes[1, 0]
        shell1_cols = [col for col in df.columns if col.startswith('shell1_') and col.endswith(('V', 'Cr', 'Ti', 'W', 'Zr'))]
        if shell1_cols and 'barrier_mean' in df.columns:
            for col in shell1_cols[:5]:  # Limit to 5 elements
                elem = col.split('_')[-1]
                valid_data = df[[col, 'barrier_mean']].dropna()
                if len(valid_data) > 3:
                    ax.scatter(valid_data[col], valid_data['barrier_mean'], 
                             label=elem, alpha=0.6, s=30)
            ax.set_xlabel('1st Shell Element Fraction')
            ax.set_ylabel('Mean Barrier (eV)')
            ax.set_title('1st Shell Composition vs Barrier')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 4. Coordination number effect
        ax = axes[1, 1]
        coord_cols = [col for col in df.columns if 'coord' in col]
        if coord_cols and 'barrier_mean' in df.columns:
            for col in coord_cols[:3]:  # First 3 shells
                shell_num = col.split('shell')[1][0] if 'shell' in col else '?'
                valid_data = df[[col, 'barrier_mean']].dropna()
                if len(valid_data) > 3:
                    ax.scatter(valid_data[col], valid_data['barrier_mean'],
                             label=f'Shell {shell_num}', alpha=0.6, s=30)
            ax.set_xlabel('Coordination Number')
            ax.set_ylabel('Mean Barrier (eV)')
            ax.set_title('Coordination Number vs Barrier')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return barrier_correlations
    
    def cluster_environments(
        self,
        n_clusters: int = 5,
        method: str = 'kmeans',
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Cluster local environments and visualize barrier distributions per cluster.
        
        Args:
            n_clusters: Number of clusters for KMeans
            method: Clustering method ('kmeans' or 'dbscan')
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not HAS_SKLEARN or not HAS_PANDAS:
            print("Warning: sklearn and pandas required for clustering analysis")
            return None
        
        df = self.export_to_dataframe()
        if df is None or len(df) < n_clusters:
            print("Insufficient data for clustering")
            return None
        
        # Select features for clustering (composition features)
        feature_cols = []
        for col in df.columns:
            if (col.startswith('shell') and ('_' in col) and 
                any(col.endswith(elem) for elem in ['V', 'Cr', 'Ti', 'W', 'Zr', 'coord'])):
                feature_cols.append(col)
        
        if not feature_cols:
            print("No suitable features found for clustering")
            return None
        
        # Prepare data (fill NaN with 0 for composition fractions)
        X = df[feature_cols].fillna(0).values
        
        # Dimensionality reduction for visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=min(2, X.shape[1]))
            X_viz = pca.fit_transform(X)
        else:
            X_viz = X
        
        # Clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        clusters = clusterer.fit_predict(X)
        df['cluster'] = clusters
        
        # Create visualization
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Cluster visualization in PCA space
        ax1 = fig.add_subplot(gs[0, :])
        scatter = ax1.scatter(X_viz[:, 0], X_viz[:, 1], c=clusters, 
                             cmap='tab10', alpha=0.6, s=50)
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        ax1.set_title('Local Environment Clusters (PCA Visualization)')
        plt.colorbar(scatter, ax=ax1, label='Cluster ID')
        ax1.grid(alpha=0.3)
        
        # 2. Barrier distributions per cluster
        ax2 = fig.add_subplot(gs[1, 0])
        cluster_ids = sorted(df['cluster'].unique())
        barrier_data = [df[df['cluster'] == cid]['barrier_mean'].dropna() 
                       for cid in cluster_ids]
        
        bp = ax2.boxplot(barrier_data, labels=cluster_ids, patch_artist=True)
        for patch, cid in zip(bp['boxes'], cluster_ids):
            color = plt.cm.tab10(cid / max(cluster_ids))
            patch.set_facecolor(color)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Mean Barrier (eV)')
        ax2.set_title('Barrier Distribution by Cluster')
        ax2.grid(alpha=0.3, axis='y')
        
        # 3. Cluster statistics table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        stats_text = "Cluster Statistics:\n\n"
        for cid in cluster_ids:
            cluster_data = df[df['cluster'] == cid]
            n_sites = len(cluster_data)
            mean_barrier = cluster_data['barrier_mean'].mean()
            std_barrier = cluster_data['barrier_mean'].std()
            stats_text += f"Cluster {cid}: n={n_sites}, "
            stats_text += f"μ={mean_barrier:.3f}±{std_barrier:.3f} eV\n"
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')
        
        plt.suptitle(f'Local Environment Clustering ({method.upper()})', 
                    fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return df, clusters


def integrate_heterogeneity_analysis(
    workflow_results: Dict,
    optimized_structures: List[Atoms],
    output_dir: Path,
    shell_cutoffs: List[float] = [2.8, 3.5, 4.0]
) -> Dict:
    """
    Convenience function to run complete heterogeneity analysis on workflow results.
    
    Args:
        workflow_results: Results from HybridNEBWorkflow
        optimized_structures: List of optimized structures
        output_dir: Directory to save analysis results
        shell_cutoffs: Cutoff distances for neighbor shells
        
    Returns:
        Dictionary containing heterogeneity analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    het_analyzer = NEBHeterogeneityAnalyzer()
    
    # Process results for each structure
    neb_results = workflow_results.get('analysis_results', {}).get('neb_results', [])
    
    # Group NEB results by structure
    structure_groups = defaultdict(list)
    for result in neb_results:
        struct_idx = result.get('structure_index', 0)
        structure_groups[struct_idx].append(result)
    
    # Add each structure's results
    for struct_idx, struct_results in structure_groups.items():
        if struct_idx < len(optimized_structures):
            atoms = optimized_structures[struct_idx]
            composition = workflow_results.get('compositions', [{}])[struct_idx] if struct_idx < len(workflow_results.get('compositions', [])) else {}
            
            structure_info = {
                'composition': composition,
                'formula': atoms.get_chemical_formula(),
                'n_atoms': len(atoms)
            }
            
            het_analyzer.add_structure_results(
                atoms=atoms,
                neb_results=struct_results,
                structure_info=structure_info,
                shell_cutoffs=shell_cutoffs
            )
    
    # Calculate and save results
    print("\n=== Heterogeneity Analysis ===")
    global_het = het_analyzer.calculate_global_heterogeneity()
    
    print(f"\nGlobal Heterogeneity Metrics:")
    print(f"  Site-to-site std: {global_het.get('site_mean_barriers_std', 0):.4f} eV")
    print(f"  Site-to-site range: {global_het.get('site_mean_barriers_range', 0):.4f} eV")
    print(f"  Avg within-site range: {global_het.get('avg_within_site_range', 0):.4f} eV")
    print(f"  Global mean barrier: {global_het.get('global_mean_barrier', 0):.4f} eV")
    print(f"  Coefficient of variation: {global_het.get('site_mean_barriers_cv', 0):.4f}")
    
    # Export dataset
    het_analyzer.export_dataset(output_dir / "heterogeneity_dataset.json")
    print(f"\nDataset exported to: {output_dir / 'heterogeneity_dataset.json'}")
    
    # Export to DataFrame if pandas available
    df = het_analyzer.export_to_dataframe()
    if df is not None:
        df.to_csv(output_dir / "heterogeneity_dataset.csv", index=False)
        print(f"DataFrame exported to: {output_dir / 'heterogeneity_dataset.csv'}")
    
    # Create visualizations
    het_analyzer.plot_heterogeneity_overview(
        save_path=output_dir / "heterogeneity_overview.png"
    )
    print(f"Overview plot saved to: {output_dir / 'heterogeneity_overview.png'}")
    
    het_analyzer.analyze_environment_barrier_correlation(
        save_path=output_dir / "environment_barrier_correlation.png"
    )
    print(f"Correlation plot saved to: {output_dir / 'environment_barrier_correlation.png'}")
    
    # Clustering analysis
    if len(het_analyzer.vacancy_sites) >= 5:
        het_analyzer.cluster_environments(
            n_clusters=min(5, len(het_analyzer.vacancy_sites) // 2),
            save_path=output_dir / "environment_clusters.png"
        )
        print(f"Clustering plot saved to: {output_dir / 'environment_clusters.png'}")
    
    return {
        'global_heterogeneity': global_het,
        'n_vacancy_sites': len(het_analyzer.vacancy_sites),
        'analyzer': het_analyzer
    }



