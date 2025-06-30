"""Plotting utilities for visualization of analysis results."""
from typing import Optional, Union, List, Dict, Any
import numpy as np
from ase import Atoms
import logging

logger = logging.getLogger(__name__)

# Conditional imports for plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy import stats
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not installed. Plotting functions will be unavailable.")

try:
    import nglview as nv
    from nglview.color import ColormakerRegistry
    NGLVIEW_AVAILABLE = True
except ImportError:
    NGLVIEW_AVAILABLE = False
    logger.warning("NGLView not installed. 3D visualization will be unavailable.")


def plot_force_error_histogram(
    errors: np.ndarray,
    bins: int = 50,
    use_log_scale: bool = True,
    title: str = "Force Error Distribution",
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot a histogram of force error magnitudes.
    
    Parameters
    ----------
    errors : np.ndarray
        1D array of force error magnitudes.
    bins : int, optional
        Number of histogram bins (default: 50).
    use_log_scale : bool, optional
        Whether to use log scale on y-axis (default: True).
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : Optional[str]
        If provided, saves the figure to this path.
        
    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure object, or None if matplotlib not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot create histogram - matplotlib not installed")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    counts, bin_edges, patches = ax.hist(
        errors, bins=bins, density=True, alpha=0.7, 
        color='steelblue', edgecolor='black', linewidth=0.5
    )
    
    # Add statistics
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    q95_err = np.quantile(errors, 0.95)
    
    # Add vertical lines for statistics
    ax.axvline(mean_err, color='red', linestyle='--', 
               label=f'Mean: {mean_err:.3f}')
    ax.axvline(median_err, color='green', linestyle='--',
               label=f'Median: {median_err:.3f}')
    ax.axvline(q95_err, color='orange', linestyle='--',
               label=f'Q95: {q95_err:.3f}')
    
    if use_log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('Force Error Magnitude (eV/Å)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved histogram to {save_path}")
    
    return fig


def plot_qq(
    errors: np.ndarray,
    dist: str = "norm",
    title: str = "Q-Q Plot vs. Normal Distribution",
    figsize: tuple = (6, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Create a Q-Q plot of errors against a theoretical distribution.
    
    Parameters
    ----------
    errors : np.ndarray
        1D array of error values.
    dist : str, optional
        Theoretical distribution to compare against (default: 'norm').
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    save_path : Optional[str]
        If provided, saves the figure to this path.
        
    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure object, or None if matplotlib not available.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot create Q-Q plot - matplotlib not installed")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create Q-Q plot
    stats.probplot(errors, dist=dist, plot=ax)
    
    # Customize appearance
    ax.set_title(title, fontsize=14)
    ax.get_lines()[0].set_markerfacecolor('steelblue')
    ax.get_lines()[0].set_markeredgecolor('darkblue')
    ax.get_lines()[0].set_markersize(6)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)
    
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add text box with deviation measure
    _, (slope, intercept, r) = stats.probplot(errors, dist=dist, plot=None)
    textstr = f'R² = {r**2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved Q-Q plot to {save_path}")
    
    return fig


def plot_structure_ngl(
    atoms: Atoms,
    errors: np.ndarray,
    cmap_name: str = "viridis",
    use_log_scale: bool = True,
    size: tuple = (800, 600)
) -> Optional[Any]:
    """Create an interactive 3D plot using NGLView.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object to visualize.
    errors : np.ndarray
        Per-atom error values for coloring.
    cmap_name : str, optional
        Matplotlib colormap name (default: 'viridis').
    use_log_scale : bool, optional
        Whether to use log scale for errors (default: True).
    size : tuple, optional
        Widget size (width, height).
        
    Returns
    -------
    Optional[Any]
        NGLWidget instance, or None if nglview not available.
    """
    if not NGLVIEW_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot create 3D visualization - nglview or matplotlib not installed")
        return None
    
    # Handle log scale
    if use_log_scale:
        display_errors = np.log10(errors + 1e-9)
        scale_label = "log10"
    else:
        display_errors = errors
        scale_label = "linear"
    
    # Normalize errors for coloring
    min_err = np.min(display_errors)
    max_err = np.max(display_errors)
    
    if max_err > min_err:
        norm_errors = (display_errors - min_err) / (max_err - min_err)
    else:
        norm_errors = np.zeros_like(display_errors)
    
    # Get colormap and create color list
    cmap = plt.get_cmap(cmap_name)
    colors = [list(cmap(val)[:3]) for val in norm_errors]
    
    # Create unique color scheme name
    scheme_name = f"error_colors_{np.random.randint(0, 10000)}"
    
    # Register color scheme
    if scheme_name not in ColormakerRegistry.get_schemes():
        ColormakerRegistry.add_selection_scheme(scheme_name, colors)
    
    # Create view
    view = nv.show_ase(atoms, default_representation=False)
    view.add_ball_and_stick(color_scheme=scheme_name)
    view.center()
    view._set_size(f"{size[0]}px", f"{size[1]}px")
    
    # Add title/info
    if use_log_scale:
        title_text = f"Error range ({scale_label}): [{min_err:.2f}, {max_err:.2f}]"
    else:
        title_text = f"Error range: [{min_err:.3f}, {max_err:.3f}]"
    
    view.parameters = {"title": title_text}
    
    return view


def plot_error_correlation(
    metric1: np.ndarray,
    metric2: np.ndarray,
    metric1_name: str = "Metric 1",
    metric2_name: str = "Metric 2",
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot correlation between two error metrics.
    
    Parameters
    ----------
    metric1, metric2 : np.ndarray
        Arrays of metric values to correlate.
    metric1_name, metric2_name : str
        Names for axis labels.
    figsize : tuple
        Figure size.
    save_path : Optional[str]
        If provided, saves the figure.
        
    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure object.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot create correlation plot - matplotlib not installed")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(metric1, metric2, alpha=0.6, s=50, 
                        c=metric1 + metric2, cmap='plasma')
    
    # Add trend line
    z = np.polyfit(metric1, metric2, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(metric1.min(), metric1.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation
    corr = np.corrcoef(metric1, metric2)[0, 1]
    
    # Labels and title
    ax.set_xlabel(metric1_name, fontsize=12)
    ax.set_ylabel(metric2_name, fontsize=12)
    ax.set_title(f'Correlation: r = {corr:.3f}', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Combined Metric', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved correlation plot to {save_path}")
    
    return fig


def plot_spatial_errors(
    atoms: Atoms,
    errors: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
    view_axis: str = 'z',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """Plot spatial distribution of errors in 2D projection.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    errors : np.ndarray
        Per-atom error values.
    cluster_labels : Optional[np.ndarray]
        Cluster labels for each atom (-1 for noise, >=0 for clusters).
    view_axis : str
        Axis to project along ('x', 'y', or 'z').
    figsize : tuple
        Figure size.
    save_path : Optional[str]
        If provided, saves the figure.
        
    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure object.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Cannot create spatial plot - matplotlib not installed")
        return None
    
    positions = atoms.positions
    
    # Determine projection axes
    axis_map = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}
    if view_axis not in axis_map:
        view_axis = 'z'
    ax1_idx, ax2_idx = axis_map[view_axis]
    ax_labels = ['x', 'y', 'z']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Error magnitude
    scatter1 = ax1.scatter(
        positions[:, ax1_idx], positions[:, ax2_idx],
        c=errors, s=100, cmap='hot', edgecolors='black', linewidth=0.5
    )
    ax1.set_xlabel(f'{ax_labels[ax1_idx]} (Å)', fontsize=12)
    ax1.set_ylabel(f'{ax_labels[ax2_idx]} (Å)', fontsize=12)
    ax1.set_title('Error Magnitude Distribution', fontsize=14)
    ax1.set_aspect('equal')
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Error Magnitude', fontsize=10)
    
    # Right plot: Clusters (if provided)
    if cluster_labels is not None:
        # Create custom colormap for clusters
        unique_labels = np.unique(cluster_labels)
        n_colors = len(unique_labels[unique_labels >= 0])
        
        if n_colors > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
            cluster_colors = np.zeros((len(cluster_labels), 4))
            
            # Color assignment
            color_idx = 0
            for label in unique_labels:
                if label >= 0:
                    mask = cluster_labels == label
                    cluster_colors[mask] = colors[color_idx]
                    color_idx += 1
                elif label == -1:
                    mask = cluster_labels == -1
                    cluster_colors[mask] = [0.7, 0.7, 0.7, 1]  # Gray for noise
                else:
                    mask = cluster_labels == -2
                    cluster_colors[mask] = [0.9, 0.9, 0.9, 0.3]  # Light gray
            
            scatter2 = ax2.scatter(
                positions[:, ax1_idx], positions[:, ax2_idx],
                c=cluster_colors, s=100, edgecolors='black', linewidth=0.5
            )
            ax2.set_xlabel(f'{ax_labels[ax1_idx]} (Å)', fontsize=12)
            ax2.set_ylabel(f'{ax_labels[ax2_idx]} (Å)', fontsize=12)
            ax2.set_title('Error Clusters', fontsize=14)
            ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved spatial plot to {save_path}")
    
    return fig 