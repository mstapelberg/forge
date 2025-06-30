from typing import Optional
import numpy as np
from ase import Atoms

# We will conditionally import plotting libraries to provide better error messages
# if they are not installed. This prevents the entire module from failing if
# a user only wants to use the non-plotting parts of the analysis toolkit.
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy import stats
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import nglview as nv
    from nglview.color import ColormakerRegistry
    NGLVIEW_AVAILABLE = True
except ImportError:
    NGLVIEW_AVAILABLE = False

def plot_structure_ngl(atoms: Atoms, errors: np.ndarray, cmap_name: str = 'viridis'):
    """
    Creates an interactive 3D plot of a structure using NGLView, with
    atoms colored by the logarithm of their error values.

    Args:
        atoms: The ASE Atoms object to display.
        errors: A 1D numpy array of per-atom error magnitudes.
        cmap_name: The name of the matplotlib colormap to use.

    Returns:
        An NGLWidget instance for display in a Jupyter notebook.
    """
    if not NGLVIEW_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("[WARN] nglview or matplotlib not installed. Skipping plot. "
              "`pip install nglview matplotlib`")
        return None

    # Use log scale for better color differentiation, handle zeros
    log_errors = np.log10(errors + 1e-9) 
    min_log_err = np.min(log_errors)
    max_log_err = np.max(log_errors)

    # Normalize log errors to [0, 1] for color mapping
    norm_errors = (log_errors - min_log_err) / (max_log_err - min_log_err + 1e-9)

    # Get the colormap from matplotlib
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm_errors)

    # NGLView expects colors in a list of lists/tuples format
    color_list = [list(c[:3]) for c in colors] # Drop alpha channel

    # Create a unique name for the custom color scheme
    scheme_name = f"custom_log_errors_{np.random.randint(0, 10000)}"
    
    # Register the custom color scheme with NGLView
    if scheme_name not in ColormakerRegistry.get_schemes():
        ColormakerRegistry.add_selection_scheme(scheme_name, color_list)

    # Create and configure the NGLView widget
    view = nv.show_ase(atoms, default_representation=False)
    view.add_ball_and_stick(color_scheme=scheme_name)
    view.center()
    
    # Add a title to the view with the error range
    view.parameters = {
        "title": f"Log10(Error) Range: [{min_log_err:.2f}, {max_log_err:.2f}]"
    }
    
    return view

def plot_force_error_histogram(
    errors: np.ndarray, bins: int = 50, use_log_scale: bool = True, title: str = 'Force Error Distribution'
) -> Optional[plt.Figure]:
    """
    Plots a histogram of force error magnitudes.

    Args:
        errors: 1D array of scalar force error magnitudes.
        bins: Number of bins for the histogram.
        use_log_scale: If True, the y-axis will be on a logarithmic scale.
        title: The title of the plot.

    Returns:
        A matplotlib Figure object, or None if matplotlib is not installed.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib not installed. Skipping histogram plot. `pip install matplotlib`")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(errors, bins=bins, density=True, alpha=0.8)
    if use_log_scale:
        ax.set_yscale('log')
    ax.set_xlabel('Force Error Magnitude (|ΔF|)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    plt.show()
    return fig

def plot_qq(errors: np.ndarray, title: str = 'Q-Q Plot vs. Normal Distribution') -> Optional[plt.Figure]:
    """
    Creates a Quantile-Quantile (Q-Q) plot of errors against a normal distribution.

    This plot is useful for visually assessing if the error distribution is
    heavy-tailed compared to a normal distribution.

    Args:
        errors: 1D array of scalar error values.
        title: The title of the plot.

    Returns:
        A matplotlib Figure object, or None if matplotlib is not installed.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib or scipy not installed. Skipping Q-Q plot. "
              "`pip install matplotlib scipy`")
        return None
        
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title(title, fontsize=14)
    ax.get_lines()[0].set_markerfacecolor('C0')
    ax.get_lines()[0].set_markeredgecolor('C0')
    ax.get_lines()[0].set_markersize(4.0)
    ax.get_lines()[1].set_linewidth(2.0)
    ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12)
    ax.set_ylabel('Sample Quantiles (Errors)', fontsize=12)
    fig.tight_layout()
    plt.show()
    return fig 