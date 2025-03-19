import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class PotentialResultsAnalyzer:
    def __init__(self, json_file_path: str):
        """Initialize the analyzer with a JSON file path."""
        self.data = self._load_json(json_file_path)
        self.structures = self._extract_structures()
        
    def _load_json(self, file_path: str) -> Dict:
        """Load JSON data from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _extract_structures(self) -> List[Dict]:
        """Extract all structure data from the JSON."""
        # Assuming the first element contains the per_structure data
        return self.data[0]['per_structure']
    
    def plot_inference_time_vs_atoms(self, save_path: str = None, plot_type: str = "boxplot"):
        """
        Plot inference time vs number of atoms.
        
        Parameters:
        - save_path: Path to save the figure (optional)
        - plot_type: Type of plot to create ("boxplot", "scatter", or "both")
        """
        # Create a DataFrame for easier manipulation
        data = []
        for s in self.structures:
            data.append({
                'n_atoms': s['n_atoms'],
                'inference_time': s['inference_time']
            })
        df = pd.DataFrame(data)
        
        # Set up the figure
        plt.figure(figsize=(12, 7))
        
        if plot_type in ["boxplot", "both"]:
            # Create boxplot
            ax = plt.subplot(1, 2 if plot_type == "both" else 1, 1)
            # Group by number of atoms
            atom_counts = sorted(df['n_atoms'].unique())
            boxplot_data = [df[df['n_atoms'] == count]['inference_time'].values for count in atom_counts]
            
            # Create the boxplot
            bp = ax.boxplot(boxplot_data, patch_artist=True)
            
            # Set colors
            for box in bp['boxes']:
                box.set(facecolor='lightblue', alpha=0.7)
            
            # Set x-tick labels to atom counts
            ax.set_xticklabels(atom_counts)
            ax.set_xlabel('Number of Atoms')
            ax.set_ylabel('Inference Time (s)')
            ax.set_title('Inference Time vs Number of Atoms')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Note: Trend line removed as requested
        
        if plot_type in ["scatter", "both"]:
            # Create scatter plot
            ax = plt.subplot(1, 2 if plot_type == "both" else 1, 2 if plot_type == "both" else 1)
            ax.scatter(df['n_atoms'], df['inference_time'], alpha=0.7)
            ax.set_xlabel('Number of Atoms')
            ax.set_ylabel('Inference Time (s)')
            ax.set_title('Inference Time vs Number of Atoms (Scatter)')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add a trend line based on all data points
            if len(df['n_atoms'].unique()) > 1:
                z = np.polyfit(df['n_atoms'], df['inference_time'], 1)
                p = np.poly1d(z)
                
                # Plot the trend line
                ax.plot(sorted(df['n_atoms']), p(sorted(df['n_atoms'])), 'r--', 
                        label=f'Trend: {z[0]:.2e}x + {z[1]:.2e}')
                ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate and return scaling information
        if len(df['n_atoms'].unique()) > 1:
            # Group by atom count and calculate means
            means = df.groupby('n_atoms')['inference_time'].mean().reset_index()
            
            # Fit trend line to means
            z = np.polyfit(means['n_atoms'], means['inference_time'], 1)
            
            print(f"Scaling analysis:")
            print(f"Linear fit (based on means): Inference time = {z[0]:.4e} × (number of atoms) + {z[1]:.4e}")
            
            # Calculate R-squared to assess fit quality
            p = np.poly1d(z)
            residuals = means['inference_time'] - p(means['n_atoms'])
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((means['inference_time'] - means['inference_time'].mean())**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"R² value: {r_squared:.4f}")
            print(f"This suggests {'linear' if 0.8 < z[0]/means['inference_time'].mean() < 1.2 and r_squared > 0.9 else 'non-linear'} scaling with system size.")
            
            return z
        return None
    
    def get_highest_error_structures(self, metric: str, n: int = 10) -> pd.DataFrame:
        """
        Get the N structures with the highest errors for a specific metric.
        
        Parameters:
        - metric: One of 'energy_error_per_atom', 'force_error', 'stress_error', 
                 'relative_energy_error_percent', 'relative_force_error_percent', 
                 'relative_stress_error_percent'
        - n: Number of structures to return
        
        Returns:
        - DataFrame with the top N structures sorted by the specified error metric
        """
        valid_metrics = [
            'energy_error_per_atom', 'force_error', 'stress_error',
            'relative_energy_error_percent', 'relative_force_error_percent', 
            'relative_stress_error_percent'
        ]
        
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}")
        
        # Create a DataFrame for easier manipulation
        df = pd.DataFrame(self.structures)
        
        # Sort by the specified metric in descending order
        sorted_df = df.sort_values(by=metric, ascending=False).head(n)
        
        # Select relevant columns
        if 'energy' in metric:
            true_col = 'true_energy_per_atom'
            pred_col = 'predicted_energy_per_atom'
        elif 'force' in metric:
            true_col = None  # Force is a vector, so we don't have a simple scalar value
            pred_col = None
        elif 'stress' in metric:
            true_col = None  # Stress is a tensor, so we don't have a simple scalar value
            pred_col = None
        
        # Create the result DataFrame
        result = sorted_df[['structure_index', 'config_type', metric]].copy()
        
        # Add true and predicted values if applicable
        if true_col and pred_col:
            result['true_value'] = sorted_df[true_col]
            result['predicted_value'] = sorted_df[pred_col]
        
        return result
    
    def plot_error_histograms_by_config(self, metric: str, bins: int = 20, 
                                        figsize: Tuple[int, int] = (12, 8),
                                        save_path: str = None,
                                        separate_configs: List[str] = None):
        """
        Plot histograms of errors grouped by config_type.
        
        Parameters:
        - metric: The error metric to plot
        - bins: Number of bins for the histogram
        - figsize: Figure size as (width, height)
        - save_path: Path to save the figure (optional)
        - separate_configs: List of config_types to plot in a separate histogram
        """
        valid_metrics = [
            'energy_error_per_atom', 'force_error', 'stress_error',
            'relative_energy_error_percent', 'relative_force_error_percent', 
            'relative_stress_error_percent'
        ]
        
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}")
        
        # Create a DataFrame
        df = pd.DataFrame(self.structures)
        
        # Get unique config types
        all_config_types = sorted(df['config_type'].unique())
        
        # Separate configs into two groups if specified
        if separate_configs:
            separate_configs = [c for c in separate_configs if c in all_config_types]
            main_configs = [c for c in all_config_types if c not in separate_configs]
        else:
            separate_configs = []
            main_configs = all_config_types
        
        # Determine if we need one or two plots
        has_two_plots = len(separate_configs) > 0
        
        # Set up the figure
        fig = plt.figure(figsize=figsize)
        
        # Function to create a histogram for a set of config types
        def create_histogram(ax, config_types, title_suffix=""):
            # Create a color map for this set of configs
            colors = plt.cm.tab10(np.linspace(0, 1, len(config_types)))
            
            # Plot histograms for each config type
            for i, config in enumerate(config_types):
                config_data = df[df['config_type'] == config][metric]
                if len(config_data) > 0:  # Only plot if we have data
                    ax.hist(config_data, bins=bins, alpha=0.7, 
                            label=f'{config} (n={len(config_data)})',
                            color=colors[i])
            
            # Add labels and title
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {metric} by Configuration Type {title_suffix}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create the main plot
        if has_two_plots:
            ax1 = fig.add_subplot(1, 2, 1)
            create_histogram(ax1, main_configs)
            
            ax2 = fig.add_subplot(1, 2, 2)
            create_histogram(ax2, separate_configs, "(Separated)")
            
            # Adjust x-axis limits for each plot separately based on its own data
            for ax, configs in [(ax1, main_configs), (ax2, separate_configs)]:
                subset_df = df[df['config_type'].isin(configs)]
                if len(subset_df) > 0:
                    # Only limit x-axis if there are extreme outliers
                    if subset_df[metric].max() > 5 * subset_df[metric].quantile(0.95):
                        ax.set_xlim(0, subset_df[metric].quantile(0.95) * 1.5)
                        ax.text(0.5, -0.1, f"Note: x-axis limited. Max value: {subset_df[metric].max():.4f}", 
                               ha='center', fontsize=8, transform=ax.transAxes)
        else:
            ax = fig.add_subplot(1, 1, 1)
            create_histogram(ax, main_configs)
            
            # Adjust x-axis to focus on the main distribution
            if df[metric].max() > 5 * df[metric].quantile(0.95):
                ax.set_xlim(0, df[metric].quantile(0.95) * 1.5)
                ax.text(0.5, -0.1, f"Note: x-axis limited. Max value: {df[metric].max():.4f}", 
                       ha='center', fontsize=8, transform=ax.transAxes)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_boxplots_by_config(self, metric: str, figsize: Tuple[int, int] = (12, 8),
                                     save_path: str = None):
        """
        Plot boxplots of errors grouped by config_type.
        
        Parameters:
        - metric: The error metric to plot
        - figsize: Figure size as (width, height)
        - save_path: Path to save the figure (optional)
        """
        valid_metrics = [
            'energy_error_per_atom', 'force_error', 'stress_error',
            'relative_energy_error_percent', 'relative_force_error_percent', 
            'relative_stress_error_percent'
        ]
        
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}")
        
        # Create a DataFrame
        df = pd.DataFrame(self.structures)
        
        # Set up the plot
        plt.figure(figsize=figsize)
        
        # Create the boxplot
        sns.boxplot(x='config_type', y=metric, data=df)
        
        # Add labels and title
        plt.xlabel('Configuration Type')
        plt.ylabel(metric)
        plt.title(f'Distribution of {metric} by Configuration Type')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def summarize_by_config_type(self) -> pd.DataFrame:
        """
        Create a summary of error metrics grouped by configuration type.
        
        Returns:
        - DataFrame with summary statistics for each config_type
        """
        df = pd.DataFrame(self.structures)
        
        # Define the metrics to summarize
        metrics = [
            'energy_error_per_atom', 'force_error', 'stress_error',
            'relative_energy_error_percent', 'relative_force_error_percent', 
            'relative_stress_error_percent'
        ]
        
        # Group by config_type and calculate statistics
        summary = df.groupby('config_type')[metrics].agg(['mean', 'std', 'min', 'max', 'count'])
        
        # Flatten the multi-index columns
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        
        return summary

# Example usage
if __name__ == "__main__":
    num_structures = 100

    # Replace with your actual file path
    analyzer = PotentialResultsAnalyzer("gen6_results_b1_warmup_all.json")
    
    # 1. Plot inference time vs number of atoms
    analyzer.plot_inference_time_vs_atoms(save_path="inference_time_vs_atoms.png")
    
    # 2. Get structures with highest energy errors
    top_energy_errors = analyzer.get_highest_error_structures('energy_error_per_atom', n=num_structures)
    print(f"Top {num_structures} structures with highest energy errors per atom:")
    print(top_energy_errors)
    
    # Get structures with highest force errors
    top_force_errors = analyzer.get_highest_error_structures('force_error', n=num_structures)
    print(f"\nTop {num_structures} structures with highest force errors:")
    print(top_force_errors)
    
    # 3. Plot histograms of errors by config type
    analyzer.plot_error_histograms_by_config('energy_error_per_atom', 
                                           save_path="energy_error_histograms.png",
                                           separate_configs=["default"])
    analyzer.plot_error_histograms_by_config('force_error',
                                           save_path="force_error_histograms.png",
                                           separate_configs=["default"])
    
    # Additional useful visualizations
    analyzer.plot_error_boxplots_by_config('energy_error_per_atom',
                                         save_path="energy_error_boxplots.png")

    analyzer.plot_error_boxplots_by_config('force_error',
                                         save_path="force_error_boxplots.png")
    
    # Get summary statistics by config type
    summary = analyzer.summarize_by_config_type()
    print("\nSummary statistics by configuration type:")
    print(summary)
