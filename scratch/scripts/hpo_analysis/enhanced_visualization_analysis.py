#!/usr/bin/env python3
"""
Enhanced visualization and analysis for Allegro HPO results.
Provides both interactive and static visualizations with integrated statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
import json
import re
from adjustText import adjust_text

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path("analysis_outputs/model_summary_analysis.csv")
OUTPUT_DIR = Path("analysis_outputs/enhanced_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# New: Production run constraints
TIMESTEP_THRESHOLD = 5.0  # timesteps/s
BENCHMARK_ATOMS = 8192  # atoms used for the benchmark throughput metric

# Column names
ERROR_COL = "mean_Force_Error"
SPEED_COL_KATOM = "mean_KAtomSteps_s"  # Renamed for clarity
TIMESTEPS_COL = "mean_Timesteps_s"      # NEW: Primary speed metric
MEMORY_COL = "mean_Memory_MiB"
ERROR_STD_COL = "std_Force_Error"
SPEED_STD_COL_KATOM = "std_KAtomSteps_s"  # Renamed for clarity
TIMESTEPS_STD_COL = "std_Timesteps_s"      # NEW
MEMORY_STD_COL = "std_Memory_MiB"
HPO_ID_COL = "hpo_id"
CONFIG_COL = "config_name"

# Significance level for statistical tests
ALPHA = 0.05

# Color scheme for publications
COLORS = {
    'pareto': '#FF4136',
    'dominated': '#0074D9',
    'selected': '#2ECC40',
    'reference': '#FF851B',
    'significant': '#B10DC9',
    'not_significant': '#AAAAAA'
}

def parse_hyperparameters(config_name):
    """Extract hyperparameters from config name."""
    pattern = r"num_layers-(\d+)_l_max-(\d+)_num_scalar_features-(\d+)_num_tensor_features-(\d+)_mlp_width-(\d+)"
    match = re.search(pattern, config_name)
    if match:
        return {
            'num_layers': int(match.group(1)),
            'l_max': int(match.group(2)),
            'num_scalar_features': int(match.group(3)),
            'num_tensor_features': int(match.group(4)),
            'mlp_width': int(match.group(5))
        }
    return None

def estimate_multinode_scaling(single_gpu_speed, num_atoms_single=8192, 
                             target_atoms=250000, num_gpus=128):
    """
    Estimate multi-node performance based on single GPU benchmarks.
    This is a rough approximation - actual scaling depends on many factors.
    """
    # Weak scaling assumption: each GPU handles similar atom count
    atoms_per_gpu = target_atoms / num_gpus
    
    # Estimate scaling factors
    # Communication overhead increases with node count
    comm_efficiency = 0.7  # Typical multi-node efficiency
    
    # Larger systems may have different computation patterns
    size_scaling = np.sqrt(num_atoms_single / atoms_per_gpu)
    
    # Rough estimate
    estimated_speed = single_gpu_speed * num_gpus * comm_efficiency * size_scaling
    
    return {
        'estimated_speed': estimated_speed,
        'atoms_per_gpu': atoms_per_gpu,
        'scaling_efficiency': comm_efficiency,
        'warning': 'This is a rough estimate. Actual performance depends on implementation details.'
    }

def perform_statistical_grouping(df, metric_col, metric_std_col=None):
    """
    Group models by statistical significance using pairwise t-tests.
    Returns group assignments where models in the same group are not significantly different.
    """
    n_models = len(df)
    p_matrix = np.ones((n_models, n_models))
    
    # Perform pairwise t-tests
    for i in range(n_models):
        for j in range(i+1, n_models):
            mean1, mean2 = df.iloc[i][metric_col], df.iloc[j][metric_col]
            
            if metric_std_col and metric_std_col in df.columns:
                std1, std2 = df.iloc[i][metric_std_col], df.iloc[j][metric_std_col]
                # Approximate t-test assuming sample sizes
                n_samples = 9  # From your data
                se = np.sqrt(std1**2/n_samples + std2**2/n_samples)
                if se > 0:
                    t_stat = abs(mean1 - mean2) / se
                    df_free = 2 * n_samples - 2
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_free))
                else:
                    p_val = 1.0
            else:
                # If no std provided, use non-parametric test
                p_val = 1.0  # Conservative assumption
            
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
    
    # Group models using clustering on significance
    # Models are connected if p > ALPHA (not significantly different)
    adjacency = (p_matrix > ALPHA).astype(int)
    
    # Use DBSCAN for clustering
    clustering = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')
    distance_matrix = 1 - adjacency
    groups = clustering.fit_predict(distance_matrix)
    
    return groups, p_matrix

def create_enhanced_pareto_plot(df, title_suffix="", filename_suffix="", y_axis_col=SPEED_COL_KATOM, y_axis_label="Throughput (k-atom-steps/s) →"):
    """Create publication-ready 2D Pareto plot with statistical significance coloring."""
    # Find Pareto optimal models
    pareto_mask = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                # Model j dominates model i if better in both metrics
                if (df.iloc[j][ERROR_COL] <= df.iloc[i][ERROR_COL] and 
                    df.iloc[j][y_axis_col] >= df.iloc[i][y_axis_col] and
                    (df.iloc[j][ERROR_COL] < df.iloc[i][ERROR_COL] or 
                     df.iloc[j][y_axis_col] > df.iloc[i][y_axis_col])):
                    pareto_mask[i] = False
                    break
    
    # Perform statistical grouping for error
    error_groups, _ = perform_statistical_grouping(df, ERROR_COL, ERROR_STD_COL)
    
    # Create static plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot dominated points
    dominated = df[~pareto_mask]
    scatter_dominated = ax.scatter(dominated[ERROR_COL], dominated[y_axis_col],
                                 c=error_groups[~pareto_mask], cmap='viridis',
                                 alpha=0.6, s=60, edgecolors='none',
                                 label='Dominated')
    
    # Plot Pareto optimal points
    pareto = df[pareto_mask]
    ax.scatter(pareto[ERROR_COL], pareto[y_axis_col],
              c=error_groups[pareto_mask], cmap='viridis',
              s=120, edgecolors='red', linewidth=2,
              label='Pareto Optimal')
    
    # Add Pareto frontier line
    pareto_sorted = pareto.sort_values(ERROR_COL)
    ax.plot(pareto_sorted[ERROR_COL], pareto_sorted[y_axis_col],
           'r--', alpha=0.5, linewidth=1)
    
    # Annotations for top models, now using adjust_text for clarity
    top_speed = df.nlargest(3, y_axis_col)
    top_accuracy = df.nsmallest(3, ERROR_COL)
    pareto_points_to_label = df[pareto_mask] # Label all pareto points
    
    points_to_label = pd.concat([top_speed, top_accuracy, pareto_points_to_label]).drop_duplicates()
    
    texts = []
    for idx, row in points_to_label.iterrows():
        texts.append(ax.text(row[ERROR_COL], row[y_axis_col], row[HPO_ID_COL], fontsize=8))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), ax=ax)
    
    ax.set_xlabel('Force Error (Å) →', fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    ax.set_title('Model Performance: Accuracy vs Speed Trade-off' + title_suffix, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar for statistical groups
    cbar = plt.colorbar(scatter_dominated, ax=ax)
    cbar.set_label('Statistical Significance Group', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'pareto_plot_with_significance{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return pareto_mask, error_groups

def create_interactive_pareto_plot(df, title_suffix="", filename_suffix="", y_axis_col=TIMESTEPS_COL, y_axis_label="Speed (timesteps/s) →"):
    """Create an interactive 2D Pareto plot with draggable labels, styled to match matplotlib."""

    # --- 1. Prepare data (filter for log scale if needed) ---
    df_plot = df.copy()
    if y_axis_col == 'mean_Timesteps_s':
        # Ensure y-values are positive for log scale
        df_plot = df_plot[df_plot[y_axis_col] > 0].copy()

    # --- 2. Find Pareto optimal models & Perform Statistical Grouping on the plotting data ---
    pareto_mask = np.ones(len(df_plot), dtype=bool)
    for i in range(len(df_plot)):
        for j in range(len(df_plot)):
            if i != j:
                if (df_plot.iloc[j][ERROR_COL] <= df_plot.iloc[i][ERROR_COL] and
                    df_plot.iloc[j][y_axis_col] >= df_plot.iloc[i][y_axis_col] and
                    (df_plot.iloc[j][ERROR_COL] < df_plot.iloc[i][ERROR_COL] or
                     df_plot.iloc[j][y_axis_col] > df_plot.iloc[i][y_axis_col])):
                    pareto_mask[i] = False
                    break

    error_groups, _ = perform_statistical_grouping(df_plot, ERROR_COL, ERROR_STD_COL)
    df_plot['error_group'] = error_groups

    # --- 3. Separate data for plotting ---
    pareto_df = df_plot[pareto_mask].sort_values(by=ERROR_COL)
    dominated_df = df_plot[~pareto_mask]

    # --- 4. Create the plot using Graph Objects for fine control ---
    fig = go.Figure()

    # Plot dominated points
    fig.add_trace(go.Scatter(
        x=dominated_df[ERROR_COL], y=dominated_df[y_axis_col],
        mode='markers',
        marker=dict(
            size=10,
            color=dominated_df['error_group'],
            colorscale='Viridis',
            showscale=False,  # Colorbar removed
            opacity=0.6
        ),
        text=[f"ID: {row[HPO_ID_COL]}" for _, row in dominated_df.iterrows()],
        hoverinfo='text',
        name='Dominated Models'
    ))

    # Plot Pareto optimal points
    fig.add_trace(go.Scatter(
        x=pareto_df[ERROR_COL], y=pareto_df[y_axis_col],
        mode='markers',
        marker=dict(
            size=16,
            color=pareto_df['error_group'],
            colorscale='Viridis',
            line=dict(width=4, color='red')
        ),
        text=[f"ID: {row[HPO_ID_COL]}" for _, row in pareto_df.iterrows()],
        hoverinfo='text',
        name='Pareto Optimal'
    ))
    
    # Plot the Pareto frontier line
    fig.add_trace(go.Scatter(
        x=pareto_df[ERROR_COL], y=pareto_df[y_axis_col],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Pareto Frontier'
    ))

    # --- 5. Add draggable annotations for key models ---
    top_speed = df_plot.nlargest(3, y_axis_col)
    top_accuracy = df_plot.nsmallest(3, ERROR_COL)
    points_to_label = pd.concat([top_speed, top_accuracy, pareto_df]).drop_duplicates()

    for _, row in points_to_label.iterrows():
        fig.add_annotation(
            x=row[ERROR_COL], y=row[y_axis_col],
            text=row[HPO_ID_COL],
            showarrow=True, arrowhead=1,
            standoff=20,  # Distance from point in pixels - works better with log scale
            font=dict(size=18)  # Increased data label font size
        )
        
    # --- 6. Final layout styling to match matplotlib ---
    fig.update_layout(
        title=dict(
            text='Interactive Pareto Plot: Accuracy vs Speed' + title_suffix,
            font=dict(size=26)  # Increased title font size
        ),
        xaxis_title='Force Error (Å) →',
        yaxis_title=y_axis_label,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.5)',
            font=dict(size=20)  # Increased legend font size
        ),
        template='plotly_white',  # Clean background
        showlegend=True
    )
    
    # Update axes font sizes
    fig.update_xaxes(
        titlefont=dict(size=26),  # Axis title font size
        tickfont=dict(size=18)    # Tick label font size
    )
    fig.update_yaxes(
        titlefont=dict(size=26),  # Axis title font size
        tickfont=dict(size=18)    # Tick label font size
    )
    
    # Log scale disabled - using linear scale for better label positioning
    # if y_axis_col == 'mean_Timesteps_s':
    #     # Constrain the y-axis to a reasonable range to avoid distortion from outliers.
    #     # The range is based on the production threshold and a sane upper limit.
    #     fig.update_yaxes(
    #         type="log",
    #         range=[np.log10(TIMESTEP_THRESHOLD), np.log10(1000)]
    #     )

    # Add config to enable editing and save
    config = {'editable': True, 'scrollZoom': True}
    fig.write_html(OUTPUT_DIR / f'interactive_pareto_plot{filename_suffix}.html', config=config)

def create_interactive_3d_visualization(df, title_suffix="", filename_suffix=""):
    """Create interactive 3D plot with error, speed, and memory."""
    # Parse hyperparameters
    hp_data = []
    for _, row in df.iterrows():
        hp = parse_hyperparameters(row[CONFIG_COL])
        if hp:
            hp_data.append(hp)
        else:
            hp_data.append({k: np.nan for k in ['num_layers', 'l_max', 
                          'num_scalar_features', 'num_tensor_features', 'mlp_width']})
    
    hp_df = pd.DataFrame(hp_data)
    df_combined = pd.concat([df, hp_df], axis=1)
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Find Pareto optimal in 3D (minimize error and memory, maximize speed)
    pareto_3d = find_pareto_optimal_3d(df, ERROR_COL, SPEED_COL_KATOM, MEMORY_COL)
    
    # Add all points
    fig.add_trace(go.Scatter3d(
        x=df[ERROR_COL],
        y=df[SPEED_COL_KATOM], 
        z=df[MEMORY_COL],
        mode='markers',
        marker=dict(
            size=8,
            color=df_combined['mlp_width'],
            colorscale='Viridis',
            showscale=False,  # Colorbar removed
            colorbar=dict(title="MLP Width"),
            line=dict(width=1, color='DarkSlateGray')
        ),
        text=[f"ID: {row[HPO_ID_COL]}<br>" +
              f"Error: {row[ERROR_COL]:.4f}<br>" +
              f"Speed: {row[SPEED_COL_KATOM]:.1f}<br>" +
              f"Memory: {row[MEMORY_COL]:.0f}<br>" +
              f"Layers: {row['num_layers']}<br>" +
              f"L_max: {row['l_max']}<br>" +
              f"Scalar feat: {row['num_scalar_features']}<br>" +
              f"Tensor feat: {row['num_tensor_features']}<br>" +
              f"MLP width: {row['mlp_width']}"
              for _, row in df_combined.iterrows()],
        name='All Models',
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Highlight Pareto optimal points
    fig.add_trace(go.Scatter3d(
        x=df.loc[pareto_3d][ERROR_COL],
        y=df.loc[pareto_3d][SPEED_COL_KATOM],
        z=df.loc[pareto_3d][MEMORY_COL],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        name='3D Pareto Optimal',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='3D Model Performance: Error vs Speed vs Memory' + title_suffix,
        scene=dict(
            xaxis_title='Force Error (Å)',
            yaxis_title='Throughput (k-atom-steps/s)',
            zaxis_title='Memory (MiB)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1000,
        height=800
    )
    
    fig.write_html(OUTPUT_DIR / f'interactive_3d_pareto{filename_suffix}.html')
    
    return fig

def find_pareto_optimal_3d(df, error_col, speed_col, memory_col):
    """Find Pareto optimal points in 3D (minimize error and memory, maximize speed)."""
    pareto_mask = np.ones(len(df), dtype=bool)
    
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                # Check if j dominates i
                if (df.iloc[j][error_col] <= df.iloc[i][error_col] and
                    df.iloc[j][speed_col] >= df.iloc[i][speed_col] and
                    df.iloc[j][memory_col] <= df.iloc[i][memory_col] and
                    (df.iloc[j][error_col] < df.iloc[i][error_col] or
                     df.iloc[j][speed_col] > df.iloc[i][speed_col] or
                     df.iloc[j][memory_col] < df.iloc[i][memory_col])):
                    pareto_mask[i] = False
                    break
    
    return pareto_mask

def create_speed_constrained_analysis(df, title_suffix="", filename_suffix=""):
    """Create plots showing best models for different speed requirements."""
    # Define speed thresholds (k-atom-steps/s)
    speed_thresholds = [50, 100, 150, 200, 250, 300]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, min_speed in enumerate(speed_thresholds):
        ax = axes[idx]
        
        # Filter models meeting speed requirement
        qualified = df[df[SPEED_COL_KATOM] >= min_speed].copy()
        
        if len(qualified) == 0:
            ax.text(0.5, 0.5, f'No models ≥ {min_speed} k-atom-steps/s',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Min Speed: {min_speed} k-atom-steps/s')
            continue
        
        # Sort by error
        qualified = qualified.sort_values(ERROR_COL)
        
        # Statistical analysis
        if len(qualified) > 1:
            best_model = qualified.iloc[0]
            # Test if other models are significantly worse
            p_values = []
            for i in range(1, min(10, len(qualified))):  # Top 10 models
                other_model = qualified.iloc[i]
                # Simple t-test approximation
                mean_diff = other_model[ERROR_COL] - best_model[ERROR_COL]
                se = np.sqrt(best_model[ERROR_STD_COL]**2 + other_model[ERROR_STD_COL]**2) / 3
                if se > 0:
                    t_stat = mean_diff / se
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=16))
                else:
                    p_val = 0
                p_values.append(p_val)
            
            # Plot top models
            top_n = min(10, len(qualified))
            y_pos = np.arange(top_n)
            colors = ['green' if i == 0 else 'red' if p_values[i-1] < ALPHA else 'gray' 
                     for i in range(top_n)]
            
            bars = ax.barh(y_pos, qualified[ERROR_COL].iloc[:top_n], color=colors, alpha=0.7)
            
            # Add error bars
            ax.errorbar(qualified[ERROR_COL].iloc[:top_n], y_pos,
                       xerr=qualified[ERROR_STD_COL].iloc[:top_n],
                       fmt='none', color='black', capsize=3)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([qualified[HPO_ID_COL].iloc[i] for i in range(top_n)], fontsize=8)
            ax.set_xlabel('Force Error (Å)')
            ax.set_title(f'Min Speed: {min_speed} k-atom-steps/s\n({len(qualified)} models qualify)')
            ax.grid(axis='x', alpha=0.3)
            
            # Add significance indicators
            for i in range(1, top_n):
                if p_values[i-1] < ALPHA:
                    ax.text(qualified[ERROR_COL].iloc[i] + 0.001, i, '*', 
                           fontsize=12, va='center')
    
    plt.suptitle('Best Models for Different Speed Requirements' + title_suffix +
                '\n(* indicates significantly worse than best model)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'speed_constrained_analysis{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hyperparameter_analysis(df, title_suffix="", filename_suffix=""):
    """Analyze how hyperparameters affect performance."""
    # Parse hyperparameters
    hp_data = []
    for _, row in df.iterrows():
        hp = parse_hyperparameters(row[CONFIG_COL])
        if hp:
            hp['error'] = row[ERROR_COL]
            hp['speed'] = row[SPEED_COL_KATOM]
            hp['memory'] = row[MEMORY_COL]
            hp_data.append(hp)
    
    hp_df = pd.DataFrame(hp_data)
    
    # Create subplots for each hyperparameter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    hp_names = ['num_layers', 'l_max', 'num_scalar_features', 
                'num_tensor_features', 'mlp_width']
    
    for idx, hp_name in enumerate(hp_names):
        ax = axes[idx]
        
        # Group by hyperparameter value
        grouped = hp_df.groupby(hp_name).agg({
            'error': ['mean', 'std', 'count'],
            'speed': ['mean', 'std']
        }).reset_index()
        
        # Create grouped bar plot
        x = np.arange(len(grouped))
        width = 0.35
        
        ax2 = ax.twinx()
        
        # Error bars
        bars1 = ax.bar(x - width/2, grouped['error']['mean'], width, 
                       yerr=grouped['error']['std'], capsize=5,
                       label='Error', color='steelblue', alpha=0.7)
        
        # Speed bars  
        bars2 = ax2.bar(x + width/2, grouped['speed']['mean'], width,
                        yerr=grouped['speed']['std'], capsize=5,
                        label='Speed', color='darkorange', alpha=0.7)
        
        ax.set_xlabel(hp_name.replace('_', ' ').title())
        ax.set_ylabel('Force Error (Å)', color='steelblue')
        ax2.set_ylabel('Throughput (k-atom-steps/s)', color='darkorange')
        
        ax.set_xticks(x)
        ax.set_xticklabels(grouped[hp_name])
        
        # Add sample sizes
        for i, count in enumerate(grouped['error']['count']):
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                   ha='center', va='top', fontsize=8)
        
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        
        ax.set_title(f'Effect of {hp_name.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
    
    # Remove last subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Hyperparameter Effects on Performance' + title_suffix, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'hyperparameter_analysis{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_scaling_estimation_tool(df, title_suffix="", filename_suffix=""):
    """Create interactive tool for estimating multi-node performance."""
    # Create interactive plot with scaling estimations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Single GPU Performance', 'Estimated 16-node Performance',
                       'Estimated 32-node Performance', 'Scaling Efficiency'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Single GPU performance
    fig.add_trace(
        go.Scatter(x=df[ERROR_COL], y=df[SPEED_COL_KATOM],
                  mode='markers', name='Single GPU',
                  marker=dict(size=8, color='blue'),
                  text=df[HPO_ID_COL],
                  hovertemplate='%{text}<br>Error: %{x:.4f}<br>Speed: %{y:.1f}'),
        row=1, col=1
    )
    
    # Estimate multi-node performance
    for config, target in [(16*8, 250000), (32*8, 600000)]:
        row, col = (1, 2) if config == 128 else (2, 1)
        
        estimated_speeds = []
        for _, model in df.iterrows():
            est = estimate_multinode_scaling(
                model[SPEED_COL_KATOM], 
                target_atoms=target,
                num_gpus=config
            )
            estimated_speeds.append(est['estimated_speed'])
        
        fig.add_trace(
            go.Scatter(x=df[ERROR_COL], y=estimated_speeds,
                      mode='markers', 
                      name=f'{config} GPUs ({target//1000}k atoms)',
                      marker=dict(size=8),
                      text=df[HPO_ID_COL],
                      hovertemplate='%{text}<br>Error: %{x:.4f}<br>Est. Speed: %{y:.1f}'),
            row=row, col=col
        )
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Force Error (Å)", row=row, col=col)
            fig.update_yaxes(title_text="Throughput (k-atom-steps/s)", row=row, col=col)
    
    fig.update_layout(height=800, showlegend=True,
                     title_text="Multi-node Performance Estimation (Rough Approximation)" + title_suffix)
    
    fig.write_html(OUTPUT_DIR / f'scaling_estimation_tool{filename_suffix}.html')
    
    # Also create a summary table
    scaling_summary = []
    for _, model in df.iterrows():
        row_data = {'hpo_id': model[HPO_ID_COL], 
                   'error': model[ERROR_COL],
                   'single_gpu_speed': model[SPEED_COL_KATOM]}
        
        for config, target in [(128, 250000), (256, 600000)]:
            est = estimate_multinode_scaling(
                model[SPEED_COL_KATOM],
                target_atoms=target,
                num_gpus=config
            )
            row_data[f'est_speed_{config}gpu'] = est['estimated_speed']
            row_data[f'est_timesteps_s_{config}gpu'] = est['estimated_speed'] * 1000 / target
        
        scaling_summary.append(row_data)
    
    scaling_df = pd.DataFrame(scaling_summary)
    scaling_df.to_csv(OUTPUT_DIR / f'estimated_scaling_performance{filename_suffix}.csv', index=False)
    
    return scaling_df

def create_recommendation_report(df, scaling_df, is_filtered=False):
    """Create a summary report with recommendations."""
    report = []
    report.append("# Allegro HPO Analysis Report\n")
    if is_filtered:
        report.append(f"## (Filtered for Production: >= {TIMESTEP_THRESHOLD} timesteps/s on {BENCHMARK_ATOMS} atoms)\n")
    
    report.append("## Executive Summary\n")
    
    # Find best models for different scenarios
    scenarios = [
        ("Best Accuracy (regardless of speed)", df.nsmallest(1, ERROR_COL)),
        ("Best Speed (regardless of accuracy)", df.nlargest(1, TIMESTEPS_COL)),
        ("Best Accuracy with Speed > 100 k-atom-steps/s", 
         df[df[SPEED_COL_KATOM] > 100].nsmallest(1, ERROR_COL)),
        ("Best Memory Efficiency with Good Accuracy",
         df[df[ERROR_COL] < df[ERROR_COL].quantile(0.25)].nsmallest(1, MEMORY_COL))
    ]
    
    report.append("### Top Model Recommendations\n")
    for scenario, best in scenarios:
        if len(best) > 0:
            model = best.iloc[0]
            report.append(f"\n**{scenario}:**")
            report.append(f"- Model: {model[HPO_ID_COL]}")
            report.append(f"- Force Error: {model[ERROR_COL]:.4f} ± {model[ERROR_STD_COL]:.4f} Å")
            report.append(f"- Speed: {model[SPEED_COL_KATOM]:.1f} ± {model[SPEED_STD_COL_KATOM]:.1f} k-atom-steps/s")
            report.append(f"- Timesteps/s: {model[TIMESTEPS_COL]:.2f}")
            report.append(f"- Memory: {model[MEMORY_COL]:.0f} MiB")
            
            # Add scaling estimates
            scaling_row = scaling_df[scaling_df['hpo_id'] == model[HPO_ID_COL]].iloc[0]
            report.append(f"- Estimated 16-node speed: {scaling_row['est_speed_128gpu']:.1f} k-atom-steps/s")
            report.append(f"- Estimated 32-node speed: {scaling_row['est_speed_256gpu']:.1f} k-atom-steps/s")
    
    report.append("\n## Statistical Analysis\n")
    report.append("- Models have been grouped by statistical significance (α=0.05)")
    report.append("- Error bars represent standard deviation across runs")
    report.append("- Multi-node estimates are rough approximations - actual benchmarking recommended")
    
    report.append("\n## Recommendations for Production\n")
    report.append("1. **Benchmark top candidates on actual multi-node setup**")
    report.append("   - Single GPU results don't capture communication overhead")
    report.append("   - Scaling behavior varies by model architecture")
    report.append("\n2. **Consider these factors:**")
    report.append("   - Larger models (more parameters) may have worse multi-node scaling")
    report.append("   - Memory usage becomes critical at scale")
    report.append("   - Some models may be unstable at larger scales")
    report.append("\n3. **Suggested validation approach:**")
    report.append("   - Test top 3-5 models from each category")
    report.append("   - Run short simulations (1000 steps) to measure actual throughput")
    report.append("   - Verify force accuracy on your specific systems")
    
    # Save report
    filename = "recommendation_report_filtered.md" if is_filtered else "recommendation_report.md"
    with open(OUTPUT_DIR / filename, 'w') as f:
        f.write('\n'.join(report))
    
    return report

def main():
    """Run all analyses and create visualizations."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # --- New: Filter for models benchmarked at the correct atom count ---
    print(f"\nFiltering for models benchmarked at {BENCHMARK_ATOMS} atoms.")
    original_count = len(df)
    if 'benchmark_atoms' in df.columns:
        # Drop rows where the benchmark atom count is not what we expect
        df.dropna(subset=['benchmark_atoms'], inplace=True)
        df = df[df['benchmark_atoms'] == BENCHMARK_ATOMS].copy()
        print(f"Retained {len(df)} out of {original_count} models after filtering for atom count.")
        if len(df) == 0:
            print(f"Error: No models found that were benchmarked at {BENCHMARK_ATOMS} atoms. Exiting.")
            return
    else:
        print("Warning: 'benchmark_atoms' column not found. Assuming all models were benchmarked at the correct size.")

    # Apply production throughput constraint
    print(f"\nApplying production throughput constraint: >= {TIMESTEP_THRESHOLD} timesteps/s")
    
    # Filter based on the pre-calculated mean_Timesteps_s column
    original_model_count = len(df)
    # Ensure the column exists and handle potential NaN values from models that weren't benchmarked
    if 'mean_Timesteps_s' not in df.columns:
        print(f"Error: Required column 'mean_Timesteps_s' not found in the data. Exiting.")
        return
        
    df.dropna(subset=[TIMESTEPS_COL], inplace=True)
    
    # Simplified filtering logic
    MAX_TIMESTEPS = 1000  # Set a reasonable upper limit for timesteps/s
    count_before_filter = len(df)

    # Apply both lower and upper bound filters at once
    df_filtered = df[
        (df[TIMESTEPS_COL] >= TIMESTEP_THRESHOLD) & 
        (df[TIMESTEPS_COL] <= MAX_TIMESTEPS)
    ].copy()
    
    filtered_out_count = count_before_filter - len(df_filtered)
    
    print(f"Filtered out {filtered_out_count} models that were outside the range [{TIMESTEP_THRESHOLD:.1f}, {MAX_TIMESTEPS}] timesteps/s.")
    print(f"{len(df_filtered)} models will be analyzed.")

    if df_filtered.empty:
        print("\nNo models meet the production requirement. Exiting analysis.")
        return
        
    title_suffix = f"\n(Models with ≥ {TIMESTEP_THRESHOLD} timesteps/s on {BENCHMARK_ATOMS} atoms)"
    
    # --- New logic for Pareto plot ---
    y_axis_col_pareto = 'mean_Timesteps_s'
    y_axis_label_pareto = 'Throughput (timesteps/s) →'
    filename_suffix_pareto = "_production_filtered_timesteps"

    print("\nCreating enhanced Pareto plot...")
    create_enhanced_pareto_plot(df_filtered, title_suffix, filename_suffix_pareto, y_axis_col_pareto, y_axis_label_pareto)
    
    print("Creating interactive Pareto plot with draggable labels...")
    create_interactive_pareto_plot(df_filtered, title_suffix, filename_suffix_pareto, y_axis_col_pareto, y_axis_label_pareto)
    
    # --- Other plots use standard filenames ---
    filename_suffix = "_production_filtered"
    
    print("Creating interactive 3D visualization...")
    create_interactive_3d_visualization(df_filtered, title_suffix, filename_suffix)
    
    print("Creating speed-constrained analysis...")
    create_speed_constrained_analysis(df_filtered, title_suffix, filename_suffix)
    
    print("Creating hyperparameter analysis...")
    create_hyperparameter_analysis(df_filtered, title_suffix, filename_suffix)
    
    print("Creating scaling estimation tool...")
    scaling_df = create_scaling_estimation_tool(df_filtered, title_suffix, filename_suffix)
    
    print("Creating recommendation report...")
    report = create_recommendation_report(df_filtered, scaling_df, is_filtered=True)
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")
    print("\nKey outputs (filtered for production):")
    print(f"- pareto_plot_with_significance{filename_suffix_pareto}.png (Static)")
    print(f"- interactive_pareto_plot{filename_suffix_pareto}.html (Interactive with draggable labels)")
    print("- interactive_3d_pareto_production_filtered.html")
    print("- speed_constrained_analysis_production_filtered.png")
    print("- hyperparameter_analysis_production_filtered.png")
    print("- scaling_estimation_tool_production_filtered.html")
    print("- recommendation_report_filtered.md")

if __name__ == "__main__":
    main() 