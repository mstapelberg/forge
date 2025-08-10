"""Analyze and visualize Warren–Cowley (WC) sweep results.

This script loads WC sweep CSV series for multiple supercell sizes, produces
several plots (metric evolution, final metric vs. size, normalized analyses),
and prints concise recommendations. It also generates a stepwise-normalized
convergence plot where each supercell's metric at every step is normalized to
the largest supercell's metric at the same step.
"""

import pandas as pd 
import matplotlib.pyplot as plt
import glob 
import os
import numpy as np
import seaborn as sns


def plot_stepwise_normalized(df: pd.DataFrame, plot_dir: str) -> str:
    """Plot stepwise-normalized WC metrics for each supercell size.

    Each supercell's metric at a given step is normalized by the metric of the
    largest supercell at the same step, i.e. normalized(step, n) =
    metric(step, n) / metric(step, n_max). This highlights the convergence of
    smaller supercells toward the behavior of the largest one across steps.

    Args:
        df (pd.DataFrame): Long-form DataFrame with columns `step`, `metric`,
            and `supercell_size` containing all series.
        plot_dir (str): Directory where the plot image will be saved.

    Returns:
        str: Absolute path to the saved PNG figure.

    Raises:
        ValueError: If the largest supercell reference series is missing or
            if there are no overlapping steps to normalize against.

    Examples:
        >>> # Assuming `df` is prepared with required columns
        >>> path = plot_stepwise_normalized(df, "./plots")
        >>> os.path.exists(path)
        True
    """

    # Identify largest supercell and build reference series by step
    unique_sizes = sorted(df["supercell_size"].unique())
    if not unique_sizes:
        raise ValueError("No supercell sizes found in DataFrame.")
    max_size = max(unique_sizes)

    ref = (
        df[df["supercell_size"] == max_size][["step", "metric"]]
        .drop_duplicates(subset=["step"]).rename(columns={"metric": "ref_metric"})
    )
    if ref.empty:
        raise ValueError("Reference series (largest supercell) is empty.")

    # Prepare seaborn viridis colors
    num_sizes = len(unique_sizes)
    colors = sns.color_palette("viridis", n_colors=num_sizes)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect all normalized values to compute dynamic y-limits
    all_norm_values: list[float] = []

    for idx, size in enumerate(unique_sizes):
        series = df[df["supercell_size"] == size][["step", "metric"]]
        merged = series.merge(ref, on="step", how="inner")
        if merged.empty:
            # Skip sizes without overlapping steps
            continue
        # Avoid division by zero; filter steps with ref_metric == 0
        merged = merged[merged["ref_metric"] != 0]
        if merged.empty:
            continue
        merged["normalized_metric"] = merged["metric"] / merged["ref_metric"]

        all_norm_values.extend(merged["normalized_metric"].tolist())

        label = f"{size}×{size}×{size} supercell"
        ax.plot(
            merged["step"],
            merged["normalized_metric"],
            color=colors[idx],
            linewidth=2,
            alpha=0.9,
            label=label,
        )

    # Reference line at 1.0
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Reference = largest")

    ax.set_xlabel("Step", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Metric vs largest (per-step)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Stepwise Normalized WC Metric by Supercell Size",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Dynamic y-limits with small padding; clamp lower bound at 0 if values are non-negative
    if all_norm_values:
        y_min = min(all_norm_values)
        y_max = max(all_norm_values)
        if y_min == y_max:
            # Degenerate case; expand slightly around the constant value
            pad = 0.05 * (1.0 if y_max == 0 else abs(y_max))
            ax.set_ylim(y_max - pad, y_max + pad)
        else:
            pad = 0.05 * (y_max - y_min)
            lower = max(0.0, y_min - pad) if y_min >= 0 else y_min - pad
            upper = y_max + pad
            ax.set_ylim(lower, upper)

    # Legend outside the plot for clarity
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11, frameon=True)

    plt.tight_layout()
    output_path = os.path.join(plot_dir, "wc_sweep_stepwise_normalized.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Stepwise normalized plot saved to: {output_path}")
    return output_path



# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

data_path = '../../data/wc_sweep_out/'

# Load the data from all the csvs 
csv_files = glob.glob(os.path.join(data_path, 'wc_series_n*.csv'))
csv_files.sort()  # Sort to ensure consistent ordering

# Create a comprehensive dataframe with supercell size information
df_list = []
for file in csv_files:
    # Extract supercell size from filename
    filename = os.path.basename(file)
    if 'wc_series_n' in filename:
        # Extract the number after 'n' and before '.csv'
        n_size = int(filename.replace('wc_series_n', '').replace('.csv', ''))
        
        # Read the CSV
        temp_df = pd.read_csv(file)
        temp_df['supercell_size'] = n_size
        temp_df['supercell_dim'] = f"{n_size}×{n_size}×{n_size}"
        df_list.append(temp_df)

# Combine all dataframes
df = pd.concat(df_list, ignore_index=True)

# Create output directory for plots
plot_dir = '../../data/wc_sweep_plots'
os.makedirs(plot_dir, exist_ok=True)

# Create the main analysis plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create a color map for the different supercell sizes
n_sizes = len(df['supercell_size'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, n_sizes))

# Plot each supercell size separately
for i, (size, group) in enumerate(df.groupby('supercell_size')):
    color = colors[i]
    label = f"{size}×{size}×{size} supercell"
    
    # Plot the metric vs step
    ax.plot(group['step'], group['metric'], 
            color=color, linewidth=2, marker='o', markersize=4,
            label=label, alpha=0.8)

# Customize the plot
ax.set_xlabel('Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
ax.set_title('WC Sweep Analysis: Metric Evolution Across Different Supercell Sizes', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Customize legend
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
          fontsize=11, frameon=True, fancybox=True, shadow=True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the main analysis plot
output_path = os.path.join(plot_dir, 'wc_sweep_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Main analysis plot saved to: {output_path}")

# Display the plot
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("WC SWEEP ANALYSIS SUMMARY")
print("="*60)

for size, group in df.groupby('supercell_size'):
    print(f"\n{size}×{size}×{size} Supercell:")
    print(f"  Total steps: {len(group)}")
    print(f"  Final metric value: {group['metric'].iloc[-1]:.6f}")
    print(f"  Metric range: {group['metric'].min():.6f} - {group['metric'].max():.6f}")
    print(f"  Mean metric: {group['metric'].mean():.6f}")
    print(f"  Std metric: {group['metric'].std():.6f}")

# Additional analysis: correlation between supercell size and final metric
final_metrics = df.groupby('supercell_size')['metric'].last().reset_index()
correlation = final_metrics['supercell_size'].corr(final_metrics['metric'])
print(f"\nCorrelation between supercell size and final metric: {correlation:.4f}")

# Create a second plot showing final metric vs supercell size
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(final_metrics['supercell_size'], final_metrics['metric'], 
          marker='s', markersize=8, linewidth=2, color='red')
ax2.set_xlabel('Supercell Size (n×n×n)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Final Metric Value', fontsize=14, fontweight='bold')
ax2.set_title('Final Metric Value vs Supercell Size', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# Save the final metrics plot
output_path2 = os.path.join(plot_dir, 'wc_sweep_final_metrics.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Final metrics plot saved to: {output_path2}")

plt.show()

# Create normalized metrics analysis
print("\n" + "="*60)
print("NORMALIZED METRIC ANALYSIS")
print("="*60)

# Get the largest supercell size and its metric
max_size = final_metrics['supercell_size'].max()
max_metric = final_metrics[final_metrics['supercell_size'] == max_size]['metric'].iloc[0]

print(f"Largest supercell size: {max_size}×{max_size}×{max_size} ({max_size**3} atoms)")
print(f"Reference metric value: {max_metric:.6f}")

# Calculate normalized metrics (relative to largest supercell)
final_metrics['normalized_metric'] = final_metrics['metric'] / max_metric
final_metrics['atoms'] = final_metrics['supercell_size']**3
final_metrics['relative_error'] = abs(1 - final_metrics['normalized_metric']) * 100

# Sort by supercell size for better visualization
final_metrics_sorted = final_metrics.sort_values('supercell_size')

# Create separate plots for better readability
print("\n" + "="*60)
print("CREATING SEPARATE NORMALIZED METRICS PLOTS")
print("="*60)

# Plot 1: Normalized metric vs supercell size (separate)
fig3a, ax3a = plt.subplots(figsize=(10, 6))
ax3a.plot(final_metrics_sorted['supercell_size'], final_metrics_sorted['normalized_metric'], 
           marker='o', markersize=8, linewidth=2, color='blue')
ax3a.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Reference (largest supercell)')
ax3a.axhline(y=0.95, color='orange', linestyle=':', alpha=0.7, label='95% threshold')
ax3a.axhline(y=0.90, color='green', linestyle=':', alpha=0.7, label='90% threshold')
ax3a.axhline(y=1.05, color='orange', linestyle=':', alpha=0.7, label='105% threshold')
ax3a.axhline(y=1.10, color='green', linestyle=':', alpha=0.7, label='110% threshold')

ax3a.set_xlabel('Supercell Size (n×n×n)', fontsize=14, fontweight='bold')
ax3a.set_ylabel('Normalized Metric (relative to largest)', fontsize=14, fontweight='bold')
ax3a.set_title('Normalized Metric vs Supercell Size', fontsize=16, fontweight='bold')
ax3a.grid(True, alpha=0.3, linestyle='--')
ax3a.legend(fontsize=11)
# Dynamic y-limits based on data
_y = final_metrics_sorted['normalized_metric'].to_numpy()
if _y.size:
    _ymin, _ymax = float(np.min(_y)), float(np.max(_y))
    if _ymin == _ymax:
        _pad = 0.05 * (1.0 if _ymax == 0 else abs(_ymax))
        ax3a.set_ylim(_ymax - _pad, _ymax + _pad)
    else:
        _pad = 0.05 * (_ymax - _ymin)
        _lower = max(0.0, _ymin - _pad) if _ymin >= 0 else _ymin - _pad
        _upper = _ymax + _pad
        ax3a.set_ylim(_lower, _upper)

# Save the first normalized plot
output_path3a = os.path.join(plot_dir, 'wc_sweep_normalized_metric.png')
plt.savefig(output_path3a, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Normalized metric plot saved to: {output_path3a}")
plt.show()

# Plot 2: Relative error percentage vs supercell size (separate)
fig3b, ax3b = plt.subplots(figsize=(10, 6))
ax3b.plot(final_metrics_sorted['supercell_size'], final_metrics_sorted['relative_error'], 
           marker='s', markersize=8, linewidth=2, color='purple')
ax3b.axhline(y=5.0, color='orange', linestyle=':', alpha=0.7, label='5% error threshold')
ax3b.axhline(y=10.0, color='green', linestyle=':', alpha=0.7, label='10% error threshold')

ax3b.set_xlabel('Supercell Size (n×n×n)', fontsize=14, fontweight='bold')
ax3b.set_ylabel('Relative Error (%)', fontsize=14, fontweight='bold')
ax3b.set_title('Relative Error vs Supercell Size', fontsize=16, fontweight='bold')
ax3b.grid(True, alpha=0.3, linestyle='--')
ax3b.legend(fontsize=11)

# Save the second normalized plot
output_path3b = os.path.join(plot_dir, 'wc_sweep_relative_error.png')
plt.savefig(output_path3b, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Relative error plot saved to: {output_path3b}")
plt.show()

print(f"\nAll plots saved to: {plot_dir}")

# Create stepwise normalized convergence plot (per-step normalization to largest supercell)
try:
    stepwise_path = plot_stepwise_normalized(df, plot_dir)
except ValueError as err:
    print(f"Skipping stepwise normalized plot: {err}")

# Print analysis results
print("\n" + "="*60)
print("OPTIMAL SUPERCELL SIZE RECOMMENDATIONS")
print("="*60)

# Find supercells that meet different symmetric accuracy thresholds around 1.0
accuracy_thresholds = [0.95, 0.90, 0.85]
for acc in accuracy_thresholds:
    tol_percent = (1.0 - acc) * 100.0  # e.g., 95% accuracy => 5% tolerance
    # Smallest supercell whose relative error is within tolerance
    candidates = final_metrics_sorted[final_metrics_sorted['relative_error'] <= tol_percent]
    print(f"\nWithin {tol_percent:.0f}% of largest (±{tol_percent:.0f}% band around 1.0):")
    if not candidates.empty:
        smallest = candidates.iloc[0]
        print(f"  Smallest supercell: {smallest['supercell_size']}×{smallest['supercell_size']}×{smallest['supercell_size']}")
        print(f"  Atoms: {smallest['atoms']}")
        print(f"  Normalized metric: {smallest['normalized_metric']:.4f}")
        print(f"  Relative error: {smallest['relative_error']:.2f}%")
        print(f"  Computational savings: {((max_size**3 - smallest['atoms']) / max_size**3 * 100):.1f}% fewer atoms")
    else:
        print("  No supercell meets this tolerance.")

# Show detailed comparison table
print("\n" + "="*80)
print("DETAILED COMPARISON TABLE")
print("="*80)
print(f"{'Size':<8} {'Atoms':<8} {'Metric':<12} {'Norm.':<8} {'Error%':<8} {'Savings%':<10}")
print("-" * 80)

for _, row in final_metrics_sorted.iterrows():
    savings = ((max_size**3 - row['atoms']) / max_size**3 * 100) if row['atoms'] < max_size**3 else 0
    print(f"{row['supercell_size']:<8} {row['atoms']:<8} {row['metric']:<12.6f} {row['normalized_metric']:<8.4f} {row['relative_error']:<8.2f} {savings:<10.1f}")
