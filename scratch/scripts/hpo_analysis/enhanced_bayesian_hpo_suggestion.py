#!/usr/bin/env python3
"""
Enhanced Bayesian HPO suggestion with statistical significance analysis.
Suggests new hyperparameters while considering which models are statistically different.
"""

import pandas as pd
import numpy as np
import re
from scipy import stats
from skopt import Optimizer
from skopt.space import Integer, Categorical
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
CSV_PATH = "analysis_outputs/model_summary_analysis.csv"
OUTPUT_DIR = Path("analysis_outputs/enhanced_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# New: Production run constraints
TIMESTEP_THRESHOLD = 5.0  # timesteps/s
BENCHMARK_ATOMS = 8192  # atoms used for the benchmark throughput metric

# Column names
HPO_ID_COL = "hpo_id"
CONFIG_COL = "config_name"
ERROR_COL = "mean_Force_Error"
ERROR_STD_COL = "std_Force_Error"
SPEED_COL = "mean_KAtomSteps_s"
SPEED_STD_COL = "std_KAtomSteps_s"
TIMESTEPS_COL = "mean_Timesteps_s"  # NEW
MEMORY_COL = "mean_Memory_MiB"

# Bayesian optimization settings
N_SUGGESTIONS = 10
ALPHA = 0.05  # Statistical significance level

# Define the hyperparameter search space
param_space = [
    Integer(1, 2, name='num_layers'),
    Integer(1, 3, name='l_max'),
    Categorical([32, 64, 128], name='num_scalar_features'),
    Categorical([16, 32, 64], name='num_tensor_features'),
    Categorical([128, 256, 512], name='mlp_width')
]
param_names = [s.name for s in param_space]

# Weights for multi-objective optimization
WEIGHT_ERROR = 2.0
WEIGHT_SPEED = 1.0
WEIGHT_MEMORY = 0.5

# Pattern for parsing hyperparameters
hpo_pattern = re.compile(
    r"num_layers-(\d+)_l_max-(\d+)_num_scalar_features-(\d+)_num_tensor_features-(\d+)_mlp_width-(\d+)"
)

def parse_hpo_config(config_str):
    """Parse hyperparameter values from config string."""
    match = hpo_pattern.search(config_str)
    if match:
        return [
            int(match.group(1)),  # num_layers
            int(match.group(2)),  # l_max
            int(match.group(3)),  # num_scalar_features
            int(match.group(4)),  # num_tensor_features
            int(match.group(5))   # mlp_width
        ]
    return None

def calculate_objective_score(error, speed, memory, timesteps_s):
    """
    Calculate objective score for optimization.
    Lower is better (optimizer minimizes).
    """
    # New: Heavily penalize models that don't meet the production throughput requirement
    if timesteps_s < TIMESTEP_THRESHOLD:
        return 1e6  # Return a large penalty

    # Normalize metrics to similar scales
    error_normalized = error / 0.2  # Typical error range 0.15-0.25
    speed_normalized = speed / 100   # Typical speed range 0-400
    memory_normalized = memory / 10000  # Typical memory range 5000-20000
    
    # Minimize error and memory, maximize speed
    score = (WEIGHT_ERROR * error_normalized + 
             WEIGHT_MEMORY * memory_normalized - 
             WEIGHT_SPEED * speed_normalized)
    
    return score

def identify_statistically_equivalent_models(df):
    """
    Identify groups of models that are statistically equivalent in performance.
    Returns a dictionary mapping each model to its equivalence group.
    """
    n_models = len(df)
    
    # Calculate pairwise p-values for error metric
    p_matrix = np.ones((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Approximate t-test for error
            mean1, std1 = df.iloc[i][ERROR_COL], df.iloc[i][ERROR_STD_COL]
            mean2, std2 = df.iloc[j][ERROR_COL], df.iloc[j][ERROR_STD_COL]
            
            n_samples = 9  # From the data
            se = np.sqrt(std1**2/n_samples + std2**2/n_samples)
            
            if se > 0:
                t_stat = abs(mean1 - mean2) / se
                df_free = 2 * n_samples - 2
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_free))
            else:
                p_val = 1.0
            
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
    
    # Group models using connected components
    # Models are in the same group if p > ALPHA
    groups = {}
    group_id = 0
    assigned = set()
    
    for i in range(n_models):
        if i not in assigned:
            # Start new group
            group_members = [i]
            assigned.add(i)
            
            # Find all models statistically equivalent to this one
            queue = [i]
            while queue:
                current = queue.pop(0)
                for j in range(n_models):
                    if j not in assigned and p_matrix[current, j] > ALPHA:
                        group_members.append(j)
                        assigned.add(j)
                        queue.append(j)
            
            # Assign group
            for member in group_members:
                groups[df.iloc[member][HPO_ID_COL]] = group_id
            group_id += 1
    
    return groups, p_matrix

def analyze_hyperparameter_coverage(df, param_space):
    """
    Analyze which regions of hyperparameter space have been explored.
    Returns suggestions for underexplored regions.
    """
    # Parse all configurations
    explored_configs = []
    for _, row in df.iterrows():
        config = parse_hpo_config(row[CONFIG_COL])
        if config:
            explored_configs.append(config)
    
    explored_configs = np.array(explored_configs)
    
    # Create coverage heatmap
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    param_pairs = [
        ('num_layers', 'l_max'),
        ('num_scalar_features', 'num_tensor_features'),
        ('num_scalar_features', 'mlp_width'),
        ('num_tensor_features', 'mlp_width'),
        ('l_max', 'mlp_width'),
        ('num_layers', 'mlp_width')
    ]
    
    for idx, (param1, param2) in enumerate(param_pairs):
        ax = axes[idx]
        
        # Get parameter indices
        p1_idx = param_names.index(param1)
        p2_idx = param_names.index(param2)
        
        # Create 2D histogram
        p1_values = explored_configs[:, p1_idx]
        p2_values = explored_configs[:, p2_idx]
        
        # Get unique values for each parameter
        p1_unique = sorted(set(p1_values))
        p2_unique = sorted(set(p2_values))
        
        # Create count matrix
        count_matrix = np.zeros((len(p2_unique), len(p1_unique)), dtype=int)
        for p1, p2 in zip(p1_values, p2_values):
            i = p2_unique.index(p2)
            j = p1_unique.index(p1)
            count_matrix[i, j] += 1
        
        # Plot heatmap
        sns.heatmap(count_matrix, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=p1_unique, yticklabels=p2_unique,
                   ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel(param1.replace('_', ' ').title())
        ax.set_ylabel(param2.replace('_', ' ').title())
        ax.set_title(f'{param1} vs {param2}')
    
    plt.suptitle('Hyperparameter Space Coverage', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hyperparameter_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find underexplored regions
    all_possible = []
    # Get parameter ranges/categories
    p1_vals = range(param_space[0].low, param_space[0].high + 1)  # num_layers: 1-2
    p2_vals = range(param_space[1].low, param_space[1].high + 1)  # l_max: 1-3
    p3_vals = param_space[2].categories  # num_scalar_features
    p4_vals = param_space[3].categories  # num_tensor_features
    p5_vals = param_space[4].categories  # mlp_width
    
    for p1 in p1_vals:
        for p2 in p2_vals:
            for p3 in p3_vals:
                for p4 in p4_vals:
                    for p5 in p5_vals:
                        all_possible.append([p1, p2, p3, p4, p5])
    
    explored_set = set(map(tuple, explored_configs))
    all_possible_set = set(map(tuple, all_possible))
    unexplored = all_possible_set - explored_set
    
    return unexplored

def generate_suggestions_with_significance(df, n_suggestions=10):
    """
    Generate new hyperparameter suggestions considering statistical significance.
    """
    # Load and process existing data
    X_observed = []
    y_observed = []
    
    for _, row in df.iterrows():
        config = parse_hpo_config(row[CONFIG_COL])
        if config:
            X_observed.append(config)
            
            # Use the pre-calculated timesteps/s value
            timesteps_s = row[TIMESTEPS_COL] if pd.notna(row[TIMESTEPS_COL]) else 0

            score = calculate_objective_score(
                row[ERROR_COL], 
                row[SPEED_COL],
                row[MEMORY_COL],
                timesteps_s  # Pass new argument
            )
            y_observed.append(score)
    
    # Identify statistical groups
    groups, p_matrix = identify_statistically_equivalent_models(df)
    
    # Create optimizer
    optimizer = Optimizer(
        dimensions=param_space,
        random_state=42,
        acq_func="EI",  # Expected Improvement
        n_initial_points=10
    )
    
    # Inform optimizer with existing data
    if X_observed:
        optimizer.tell(X_observed, y_observed)
    
    # Get suggestions
    suggestions = optimizer.ask(n_points=n_suggestions)
    
    # Analyze suggestions
    suggestion_analysis = []
    for i, suggestion in enumerate(suggestions):
        config_dict = dict(zip(param_names, suggestion))
        
        # Find most similar existing models
        similarities = []
        for j, existing_config in enumerate(X_observed):
            # Calculate distance in parameter space
            distance = 0
            for k in range(len(suggestion)):
                if k < 2:  # Integer parameters
                    distance += abs(suggestion[k] - existing_config[k])
                else:  # Categorical parameters
                    distance += 0 if suggestion[k] == existing_config[k] else 1
            similarities.append((distance, j))
        
        similarities.sort()
        closest_idx = similarities[0][1]
        closest_model = df.iloc[closest_idx]
        
        # Calculate expected score based on closest model performance
        closest_error = closest_model[ERROR_COL]
        closest_speed = closest_model[SPEED_COL] 
        closest_memory = closest_model[MEMORY_COL]
        
        # Use pre-calculated timesteps/s for the closest model
        closest_timesteps_s = closest_model[TIMESTEPS_COL] if pd.notna(closest_model[TIMESTEPS_COL]) else 0

        expected_score = calculate_objective_score(closest_error, closest_speed, closest_memory, closest_timesteps_s)
        
        suggestion_analysis.append({
            'suggestion_id': i + 1,
            'config': config_dict,
            'expected_score': expected_score,
            'closest_model': closest_model[HPO_ID_COL],
            'closest_model_error': closest_model[ERROR_COL],
            'closest_model_speed': closest_model[SPEED_COL],
            'distance_to_closest': similarities[0][0]
        })
    
    return suggestion_analysis, groups

def create_suggestion_report(suggestion_analysis, groups, df):
    """Create a detailed report of suggestions with statistical context."""
    report = []
    report.append("# Enhanced Bayesian HPO Suggestions Report\n")
    
    report.append("## Statistical Analysis of Existing Models\n")
    report.append(f"Constraint Applied: Models must meet **>= {TIMESTEP_THRESHOLD} timesteps/s** on {BENCHMARK_ATOMS} atoms.\n")
    
    # Group statistics
    group_stats = {}
    for hpo_id, group in groups.items():
        if group not in group_stats:
            group_stats[group] = []
        model_data = df[df[HPO_ID_COL] == hpo_id].iloc[0]
        group_stats[group].append({
            'hpo_id': hpo_id,
            'error': model_data[ERROR_COL],
            'speed': model_data[SPEED_COL]
        })
    
    report.append(f"Found {len(group_stats)} statistically distinct performance groups:\n")
    
    for group_id, models in sorted(group_stats.items()):
        avg_error = np.mean([m['error'] for m in models])
        avg_speed = np.mean([m['speed'] for m in models])
        report.append(f"**Group {group_id + 1}** ({len(models)} models):")
        report.append(f"- Average Error: {avg_error:.4f} Å")
        report.append(f"- Average Speed: {avg_speed:.1f} k-atom-steps/s")
        report.append(f"- Members: {', '.join([m['hpo_id'] for m in models[:5]])}{'...' if len(models) > 5 else ''}")
        report.append("")
    
    report.append("\n## New Hyperparameter Suggestions\n")
    
    # Sort suggestions by expected improvement
    suggestion_analysis.sort(key=lambda x: x['expected_score'])
    
    report.append("Suggestions ranked by expected performance (considering error, speed, and memory):\n")
    
    for i, suggestion in enumerate(suggestion_analysis[:10]):
        report.append(f"### Suggestion {i + 1}")
        report.append(f"**Configuration:**")
        for param, value in suggestion['config'].items():
            report.append(f"- {param}: {value}")
        
        report.append(f"\n**Analysis:**")
        report.append(f"- Expected objective score: {suggestion['expected_score']:.3f}")
        report.append(f"- Most similar existing model: {suggestion['closest_model']}")
        report.append(f"  - Error: {suggestion['closest_model_error']:.4f} Å")
        report.append(f"  - Speed: {suggestion['closest_model_speed']:.1f} k-atom-steps/s")
        report.append(f"- Parameter distance to closest: {suggestion['distance_to_closest']}")
        report.append("")
    
    report.append("\n## Recommendations\n")
    report.append("1. **Priority Testing:** Start with suggestions 1-3 as they have the best expected performance")
    report.append("2. **Exploration vs Exploitation:** Some suggestions explore new parameter regions")
    report.append("3. **Statistical Validation:** After testing, use significance tests to confirm improvements")
    report.append("4. **Multi-node Testing:** Promising candidates should be tested at production scale")
    
    # Save report
    with open(OUTPUT_DIR / 'bayesian_suggestions_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    # Save suggestions as CSV
    suggestions_df = pd.DataFrame([
        {
            'suggestion_id': s['suggestion_id'],
            **s['config'],
            'expected_score': s['expected_score'],
            'closest_model': s['closest_model'],
            'distance_to_closest': s['distance_to_closest']
        }
        for s in suggestion_analysis
    ])
    suggestions_df.to_csv(OUTPUT_DIR / 'bayesian_suggestions.csv', index=False)
    
    return report

def main():
    """Run enhanced Bayesian optimization with statistical analysis."""
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    # --- New: Filter for models benchmarked at the correct atom count ---
    print(f"\nFiltering for models benchmarked at {BENCHMARK_ATOMS} atoms to inform the optimizer.")
    original_count = len(df)
    if 'benchmark_atoms' in df.columns:
        df.dropna(subset=['benchmark_atoms'], inplace=True)
        df = df[df['benchmark_atoms'] == BENCHMARK_ATOMS].copy()
        print(f"Retained {len(df)} out of {original_count} models after filtering for atom count.")
        if len(df) == 0:
            print(f"Error: No models found that were benchmarked at {BENCHMARK_ATOMS} atoms. Exiting.")
            return
    else:
        print("Warning: 'benchmark_atoms' column not found. Assuming all models were benchmarked at the correct size.")
    
    print("Analyzing hyperparameter coverage...")
    unexplored = analyze_hyperparameter_coverage(df, param_space)
    print(f"Found {len(unexplored)} unexplored configurations")
    
    print("Generating suggestions with statistical significance analysis...")
    suggestions, groups = generate_suggestions_with_significance(df, N_SUGGESTIONS)
    
    print("Creating detailed report...")
    report = create_suggestion_report(suggestions, groups, df)
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")
    print("\nKey outputs:")
    print("- hyperparameter_coverage.png: Visual of explored parameter space")
    print("- bayesian_suggestions.csv: New hyperparameter suggestions")
    print("- bayesian_suggestions_report.md: Detailed analysis and recommendations")

if __name__ == "__main__":
    main() 