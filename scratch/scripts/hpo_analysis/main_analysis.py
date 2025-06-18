import pandas as pd
import numpy as np
from scipy import stats
import re
import json
from pathlib import Path
from itertools import combinations
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Configuration ---
WANDB_CSV_PATH = Path("../../data/hpo_analysis/new_allegro/wandb_export_2025-05-17T11_11_30.422-04_00.csv")
SPEED_DATA_BASE_DIR = Path("../../data/hpo_analysis/new_allegro/")
FORCE_RMSE_COLUMN = 'test0_epoch/forces_mae'
MODEL_NAME_COLUMN = 'Name'
ALPHA = 0.05  # Significance level
MIN_SAMPLES_PER_CONFIG = 2 # Minimum samples needed for a config to be analyzed
BENCHMARK_ATOMS = 8192 # The *only* atom count to consider for speed/memory benchmarks
OUTPUT_DIR = Path("./analysis_outputs") # Directory to save outputs

# --- Define simple, filesystem-safe aliases for metrics ---
METRIC_ALIAS_FORCE_ERROR = "Force_Error"
METRIC_ALIAS_KATOM_STEPS_S = "KAtomSteps_s"
METRIC_ALIAS_TIMESTEPS_S = "Timesteps_s"
METRIC_ALIAS_MEMORY = "Memory_MiB"

# Define the columns that will be created in the summary table
# These will be the primary metrics for Pareto analysis etc.
PARETO_ERROR_METRIC = f"mean_{METRIC_ALIAS_FORCE_ERROR}"
PARETO_SPEED_METRIC = f"mean_{METRIC_ALIAS_TIMESTEPS_S}" # Using Timesteps/s as the primary speed metric
PARETO_MEMORY_METRIC = f"mean_{METRIC_ALIAS_MEMORY}"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Functions ---
def get_base_config_and_id_from_name(name_str):
    """
    Extracts base configuration string and HPO ID from a model name.
    Example: "hpo_107_params_fold2_seed2" -> ("hpo_107_params", "hpo_107")
    """
    match_base = re.match(r"^(hpo_\d+.*?)(?:_fold\d_seed\d|_fold\d|_seed\d)?$", name_str)
    base_config_str = name_str
    if match_base:
        base_config_str = match_base.group(1)
    
    match_id = re.match(r"^(hpo_\d+)", base_config_str)
    config_id = None
    if match_id:
        config_id = match_id.group(1)
    else:
        logging.warning(f"Could not extract HPO ID from base config string: {base_config_str} (original name: {name_str})")
        
    return base_config_str, config_id

def extract_hpo_id_from_potential_file(potential_file_path_str):
    """
    Extracts 'hpo_XXX' from a potential file path string.
    Example: "/path/to/hpo_107_compiled.nequip.pt2" -> "hpo_107"
    """
    if not potential_file_path_str:
        return None
    name_part = Path(potential_file_path_str).name
    match = re.match(r"^(hpo_\d+)_compiled\.nequip\.pt2$", name_part)
    if match:
        return match.group(1)
    logging.warning(f"Could not extract HPO ID from potential file name: {name_part}")
    return None

def cohen_d(group1, group2):
    """Calculate Cohen's d for independent samples."""
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1:
        return np.nan
        
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1) # ddof=1 for sample std dev
    
    # Handle cases where one or both groups have zero variance
    if (n1 > 1 and std1 == 0) and (n2 > 1 and std2 == 0):
        return 0.0 if mean1 == mean2 else np.inf * np.sign(mean1 - mean2)
    if n1 > 1 and std1 == 0: # Only group1 has zero variance
        # If means are different, effect size is infinite. If same, it's 0.
        # This is a simplification; could use a very small epsilon for std1.
        return 0.0 if mean1 == mean2 else np.inf * np.sign(mean1 - mean2)
    if n2 > 1 and std2 == 0: # Only group2 has zero variance
        return 0.0 if mean1 == mean2 else np.inf * np.sign(mean1 - mean2)


    # Pooled standard deviation
    # Denominator for pooled_std: (n1 - 1) + (n2 - 1) = n1 + n2 - 2
    # This term must be > 0, which is ensured by n1 > 1 and n2 > 1 check earlier.
    # If either n1 or n2 is 1, cohen_d returns nan.
    pooled_std_squared = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    
    if pooled_std_squared <= 0: # Avoid sqrt of zero or negative if stds were tiny
        # This case should ideally be caught by earlier std==0 checks,
        # but as a fallback:
        return 0.0 if mean1 == mean2 else np.inf * np.sign(mean1 - mean2)
        
    pooled_std = np.sqrt(pooled_std_squared)
    
    return (mean1 - mean2) / pooled_std

# --- Main Analysis Logic ---
def load_performance_data(csv_path, name_col, rmse_col):
    """Loads and processes performance data from W&B CSV."""
    try:
        perf_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"Performance data CSV not found at: {csv_path}")
        return {}
        
    perf_df.dropna(subset=[rmse_col], inplace=True)
    
    model_performance = {}
    for _, row in perf_df.iterrows():
        name = row[name_col]
        force_rmse = row[rmse_col]
        
        base_config_str, config_id = get_base_config_and_id_from_name(name)
        
        if not config_id: # Skip if we couldn't get a valid ID
            continue

        if base_config_str not in model_performance:
            model_performance[base_config_str] = {'id': config_id, 'force_rmse_values': []}
        
        # Ensure the ID is consistent if the base_config_str was already seen
        # (e.g. if regex had an issue with one variant of name)
        if model_performance[base_config_str]['id'] != config_id:
            logging.warning(f"Inconsistent HPO ID for base_config '{base_config_str}': "
                            f"'{model_performance[base_config_str]['id']}' vs '{config_id}'. Using first one.")
        
        model_performance[base_config_str]['force_rmse_values'].append(force_rmse)
        
    # Filter out models with insufficient data points
    valid_model_performance = {
        cfg: data for cfg, data in model_performance.items()
        if len(data['force_rmse_values']) >= MIN_SAMPLES_PER_CONFIG
    }
    logging.info(f"Loaded performance data for {len(valid_model_performance)} distinct model configurations "
                 f"with at least {MIN_SAMPLES_PER_CONFIG} samples each.")
    return valid_model_performance

def load_speed_data_ext(base_dir, benchmark_atom_count):
    """
    Loads and aggregates speed and memory benchmark data, focusing exclusively
    on benchmarks run with a specific number of atoms.

    Args:
        base_dir (Path): The base directory containing the benchmark JSON files.
        benchmark_atom_count (int): The exact number of atoms to filter for.

    Returns a dict: hpo_id -> { ... aggregated metrics ... }
    """
    speed_data_files = list(base_dir.glob("all_benchmarks_consolidated_results_seed_*/all_models_benchmark_summary.json"))
    if not speed_data_files:
        logging.warning(f"No speed benchmark JSON files found in {base_dir} matching the pattern.")
        return {}

    # Step 1: Collect all benchmark runs for each HPO ID from all seed files
    all_runs_by_hpo_id = defaultdict(list)
    for speed_file_path in speed_data_files:
        logging.info(f"Processing speed file: {speed_file_path}")
        try:
            with open(speed_file_path, 'r') as f:
                benchmarks_this_seed = json.load(f)
        except Exception as e:
            logging.error(f"Error reading or parsing speed file {speed_file_path}: {e}")
            continue

        for run in benchmarks_this_seed:
            potential_file = run.get('potential_file')
            hpo_id = extract_hpo_id_from_potential_file(potential_file)
            
            # --- The critical change is here ---
            # We ONLY consider runs that match the specified benchmark_atom_count
            if hpo_id and run.get('num_atoms') == benchmark_atom_count:
                if all(k in run and run[k] is not None for k in ['katom_steps_per_s_log', 'peak_gpu_memory_mib']):
                    all_runs_by_hpo_id[hpo_id].append({
                        'katom_steps_per_s_log': run['katom_steps_per_s_log'],
                        'peak_gpu_memory_mib': run['peak_gpu_memory_mib']
                    })
                else:
                    logging.debug(f"Skipping run for HPO ID {hpo_id} at {benchmark_atom_count} atoms due to missing critical data.")

    # Step 2: Aggregate the collected metrics for each HPO ID
    aggregated_metrics = {}
    for hpo_id, runs in all_runs_by_hpo_id.items():
        if not runs:
            continue

        katom_sps_values = [r['katom_steps_per_s_log'] for r in runs]
        peak_mem_values = [r['peak_gpu_memory_mib'] for r in runs]
        
        # New: Also calculate timesteps/s values
        timesteps_s_values = [(k * 1000 / benchmark_atom_count) for k in katom_sps_values]

        aggregated_metrics[hpo_id] = {
            'katom_steps_per_s_values': katom_sps_values,
            'mean_katom_steps_per_s': np.mean(katom_sps_values) if katom_sps_values else np.nan,
            'std_katom_steps_per_s': np.std(katom_sps_values, ddof=1) if len(katom_sps_values) > 1 else 0.0,
            
            'timesteps_per_s_values': timesteps_s_values,
            'mean_timesteps_per_s': np.mean(timesteps_s_values) if timesteps_s_values else np.nan,
            'std_timesteps_per_s': np.std(timesteps_s_values, ddof=1) if len(timesteps_s_values) > 1 else 0.0,
            
            'peak_gpu_memory_mib_values': peak_mem_values,
            'mean_peak_gpu_memory_mib': np.mean(peak_mem_values) if peak_mem_values else np.nan,
            'std_peak_gpu_memory_mib': np.std(peak_mem_values, ddof=1) if len(peak_mem_values) > 1 else 0.0,
            
            'num_samples': len(runs)
        }
        if not katom_sps_values:
             logging.warning(f"No valid 'katom_steps_per_s_log' found for HPO ID {hpo_id} at {benchmark_atom_count} atoms.")
        if not peak_mem_values:
             logging.warning(f"No valid 'peak_gpu_memory_mib' found for HPO ID {hpo_id} at {benchmark_atom_count} atoms.")

    logging.info(f"Loaded and aggregated speed/memory data for {len(aggregated_metrics)} distinct HPO IDs from {benchmark_atom_count}-atom benchmarks.")
    return aggregated_metrics

def create_summary_table(model_performance_data, aggregated_ext_metrics, benchmark_atom_count):
    """
    Creates a summary table of model performance, speed, and memory.
    """
    model_summary_data = []
    
    # Define the column names using the aliases for clarity
    mean_error_col = f"mean_{METRIC_ALIAS_FORCE_ERROR}"
    std_error_col = f"std_{METRIC_ALIAS_FORCE_ERROR}"

    mean_ksps_col = f"mean_{METRIC_ALIAS_KATOM_STEPS_S}"
    std_ksps_col = f"std_{METRIC_ALIAS_KATOM_STEPS_S}"
    
    mean_tsps_col = f"mean_{METRIC_ALIAS_TIMESTEPS_S}"
    std_tsps_col = f"std_{METRIC_ALIAS_TIMESTEPS_S}"

    mean_mem_col = f"mean_{METRIC_ALIAS_MEMORY}"
    std_mem_col = f"std_{METRIC_ALIAS_MEMORY}"
    
    # Keys in aggregated_ext_metrics from load_speed_data_ext
    agg_ksps_mean_key = 'mean_katom_steps_per_s'
    agg_ksps_std_key = 'std_katom_steps_per_s'
    agg_tsps_mean_key = 'mean_timesteps_per_s'
    agg_tsps_std_key = 'std_timesteps_per_s'
    agg_mem_mean_key = 'mean_peak_gpu_memory_mib'
    agg_mem_std_key = 'std_peak_gpu_memory_mib'
    
    agg_ksps_values_key = 'katom_steps_per_s_values'
    agg_mem_values_key = 'peak_gpu_memory_mib_values'

    for config_name, perf_data in model_performance_data.items():
        hpo_id = perf_data['id']
        rmse_values = perf_data['force_rmse_values']
        
        summary_row = {
            'config_name': config_name,
            'hpo_id': hpo_id,
            mean_error_col: np.mean(rmse_values),
            std_error_col: np.std(rmse_values, ddof=1),
            'num_rmse_samples': len(rmse_values),
            'benchmark_atoms': np.nan,
            'num_speed_samples': np.nan,
        }
        
        # Initialize speed/memory metrics to NaN
        for col in [mean_ksps_col, std_ksps_col, mean_tsps_col, std_tsps_col, mean_mem_col, std_mem_col]:
            summary_row[col] = np.nan
            
        if hpo_id in aggregated_ext_metrics:
            hpo_metrics = aggregated_ext_metrics[hpo_id]
            summary_row['benchmark_atoms'] = benchmark_atom_count
            summary_row['num_speed_samples'] = hpo_metrics.get('num_samples')
            
            # Populate from the aggregated data dict
            summary_row[mean_ksps_col] = hpo_metrics.get(agg_ksps_mean_key)
            summary_row[std_ksps_col] = hpo_metrics.get(agg_ksps_std_key)
            summary_row[mean_tsps_col] = hpo_metrics.get(agg_tsps_mean_key)
            summary_row[std_tsps_col] = hpo_metrics.get(agg_tsps_std_key)
            summary_row[mean_mem_col] = hpo_metrics.get(agg_mem_mean_key)
            summary_row[std_mem_col] = hpo_metrics.get(agg_mem_std_key)

        model_summary_data.append(summary_row)
        
    summary_df = pd.DataFrame(model_summary_data)
    
    # Reorder columns for better readability
    cols_order = [
        'config_name', 'hpo_id', 
        mean_error_col, std_error_col, 'num_rmse_samples',
        mean_ksps_col, std_ksps_col,
        mean_tsps_col, std_tsps_col,
        mean_mem_col, std_mem_col,
        'benchmark_atoms', 'num_speed_samples'
    ]
    
    final_cols = [col for col in cols_order if col in summary_df.columns]
    if not summary_df.empty:
        summary_df = summary_df[final_cols]

    logging.info("Generated model summary table with both k-atom-steps/s and timesteps/s metrics.")
    return summary_df

def perform_statistical_comparisons(model_performance_data, alpha):
    """Performs pairwise t-tests and Cohen's d calculations for Force RMSE."""
    comparison_results = []
    valid_config_names = list(model_performance_data.keys())
    
    if len(valid_config_names) < 2:
        logging.info("Not enough model configurations (need at least 2) to perform pairwise comparisons.")
        return pd.DataFrame()

    for config1_name, config2_name in combinations(valid_config_names, 2):
        data1 = np.array(model_performance_data[config1_name]['force_rmse_values'])
        data2 = np.array(model_performance_data[config2_name]['force_rmse_values'])
        
        # Welch's t-test (does not assume equal variance)
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
        d_cohen = cohen_d(data1, data2)
        
        comparison_results.append({
            'config1': config1_name,
            'config2': config2_name,
            'mean_rmse_config1': np.mean(data1),
            'mean_rmse_config2': np.mean(data2),
            'std_rmse_config1': np.std(data1, ddof=1),
            'std_rmse_config2': np.std(data2, ddof=1),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': d_cohen,
            'significant_force_rmse': p_value < alpha,
            'better_config': config1_name if np.mean(data1) < np.mean(data2) else config2_name # Lower RMSE is better
        })
        
    logging.info(f"Performed {len(comparison_results)} pairwise statistical comparisons for Force RMSE.")
    return pd.DataFrame(comparison_results)

def perform_metric_statistical_comparisons(model_summary_df, value_lists_source_dict, hpo_id_col, config_name_col, metric_values_key, metric_name, alpha, lower_is_better=True):
    """
    Performs pairwise t-tests and Cohen's d for a generic metric.
    
    model_summary_df: DataFrame linking config_name to hpo_id.
    value_lists_source_dict: Dict mapping hpo_id to a dict containing the list of raw metric values (e.g., output of load_speed_data_ext).
    hpo_id_col: Name of the HPO ID column in model_summary_df.
    config_name_col: Name of the configuration name column in model_summary_df.
    metric_values_key: The key in value_lists_source_dict[hpo_id] that holds the list of raw values for the metric.
    metric_name: A descriptive name for the metric (e.g., "Speed at Max Atoms").
    alpha: Significance level.
    lower_is_better: True if lower values of the metric are better.
    """
    comparison_results = []
    
    # Prepare data: map config_name to list of metric values
    metric_data_by_config = {}
    for _, row in model_summary_df.iterrows():
        config_name = row[config_name_col]
        hpo_id = row[hpo_id_col]
        if hpo_id in value_lists_source_dict and \
           metric_values_key in value_lists_source_dict[hpo_id] and \
           len(value_lists_source_dict[hpo_id][metric_values_key]) >= MIN_SAMPLES_PER_CONFIG:
            metric_data_by_config[config_name] = np.array(value_lists_source_dict[hpo_id][metric_values_key])
        else:
            logging.debug(f"Skipping config {config_name} (HPO ID: {hpo_id}) for {metric_name} stats due to insufficient data or missing key '{metric_values_key}'.")

    valid_config_names = list(metric_data_by_config.keys())

    if len(valid_config_names) < 2:
        logging.info(f"Not enough model configurations (need at least 2 with sufficient data) for {metric_name} to perform pairwise comparisons.")
        return pd.DataFrame()

    for config1_name, config2_name in combinations(valid_config_names, 2):
        data1 = metric_data_by_config[config1_name]
        data2 = metric_data_by_config[config2_name]
        
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
        d_cohen = cohen_d(data1, data2) # group1 - group2
        
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)

        if lower_is_better:
            better_config = config1_name if mean1 < mean2 else config2_name
        else: # Higher is better
            better_config = config1_name if mean1 > mean2 else config2_name
            
        # Use the metric_name (which should now be a simple alias) for column naming in comparison CSV
        # Ensure metric_name is filesystem-safe if it's not already an alias
        safe_metric_name_for_col = re.sub(r'[^\w\-_]', '_', metric_name) # Basic sanitization
            
        comparison_results.append({
            'config1': config1_name,
            'config2': config2_name,
            f'mean_{safe_metric_name_for_col}_config1': mean1,
            f'mean_{safe_metric_name_for_col}_config2': mean2,
            f'std_{safe_metric_name_for_col}_config1': np.std(data1, ddof=1),
            f'std_{safe_metric_name_for_col}_config2': np.std(data2, ddof=1),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': d_cohen,
            f'significant_diff': p_value < alpha,
            'better_config': better_config
        })
        
    logging.info(f"Performed {len(comparison_results)} pairwise statistical comparisons for {metric_name}.")
    return pd.DataFrame(comparison_results)

# --- Pareto Analysis and Visualization ---
def find_pareto_optimal_models_multi_objective(df, objectives_cols, objectives_minimize_flags):
    """
    Finds Pareto optimal models from a DataFrame for multiple objectives.
    objectives_cols: List of column names for objectives.
    objectives_minimize_flags: List of booleans (True if objective should be minimized, False if maximized).
    """
    if len(objectives_cols) != len(objectives_minimize_flags):
        raise ValueError("objectives_cols and objectives_minimize_flags must have the same length.")

    df_cleaned = df.dropna(subset=objectives_cols).copy()
    if df_cleaned.empty:
        logging.warning(f"DataFrame is empty after dropping NaNs for Pareto analysis columns: {objectives_cols}")
        return pd.DataFrame()

    num_objectives = len(objectives_cols)
    is_pareto = np.ones(df_cleaned.shape[0], dtype=bool)
    
    points = df_cleaned[objectives_cols].values

    for i in range(points.shape[0]):
        if not is_pareto[i]:
            continue
        for j in range(points.shape[0]):
            if i == j:
                continue
            
            # Check if point j dominates point i
            dominates = True
            strictly_better_on_one = False
            for k in range(num_objectives):
                val_i = points[i, k]
                val_j = points[j, k]
                
                if objectives_minimize_flags[k]: # Minimize objective k
                    if val_j > val_i: # j is worse than i on this objective
                        dominates = False
                        break
                    if val_j < val_i: # j is strictly better than i on this objective
                        strictly_better_on_one = True
                else: # Maximize objective k
                    if val_j < val_i: # j is worse than i on this objective
                        dominates = False
                        break
                    if val_j > val_i: # j is strictly better than i on this objective
                        strictly_better_on_one = True
            
            if dominates and strictly_better_on_one:
                is_pareto[i] = False # Point i is dominated by point j
                break
                
    pareto_df = df_cleaned[is_pareto].copy()
    # Sort by first objective for consistent output, then second, etc.
    # Ascending based on minimize_flags
    sort_ascending_flags = objectives_minimize_flags
    pareto_df = pareto_df.sort_values(by=objectives_cols, ascending=sort_ascending_flags)

    logging.info(f"Found {len(pareto_df)} Pareto optimal models for {num_objectives} objectives.")
    return pareto_df

def plot_3d_pareto_projections(df_all_models, pareto_3d_df, objectives_cols, objectives_names, objectives_minimize_flags, output_dir_path):
    """
    Plots 2D projections of a 3D Pareto front.
    df_all_models: DataFrame with all models.
    pareto_3d_df: DataFrame with 3-objective Pareto optimal models.
    objectives_cols: List of 3 column names [err_col, speed_col, mem_col].
    objectives_names: List of 3 pretty names for axes [err_name, speed_name, mem_name].
    objectives_minimize_flags: List of 3 booleans [err_min, speed_min, mem_min].
    output_dir_path: Path object for the output directory.
    """
    if len(objectives_cols) != 3 or len(objectives_names) != 3 or len(objectives_minimize_flags) !=3:
        logging.error("Pareto projection plotting requires exactly 3 objectives.")
        return

    df_plot_all = df_all_models.dropna(subset=objectives_cols)
    if df_plot_all.empty:
        logging.warning("Cannot plot Pareto projections: input DataFrame for all models is empty after NaNs removed.")
        return

    projection_pairs = list(combinations(range(3), 2)) # (0,1), (0,2), (1,2)

    for idx1, idx2 in projection_pairs:
        col1, col2 = objectives_cols[idx1], objectives_cols[idx2]
        name1, name2 = objectives_names[idx1], objectives_names[idx2]
        min1, min2 = objectives_minimize_flags[idx1], objectives_minimize_flags[idx2]

        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df_plot_all, x=col1, y=col2, label='All Models', alpha=0.5, s=30)

        if not pareto_3d_df.empty:
            df_plot_pareto = pareto_3d_df.dropna(subset=[col1, col2])
            if not df_plot_pareto.empty:
                sns.scatterplot(data=df_plot_pareto, x=col1, y=col2, color='red', label='3D Pareto Optimal', s=80, edgecolor='black', zorder=5)
        
        xlabel_text = f"{name1} ({'Lower is Better' if min1 else 'Higher is Better'})" # Use pretty names for labels
        ylabel_text = f"{name2} ({'Lower is Better' if min2 else 'Higher is Better'})"
        plt.xlabel(xlabel_text)
        plt.ylabel(ylabel_text)
        plt.title(f"2D Projection: {name1} vs. {name2}") # Use pretty names for title
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Use the column names (which are now simple aliases) for filename generation
        fname1_safe = re.sub(r'[^\w\-_.]', '_', col1) # col1 should be like 'mean_Force_Error'
        fname2_safe = re.sub(r'[^\w\-_.]', '_', col2)
        fname1_safe = re.sub(r'_+', '_', fname1_safe)
        fname2_safe = re.sub(r'_+', '_', fname2_safe)
        
        plot_filename = output_dir_path / f"pareto_projection_{fname1_safe}_vs_{fname2_safe}.png"
        
        try:
            plt.savefig(plot_filename)
            logging.info(f"Saved Pareto projection plot to {plot_filename}")
        except Exception as e:
            logging.error(f"Failed to save Pareto projection plot {plot_filename}: {e}")
        plt.close()

def main():
    """Main analysis script execution."""
    logging.info("Starting HPO analysis...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Create output directory

    # 1. Load performance data
    valid_model_performance = load_performance_data(WANDB_CSV_PATH, MODEL_NAME_COLUMN, FORCE_RMSE_COLUMN)
    if not valid_model_performance:
        logging.error("No valid performance data loaded. Exiting.")
        return

    # 2. Load extended speed and memory data, strictly for the specified atom count
    aggregated_speed_memory_data = load_speed_data_ext(SPEED_DATA_BASE_DIR, BENCHMARK_ATOMS)
    if not aggregated_speed_memory_data:
        logging.warning(f"No speed/memory data loaded for {BENCHMARK_ATOMS}-atom benchmarks. Some parts of the analysis might be skipped or incomplete.")

    # 3. Create summary table
    model_summary_df = create_summary_table(valid_model_performance, aggregated_speed_memory_data, BENCHMARK_ATOMS)
    
    if model_summary_df.empty:
        logging.error("Model summary table is empty. Cannot proceed with Pareto or full statistical analysis.")
        # We might still be able to do RMSE stats if valid_model_performance exists
        # For now, let's assume if summary_df is empty, we stop detailed analysis.
        # Perform RMSE statistical comparisons
        rmse_comparison_df = perform_statistical_comparisons(valid_model_performance, ALPHA) # Original function
        if not rmse_comparison_df.empty:
            print("\n--- Pairwise Comparison Results (Force RMSE) ---")
            rmse_comparison_df_sorted = rmse_comparison_df.sort_values(by=['p_value', 'cohen_d'], key=lambda x: abs(x) if x.name == 'cohen_d' else x, ascending=[True, False])
            print(rmse_comparison_df_sorted.to_string())
            rmse_comparison_path = OUTPUT_DIR / "pairwise_comparison_force_rmse.csv"
            rmse_comparison_df_sorted.to_csv(rmse_comparison_path, index=False)
            logging.info(f"Saved Force RMSE pairwise comparison results to {rmse_comparison_path}")
        else:
            logging.info("Force RMSE pairwise comparison table is empty.")
        logging.info("Analysis complete (limited due to empty model summary).")
        return
        
    print("\n--- Model Performance, Speed, and Memory Summary ---")
    print(model_summary_df.to_string())
    summary_path = OUTPUT_DIR / "model_summary_analysis.csv"
    model_summary_df.to_csv(summary_path, index=False)
    logging.info(f"Saved model summary to {summary_path}")

    # 4. Perform statistical comparisons
    # 4.1 Force RMSE (original function)
    rmse_comparison_df = perform_statistical_comparisons(valid_model_performance, ALPHA)
    if not rmse_comparison_df.empty:
        print("\n--- Pairwise Comparison Results (Force RMSE) ---")
        rmse_comparison_df_sorted = rmse_comparison_df.sort_values(by=['p_value', 'cohen_d'], key=lambda x: abs(x) if x.name == 'cohen_d' else x, ascending=[True, False])
        print(rmse_comparison_df_sorted.to_string())
        rmse_comparison_path = OUTPUT_DIR / "pairwise_comparison_force_rmse.csv"
        rmse_comparison_df_sorted.to_csv(rmse_comparison_path, index=False)
        logging.info(f"Saved Force RMSE pairwise comparison results to {rmse_comparison_path}")
    else:
        logging.info("Force RMSE pairwise comparison table is empty.")

    # 4.2 Timesteps per second
    tsps_metric_values_key = 'timesteps_per_s_values'
    tsps_metric_name = METRIC_ALIAS_TIMESTEPS_S
    if f"mean_{tsps_metric_name}" in model_summary_df.columns:
        tsps_comparison_df = perform_metric_statistical_comparisons(
            model_summary_df, aggregated_speed_memory_data, 'hpo_id', 'config_name',
            tsps_metric_values_key, tsps_metric_name, ALPHA, lower_is_better=False
        )
        if not tsps_comparison_df.empty:
            print(f"\n--- Pairwise Comparison Results ({tsps_metric_name}) ---")
            tsps_comparison_df_sorted = tsps_comparison_df.sort_values(by=['p_value', 'cohen_d'], key=lambda x: abs(x) if x.name == 'cohen_d' else x, ascending=[True, False])
            print(tsps_comparison_df_sorted.to_string())
            tsps_comp_filename = f"pairwise_comparison_{tsps_metric_name}.csv"
            tsps_comp_path = OUTPUT_DIR / tsps_comp_filename
            tsps_comparison_df_sorted.to_csv(tsps_comp_path, index=False)
            logging.info(f"Saved {tsps_metric_name} pairwise comparison results to {tsps_comp_path}")
        else:
            logging.info(f"{tsps_metric_name} pairwise comparison table is empty.")
    else:
        logging.warning(f"Column mean_{tsps_metric_name} not found in summary, skipping statistical tests for it.")

    # 4.3 Memory at Max Atoms
    memory_metric_values_key = 'peak_gpu_memory_mib_values'
    memory_metric_name_for_csv_cols = METRIC_ALIAS_MEMORY
    if f"mean_{memory_metric_name_for_csv_cols}" in model_summary_df.columns: 
        memory_comparison_df = perform_metric_statistical_comparisons(
            model_summary_df, aggregated_speed_memory_data, 'hpo_id', 'config_name',
            memory_metric_values_key, memory_metric_name_for_csv_cols, ALPHA, lower_is_better=True
        )
        if not memory_comparison_df.empty:
            print(f"\n--- Pairwise Comparison Results ({memory_metric_name_for_csv_cols}) ---")
            memory_comparison_df_sorted = memory_comparison_df.sort_values(by=['p_value', 'cohen_d'], key=lambda x: abs(x) if x.name == 'cohen_d' else x, ascending=[True, False])
            print(memory_comparison_df_sorted.to_string())
            mem_comp_filename = f"pairwise_comparison_{memory_metric_name_for_csv_cols}.csv"
            mem_comp_path = OUTPUT_DIR / mem_comp_filename
            memory_comparison_df_sorted.to_csv(mem_comp_path, index=False)
            logging.info(f"Saved {memory_metric_name_for_csv_cols} pairwise comparison results to {mem_comp_path}")
        else:
            logging.info(f"{memory_metric_name_for_csv_cols} pairwise comparison table is empty.")
    else:
        logging.warning(f"Column mean_{memory_metric_name_for_csv_cols} not found in summary, skipping statistical tests for {METRIC_ALIAS_MEMORY}.")


    # 5. Multi-Objective Pareto Analysis (Error, Speed, Memory)
    # Using Timesteps/s as the primary speed metric, as defined in PARETO_SPEED_METRIC
    objectives_cols_3d = [PARETO_ERROR_METRIC, PARETO_SPEED_METRIC, PARETO_MEMORY_METRIC]
    objectives_minimize_flags_3d = [True, False, True] # Error (min), Speed (max), Memory (min)
    objectives_pretty_names_3d = ["Force RMSE", "Throughput (timesteps/s)", "Memory (MiB)"]

    # Ensure all objective columns are present in the summary DataFrame
    missing_obj_cols = [col for col in objectives_cols_3d if col not in model_summary_df.columns]
    if missing_obj_cols:
        logging.error(f"Missing one or more objective columns for 3D Pareto analysis: {missing_obj_cols}. Skipping 3D Pareto.")
    else:
        pareto_3d_optimal_df = find_pareto_optimal_models_multi_objective(
            model_summary_df, 
            objectives_cols_3d, 
            objectives_minimize_flags_3d
        )
        
        if not pareto_3d_optimal_df.empty:
            print("\n--- 3-Objective Pareto Optimal Models (Error, Speed, Memory) ---")
            print(pareto_3d_optimal_df[['config_name', 'hpo_id'] + objectives_cols_3d].to_string())
            pareto_3d_path = OUTPUT_DIR / "pareto_3d_optimal_models.csv"
            pareto_3d_optimal_df.to_csv(pareto_3d_path, index=False)
            logging.info(f"Saved 3D Pareto optimal models to {pareto_3d_path}")

            plot_3d_pareto_projections(
                model_summary_df, 
                pareto_3d_optimal_df, 
                objectives_cols_3d, 
                objectives_pretty_names_3d,
                objectives_minimize_flags_3d,
                OUTPUT_DIR
            )
        else:
            logging.info("No 3-Objective Pareto optimal models found or data was insufficient.")
    
    # Clean up old 2-objective Pareto (optional, or adapt if still needed)
    # The previous 2-objective Pareto plot (Error vs. old Speed metric) might be redundant or misleading now.
    # For now, I will comment out the old `plot_pareto_front` call. If you want a specific
    # 2D Pareto (e.g. just Error vs New Speed), we can add a dedicated call.

    # if PARETO_SPEED_METRIC in model_summary_df.columns and PARETO_ERROR_METRIC in model_summary_df.columns:
    #     # This would be a 2D Pareto on the new speed metric if desired
    #     pareto_2d_error_speed_df = find_pareto_optimal_models_multi_objective(
    #         model_summary_df,
    #         [PARETO_ERROR_METRIC, PARETO_SPEED_METRIC],
    #         [True, False] # RMSE (min), New Speed (max)
    #     )
    #     if not pareto_2d_error_speed_df.empty:
    #         plot_output_path = OUTPUT_DIR / f"pareto_2D_plot_{PARETO_ERROR_METRIC}_vs_{PARETO_SPEED_METRIC}.png"
    #         # Need to adapt plot_pareto_front or use a generic plotting function for 2D
    #         # plot_pareto_front(model_summary_df, pareto_2d_error_speed_df, PARETO_ERROR_METRIC, PARETO_SPEED_METRIC, ...)
    #         logging.info("Consider adding a specific 2D Pareto plot for Error vs. New Speed if needed.")
    # else:
    #     logging.info(f"Skipping 2D Pareto plot for {PARETO_ERROR_METRIC} vs {PARETO_SPEED_METRIC} due to missing columns.")


    logging.info("Analysis complete.")

if __name__ == "__main__":
    main()
