import numpy as np
import matplotlib.pyplot as plt
import glob
from nequip.ase import NequIPCalculator
from forge.core.database import DatabaseManager
from tqdm import tqdm
import pandas as pd
from scipy.stats import kurtosis, spearmanr
import seaborn as sns

# Note: 'pred' and 'target' are assumed to be NumPy arrays.

def tail_mse_numpy(pred, target, quantile=0.9):
    err = pred - target
    abs_err = np.abs(err)
    if abs_err.size == 0: return 0.0
    threshold = np.quantile(abs_err, quantile)
    tail_mask = abs_err >= threshold
    if not np.any(tail_mask): return 0.0
    tail_err = err[tail_mask]
    return np.mean(tail_err**2)

def tail_huber_loss_numpy(pred, target, delta=1.0, quantile=0.9):
    err = pred - target
    err_norm = np.linalg.norm(err, axis=-1)
    if err_norm.size == 0: return 0.0
    threshold = np.quantile(err_norm, quantile)
    tail_mask = err_norm >= threshold
    if not np.any(tail_mask): return 0.0
    
    tail_preds = pred[tail_mask]
    tail_targets = target[tail_mask]
    
    tail_err = tail_preds - tail_targets
    abs_tail_err = np.abs(tail_err)
    quadratic = np.minimum(abs_tail_err, delta)
    linear = abs_tail_err - quadratic
    huber_losses = 0.5 * quadratic**2 + delta * linear
    return np.mean(huber_losses)

def focal_mse_loss_numpy(pred, target, beta=1.0, gamma=2.0, eps=1e-6):
    err = pred - target
    sigmoid = 1 / (1 + np.exp(-beta * np.abs(err)))
    w = (sigmoid + eps) ** gamma
    focal_loss = w * err**2
    return np.mean(focal_loss)

def force_angle_loss_numpy(pred, target, eps=1e-8):
    pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    cos_theta = np.sum(pred * target, axis=-1, keepdims=True) / \
                (np.maximum(pred_norm, eps) * np.maximum(target_norm, eps))
    angle_error = 1.0 - cos_theta
    return np.mean(angle_error)

def stress_shear_mae_numpy(pred, target):
    shear_errors = [
        np.abs(pred[:, 0, 1] - target[:, 0, 1]),
        np.abs(pred[:, 0, 2] - target[:, 0, 2]),
        np.abs(pred[:, 1, 2] - target[:, 1, 2])
    ]
    mean_abs_err_per_sample = np.mean(np.stack(shear_errors, axis=0), axis=0)
    return np.mean(mean_abs_err_per_sample)

def _to_voigt_numpy(stress):
    return np.stack([
        stress[:, 0, 0], stress[:, 1, 1], stress[:, 2, 2],
        stress[:, 1, 2], stress[:, 0, 2], stress[:, 0, 1]
    ], axis=-1)

def stress_angle_loss_numpy(pred, target, eps=1e-8):
    voigt_pred = _to_voigt_numpy(pred)
    voigt_target = _to_voigt_numpy(target)
    
    pred_norm = np.linalg.norm(voigt_pred, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(voigt_target, axis=-1, keepdims=True)
    
    cos_phi = np.sum(voigt_pred * voigt_target, axis=-1, keepdims=True) / \
              (np.maximum(pred_norm, eps) * np.maximum(target_norm, eps))
    angle_error = 1.0 - cos_phi
    return np.mean(angle_error)

def voigt_to_matrix(voigt_stress):
    """Converts a 6-element Voigt stress vector to a 3x3 matrix."""
    s = voigt_stress
    return np.array([[s[0], s[5], s[4]],
                     [s[5], s[1], s[3]],
                     [s[4], s[3], s[2]]])

def gini_coefficient(x):
    """Calculate the Gini coefficient of a numpy array."""
    x = np.abs(x.flatten())
    if np.amin(x) < 0:
        # Values cannot be negative:
        x -= np.amin(x)
    # Values cannot be 0:
    x += 0.0000001
    # Values must be sorted:
    x = np.sort(x)
    index = np.arange(1, x.shape[0] + 1)
    n = x.shape[0]
    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

# --- Example Usage ---

if __name__ == "__main__":
    db = DatabaseManager(debug=False)

    print("[INFO] Finding relevant structure IDs...")
    # Find structures that are not dimers and are part of the training set
    structure_ids = db.find_structures_by_metadata(metadata_filters={'generation': 0}, operator='>=')
    dimer_ids = db.find_structures_by_metadata(metadata_filters={'config_type': 'dimer'})
    valid_structure_ids = [sid for sid in structure_ids if sid not in dimer_ids]
    
    print(f"[INFO] Found {len(valid_structure_ids)} structures to process.")
    
    # Get all atoms objects with their ground truth calculations
    atoms_list = db.get_batch_atoms_with_calculation(valid_structure_ids, calculator='vasp')

    print("[INFO] Loading ensemble of calculators...")
    ens_calc = []
    # This path might need to be adjusted depending on where you run the script
    compiled_paths = glob.glob('../data/potentials/compiled_models/*.pt2')
    for compiled_path in compiled_paths:
        ens_calc.append(NequIPCalculator.from_compiled_model(compile_path=compiled_path, device='cuda'))
    print(f"[INFO] Loaded {len(ens_calc)} calculators.")

    results_data = []

    print("[INFO] Calculating metrics for each structure...")
    for atoms in tqdm(atoms_list):
        if 'forces' not in atoms.arrays or 'stress' not in atoms.info:
            continue

        ref_forces = atoms.arrays['forces']
        ref_stress_voigt = atoms.info['stress']
        if ref_stress_voigt.shape != (6,):
            continue # Skip if stress is not in Voigt format
        ref_stress = voigt_to_matrix(ref_stress_voigt)

        # Get predictions from the ensemble
        pred_forces_list = []
        pred_stress_list = []
        for calc in ens_calc:
            atoms.calc = calc
            pred_forces_list.append(atoms.get_forces())
            pred_stress_list.append(atoms.get_stress(voigt=False))
        
        # Calculate ensemble average predictions
        avg_pred_forces = np.mean(np.array(pred_forces_list), axis=0)
        avg_pred_stress = np.mean(np.array(pred_stress_list), axis=0)
        
        # --- Calculate Force Error Statistics ---
        force_errors = avg_pred_forces - ref_forces
        force_error_norms = np.linalg.norm(force_errors, axis=1)
        
        # --- Store results for this structure ---
        struct_results = {
            'id': atoms.info['structure_id'],
            # Force Error Stats
            'error_kurtosis': kurtosis(force_error_norms),
            'error_gini': gini_coefficient(force_error_norms),
            # Force Losses
            'rmse_force': np.sqrt(np.mean(force_errors**2)),
            'tail_mse_force': tail_mse_numpy(avg_pred_forces, ref_forces),
            'tail_huber_force': tail_huber_loss_numpy(avg_pred_forces, ref_forces),
            'focal_mse_force': focal_mse_loss_numpy(avg_pred_forces, ref_forces),
            'angle_loss_force': force_angle_loss_numpy(avg_pred_forces, ref_forces),
            # Stress Losses (needs unsqueezing to add a batch dimension of 1)
            'shear_mae_stress': stress_shear_mae_numpy(avg_pred_stress[None, ...], ref_stress[None, ...]),
            'angle_loss_stress': stress_angle_loss_numpy(avg_pred_stress[None, ...], ref_stress[None, ...]),
        }
        results_data.append(struct_results)

    # Convert results to a pandas DataFrame for easier analysis
    df = pd.DataFrame(results_data).dropna()
    print("\n--- Analysis Results ---")
    print(df.head())

    # --- Plotting ---
    """
     loss_metrics = [
        'rmse_force', 'tail_mse_force', 'tail_huber_force', 
        'focal_mse_force', 'angle_loss_force', 'shear_mae_stress', 'angle_loss_stress'
    ]

    """
    loss_metrics = [
        'rmse_force', 'tail_mse_force', 'tail_huber_force', 
        'focal_mse_force', 'angle_loss_force', 'angle_loss_stress'
    ]
    
    # 1. Violin Plot for Distribution Comparison
    print("\n[INFO] Generating violin plot for distribution comparison...")
    # Melt the DataFrame to a long format suitable for seaborn
    df_melted = df.melt(id_vars=['id'], value_vars=loss_metrics, 
                        var_name='Metric', value_name='Value')

    plt.figure(figsize=(18, 10))
    sns.violinplot(x='Metric', y='Value', data=df_melted, cut=0)
    plt.yscale('log') # Use log scale to handle different value ranges
    plt.title('Comparison of Loss Metric Distributions', fontsize=20)
    plt.xlabel('Loss Metric', fontsize=14)
    plt.ylabel('Metric Value (Log Scale)', fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('loss_metrics_violin_plot.png')
    plt.close()

    # 2. Scatter plots vs Kurtosis and Gini
    stats_to_compare = ['error_kurtosis', 'error_gini']
    for stat_col in stats_to_compare:
        print(f"[INFO] Generating scatter plots vs {stat_col}...")
        num_metrics = len(loss_metrics)
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle(f'Loss Metrics vs. {stat_col.replace("_", " ").title()}', fontsize=20)
        axes = axes.flatten()

        for i, metric in enumerate(loss_metrics):
            ax = axes[i]
            ax.scatter(df[stat_col], df[metric], alpha=0.3, edgecolors='k', s=20)
            ax.set_xlabel(stat_col.replace("_", " ").title())
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Fit and plot a trendline
            m, b = np.polyfit(df[stat_col], df[metric], 1)
            ax.plot(df[stat_col], m*df[stat_col] + b, color='red', linestyle='--', linewidth=2)

        # Hide any unused subplots
        for i in range(num_metrics, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f'scatter_vs_{stat_col}.png')
        plt.close()

    # 3. Correlation Summary
    print("\n--- Correlation Summary (Spearman's Rho) ---")
    correlation_results = []
    for stat_col in stats_to_compare:
        for metric in loss_metrics:
            rho, pval = spearmanr(df[stat_col], df[metric])
            correlation_results.append({
                'Comparison': f'{metric}_vs_{stat_col}',
                'Spearman_Rho': rho,
                'P-Value': pval
            })
            
    corr_df = pd.DataFrame(correlation_results).sort_values('Spearman_Rho', ascending=False)
    print("Correlation coefficients close to 1.0 indicate that the loss function is highly")
    print("sensitive to the error concentration metric (Kurtosis or Gini).")
    print(corr_df.to_string())
    print("\n[INFO] Script finished. Check for 'loss_metrics_violin_plot.png' and scatter plot images.")