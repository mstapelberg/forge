import argparse
import glob
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.stats import kurtosis

from nequip.ase import NequIPCalculator
from forge.core.database import DatabaseManager
from forge.analysis.training import ErrorAnalyser
from ase.io import write

# Loss functions from compare_custom_loss_metrics.py
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
        x -= np.amin(x)
    x += 1e-9 # Values cannot be 0
    x = np.sort(x)
    index = np.arange(1, x.shape[0] + 1)
    n = x.shape[0]
    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

def analyze(args):
    """
    Phase 1: Run analysis, compute metrics, and save data.
    """
    print("--- Starting Phase 1: Analysis ---")
    
    print("[INFO] Initializing DatabaseManager...")
    db = DatabaseManager(debug=False)

    print("[INFO] Loading ensemble of calculators...")
    ens_calc = []
    # Note: You might need to adjust this path depending on your project structure
    compiled_paths = glob.glob('forge/scratch/data/potentials/compiled_models/*.pt2')
    if not compiled_paths:
        print("[ERROR] No compiled models found. Please check the path.")
        return
        
    for compiled_path in compiled_paths:
        ens_calc.append(NequIPCalculator.from_compiled_model(compile_path=compiled_path, device=args.device))
    print(f"[INFO] Loaded {len(ens_calc)} calculators.")

    print("[INFO] Initializing ErrorAnalyser...")
    analyser = ErrorAnalyser(db, ens_calc)

    print("[INFO] Finding relevant structure IDs...")
    structure_ids = db.find_structures_by_metadata(metadata_filters={'generation': 0}, operator='>=')
    dimer_ids = db.find_structures_by_metadata(metadata_filters={'config_type': 'dimer'})
    valid_structure_ids = [sid for sid in structure_ids if sid not in dimer_ids]
    print(f"[INFO] Found {len(valid_structure_ids)} structures to process.")

    print("[INFO] Running base analysis with ErrorAnalyser...")
    df_struct, df_atom = analyser.run(structure_ids=valid_structure_ids, batch_size=args.batch_size)

    print("[INFO] Computing custom loss metrics...")
    custom_metrics_data = []
    for structure_id in tqdm(df_struct['structure_id']):
        cache_item = analyser.results_cache.get(int(structure_id))
        if not cache_item:
            continue

        atoms = cache_item['atoms'].copy()
        if 'forces' not in atoms.arrays:
            continue

        ref_forces = atoms.arrays['forces']
        
        # Re-calculate ensemble predictions to get the mean
        pred_forces_list = []
        for calc in ens_calc:
            atoms.calc = calc
            pred_forces_list.append(atoms.get_forces())
        avg_pred_forces = np.mean(np.array(pred_forces_list), axis=0)
        
        force_errors = avg_pred_forces - ref_forces
        force_error_norms = np.linalg.norm(force_errors, axis=1)

        custom_metrics = {
            'structure_id': structure_id,
            'error_kurtosis': kurtosis(force_error_norms),
            'error_gini': gini_coefficient(force_error_norms),
            'tail_mse_force': tail_mse_numpy(avg_pred_forces, ref_forces),
            'tail_huber_force': tail_huber_loss_numpy(avg_pred_forces, ref_forces),
            'focal_mse_force': focal_mse_loss_numpy(avg_pred_forces, ref_forces),
            'angle_loss_force': force_angle_loss_numpy(avg_pred_forces, ref_forces),
        }
        custom_metrics_data.append(custom_metrics)

    df_custom = pd.DataFrame(custom_metrics_data)
    
    # Merge custom metrics into the main structure dataframe
    df_struct_full = pd.merge(df_struct, df_custom, on='structure_id')

    print("[INFO] Saving analysis results...")
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    df_struct_full.to_csv(output_path / 'structural_metrics.csv', index=False)
    # df_atom is large, consider parquet for efficiency
    if not df_atom.empty:
        df_atom.to_parquet(output_path / 'atomic_metrics.parquet')
    
    with open(output_path / 'results_cache.pkl', 'wb') as f:
        pickle.dump(analyser.results_cache, f)

    print(f"--- Analysis complete. Results saved to: {output_path.resolve()} ---")


def export(args):
    """
    Phase 2: Load analysis data and export top N structures to an .extxyz file.
    """
    print("--- Starting Phase 2: Export for Visualization ---")
    analysis_path = Path(args.analysis_path)
    
    print(f"[INFO] Loading data from {analysis_path}...")
    df_struct_csv_path = analysis_path / 'structural_metrics.csv'
    df_struct_pkl_path = analysis_path / 'df_struct.pkl'
    cache_path = analysis_path / 'results_cache.pkl'

    if not cache_path.exists():
        print(f"[ERROR] 'results_cache.pkl' not found in {analysis_path}.")
        return

    if df_struct_csv_path.exists():
        print("[INFO] Found 'structural_metrics.csv'. Loading data from CSV.")
        df_struct = pd.read_csv(df_struct_csv_path)
    elif df_struct_pkl_path.exists():
        print("[INFO] Found 'df_struct.pkl'. Loading data from pickle.")
        df_struct = pd.read_pickle(df_struct_pkl_path)
    else:
        print(f"[ERROR] Structural data ('structural_metrics.csv' or 'df_struct.pkl') not found in {analysis_path}.")
        return
        
    with open(cache_path, 'rb') as f:
        results_cache = pickle.load(f)
        
    if args.metric not in df_struct.columns:
        print(f"[ERROR] Metric '{args.metric}' not found in the structural data.")
        print(f"Available metrics: {list(df_struct.columns)}")
        return

    print(f"[INFO] Sorting structures by '{args.metric}'...")
    df_sorted = df_struct.sort_values(by=args.metric, ascending=False)
    
    top_n_ids = df_sorted.head(args.top_n)['structure_id'].tolist()
    print(f"[INFO] Selected top {args.top_n} structure IDs: {top_n_ids}")

    atoms_to_write = []
    for sid in top_n_ids:
        # The cache keys might be integers
        sid_int = int(sid)
        if sid_int in results_cache:
            # We need to use the analyser's method to populate atoms with error data
            atoms = results_cache[sid_int]['atoms'].copy()
            atoms.info['structure_id'] = sid_int
            atoms.info['metric_name'] = args.metric
            atoms.info['metric_value'] = df_sorted[df_sorted['structure_id'] == sid_int][args.metric].iloc[0]
            atoms.arrays['force_error_magnitude'] = results_cache[sid_int]['force_error_magnitudes']
            # Recompute force error vectors as they are not stored directly
            if 'model_forces' in atoms.arrays and 'forces' in atoms.arrays:
                atoms.arrays['force_error_vectors'] = atoms.arrays['forces'] - atoms.arrays['model_forces']

            atoms_to_write.append(atoms)
        else:
            print(f"[WARN] Structure ID {sid_int} not found in results_cache.")
            
    if not atoms_to_write:
        print("[ERROR] No structures to write. Exiting.")
        return

    print(f"[INFO] Writing {len(atoms_to_write)} structures to {args.output_xyz}...")
    write(args.output_xyz, atoms_to_write)
    
    print(f"--- Export complete. File saved to: {Path(args.output_xyz).resolve()} ---")


def main():
    parser = argparse.ArgumentParser(
        description="Identify and visualize defect hotspots from force-field analysis.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='phase', required=True, help='Select the phase to run')

    # --- Arguments for Phase 1: Analyze ---
    parser_analyze = subparsers.add_parser('analyze', help='Run analysis and compute metrics.')
    parser_analyze.add_argument(
        '--output-path', type=str, default='analysis_results', 
        help='Directory to save analysis results (default: analysis_results).'
    )
    parser_analyze.add_argument(
        '--batch-size', type=int, default=128, 
        help='Batch size for ErrorAnalyser (default: 128).'
    )
    parser_analyze.add_argument(
        '--device', type=str, default='cuda', 
        help='Device to run calculators on (e.g., "cuda", "cpu") (default: cuda).'
    )
    parser_analyze.set_defaults(func=analyze)

    # --- Arguments for Phase 2: Export ---
    parser_export = subparsers.add_parser('export', help='Export top N structures for visualization.')
    parser_export.add_argument(
        '--analysis-path', type=str, required=True, 
        help='Path to the directory with saved analysis results from Phase 1.'
    )
    parser_export.add_argument(
        '--output-xyz', type=str, default='hotspot_structures.extxyz', 
        help='Path for the output .extxyz file (default: hotspot_structures.extxyz).'
    )
    parser_export.add_argument(
        '--top-n', type=int, default=20, 
        help='Number of top structures to export (default: 20).'
    )
    parser_export.add_argument(
        '--metric', type=str, default='Kurtosis', 
        help='Metric to sort structures by (e.g., "MoranI", "error_gini") (default: error_kurtosis).'
    )
    parser_export.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()