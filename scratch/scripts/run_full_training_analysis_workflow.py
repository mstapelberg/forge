#!/usr/bin/env python
"""
Full workflow example for Forge Analysis Training module.

This script demonstrates:
1. Loading structures from database OR local XYZ files
2. Running error analysis with multiple metrics
3. Identifying difficult structures
4. Categorizing error causes
5. Exporting results for visualization and retraining
"""
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from ase.io import read
from typing import List
from ase import Atoms


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_atoms_from_xyz(directory: str) -> List[Atoms]:
    """Loads all atoms from .xyz files in a directory, ensuring they have structure_id.

    Args:
        directory (str): Path to the directory containing .xyz files.

    Returns:
        List[Atoms]: A list of valid ASE Atoms objects.
    """
    xyz_files = glob.glob(str(Path(directory) / '*.xyz'))
    all_atoms = []
    logger.info(f"Found {len(xyz_files)} '.xyz' files in '{directory}'")

    for fpath in xyz_files:
        try:
            atoms_list = read(fpath, index=':')
            for atoms in atoms_list:
                if 'structure_id' in atoms.info and 'REF_energy' in atoms.info and 'REF_force' in atoms.arrays:
                    all_atoms.append(atoms)
                else:
                    logger.warning(
                        f"Skipping structure in {fpath} due to missing "
                        "'structure_id', 'energy', or 'forces'."
                    )
        except Exception as e:
            logger.error(f"Error reading {fpath}: {e}")
            
    logger.info(f"Successfully loaded {len(all_atoms)} structures with required info.")
    return all_atoms


def main():
    """Run the complete analysis workflow."""
    
    parser = argparse.ArgumentParser(description="Forge Analysis Training Workflow")
    parser.add_argument(
        "--xyz-path",
        type=str,
        default=None,
        help="Path to a directory of XYZ files to use instead of querying the database."
    )
    args = parser.parse_args()

    # ========================================================================
    # 1. Setup and Initialization
    # ========================================================================
    print("=" * 80)
    print("Forge Analysis Training - Full Workflow Example")
    print("=" * 80)
    
    from forge.core.database import DatabaseManager
    from forge.analysis.training import (
        ErrorAnalyser, register_metric,
        export_structures_to_extxyz, generate_markdown_report,
        plot_force_error_histogram
    )
    
    # Initialize database (still needed for some operations)
    logger.info("Initializing database...")
    db = DatabaseManager()
    
    # ========================================================================
    # 2. Load Calculators
    # ========================================================================
    logger.info("Loading calculators...")
    
    # Example with dummy calculator - replace with your actual models
    from nequip.ase import NequIPCalculator
    import glob
    compiled_paths = glob.glob('../data/potentials/compiled_models/*.nequip.pt2')
    #compiled_path = '../data/potentials/compiled_models/base_force_angle_focal_stress_shear_rare_sampling_softadapt.nequip.pt2'
    calculators = [NequIPCalculator.from_compiled_model(compile_path=compiled_path, device='cuda') for compiled_path in compiled_paths]
    
    # ========================================================================
    # 3. Select Structures for Analysis
    # ========================================================================
    logger.info("Selecting structures for analysis...")
    
    atoms_from_xyz = None
    structure_ids = None

    if args.xyz_path:
        logger.info(f"Loading structures from XYZ files in: {args.xyz_path}")
        atoms_from_xyz = load_atoms_from_xyz(args.xyz_path)
        if not atoms_from_xyz:
            logger.error("No valid structures loaded from XYZ files. Aborting.")
            return
    else:
        logger.info("Querying structures from the database...")
        # Query structures by metadata
        structure_ids = db.find_structures_by_metadata(
            metadata_filters={'generation': 0},
            operator='>='
        )
        # Exclude specific types if needed
        dimer_ids = db.find_structures_by_metadata(
            metadata_filters={'config_type': 'dimer'}
        )
        structure_ids = [sid for sid in structure_ids if sid not in dimer_ids]
        logger.info(f"Found {len(structure_ids)} structures for analysis")

    # ========================================================================
    # 4. Define Output and Load/Run Analysis
    # ========================================================================
    output_dir = Path("analysis_output_full")
    output_dir.mkdir(exist_ok=True)
    
    results = None
    analyser = ErrorAnalyser(
        db_manager=db,
        calculators=calculators,
        ref_calc_name='vasp'  # or your reference calculator
    )

    # Check if results already exist to avoid re-running analysis
    if (output_dir / "structure_metrics.csv").exists() and \
       (output_dir / "atom_metrics.csv").exists() and \
       (output_dir / "results_cache.pkl").exists():
        
        logger.info(f"Found existing analysis in {output_dir}. Loading results.")
        from forge.analysis.training.utils import load_analysis_results
        from forge.analysis.training import AnalysisResults
        
        loaded_data = load_analysis_results(output_dir)
        results = AnalysisResults(
            structure_metrics=loaded_data['structure_metrics'],
            atom_metrics=loaded_data['atom_metrics'],
            metadata=loaded_data.get('metadata', {}),
            results_cache=loaded_data.get('results_cache', {})
        )
        analyser._last_results = results

    if results is None:
        logger.info("No existing analysis found. Running new analysis.")
        
        # ========================================================================
        # 5. Register Custom Metrics
        # ========================================================================
        logger.info("Registering custom metrics...")
        
        @register_metric("normalized_force_error")
        def normalized_force_error(pred, ref, scale=1.0):
            """Calculate normalized force errors."""
            error_vectors = pred - ref
            error_mags = np.linalg.norm(error_vectors, axis=1)
            ref_mags = np.linalg.norm(ref, axis=1)
            
            # Avoid division by zero
            mask = ref_mags > 1e-6
            normalized = np.zeros_like(error_mags)
            normalized[mask] = error_mags[mask] / ref_mags[mask]
            
            return {
                "normalized_rmse_metric": np.sqrt(np.mean(normalized**2)),
                "normalized_max_metric": np.max(normalized),
                "fraction_above_threshold_metric": np.mean(normalized > 0.1)
            }
        
        # ========================================================================
        # 6. Run Analysis
        # ========================================================================
        logger.info("Running analysis...")
        
        results = analyser.run(
            structure_ids=structure_ids,
            atoms_list=atoms_from_xyz,
            batch_size=16,
            metrics=[
                # Built-in metrics
                "force_stats",
                "energy_stats",
                "tail_mse",
                "focal_mse",
                # Custom metric
                "normalized_force_error"
            ],
            spatial_k=12,
            dbscan_eps=2.5,
            dbscan_min_samples=3,
            check_geometry_sanity=True,
            difficulty_weights={
                'w_heavy_tail': 1.5,
                'w_spatial': 1.0,
                'w_ensemble': 1.0
            }
        )
        
        # Save main analysis results only when they are newly generated
        analyser.save(output_dir)
        print(f"✓ Saved base analysis results to {output_dir}/")

    logger.info(f"Analysis results ready. Found metrics for {len(results.structure_metrics)} structures.")
    
    # ========================================================================
    # 7. Explore Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    # Summary statistics
    summary = results.summary_statistics()
    print("\nKey Metrics Summary:")
    print("-" * 40)
    for metric, stats in summary['metrics'].items():
        if 'metric' in metric and metric != 'structure_id':
            print(f"{metric:30s}: mean={stats['mean']:8.4f}, max={stats['max']:8.4f}")
    
    # ========================================================================
    # 8. Identify Difficult Structures
    # ========================================================================
    print("\n" + "-" * 80)
    print("DIFFICULT STRUCTURES")
    print("-" * 80)
    
    # Get top difficult structures
    difficult_ids = results.get_difficult_structures(top_n=10, metric='force_kurtosis_metric')
    
    print("\nTop 10 Most Difficult Structures:")
    for i, sid in enumerate(difficult_ids):
        report = results.get_structure_report(sid)
        metrics = report['structure_metrics']
        print(f"\n{i+1}. Structure {sid} ({metrics['formula']}):")
        print(f"   - Difficulty score: {metrics['difficulty_metric']:.3f}")
        print(f"   - Force RMSE: {metrics.get('force_rmse_metric', 0):.4f} eV/Å")
        print(f"   - Kurtosis: {metrics.get('force_kurtosis_metric', 0):.2f}")
        print(f"   - Error clusters: {metrics.get('n_error_clusters_metric', 0)}")
    
    # ========================================================================
    # 9. Categorize Difficulty Causes
    # ========================================================================
    print("\n" + "-" * 80)
    print("DIFFICULTY CATEGORIZATION")
    print("-" * 80)
    
    from forge.analysis.training.difficulty import categorize_difficulty_causes
    
    # Categorize all structures
    categories = []
    for _, row in results.structure_metrics.iterrows():
        cat = categorize_difficulty_causes(row.to_dict())
        categories.append(cat)
    
    cat_df = pd.DataFrame(categories)
    cat_summary = cat_df.sum()
    
    print("\nDifficulty Categories:")
    for category, count in cat_summary.items():
        percentage = 100 * count / len(cat_df) if len(cat_df) > 0 else 0
        print(f"  {category:25s}: {count:3d} structures ({percentage:5.1f}%)")
    
    # ========================================================================
    # 10. Filter Structures by Criteria
    # ========================================================================
    print("\n" + "-" * 80)
    print("STRUCTURE FILTERING")
    print("-" * 80)
    
    # High kurtosis (heavy-tailed errors)
    high_kurtosis_ids = results.filter_by_score('force_kurtosis_metric', 5.0)
    print(f"\nStructures with high kurtosis (>5.0): {len(high_kurtosis_ids)}")
    
    # Spatial clustering
    clustered_ids = results.filter_by_score('n_error_clusters_metric', 1)
    print(f"Structures with error clusters: {len(clustered_ids)}")
    
    # Combined criteria
    physics_issues = set(high_kurtosis_ids) & set(clustered_ids)
    print(f"Structures with both issues: {len(physics_issues)}")
    
    # ========================================================================
    # 9.5 Identify Problematic Structures (Bad vs. Rare)
    # ========================================================================
    print("\n" + "-" * 80)
    print("IDENTIFYING PROBLEMATIC STRUCTURES (BAD vs. RARE)")
    print("-" * 80)

    # 1. Identify "Bad" structures (geometry issues)
    bad_structure_ids = []
    if 'geometry_valid' in results.structure_metrics.columns:
        bad_structure_ids = results.structure_metrics[
            results.structure_metrics['geometry_valid'] == False
        ]['structure_id'].tolist()
        print(f"\nFound {len(bad_structure_ids)} 'bad' structures with geometry issues.")
        if bad_structure_ids:
            print("These should be excluded from future training sets.")
            print(f"Example bad IDs: {bad_structure_ids[:5]}")

    # 2. Identify "Rare" structures (difficult for the model)
    # These are structures that are valid but have high error/uncertainty.
    # We can define them as the top 5% most difficult structures.
    difficulty_threshold = results.structure_metrics['difficulty_metric'].quantile(0.95)
    rare_structure_ids = results.filter_by_score(
        'difficulty_metric',
        threshold=difficulty_threshold
    )

    # Ensure bad structures are not included in the rare list
    rare_structure_ids = [sid for sid in rare_structure_ids if sid not in bad_structure_ids]

    print(f"\nFound {len(rare_structure_ids)} 'rare' structures (top 5% difficulty, difficulty > {difficulty_threshold:.3f}).")
    if rare_structure_ids:
        print("These are candidates for enhanced sampling or active learning.")
        print(f"Example rare IDs: {rare_structure_ids[:5]}")

    # 3. Get debug information for a problematic structure
    if rare_structure_ids:
        example_id = rare_structure_ids[0]
        report = results.get_structure_report(example_id)
        print(f"\n--- Debug Report for Rare Structure {example_id} ---")
        
        metrics = report['structure_metrics']
        causes = categorize_difficulty_causes(metrics)
        
        print(f"  Formula: {metrics['formula']}")
        print(f"  Difficulty Score: {metrics['difficulty_metric']:.3f}")
        print("  Component Scores:")
        print(f"    - Heavy-tail: {metrics.get('score_heavy_tail_metric', 0):.3f}")
        print(f"    - Spatial: {metrics.get('score_spatial_metric', 0):.3f}")
        print(f"    - Ensemble: {metrics.get('score_ensemble_metric', 0):.3f}")
        print("  Likely Causes:")
        for cause, is_present in causes.items():
            if is_present:
                print(f"    - {cause}")
        print("-" * 50)

    # ========================================================================
    # 10. Visualizations
    # ========================================================================
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    # Plot error distribution
    try:
        all_errors = results.atom_metrics['force_error_mag'].values
        fig = plot_force_error_histogram(
            all_errors,
            bins=100,
            title="Force Error Distribution",
            save_path=output_dir / "force_error_distribution.png"
        )
        print("Saved force error distribution plot")
    except Exception as e:
        print(f"Could not create plots (matplotlib may not be installed): {e}")
    
    # ========================================================================
    # 11. Export Results
    # ========================================================================
    print("\n" + "-" * 80)
    print("EXPORTING RESULTS")
    print("-" * 80)
    
    # Save the bad and rare IDs to their own files for easy access
    with open(output_dir / "bad_structure_ids.json", "w") as f:
        json.dump(bad_structure_ids, f, indent=2)
    logger.info(f"Saved {len(bad_structure_ids)} bad structure IDs to {output_dir / 'bad_structure_ids.json'}")

    with open(output_dir / "rare_structure_ids.json", "w") as f:
        json.dump(rare_structure_ids, f, indent=2)
    logger.info(f"Saved {len(rare_structure_ids)} rare structure IDs to {output_dir / 'rare_structure_ids.json'}")

    # Export difficult structures for visualization
    difficult_atoms = []
    for sid in difficult_ids[:10]:
        if sid in results.results_cache:
            difficult_atoms.append(results.results_cache[sid]['atoms'])
    
    if difficult_atoms:
        export_structures_to_extxyz(
            difficult_atoms,
            results,
            output_dir / "difficult_structures.extxyz"
        )
        print("Exported difficult structures to extended XYZ")
    
    # Generate markdown report
    report = generate_markdown_report(results, output_path=output_dir / "report.md")
    print("Generated markdown report")
    
    # ========================================================================
    # 12. Integration Example
    # ========================================================================
    print("\n" + "-" * 80)
    print("TRAINING INTEGRATION")
    print("-" * 80)
    
    # Example of how to use results for retraining
    # We use the 'rare' structures identified as candidates for the next active learning cycle.
    # We also ensure we don't include 'bad' structures.
    
    current_structure_ids = results.structure_metrics['structure_id'].tolist()
    retrain_ids = [sid for sid in rare_structure_ids if sid not in bad_structure_ids]
    
    print(f"\nSelected {len(retrain_ids)} rare/difficult structures for retraining.")
    print("These IDs can be used with db_to_allegro or db_to_mace:")
    if retrain_ids:
        print(f"  structure_ids = {retrain_ids[:5]} ...")

    # You might also want to create a list of all valid IDs to keep for future training,
    # excluding the bad ones.
    all_valid_ids = [sid for sid in current_structure_ids if sid not in bad_structure_ids]
    print(f"\nTotal valid structures (excluding bad geometry): {len(all_valid_ids)}")
    
    # Example integration (commented out as it requires specific setup):
    # from forge.workflows.db_to_allegro import prepare_allegro_job
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Review the generated report.md")
    print("2. Visualize difficult_structures.extxyz in Ovito")
    print("3. Use the 'bad_structure_ids' list to clean your database")
    print("4. Use the 'rare_structure_ids' list for targeted retraining")
    
    return results


if __name__ == "__main__":
    # Run the workflow
    results = main() 