#!/usr/bin/env python
"""
Full workflow example for Forge Analysis Training module.

This script demonstrates:
1. Loading structures from database
2. Running error analysis with multiple metrics
3. Identifying difficult structures
4. Categorizing error causes
5. Exporting results for visualization and retraining
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the complete analysis workflow."""
    
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
    
    # Initialize database
    logger.info("Initializing database...")
    db = DatabaseManager()
    
    # ========================================================================
    # 2. Load Calculators
    # ========================================================================
    logger.info("Loading calculators...")
    
    # Example with dummy calculator - replace with your actual models
    from ase.calculators.emt import EMT
    calculators = [EMT()]
    
    # For real usage with multiple models:
    # from your_model import load_model
    # calculators = [
    #     load_model("model1.pth"),
    #     load_model("model2.pth"),
    #     load_model("model3.pth")
    # ]
    
    # ========================================================================
    # 3. Select Structures for Analysis
    # ========================================================================
    logger.info("Selecting structures for analysis...")
    
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
    
    # Limit for demo purposes
    if len(structure_ids) > 100:
        structure_ids = structure_ids[:100]
        logger.info("Limiting to first 100 structures for demo")
    
    # ========================================================================
    # 4. Register Custom Metrics
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
    # 5. Run Analysis
    # ========================================================================
    logger.info("Running analysis...")
    
    # Initialize analyser
    analyser = ErrorAnalyser(
        db_manager=db,
        calculators=calculators,
        ref_calc_name='vasp'  # or your reference calculator
    )
    
    # Run comprehensive analysis
    results = analyser.run(
        structure_ids=structure_ids,
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
    
    logger.info(f"Analysis complete! Analyzed {len(results.structure_metrics)} structures")
    
    # ========================================================================
    # 6. Explore Results
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
    # 7. Identify Difficult Structures
    # ========================================================================
    print("\n" + "-" * 80)
    print("DIFFICULT STRUCTURES")
    print("-" * 80)
    
    # Get top difficult structures
    difficult_ids = results.get_difficult_structures(top_n=10)
    
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
    # 8. Categorize Difficulty Causes
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
    # 9. Filter Structures by Criteria
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
    # 10. Visualizations
    # ========================================================================
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Plot error distribution
    try:
        all_errors = results.atom_metrics['force_error_mag'].values
        fig = plot_force_error_histogram(
            all_errors,
            bins=100,
            title="Force Error Distribution",
            save_path=output_dir / "force_error_distribution.png"
        )
        print("✓ Saved force error distribution plot")
    except Exception as e:
        print(f"✗ Could not create plots (matplotlib may not be installed): {e}")
    
    # ========================================================================
    # 11. Export Results
    # ========================================================================
    print("\n" + "-" * 80)
    print("EXPORTING RESULTS")
    print("-" * 80)
    
    # Save analysis results
    analyser.save(output_dir)
    print(f"✓ Saved analysis to {output_dir}/")
    
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
        print("✓ Exported difficult structures to extended XYZ")
    
    # Generate markdown report
    report = generate_markdown_report(results, output_path=output_dir / "report.md")
    print("✓ Generated markdown report")
    
    # ========================================================================
    # 12. Integration Example
    # ========================================================================
    print("\n" + "-" * 80)
    print("TRAINING INTEGRATION")
    print("-" * 80)
    
    # Example of how to use results for retraining
    retrain_ids = list(physics_issues)[:50]  # Top 50 problematic structures
    
    print(f"\nSelected {len(retrain_ids)} structures for retraining")
    print("These IDs can be used with db_to_allegro or db_to_mace:")
    print(f"  structure_ids = {retrain_ids[:5]} ...")
    
    # Example integration (commented out as it requires specific setup):
    # from forge.workflows.db_to_allegro import prepare_allegro_job
    # prepare_allegro_job(
    #     db, "retrain_job",
    #     structure_ids=retrain_ids,
    #     train_ratio=0.8,
    #     val_ratio=0.1,
    #     test_ratio=0.1
    # )
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Review the generated report.md")
    print("2. Visualize difficult_structures.extxyz in Ovito")
    print("3. Use structure IDs for targeted retraining")
    
    return results


if __name__ == "__main__":
    # Run the workflow
    results = main() 