#!/usr/bin/env python3
"""
Example: Heterogeneity analysis for potential energy landscape characterization.

This script demonstrates how to:
1. Run NEB calculations with local environment tracking
2. Analyze heterogeneity in the PEL
3. Correlate local composition with barriers
4. Export rich datasets for further analysis/ML

Optimized for 2000-atom systems with NN-only calculations.

Confidence: 9/10
"""

import torch
from pathlib import Path
from forge.workflows.hybrid_neb import HybridNEBWorkflow
from forge.workflows.neb_heterogeneity import integrate_heterogeneity_analysis


def run_heterogeneity_workflow():
    """
    Run complete workflow optimized for heterogeneity quantification.
    """
    print("=== Heterogeneity Analysis Workflow ===")
    print("Optimized for large systems (2000 atoms)")
    print("NN-only calculations for better convergence\n")
    
    # Configuration - adjust to your setup
    model_path = "../../data/potentials/allegro/exploit_rmax6.00_lmax2_layers2_mlp384_seed42.nequip.zip"
    output_dir = "../../data/pel_het_search/heterogeneity_analysis"
    
    # Your compositions to test
    compositions_to_test = [
        {'V': 0.988, 'Cr': 0.00, 'Ti': 0.012, 'W': 0.00, 'Zr': 0.00},
        {'V': 0.92, 'Cr': 0.04, 'Ti': 0.04, 'W': 0.00, 'Zr': 0.00},
        {'V': 0.76, 'Cr': 0.17, 'Ti': 0.07, 'W': 0.00, 'Zr': 0.00},
        {'V': 0.76, 'Cr': 0.12, 'Ti': 0.12, 'W': 0.00, 'Zr': 0.00},
        {'V': 0.76, 'Cr': 0.07, 'Ti': 0.17, 'W': 0.00, 'Zr': 0.00},
        {'V': 0.92, 'Cr': 0.02, 'Ti': 0.02, 'W': 0.02, 'Zr': 0.02},
        {'V': 0.76, 'Cr': 0.06, 'Ti': 0.06, 'W': 0.06, 'Zr': 0.06},
    ]
    
    # Initialize workflow
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        output_dir=output_dir,
        backend="allegro",
        species_to_type_name={'Ti': 0, 'V': 1, 'Cr': 2, 'Zr': 3, 'W': 4}
    )
    
    # ===== STEP 1: Create structures =====
    print("\n=== Step 1: Creating structures ===")
    structures = workflow.create_initial_structures(
        compositions=compositions_to_test,
        crystal_type='bcc',
        dimensions=[10, 10, 10],  # ~2000 atoms
        lattice_constant=3.01,
        cubic=True
    )
    print(f"Created {len(structures)} structures with ~{len(structures[0])} atoms each")
    
    # ===== STEP 2: Optimize with hybrid MCMC-MD =====
    print("\n=== Step 2: Optimizing structures ===")
    optimized_structures = workflow.optimize_with_hybrid_mcmc(
        structures=structures,
        temperature=873.15,  # 600°C
        n_steps=20,  # Adjust based on your system
        md_steps_per_cycle=1000,
        mc_steps_per_cycle=10000,
        convergence_window=1000,
        energy_threshold=0.002,  # 2 meV/atom
        final_cell_relax=True,
        fmax=0.05,
        steps=500
    )
    print(f"Optimized {len(optimized_structures)} structures")
    
    # ===== STEP 3: Run NEB calculations (NN only) =====
    print("\n=== Step 3: Running NEB calculations (NN only) ===")
    
    # Key parameters for heterogeneity analysis:
    # - Sample multiple vacancy sites to capture environment diversity
    # - Use only NN (better convergence on 2000-atom systems)
    # - Moderate n_nearest to balance coverage vs computation time
    
    neb_results = workflow.run_neb_calculations(
        structures=optimized_structures,
        vacancy_indices=None,  # Will auto-sample across the structure
        n_nearest=4,  # 4 NN jumps per vacancy site
        n_next_nearest=0,  # Skip NNN (convergence issues)
        num_images=5,
        neb_method="dyneb",
        climb=True,
        relax_fmax=0.01,
        relax_steps=150,
        neb_fmax=0.05,  # Reasonable for 2000 atoms
        neb_steps=250,  # Give enough steps for convergence
        save_xyz=True,
        verbose=1
    )
    
    print(f"\nCompleted {len(neb_results)} NEB calculations")
    successful = sum(1 for r in neb_results if r.get('success'))
    print(f"Success rate: {successful}/{len(neb_results)} ({successful/len(neb_results)*100:.1f}%)")
    
    # ===== STEP 4: Standard NEB analysis =====
    print("\n=== Step 4: Standard NEB analysis ===")
    analysis = workflow.analyze_results(
        save_plots=True,
        plot_barriers=True,
        plot_compositions=False  # Skip if not using composition generation
    )
    
    # ===== STEP 5: Advanced heterogeneity analysis =====
    print("\n=== Step 5: Heterogeneity analysis ===")
    
    # Define neighbor shells for local environment characterization
    # For BCC: 1st shell ~2.6Å, 2nd shell ~3.0Å, 3rd shell ~4.3Å
    shell_cutoffs = [2.8, 3.5, 4.5]  # Slightly generous cutoffs
    
    het_results = integrate_heterogeneity_analysis(
        workflow_results={
            'analysis_results': analysis,
            'compositions': compositions_to_test
        },
        optimized_structures=optimized_structures,
        output_dir=Path(output_dir) / "heterogeneity",
        shell_cutoffs=shell_cutoffs
    )
    
    # ===== STEP 6: Print summary =====
    print("\n" + "="*60)
    print("HETEROGENEITY ANALYSIS SUMMARY")
    print("="*60)
    
    global_het = het_results['global_heterogeneity']
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Vacancy sites analyzed: {het_results['n_vacancy_sites']}")
    print(f"  Total barriers measured: {global_het.get('n_total_barriers', 0)}")
    print(f"  Avg barriers per site: {global_het.get('avg_barriers_per_site', 0):.1f}")
    
    print(f"\n🎯 Global Heterogeneity Metrics:")
    print(f"  Site-to-site variation (std): {global_het.get('site_mean_barriers_std', 0):.4f} eV")
    print(f"  Site-to-site range: {global_het.get('site_mean_barriers_range', 0):.4f} eV")
    print(f"  Coefficient of variation: {global_het.get('site_mean_barriers_cv', 0):.4f}")
    
    print(f"\n📈 Within-Site Heterogeneity:")
    print(f"  Avg barrier range per site: {global_het.get('avg_within_site_range', 0):.4f} eV")
    print(f"  Avg std per site: {global_het.get('avg_within_site_std', 0):.4f} eV")
    
    print(f"\n⚡ Overall Barrier Statistics:")
    print(f"  Global mean: {global_het.get('global_mean_barrier', 0):.4f} eV")
    print(f"  Global std: {global_het.get('global_std_barrier', 0):.4f} eV")
    print(f"  Range: {global_het.get('global_min_barrier', 0):.4f} - {global_het.get('global_max_barrier', 0):.4f} eV")
    
    print(f"\n💾 Outputs:")
    print(f"  Main results: {output_dir}/")
    print(f"  Heterogeneity data: {output_dir}/heterogeneity/")
    print(f"    - heterogeneity_dataset.json  (complete dataset)")
    print(f"    - heterogeneity_dataset.csv   (flattened for analysis)")
    print(f"    - heterogeneity_overview.png  (main plots)")
    print(f"    - environment_barrier_correlation.png")
    print(f"    - environment_clusters.png    (if enough data)")
    
    print("\n" + "="*60)
    print("Workflow completed successfully!")
    print("="*60)
    
    return het_results


def analyze_existing_dataset(dataset_path: str):
    """
    Load and analyze an existing heterogeneity dataset.
    
    Args:
        dataset_path: Path to heterogeneity_dataset.json
    """
    import json
    from forge.workflows.neb_heterogeneity import NEBHeterogeneityAnalyzer
    
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nDataset contains:")
    print(f"  {data['metadata']['n_structures']} structures")
    print(f"  {data['metadata']['n_vacancy_sites']} vacancy sites")
    print(f"  {data['metadata']['n_total_barriers']} barriers")
    
    # You can now analyze this data further
    # For example, train ML models, correlation analysis, etc.
    
    return data


def example_downstream_analysis(csv_path: str):
    """
    Example of downstream analysis using the CSV export.
    
    Args:
        csv_path: Path to heterogeneity_dataset.csv
    """
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("pandas/seaborn required for this example")
        return
    
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Example 1: Correlation between 1st shell composition and barriers
    print("\n=== Example 1: 1st shell composition effects ===")
    shell1_cols = [col for col in df.columns if col.startswith('shell1_') 
                   and col.endswith(('V', 'Cr', 'Ti', 'W', 'Zr'))]
    
    if shell1_cols and 'barrier_mean' in df.columns:
        correlations = df[shell1_cols + ['barrier_mean']].corr()['barrier_mean'].drop('barrier_mean')
        print("\nCorrelations with mean barrier:")
        for col, corr in correlations.sort_values(key=abs, ascending=False).items():
            element = col.split('_')[-1]
            print(f"  {element} in 1st shell: {corr:+.4f}")
    
    # Example 2: Identify most/least heterogeneous sites
    print("\n=== Example 2: Most heterogeneous sites ===")
    top_het = df.nlargest(5, 'barrier_range')[['vacancy_index', 'vacancy_element', 
                                                 'barrier_mean', 'barrier_range']]
    print(top_het.to_string(index=False))
    
    # Example 3: Compare heterogeneity across compositions
    if 'structure_formula' in df.columns:
        print("\n=== Example 3: Heterogeneity by structure ===")
        structure_het = df.groupby('structure_formula').agg({
            'barrier_mean': ['mean', 'std'],
            'barrier_range': 'mean',
            'vacancy_index': 'count'
        })
        print(structure_het)
    
    print("\n✅ Downstream analysis complete!")
    print("You can now:")
    print("  - Train ML models to predict barriers from local environment")
    print("  - Identify composition rules for low-heterogeneity alloys")
    print("  - Quantify descriptor importance for PEL heterogeneity")
    

def main():
    """Run the heterogeneity analysis workflow."""
    
    # Option 1: Run full workflow
    results = run_heterogeneity_workflow()
    
    # Option 2: Analyze existing data (uncomment to use)
    # dataset_path = "../../data/pel_het_search/heterogeneity_analysis/heterogeneity/heterogeneity_dataset.json"
    # analyze_existing_dataset(dataset_path)
    
    # Option 3: Downstream analysis on CSV (uncomment to use)
    # csv_path = "../../data/pel_het_search/heterogeneity_analysis/heterogeneity/heterogeneity_dataset.csv"
    # example_downstream_analysis(csv_path)
    

if __name__ == "__main__":
    main()



