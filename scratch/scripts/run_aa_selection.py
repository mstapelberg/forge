import warnings
from pathlib import Path
import ase.io
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# Make sure the script can find the forge module
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from forge.analysis.aa_selection import AAAnalyzer
    from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase
    from mace.calculators.mace import MACECalculator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that 'mace-torch' is installed and that the 'forge' package is in your PYTHONPATH.")
    sys.exit(1)

def main():
    # --- Start of User Configuration ---
    
    # Directory containing .xyz trajectory files.
    #INPUT_DIR = "../data/adversarial_attacks/gen_8_no_shake_rmse_10_debug"
    INPUT_DIR = '../data/adversarial_attacks/gen_8_no_shake_rmse_all'
    
    # Directory to save VASP job folders and analysis plots.
    #OUTPUT_DIR = "../data/adversarial_attacks/gen_8_no_shake_rmse_10_debug_vasp_jobs"
    OUTPUT_DIR = "../data/adversarial_attacks/gen_8_no_shake_rmse_all_vasp_jobs"
    
    # List of paths to the MACE model files (.model) for the ensemble.
    MACE_MODEL_PATHS = [
        "../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model",
        "../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_1_pr_stagetwo.model",
        "../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_2_pr_stagetwo.model",
        "../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_3_pr_stagetwo.model",
        "../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_4_pr_stagetwo.model",
    ]
    
    # Number of diverse structures to select.
    N_SELECT = 1590
    
    # Device to run MACE on ('cpu' or 'cuda').
    DEVICE = "cuda"

    # Profile names for VASP and HPC job script generation
    VASP_PROFILE_NAME = "static" # Example: 'rlx-fast' or 'static'
    HPC_PROFILE_NAME = "PSFC-GPU"  # Example: 'Perlmutter-CPU' or 'local'

    # --- End of User Configuration ---

    # --- 1. Setup Paths and Load Atoms ---
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading all .xyz files from '{input_path}' (skipping first frame of each trajectory)...")
    all_atoms = []
    for xyz_file in input_path.glob("*.xyz"):
        try:
            # Skip the first frame (index 0) since it's the unoptimized original structure
            frames = ase.io.read(xyz_file, index="1:")
            all_atoms.extend(frames)
        except Exception as e:
            warnings.warn(f"Could not read {xyz_file.name}. Error: {e}")

    if not all_atoms:
        print("Error: No structures loaded. Exiting.")
        sys.exit(1)
    print(f"Total optimized structures loaded: {len(all_atoms)}\n")

    # --- 2. Initialize MACE (Single Model for Embeddings) ---
    print("Initializing MACE calculator (single model for embeddings generation)...")
    if not isinstance(MACE_MODEL_PATHS, list) or len(MACE_MODEL_PATHS) < 1:
        print("Error: MACE_MODEL_PATHS must be a list with at least one model.")
        sys.exit(1)
        
    # Use only the first model for embeddings generation
    calculator = MACECalculator(model_paths=MACE_MODEL_PATHS[0], device=DEVICE, default_dtype="float32")
    print(f"Using model: {MACE_MODEL_PATHS[0]}")

    # --- 3. Verify Existing Variance Values ---
    print("Verifying that loaded structures have variance values...")
    structures_with_variance = []
    for i, atoms in enumerate(all_atoms):
        if 'variance' in atoms.info:
            structures_with_variance.append(atoms)
        else:
            print(f"Warning: Structure {i} missing variance in info, skipping.")
    
    if not structures_with_variance:
        print("Error: No structures found with variance values. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(structures_with_variance)} structures with variance values.")
    all_atoms = structures_with_variance  # Use only structures with variance
    
    # --- 4. Run Selection ---
    print("Initializing AAAnalyzer for structure selection...")
    analyzer = AAAnalyzer(atoms_list=all_atoms, calculator=calculator)

    print(f"Selecting {N_SELECT} diverse structures using existing variance values...")
    selected_indices, fig = analyzer.select_diverse_structures(n_select=N_SELECT, plot=True)

    if fig:
        plot_path = output_path / "umap_selection_plot.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"\nSaved UMAP plot to {plot_path}")

    # --- 5. Create VASP Jobs ---
    selected_atoms = [all_atoms[i] for i in selected_indices]
    print(f"\nCreating {len(selected_atoms)} VASP job folders...")
    
    for i, atoms in enumerate(selected_atoms):
        job_dir = output_path / f"selection_{i:03d}"
        job_name = f"sel_{i:03d}"
        prepare_vasp_job_from_ase(
            atoms=atoms,
            vasp_profile_name=VASP_PROFILE_NAME,
            hpc_profile_name=HPC_PROFILE_NAME,
            output_dir=str(job_dir),
            job_name=job_name
        )

    print("\nAnalysis and VASP job creation complete.")

if __name__ == "__main__":
    main() 