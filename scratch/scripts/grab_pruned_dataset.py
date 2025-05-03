import argparse
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm

# Check if MACE and ASE are installed
try:
    from mace.calculators.mace import MACECalculator
except ImportError:
    print("Error: MACE-torch library not found. Please install it (e.g., pip install mace-torch).")
    exit(1)

try:
    from ase import Atoms
    from ase.io import read
except ImportError:
    print("Error: ASE library not found. Please install it (e.g., pip install ase).")
    exit(1)

# Assuming forge is installed or in the PYTHONPATH
try:
    from forge.core.database import DatabaseManager
except ImportError:
    print("Error: forge.core.database not found. Ensure FORGE is installed and accessible.")
    exit(1)

# Try importing matplotlib
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    warnings.warn("Matplotlib not found. Plotting functionality will be disabled. Install with: pip install matplotlib")

# --- Configuration ---
DEFAULT_N_SELECT = 10000
DEFAULT_OUTPUT_FILE = "selected_structure_ids.txt"
DEFAULT_PLOT_FILE = "pruned_dataset_umap.png"
# DEFAULT_BATCH_SIZE = 500 # Removed batching
DEFAULT_UMAP_COMPONENTS = 5 # Lower dim for clustering might be better than 2
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_DEVICE = 'cpu' # Change to 'cuda' if GPU is available and desired

def calculate_aggregated_descriptors(atoms_list: list[Atoms], calculator: MACECalculator, aggregation_method: str = 'mean') -> tuple[list[np.ndarray], list[int]]:
    """
    Calculates aggregated descriptors for a list of Atoms objects.

    Args:
        atoms_list: List of ASE Atoms objects.
        calculator: Initialized MACE calculator.
        aggregation_method: 'mean' or 'sum' for aggregating atom descriptors.

    Returns:
        Tuple: (list of aggregated descriptors, list of indices within the input list corresponding to valid descriptors).
               Returns empty lists if errors occur or no valid descriptors are found.
    """
    aggregated_descriptors = []
    valid_indices_in_input = [] # Indices *within the input atoms_list*

    # Use tqdm for progress indication
    for i, atoms in enumerate(tqdm(atoms_list, desc="Calculating descriptors")):
        if not atoms or len(atoms) == 0:
             # Get structure_id for warning if possible
             struct_id_info = getattr(atoms, 'info', {}).get('structure_id', f'index {i}')
             warnings.warn(f"Skipping empty Atoms object for structure {struct_id_info}.")
             continue
        try:
            # Get per-atom descriptors
            desc = calculator.get_descriptors(atoms)

            if desc is None or not isinstance(desc, np.ndarray) or desc.ndim != 2 or desc.shape[0] != len(atoms):
                struct_id_info = atoms.info.get('structure_id', 'N/A')
                warnings.warn(f"Invalid descriptor for structure {struct_id_info} (index {i}) (shape: {getattr(desc, 'shape', 'N/A')}, type: {type(desc)}). Skipping.")
                continue

            # Aggregate
            if aggregation_method == 'mean':
                agg_desc = np.mean(desc, axis=0)
            elif aggregation_method == 'sum':
                agg_desc = np.sum(desc, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            aggregated_descriptors.append(agg_desc)
            valid_indices_in_input.append(i) # Store the index *within the input list*

        except Exception as e:
            # Use structure_id if available in info for better logging
            struct_id_info = atoms.info.get('structure_id', f'index {i}')
            warnings.warn(f"Error getting descriptors for structure {struct_id_info}: {e}. Skipping.")

    return aggregated_descriptors, valid_indices_in_input

def main():
    parser = argparse.ArgumentParser(description="Select diverse structures from database using MACE descriptors, UMAP, and KMeans.")
    parser.add_argument("mace_model_path", type=str, help="Path to the pre-trained MACE model (.model file).")
    parser.add_argument("-n", "--n_select", type=int, default=DEFAULT_N_SELECT, help=f"Number of structures to select (default: {DEFAULT_N_SELECT}).")
    parser.add_argument("-o", "--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help=f"Output file for selected structure IDs (default: {DEFAULT_OUTPUT_FILE}).")
    parser.add_argument("--db_config", type=str, default=None, help="Path to database configuration YAML (uses default forge config if None).")
    # parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Number of structures to process per batch (default: {DEFAULT_BATCH_SIZE}).") # Removed
    parser.add_argument("--umap_components", type=int, default=DEFAULT_UMAP_COMPONENTS, help=f"Number of UMAP components (default: {DEFAULT_UMAP_COMPONENTS}).")
    parser.add_argument("--umap_neighbors", type=int, default=DEFAULT_UMAP_NEIGHBORS, help=f"UMAP n_neighbors (default: {DEFAULT_UMAP_NEIGHBORS}).")
    parser.add_argument("--umap_min_dist", type=float, default=DEFAULT_UMAP_MIN_DIST, help=f"UMAP min_dist (default: {DEFAULT_UMAP_MIN_DIST}).")
    parser.add_argument("--aggregation", type=str, default='mean', choices=['mean', 'sum'], help="Aggregation method for atom descriptors ('mean' or 'sum', default: mean).")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help=f"Device for MACE calculation ('cpu' or 'cuda', default: {DEFAULT_DEVICE}).")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for UMAP and KMeans.")
    parser.add_argument("--plot_file", type=str, default=DEFAULT_PLOT_FILE, help=f"Output file for UMAP plot (default: {DEFAULT_PLOT_FILE}). Set to 'None' or empty to disable plotting.")

    args = parser.parse_args()

    # Disable plotting if matplotlib is not available or plot_file is None/empty
    do_plot = matplotlib_available and args.plot_file and args.plot_file.lower() != 'none'
    if args.plot_file and args.plot_file.lower() != 'none' and not matplotlib_available:
        print("Plot file specified, but matplotlib not found. Plotting is disabled.")

    # --- 1. Initialization ---
    print("--- Initializing ---")
    if not os.path.exists(args.mace_model_path):
        print(f"Error: MACE model file not found at {args.mace_model_path}")
        exit(1)

    print(f"Loading MACE model from: {args.mace_model_path} on device: {args.device}")
    try:
        calculator = MACECalculator(model_paths=args.mace_model_path, device=args.device)
    except Exception as e:
        print(f"Error initializing MACE calculator: {e}")
        exit(1)

    print("Connecting to database...")
    try:
        # Added debug=True for more connection info if needed
        db_manager = DatabaseManager(config_path=args.db_config, debug=False)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        exit(1)

    # --- 2. Fetch Structure IDs and Atoms Objects ---
    print("Fetching structure IDs from database...")
    try:
        # Example filter - adjust as needed
        # all_structure_ids = db_manager.find_structures_by_metadata(metadata_filters={"source_type": "aa_run"})
        # Fetch ALL structure IDs for now, or apply specific filters here
        with db_manager.conn.cursor() as cur:
             cur.execute("SELECT structure_id FROM structures ORDER BY structure_id")
             all_structure_ids = [row[0] for row in cur.fetchall()]

        n_total_structures = len(all_structure_ids)
        print(f"Found {n_total_structures} structure IDs.")
        if n_total_structures == 0:
            print("No structures found in the database. Exiting.")
            db_manager.close_connection()
            exit(0)

        print(f"Fetching {n_total_structures} Atoms objects...")
        # Fetch all Atoms objects at once
        atoms_dict = db_manager.get_structures_batch(all_structure_ids)
        # Create a list of atoms objects, preserving the order of all_structure_ids
        # Handle cases where some IDs might not be found by get_structures_batch
        all_atoms_objects = []
        retrieved_ids_set = set(atoms_dict.keys())
        original_indices_mapping = {} # Map structure_id back to its original 0-based index
        for idx, sid in enumerate(all_structure_ids):
             if sid in retrieved_ids_set:
                  all_atoms_objects.append(atoms_dict[sid])
                  original_indices_mapping[sid] = idx # Store original index
             else:
                  warnings.warn(f"Structure ID {sid} was requested but not retrieved by get_structures_batch. Skipping.")
                  # Optionally append None or handle differently if needed later
                  # all_atoms_objects.append(None) # If you need to keep placeholders

        n_retrieved_atoms = len(all_atoms_objects)
        print(f"Successfully retrieved {n_retrieved_atoms} Atoms objects.")
        if n_retrieved_atoms == 0:
             print("Could not retrieve any Atoms objects. Exiting.")
             db_manager.close_connection()
             exit(1)

    except Exception as e:
        print(f"Error fetching structure data: {e}")
        db_manager.close_connection()
        exit(1)

    # --- 3. Calculate All Descriptors ---
    print(f"--- Calculating aggregated descriptors ({args.aggregation}) for {n_retrieved_atoms} structures ---")
    try:
        # Calculate descriptors for all retrieved atoms at once
        all_aggregated_descriptors, valid_indices_in_retrieved_list = calculate_aggregated_descriptors(
            all_atoms_objects, calculator, args.aggregation
        )

        # Map the valid descriptors back to their original structure IDs
        # valid_indices refers to the indices within all_atoms_objects
        descriptor_struct_ids = [all_atoms_objects[idx].info['structure_id'] for idx in valid_indices_in_retrieved_list]

        # Also keep track of the indices *within the valid descriptor list* that correspond
        # to the successfully processed atoms. These are simply range(n_valid_descriptors).
        # However, we need the original indices in all_atoms_objects for plotting later.
        original_indices_for_valid_descriptors = valid_indices_in_retrieved_list

    except Exception as e:
        warnings.warn(f"Error during descriptor calculation: {e}")
        # Attempt to continue if some descriptors were calculated, otherwise exit
        if not 'all_aggregated_descriptors' in locals() or not all_aggregated_descriptors:
            print("Exiting due to critical error during descriptor calculation.")
            db_manager.close_connection()
            exit(1)
        # Fall through if partial results exist

    n_valid_descriptors = len(all_aggregated_descriptors)
    print(f"Finished descriptor calculation. Obtained {n_valid_descriptors} valid aggregated descriptors.")

    if n_valid_descriptors == 0:
        print("Error: No valid descriptors could be calculated. Exiting.")
        db_manager.close_connection()
        exit(1)

    # Adjust n_select if fewer valid descriptors than requested
    if n_valid_descriptors < args.n_select:
         warnings.warn(f"Number of valid descriptors ({n_valid_descriptors}) is less than n_select ({args.n_select}). Will select {n_valid_descriptors} structures.")
         args.n_select = n_valid_descriptors

    # Convert to NumPy array for processing
    all_aggregated_descriptors_np = np.array(all_aggregated_descriptors)
    descriptor_struct_ids_np = np.array(descriptor_struct_ids)

    # --- 4. UMAP Dimensionality Reduction ---
    print(f"--- Performing UMAP reduction (to {args.umap_components} components) ---")
    # Scale features before UMAP
    print("Scaling features...")
    scaler = StandardScaler()
    try:
        scaled_descriptors = scaler.fit_transform(all_aggregated_descriptors_np)
    except ValueError as e:
        print(f"Error scaling descriptors: {e}. Check for NaN/infinite values.")
        # Optional: Add code here to inspect all_aggregated_descriptors_np
        # print(all_aggregated_descriptors_np[~np.isfinite(all_aggregated_descriptors_np)])
        db_manager.close_connection()
        exit(1)

    print(f"Running UMAP (n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})...")
    try:
         # Adjust neighbors if dataset is small
         effective_umap_neighbors = min(args.umap_neighbors, n_valid_descriptors - 1) if n_valid_descriptors > 1 else 1
         # Use 2 components for plotting convenience
         reducer = umap.UMAP(
             n_neighbors=effective_umap_neighbors,
             n_components=2, # Reduced to 2 for plotting
             min_dist=args.umap_min_dist,
             random_state=args.random_state,
             n_jobs=-1, # Use all available processors
             verbose=True
         )
         umap_embeddings = reducer.fit_transform(scaled_descriptors)
         print("UMAP finished.")
    except Exception as e:
         print(f"Error during UMAP: {e}")
         db_manager.close_connection()
         exit(1)

    # --- 5. KMeans Clustering ---
    print(f"--- Performing KMeans clustering (k={args.n_select}) ---")
    try:
        # Ensure n_clusters is not more than n_samples
        actual_n_clusters = min(args.n_select, n_valid_descriptors)
        if actual_n_clusters != args.n_select:
            print(f"Adjusting k for KMeans from {args.n_select} to {actual_n_clusters} (number of valid samples).")

        kmeans = KMeans(
            n_clusters=actual_n_clusters, # Use adjusted k
            random_state=args.random_state,
            n_init='auto', # Suppress warning
            verbose=1 # Show progress
        )
        print("Fitting KMeans...")
        # Fit on the UMAP embeddings
        cluster_labels = kmeans.fit_predict(umap_embeddings)
        print("KMeans finished.")
        cluster_centers = kmeans.cluster_centers_
        # cluster_labels = kmeans.labels_ # Now stored directly
    except Exception as e:
        print(f"Error during KMeans: {e}")
        db_manager.close_connection()
        exit(1)

    # --- 6. Select Representatives (Closest to Centroids) ---
    print(f"--- Selecting {actual_n_clusters} representatives (closest to cluster centroids) ---")
    try:
        # Find the index of the point in umap_embeddings closest to each cluster center
        closest_point_indices, _ = pairwise_distances_argmin_min(cluster_centers, umap_embeddings)
        # Ensure uniqueness (although typically indices should be unique if k <= n_points)
        unique_closest_point_indices = np.unique(closest_point_indices)

        if len(unique_closest_point_indices) < actual_n_clusters:
            warnings.warn(f"KMeans found only {len(unique_closest_point_indices)} unique closest points to centroids "
                          f"(expected {actual_n_clusters}). This might happen with very similar clusters.")

        # Map these indices (which correspond to rows in umap_embeddings/scaled_descriptors/all_aggregated_descriptors_np)
        # back to the original structure IDs stored in descriptor_struct_ids_np
        selected_structure_ids = descriptor_struct_ids_np[unique_closest_point_indices].tolist()
        print(f"Selected {len(selected_structure_ids)} unique structure IDs.")

    except Exception as e:
        print(f"Error during representative selection: {e}")
        db_manager.close_connection()
        exit(1)

    # --- 7. Generate Plot (if enabled) ---
    if do_plot:
        print(f"--- Generating UMAP plot to {args.plot_file} ---")
        try:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot all points in light grey
            ax.scatter(
                umap_embeddings[:, 0],
                umap_embeddings[:, 1],
                c='#cccccc', # Light grey color
                s=5,          # Smaller size for background points
                alpha=0.5,    # Semi-transparent
                label=f'All Valid Structures (n={n_valid_descriptors})'
            )

            # Overlay selected points, colored by cluster ID
            selected_embeddings = umap_embeddings[unique_closest_point_indices]
            selected_labels = cluster_labels[unique_closest_point_indices]
            scatter_selected = ax.scatter(
                selected_embeddings[:, 0],
                selected_embeddings[:, 1],
                c=selected_labels, # Color by the cluster label
                cmap='viridis',    # Colormap (can change to 'tab20' or others)
                s=30,              # Larger size for selected points
                alpha=0.8,         # Slightly more opaque
                edgecolors='k',    # Black edge color for visibility
                linewidths=0.5,
                label=f'Selected Representatives (n={len(selected_structure_ids)})'
            )

            ax.set_title('UMAP of MACE Descriptors with Selected Representatives')
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Optional: Add a colorbar - might be messy with many clusters
            # cbar = fig.colorbar(scatter_selected, ax=ax, label='Cluster ID')

            plt.tight_layout()
            plt.savefig(args.plot_file, dpi=300)
            print(f"Plot saved to {args.plot_file}")
            plt.close(fig) # Close the figure to free memory

        except Exception as e:
            warnings.warn(f"Error generating plot: {e}")
            # Ensure figure is closed if error occurred during plotting/saving
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                 plt.close(fig)

    # --- 8. Save Results ---
    print(f"--- Saving selected structure IDs to {args.output_file} ---")
    try:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(output_path, 'w') as f:
            for sid in sorted(selected_structure_ids): # Save sorted IDs
                f.write(f"{sid}\n")
        print(f"Successfully saved {len(selected_structure_ids)} IDs.")
    except Exception as e:
        print(f"Error saving output file: {e}")
        # Don't exit here, maybe user can still see the IDs printed earlier

    # --- 9. Cleanup ---
    print("--- Cleaning up ---")
    db_manager.close_connection()
    print("Database connection closed.")
    print("Script finished.")

if __name__ == "__main__":
    main()
