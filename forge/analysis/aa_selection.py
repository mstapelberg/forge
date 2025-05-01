import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
from ase import Atoms
from typing import List, Tuple, Optional, Union, Dict
import warnings
from mace.calculators.mace import MACECalculator # Assuming MACE is the primary target

class AAAnalyzer:
    """
    Analyzes results from an Adversarial Attack trajectory, focusing on
    embedding calculation and diverse structure selection using UMAP.
    """
    def __init__(self, atoms_list: List[Atoms], calculator: MACECalculator):
        """
        Initializes the analyzer with the trajectory and calculator.

        Args:
            atoms_list: List of ASE Atoms objects from the AA trajectory.
                         Assumed to have 'variance' in atoms.info for selection.
            calculator: An initialized MACE calculator instance for embedding generation.
        """
        if not atoms_list:
            raise ValueError("Input atoms_list cannot be empty.")
        if not isinstance(calculator, MACECalculator):
             # Could relax this later if supporting other calculators
             raise TypeError("calculator must be an initialized MACECalculator instance.")

        self.atoms_list: List[Atoms] = atoms_list
        self.calculator: MACECalculator = calculator
        self.n_initial_structures: int = len(atoms_list)

        # Placeholder for calculated embeddings and valid indices
        self.atomic_embeddings: List[Optional[np.ndarray]] = [None] * self.n_initial_structures
        self.embedding_valid_indices: List[int] = [] # Indices of atoms for which embeddings were successful
        self.embedding_dim: Optional[int] = None
        self._embeddings_calculated: bool = False

        print(f"AAAnalyzer initialized with {self.n_initial_structures} structures.")

    def calculate_embeddings(self, batch_size: Optional[int] = None) -> None:
        """
        Calculates per-atom embeddings for all structures using the stored calculator.

        Stores the results in self.atomic_embeddings and updates
        self.embedding_valid_indices. Structures where embedding fails will
        have None in self.atomic_embeddings.

        Args:
            batch_size: If provided, process atoms in batches (useful for large lists).
                        Currently not implemented, processes one by one. # TODO: Add batching?
        """
        print(f"Calculating MACE descriptors for {self.n_initial_structures} structures...")
        successful_embeddings = []
        valid_indices_temp = []
        temp_embeddings_list = [None] * self.n_initial_structures # Temp list during calculation

        # Simple one-by-one calculation for now
        for i, atoms in enumerate(self.atoms_list):
            try:
                # Ensure atoms object is suitable if needed
                desc = self.calculator.get_descriptors(atoms)

                if desc is None or not isinstance(desc, np.ndarray) or desc.ndim != 2 or desc.shape[0] == 0:
                     warnings.warn(f"Invalid descriptor for structure index {i} (shape: {getattr(desc, 'shape', 'N/A')}, type: {type(desc)}). Skipping.")
                     continue # Keep None in temp_embeddings_list

                # Check dimension consistency
                current_dim = desc.shape[1]
                if self.embedding_dim is None:
                    self.embedding_dim = current_dim
                elif current_dim != self.embedding_dim:
                    warnings.warn(f"Inconsistent embedding dimension for structure index {i}. Expected {self.embedding_dim}, got {current_dim}. Skipping.")
                    continue # Keep None in temp_embeddings_list

                # Store successful embedding
                temp_embeddings_list[i] = desc
                valid_indices_temp.append(i)
                successful_embeddings.append(desc) # Keep track for logging

            except Exception as e:
                warnings.warn(f"Error getting descriptors for structure index {i}: {e}. Skipping.")
                # Keep None in temp_embeddings_list

        self.atomic_embeddings = temp_embeddings_list
        self.embedding_valid_indices = valid_indices_temp
        self._embeddings_calculated = True
        num_successful = len(self.embedding_valid_indices)
        print(f"Embedding calculation finished. Successfully processed {num_successful}/{self.n_initial_structures} structures.")
        if num_successful == 0:
            warnings.warn("No successful embeddings were calculated.")
        elif self.embedding_dim is not None:
             print(f"Detected embedding dimension: {self.embedding_dim}")


    def select_diverse_structures(
        self,
        n_select: int,
        aggregation_method: str = 'mean',
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
        plot: bool = False
    ) -> Tuple[List[int], Optional[plt.Figure]]:
        """
        Selects N diverse, high-uncertainty structures using combined embeddings/variance.

        This method assumes embeddings have been calculated (it calls self.calculate_embeddings
        if not already done). It uses the 'variance' key from the atoms.info dictionary.

        Args:
            n_select: Target number of diverse structures.
            aggregation_method: 'mean' or 'sum' for aggregating atom embeddings.
            umap_neighbors: UMAP n_neighbors hyperparameter.
            umap_min_dist: UMAP min_dist hyperparameter.
            random_state: Seed for UMAP and KMeans.
            plot: If True, generate and return a plot.

        Returns:
            Tuple containing:
            - List of indices (relative to the original atoms_list) of selected structures.
            - Matplotlib Figure object if plot=True, otherwise None.

        Raises:
            ValueError: If n_select is invalid, aggregation_method is unknown, or no
                        valid structures with both embeddings and variance are found.
            RuntimeError: If embedding calculation fails for all structures.
        """
        if not self._embeddings_calculated:
             warnings.warn("Embeddings not calculated yet. Calling self.calculate_embeddings().")
             self.calculate_embeddings()
             if not self.embedding_valid_indices:
                  raise RuntimeError("Cannot select structures: Embedding calculation failed for all structures.")

        # --- Step 1: Filter based on successful embeddings AND valid variance ---
        print("Filtering structures based on valid embeddings and variance...")
        filtered_indices_for_selection = [] # Indices within the original list that are valid *for selection*
        selection_atoms = []
        selection_variances = []
        selection_embeddings = [] # List of the actual embedding arrays for valid structures

        for i in self.embedding_valid_indices: # Only iterate through structures with successful embeddings
            atoms = self.atoms_list[i]
            variance = atoms.info.get('variance')
            emb_array = self.atomic_embeddings[i] # We know this is not None

            # Check variance (must exist in info for selection)
            if variance is None or not isinstance(variance, (int, float, np.number)) or np.isnan(variance):
                warnings.warn(f"Skipping structure index {i}: Invalid or missing 'variance' in atoms.info ({variance}).")
                continue

            # All checks passed for this structure
            filtered_indices_for_selection.append(i)
            selection_atoms.append(atoms)
            selection_variances.append(variance)
            selection_embeddings.append(emb_array) # Add the actual embedding array

        n_structures_for_selection = len(filtered_indices_for_selection)
        print(f"Filtering complete. {n_structures_for_selection} structures available for selection.")

        if n_structures_for_selection == 0:
            raise ValueError("No valid structures remaining after filtering for embeddings and variance.")

        variances_np = np.array(selection_variances) # Use the filtered variances

        # --- The rest of the logic is similar to the standalone function ---
        # --- using selection_* lists and n_structures_for_selection ---

        # --- Step 2: Aggregate Per-Atom Embeddings (using selection_embeddings) ---
        aggregated_embeddings_list = []
        print(f"Aggregating per-atom embeddings for {n_structures_for_selection} structures using '{aggregation_method}'...")
        for emb_array in selection_embeddings:
            if aggregation_method == 'mean':
                aggregated_embeddings_list.append(np.mean(emb_array, axis=0))
            elif aggregation_method == 'sum':
                aggregated_embeddings_list.append(np.sum(emb_array, axis=0))

        aggregated_embeddings_np = np.array(aggregated_embeddings_list)
        print(f"Aggregation complete. Shape: {aggregated_embeddings_np.shape}")

        # --- Handle Case: Fewer Structures than n_select ---
        if n_structures_for_selection <= n_select:
             warnings.warn(f"Number of valid structures for selection ({n_structures_for_selection}) <= n_select ({n_select}). Returning all valid indices.")
             selected_original_indices = filtered_indices_for_selection
             fig = None
             # Optional Plotting (similar to standalone function, using selection data)
             if plot:
                  try:
                       # Plotting code using aggregated_embeddings_np and variances_np
                       # Ensure to use filtered_indices_for_selection for mapping back if needed
                       variance_scaler_plot = StandardScaler()
                       scaled_variances_plot = variance_scaler_plot.fit_transform(variances_np.reshape(-1, 1)).flatten()

                       reducer_plot = umap.UMAP(
                           n_neighbors=min(umap_neighbors, n_structures_for_selection - 1) if n_structures_for_selection > 1 else 1,
                           n_components=2, min_dist=umap_min_dist, random_state=random_state, n_jobs=1)
                       # UMAP on aggregated embeddings only for this simple plot case
                       embeddings_2d_plot = reducer_plot.fit_transform(aggregated_embeddings_np)

                       fig, ax = plt.subplots(figsize=(10, 8))
                       scatter = ax.scatter(embeddings_2d_plot[:, 0], embeddings_2d_plot[:, 1],
                                            c=scaled_variances_plot, # Use scaled variance for color
                                            cmap='viridis', s=50, alpha=0.7)
                       cbar = fig.colorbar(scatter, ax=ax)
                       cbar.set_label('Scaled Force Variance')
                       ax.set_title(f"UMAP (Aggregated Embeddings) - All {n_structures_for_selection} Valid Points Selected")
                       ax.set_xlabel("UMAP Dimension 1", fontsize=14)
                       ax.set_ylabel("UMAP Dimension 2", fontsize=14)
                       ax.tick_params(axis='both', which='major', labelsize=14)
                       ax.grid(True, linestyle='--', alpha=0.5)
                       plt.tight_layout()
                  except Exception as e:
                       warnings.warn(f"Plotting failed for small dataset: {e}")
                       if fig: plt.close(fig)
                       fig = None
             return selected_original_indices, fig

        # --- Steps 3 & 4: Scale and Concatenate Features ---
        print("Scaling aggregated embeddings and variances...")
        emb_scaler = StandardScaler()
        scaled_embeddings = emb_scaler.fit_transform(aggregated_embeddings_np)
        var_scaler = StandardScaler()
        scaled_variances = var_scaler.fit_transform(variances_np.reshape(-1, 1))
        combined_features = np.hstack((scaled_embeddings, scaled_variances))
        print(f"Combined feature vector shape: {combined_features.shape}")

        # --- Step 5: UMAP ---
        print(f"Running UMAP on {n_structures_for_selection} combined features...")
        effective_umap_neighbors = min(umap_neighbors, n_structures_for_selection - 1) if n_structures_for_selection > 1 else 1
        reducer = umap.UMAP(n_neighbors=effective_umap_neighbors, n_components=2, min_dist=umap_min_dist, random_state=random_state, n_jobs=1)
        embeddings_2d = reducer.fit_transform(combined_features)
        print("UMAP finished.")

        # --- Step 6: KMeans ---
        n_clusters = min(n_select, n_structures_for_selection)
        print(f"Running KMeans clustering with k={n_clusters}...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings_2d)
        print("KMeans finished.")

        # --- Step 7: Select Highest Variance from Each Cluster ---
        # Map cluster ID back to the *original* index using filtered_indices_for_selection
        selected_indices_map = {} # {cluster_id: (best_original_variance, best_original_index)}
        for i in range(n_structures_for_selection): # Iterate through filtered data for selection
            cluster_id = cluster_labels[i]
            original_variance = variances_np[i] # Variance for this structure
            original_index = filtered_indices_for_selection[i] # Get the index in the *original* list

            if cluster_id not in selected_indices_map or original_variance > selected_indices_map[cluster_id][0]:
                selected_indices_map[cluster_id] = (original_variance, original_index)

        final_selected_original_indices = sorted([data[1] for data in selected_indices_map.values()])
        num_actually_selected = len(final_selected_original_indices)

        if num_actually_selected < n_select and n_structures_for_selection > n_select:
            warnings.warn(f"Only able to select {num_actually_selected} structures (expected {n_select}).")

        # --- Step 8: Plotting ---
        fig = None
        if plot:
            print("Generating plot...")
            try:
                 # Plotting logic using embeddings_2d, cluster_labels, variances_np,
                 # final_selected_original_indices, and filtered_indices_for_selection mapping
                 fig, ax = plt.subplots(figsize=(12, 9))

                 # Scatter plot of all *valid* points used in selection, colored by cluster
                 scatter_all = ax.scatter(
                     embeddings_2d[:, 0], embeddings_2d[:, 1],
                     c=cluster_labels, cmap='viridis', s=30, alpha=0.5,
                     label=f'All Valid Points for Selection (n={n_structures_for_selection}, colored by cluster)'
                 )

                 # Get UMAP coords and original variances for the *selected* points
                 # Need to map original selected indices back to their position in the filtered lists used for UMAP/KMeans
                 selected_indices_in_selection = [filtered_indices_for_selection.index(orig_idx) for orig_idx in final_selected_original_indices]
                 selected_embeddings_2d = embeddings_2d[selected_indices_in_selection]
                 selected_original_variances = variances_np[selected_indices_in_selection]

                 # Use selected original variances for coloring the selected points
                 scatter_selected = ax.scatter(
                      selected_embeddings_2d[:, 0], selected_embeddings_2d[:, 1],
                      c=selected_original_variances, cmap='coolwarm', s=100,
                      edgecolors='black', marker='o',
                      label=f'Selected (n={num_actually_selected})', zorder=3
                 )
                 cbar = fig.colorbar(scatter_selected, ax=ax)
                 cbar.set_label('Original Force Variance of Selected', fontsize=14)
                 cbar.ax.tick_params(labelsize=14)

                 ax.set_title(f"UMAP (Combined Scaled Features), KMeans (k={n_clusters}), High-Variance Selection", fontsize=16)
                 ax.set_xlabel("UMAP Dimension 1", fontsize=14)
                 ax.set_ylabel("UMAP Dimension 2", fontsize=14)
                 ax.legend(fontsize=12) # Adjusted legend fontsize slightly
                 ax.tick_params(axis='both', which='major', labelsize=14)
                 ax.grid(True, linestyle='--', alpha=0.6)
                 plt.tight_layout()
                 # Don't call plt.show() here, return the figure object

            except Exception as e:
                warnings.warn(f"Plotting failed: {e}")
                if fig: plt.close(fig)
                fig = None

        print(f"Selected {num_actually_selected} diverse structures with original indices: {final_selected_original_indices}")
        return final_selected_original_indices, fig
