from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
from pathlib import Path

from forge.core.database import DatabaseManager
from forge.analysis.training.evaluator import Evaluator
from forge.analysis.training import metrics
from forge.analysis.training import spatial
from forge.analysis.training import difficulty
from forge.analysis.training import plotting
from ase.io import write as ase_write


class ErrorAnalyser:
    """
    Facade for the error analysis toolkit.

    This class orchestrates the process of:
    1. Loading reference structures and calculations from the database.
    2. Evaluating predictions from one or more calculators.
    3. Computing classical and heavy-tail error metrics.
    4. Analysing the spatial distribution of errors.
    5. Calculating a composite difficulty score to rank structures.
    """
    def __init__(self, db_manager: Optional[DatabaseManager], calculators: Optional[any], ref_calc_name: str = 'vasp'):
        """
        Args:
            db_manager: An active DatabaseManager instance. Can be None if loading from file.
            calculators: A single ASE calculator or a list/committee of them. Can be None if loading.
            ref_calc_name: The name of the reference calculator in the database (e.g., 'vasp').
        """
        self.db = db_manager
        self.evaluator = Evaluator(calculators) if calculators is not None else None
        self.ref_calc_name = ref_calc_name
        self.results_cache: Dict[int, Dict[str, Any]] = {}
        # Add attributes to hold the dataframes
        self.df_struct: Optional[pd.DataFrame] = None
        self.df_atom: Optional[pd.DataFrame] = None

    def run(
        self, 
        structure_ids: List[int],
        batch_size: int = 32,
        spatial_k: int = 12,
        dbscan_eps: float = 2.5,
        dbscan_min_samples: int = 3,
        difficulty_weights: Dict[str, float] = None
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Run the full analysis pipeline on a list of structures.

        Args:
            structure_ids: List of structure_ids to analyse.
            batch_size: Number of structures to process in each database/evaluator batch.
            spatial_k: Number of nearest neighbors for Moran's I calculation.
            dbscan_eps: The `eps` parameter for DBSCAN clustering of high-error atoms.
            dbscan_min_samples: The `min_samples` for DBSCAN.
            difficulty_weights: Dictionary to weigh components of the difficulty score.

        Returns:
            A tuple of (df_struct, df_atom):
            - df_struct: DataFrame with one row per structure, containing aggregated metrics.
            - df_atom: DataFrame with one row per atom, containing per-atom errors.
        """
        struct_rows = []
        atom_rows = []

        for i in tqdm(range(0, len(structure_ids), batch_size), desc="Processing Batches"):
            batch_ids = structure_ids[i:i+batch_size]
            
            atoms_list = self.db.get_batch_atoms_with_calculation(batch_ids, calculator=self.ref_calc_name)
            if not atoms_list:
                continue

            predictions_list = self.evaluator.evaluate(atoms_list)

            for atoms, preds_dict in zip(atoms_list, predictions_list):
                struct_id = atoms.info['structure_id']
                n_atoms = len(atoms)
                
                # --- Get Reference Data ---
                ref_energy = atoms.info.get('energy')
                ref_forces = atoms.arrays.get('forces')
                ref_stress = atoms.info.get('stress')

                # --- Process Predictions (currently handles first calc in ensemble) ---
                # This logic can be extended to handle multiple predictions
                if not preds_dict: continue
                pred_calc_name = list(preds_dict.keys())[0]
                preds = preds_dict[pred_calc_name]
                if preds is None: continue

                # --- Calculate Errors ---
                force_error_vectors = preds['forces'] - ref_forces
                force_error_magnitudes = np.linalg.norm(force_error_vectors, axis=1)

                # --- Metrics ---
                force_stats = metrics.force_error_stats(ref_forces, preds['forces'])
                energy_stats = metrics.energy_error_stats(ref_energy, preds['energy'], n_atoms)
                stress_stats = metrics.stress_error_stats(ref_stress, preds['stress'])
                
                # --- Spatial Analysis ---
                q95_threshold = force_stats.get('Q95', 0.0)
                spatial_stats = spatial.morans_I(atoms, force_error_magnitudes, k=spatial_k)
                dbscan_labels = spatial.find_error_clusters(
                    atoms, force_error_magnitudes, threshold=q95_threshold, 
                    eps=dbscan_eps, min_samples=dbscan_min_samples
                )

                # --- Ensemble Variance & Difficulty Score ---
                ensemble_variance = difficulty.calculate_ensemble_variance(list(preds_dict.values()), n_atoms)
                difficulty_scores = difficulty.calculate_difficulty_score(
                    force_error_magnitudes,
                    spatial_stats.get('global_I', 0.0),
                    dbscan_labels,
                    ensemble_variance.get('force_variance', 0.0),
                    n_atoms,
                    weights_dict=difficulty_weights
                )
                
                # --- Assemble DataFrames ---
                struct_row = {
                    'structure_id': struct_id,
                    'n_atoms': n_atoms,
                    'formula': atoms.get_chemical_formula(),
                    **force_stats,
                    **energy_stats,
                    **stress_stats,
                    'MoranI': spatial_stats.get('global_I'),
                    'MoranI_p': spatial_stats.get('global_p'),
                    **difficulty_scores
                }
                struct_rows.append(struct_row)
                
                for atom_idx in range(n_atoms):
                    atom_rows.append({
                        'structure_id': struct_id,
                        'atom_index': atom_idx,
                        'symbol': atoms.symbols[atom_idx],
                        'force_error_x': force_error_vectors[atom_idx, 0],
                        'force_error_y': force_error_vectors[atom_idx, 1],
                        'force_error_z': force_error_vectors[atom_idx, 2],
                        'force_error_mag': force_error_magnitudes[atom_idx],
                        'local_moran_I': spatial_stats.get('local_I', [None]*n_atoms)[atom_idx],
                        'dbscan_cluster': dbscan_labels[atom_idx]
                    })
                
                # --- Cache results for plotting ---
                self.results_cache[struct_id] = {
                    'atoms': atoms,
                    'force_error_magnitudes': force_error_magnitudes,
                    'force_error_vectors': force_error_vectors,
                    'spatial_stats': spatial_stats,
                    'dbscan_labels': dbscan_labels,
                    'force_stats': force_stats,
                }
        
        # Store dataframes as instance attributes
        self.df_struct = pd.DataFrame(struct_rows)
        self.df_atom = pd.DataFrame(atom_rows)

        return self.df_struct, self.df_atom

    def save_to_extxyz(self, filepath: str):
        """
        Saves the analysis results to a single extended XYZ file.

        This method compiles all analyzed structures into one file, enriching
        each ASE Atoms object with per-atom and per-structure metrics computed
        during the analysis. This format is ideal for visualization in tools
        like Ovito.

        Per-Atom Data Added:
        - force_error_mag: The magnitude of the force error vector.
        - force_error_vec: The 3D force error vector.
        - local_moran_I: Local Moran's I value for spatial autocorrelation.
        - dbscan_cluster: Cluster label from DBSCAN analysis.

        Per-Structure (Info) Data Added:
        - All columns from the main `df_struct` DataFrame for that structure.
        
        Args:
            filepath: The path to the output .extxyz file.
        """
        if not self.results_cache or self.df_struct is None:
            raise RuntimeError("Cannot save, analysis has not been run. Please call .run() first.")

        all_atoms_to_write = []
        struct_metrics_df = self.df_struct.set_index('structure_id')

        for struct_id, data in tqdm(self.results_cache.items(), desc="Preparing .extxyz"):
            atoms = data['atoms'].copy() # Make a copy to modify

            # --- Add per-atom data ---
            atoms.new_array('force_error_mag', data['force_error_magnitudes'])

            # Add force error vectors, reconstructing them if they are not in the cache (for backward compatibility)
            if 'force_error_vectors' in data:
                force_vectors = data['force_error_vectors']
            else:
                atom_data_df = self.df_atom[self.df_atom['structure_id'] == struct_id]
                if not atom_data_df.empty:
                    force_vectors = atom_data_df[['force_error_x', 'force_error_y', 'force_error_z']].values
                else:
                    force_vectors = np.zeros((len(atoms), 3)) # Fallback if atom data is somehow missing
            atoms.new_array('force_error_vec', force_vectors)
            
            # Add other per-atom data from df_atom if available
            atom_data = self.df_atom[self.df_atom['structure_id'] == struct_id]
            if not atom_data.empty:
                 atoms.new_array('local_moran_I', atom_data['local_moran_I'].values)
                 atoms.new_array('dbscan_cluster', atom_data['dbscan_cluster'].values)

            # --- Add per-structure data to info dict ---
            if struct_id in struct_metrics_df.index:
                struct_metrics = struct_metrics_df.loc[struct_id].to_dict()
                for key, value in struct_metrics.items():
                    # ASE info values must be basic types (str, int, float)
                    if isinstance(value, (np.ndarray, pd.Series)):
                        # Skip complex types or convert if necessary
                        continue
                    atoms.info[key] = value
            
            # --- Add the full text report as a structured dictionary ---
            atoms.info['analysis_report'] = self.get_report_dict(struct_id)

            all_atoms_to_write.append(atoms)
        
        # Write all structures to a single file
        print(f"Writing {len(all_atoms_to_write)} structures to {filepath}...")
        ase_write(filepath, all_atoms_to_write, format='extxyz')
        print("Done.")

    def save(self, directory_path: str):
        """
        Saves the complete analysis state to a specified directory.
        
        This includes the structure and atom dataframes, and the results cache
        used for plotting and reporting.

        Args:
            directory_path: Path to the directory for saving the analysis.
        """
        if self.df_struct is None or self.df_atom is None:
            raise RuntimeError("Cannot save analysis that has not been run. Please call .run() first.")

        path = Path(directory_path)
        path.mkdir(exist_ok=True, parents=True)

        self.df_struct.to_pickle(path / 'df_struct.pkl')
        self.df_atom.to_pickle(path / 'df_atom.pkl')

        with open(path / 'results_cache.pkl', 'wb') as f:
            pickle.dump(self.results_cache, f)
        
        print(f"Analysis results successfully saved to: {path.resolve()}")

    @classmethod
    def load(cls, directory_path: str, db_manager: Optional[DatabaseManager] = None, calculators: Optional[Any] = None) -> 'ErrorAnalyser':
        """
        Loads a previously saved analysis state from a directory.

        Args:
            directory_path: Path to the directory where the analysis was saved.
            db_manager: An active DatabaseManager instance. Optional.
            calculators: The calculator or list of calculators to use. Optional.
        
        Returns:
            An ErrorAnalyser instance populated with the loaded data.
        """
        path = Path(directory_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Save directory not found: {path}")

        # For now, we assume the same db and calcs are used, but we could store metadata
        # about them in the future.
        instance = cls(db_manager, calculators)

        instance.df_struct = pd.read_pickle(path / 'df_struct.pkl')
        instance.df_atom = pd.read_pickle(path / 'df_atom.pkl')
        
        with open(path / 'results_cache.pkl', 'rb') as f:
            instance.results_cache = pickle.load(f)

        print(f"Successfully loaded analysis from: {path.resolve()}")
        return instance

    def get_report_dict(self, structure_id: int) -> Dict[str, Any]:
        """
        Generates a dictionary containing the analysis report for a single structure.

        Args:
            structure_id: The ID of the structure to report on.

        Returns:
            A dictionary containing the analysis report.
        """
        if structure_id not in self.results_cache or self.df_struct is None or self.df_atom is None:
            raise ValueError(f"No results found for structure {structure_id}. Please run analysis first.")

        struct_metrics = self.df_struct.loc[self.df_struct['structure_id'] == structure_id].iloc[0]
        atom_metrics = self.df_atom.loc[self.df_atom['structure_id'] == structure_id]
        cached_atoms = self.results_cache[structure_id]['atoms']
        config_type = cached_atoms.info.get('config_type', 'N/A')
        
        kurtosis = struct_metrics.get('Kurtosis', 3.0)
        gini = struct_metrics.get('Gini', 0.0)
        moran_i = struct_metrics.get('MoranI', 0.0)

        # Hypothesis Testing
        hypothesis_supported = (kurtosis > 5.0 and gini > 0.3)
        hypothesis_details = {
            "supported": "YES" if hypothesis_supported else "NO",
            "kurtosis": f"{kurtosis:.2f}",
            "gini_coefficient": f"{gini:.2f}",
            "message": "Force errors are concentrated on a small number of 'problematic' atoms." if hypothesis_supported else "Errors are more evenly distributed."
        }
        if hypothesis_supported and moran_i > 0.1:
             hypothesis_details["spatial_clustering_message"] = f"Furthermore, a significant Moran's I ({moran_i:.2f}) indicates these atoms are spatially clustered."
        
        # Hotspot Analysis
        clusters = atom_metrics[atom_metrics['dbscan_cluster'] >= 0]
        hotspot_details = {"message": "No significant error clusters found."}
        if not clusters.empty:
            cluster_sizes = clusters['dbscan_cluster'].value_counts()
            hotspot_details = {
                "message": f"Found {len(cluster_sizes)} error hotspots (clusters of atoms with |ΔF| > Q95).",
                "cluster_count": len(cluster_sizes),
                "largest_cluster_size": int(cluster_sizes.max())
            }

        # Top 5 Problematic Atoms
        top_atoms = atom_metrics.sort_values('force_error_mag', ascending=False).head(5)
        top_atoms_list = [
            {"atom_index": row.atom_index, "symbol": row.symbol, "force_error_mag_eV_A": f"{row.force_error_mag:.3f}"}
            for _, row in top_atoms.iterrows()
        ]

        report_dict = {
            "summary": {
                "structure_id": structure_id,
                "formula": struct_metrics['formula'],
                "n_atoms": int(struct_metrics['n_atoms']),
                "config_type": config_type
            },
            "key_metrics": {
                "difficulty_score": f"{struct_metrics['difficulty']:.2f}",
                "force_rmse_eV_A": f"{struct_metrics['RMSE']:.4f}",
                "kurtosis": f"{kurtosis:.2f}",
                "gini_coefficient": f"{gini:.2f}",
                "morans_i": f"{moran_i:.2f}"
            },
            "hypothesis_test": hypothesis_details,
            "spatial_hotspot_analysis": hotspot_details,
            "top_5_problematic_atoms": top_atoms_list
        }
        return report_dict

    def generate_report(self, structure_id: int) -> str:
        """
        Generates a human-readable text report for a single structure.

        This report is designed to help diagnose model performance and test the
        hypothesis that errors are driven by a few problematic atoms.

        Args:
            structure_id: The ID of the structure to report on.

        Returns:
            A formatted string containing the analysis report.
        """
        if structure_id not in self.results_cache or self.df_struct is None or self.df_atom is None:
            raise ValueError(f"No results found for structure {structure_id}. Please run analysis first.")

        struct_metrics = self.df_struct.loc[self.df_struct['structure_id'] == structure_id].iloc[0]
        atom_metrics = self.df_atom.loc[self.df_atom['structure_id'] == structure_id]
        cached_atoms = self.results_cache[structure_id]['atoms']
        config_type = cached_atoms.info.get('config_type', 'N/A')

        # --- Hypothesis Testing ---
        kurtosis = struct_metrics.get('Kurtosis', 3.0)
        gini = struct_metrics.get('Gini', 0.0)
        moran_i = struct_metrics.get('MoranI', 0.0)
        
        hypothesis_supported = (kurtosis > 5.0 and gini > 0.3)
        hypothesis_string = (
            f"Hypothesis supported: YES. High Kurtosis ({kurtosis:.2f}) and Gini "
            f"coefficient ({gini:.2f}) suggest force errors are concentrated on a "
            f"small number of 'problematic' atoms."
        ) if hypothesis_supported else (
            f"Hypothesis supported: NO. Low Kurtosis ({kurtosis:.2f}) and Gini "
            f"coefficient ({gini:.2f}) suggest errors are more evenly distributed."
        )
        if hypothesis_supported and moran_i > 0.1:
            hypothesis_string += f"\nFurthermore, a significant Moran's I ({moran_i:.2f}) indicates these atoms are spatially clustered."

        # --- Hotspot Analysis ---
        clusters = atom_metrics[atom_metrics['dbscan_cluster'] >= 0]
        hotspot_string = "No significant error clusters found."
        if not clusters.empty:
            cluster_sizes = clusters['dbscan_cluster'].value_counts()
            hotspot_string = (
                f"Found {len(cluster_sizes)} error hotspots (clusters of atoms with |ΔF| > Q95).\n"
                f"Largest cluster size: {cluster_sizes.max()} atoms."
            )

        # --- Top 5 Problematic Atoms ---
        top_atoms = atom_metrics.sort_values('force_error_mag', ascending=False).head(5)
        top_atoms_string = "\n".join([
            f"  - Atom Index: {row.atom_index}, Symbol: {row.symbol}, Force Error: {row.force_error_mag:.3f} eV/Å"
            for _, row in top_atoms.iterrows()
        ])

        # --- Assemble Report ---
        report = f"""
============================================================
ANALYSIS REPORT FOR STRUCTURE ID: {structure_id}
============================================================
Formula: {struct_metrics['formula']} ({struct_metrics['n_atoms']} atoms)
Config Type: {config_type}

* Key Metrics:
  - Difficulty Score: {struct_metrics['difficulty']:.2f}
  - Force RMSE:       {struct_metrics['RMSE']:.4f} eV/Å
  - Kurtosis:         {kurtosis:.2f} (Normal=3.0)
  - Gini Coefficient: {gini:.2f} (0=perfect equality)
  - Moran's I:        {moran_i:.2f} (Positive = clustered errors)

* Hypothesis: 'Errors are driven by a few problematic atoms'
  - {hypothesis_string}

* Spatial Hotspot Analysis:
  - {hotspot_string}

* Top 5 Most Problematic Atoms (by |ΔF|):
{top_atoms_string}
============================================================
"""
        return report

    def force_hist(self, structure_id: int, **kwargs):
        """
        Plots a histogram of force errors for a structure.
        Requires the analysis to have been run first.
        """
        if structure_id not in self.results_cache:
            raise ValueError(f"No results found for structure {structure_id}. Please run analysis first.")
        
        errors = self.results_cache[structure_id]['force_error_magnitudes']
        return plotting.plot_force_error_histogram(errors, **kwargs)
    
    def qq_plot(self, structure_id: int, **kwargs):
        """
        Generates a Q-Q plot of force errors for a structure.
        Requires the analysis to have been run first.
        """
        if structure_id not in self.results_cache:
            raise ValueError(f"No results found for structure {structure_id}. Please run analysis first.")
            
        errors = self.results_cache[structure_id]['force_error_magnitudes']
        return plotting.plot_qq(errors, **kwargs)

def load_analysis_from_pickle(directory_path: str) -> (pd.DataFrame, pd.DataFrame, Dict[int, Any]):
    """
    Loads saved analysis dataframes and cache from a directory.

    This is a convenience function to quickly load the results of a previous
    analysis run from the pickle files created by `ErrorAnalyser.save()`.

    Args:
        directory_path: Path to the directory where the analysis was saved.

    Returns:
        A tuple of (df_struct, df_atom, results_cache).
    """
    path = Path(directory_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Save directory not found: {path}")

    df_struct = pd.read_pickle(path / 'df_struct.pkl')
    df_atom = pd.read_pickle(path / 'df_atom.pkl')
    
    with open(path / 'results_cache.pkl', 'rb') as f:
        results_cache = pickle.load(f)

    print(f"Successfully loaded analysis data from: {path.resolve()}")
    return df_struct, df_atom, results_cache 