"""Main ErrorAnalyser class for orchestrating structure analysis."""
from typing import List, Dict, Optional, Any, Union
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import logging
from ase import Atoms

from forge.core.database import DatabaseManager
from .evaluator import Evaluator
from .results import AnalysisResults
from ..metrics import (
    MetricRegistry, get_registry, register_metric,
    ForceErrorStats, EnergyErrorStats, StressErrorStats,
    MoransI, ErrorClustering, analyze_spatial_patterns
)
from ..difficulty import calculate_difficulty_score, calculate_ensemble_variance
from ..utils import check_geometry, save_analysis_results

logger = logging.getLogger(__name__)


class ErrorAnalyser:
    """Facade for the error analysis toolkit.
    
    This class orchestrates the process of:
    1. Loading reference structures and calculations from the database
    2. Evaluating predictions from one or more calculators
    3. Computing classical and heavy-tail error metrics
    4. Analysing the spatial distribution of errors
    5. Calculating a composite difficulty score to rank structures
    
    Parameters
    ----------
    db_manager : DatabaseManager
        An active DatabaseManager instance.
    calculators : Any
        A single ASE calculator or a list/tuple of calculators.
    ref_calc_name : str, optional
        The name of the reference calculator in the database (default: 'vasp').
    metric_registry : Optional[MetricRegistry]
        Custom metric registry. If None, uses global registry.
    ref_energy_key : str, optional
        The key for reference energy in the structure info (default: 'energy').
    ref_forces_key : str, optional
        The key for reference forces in the structure arrays (default: 'forces').
    ref_stress_key : str, optional
        The key for reference stress in the structure info (default: 'stress').
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        calculators: Any,
        ref_calc_name: str = 'vasp',
        metric_registry: Optional[MetricRegistry] = None,
        ref_energy_key: str = 'energy',
        ref_forces_key: str = 'forces',
        ref_stress_key: str = 'stress'
    ):
        """Initialize the analyser."""
        self.db = db_manager
        self.evaluator = Evaluator(calculators) if calculators is not None else None
        self.ref_calc_name = ref_calc_name
        self.metric_registry = metric_registry or get_registry()
        self.ref_energy_key = ref_energy_key
        self.ref_forces_key = ref_forces_key
        self.ref_stress_key = ref_stress_key
        # Initialize default metrics if not already registered
        self._register_default_metrics()
        
        # Track analysis state
        self._last_results: Optional[AnalysisResults] = None
        
    def _register_default_metrics(self) -> None:
        """Register default metrics if not already present."""
        # Check if key metrics are already registered
        if "force_stats" not in self.metric_registry.list_metrics():
            # Register statistical metrics
            self.metric_registry.register("force_stats", ForceErrorStats())
            self.metric_registry.register("energy_stats", EnergyErrorStats())
            self.metric_registry.register("stress_stats", StressErrorStats())
    
    def register_metric(self, name: str, metric: Any, **kwargs) -> None:
        """Register a custom metric.
        
        Parameters
        ----------
        name : str
            Name for the metric.
        metric : Any
            Metric function or class implementing calculate method.
        **kwargs
            Additional arguments for metric registration.
        """
        self.metric_registry.register(name, metric, **kwargs)
        logger.info(f"Registered custom metric: {name}")
    
    def run(
        self,
        structure_ids: Optional[List[int]] = None,
        atoms_list: Optional[List[Atoms]] = None,
        batch_size: int = 32,
        metrics: Optional[List[str]] = None,
        spatial_k: int = 12,
        dbscan_eps: float = 2.5,
        dbscan_min_samples: int = 3,
        difficulty_weights: Optional[Dict[str, float]] = None,
        check_geometry_sanity: bool = True,
        geometry_cutoff: float = 1.2
    ) -> AnalysisResults:
        """Run the full analysis pipeline on a list of structures.
        
        This method can be run in two modes:
        1. By providing `structure_ids`: Fetches structures from the database.
        2. By providing `atoms_list`: Uses a pre-loaded list of Atoms objects.
        
        If both are provided, `atoms_list` takes precedence.

        Parameters
        ----------
        structure_ids : Optional[List[int]]
            List of structure_ids to analyse from the database.
        atoms_list : Optional[List[Atoms]]
            A list of ASE Atoms objects to analyze directly.
        batch_size : int, optional
            Number of structures to process in each batch (default: 32).
        metrics : Optional[List[str]]
            List of metric names to calculate. If None, uses defaults.
        spatial_k : int, optional
            Number of nearest neighbors for Moran's I (default: 12).
        dbscan_eps : float, optional
            DBSCAN epsilon parameter (default: 2.5).
        dbscan_min_samples : int, optional
            DBSCAN min_samples parameter (default: 3).
        difficulty_weights : Optional[Dict[str, float]]
            Weights for difficulty score components.
        check_geometry_sanity : bool, optional
            Whether to check geometry sanity (default: True).
        geometry_cutoff : float, optional
            Cutoff for geometry checks (default: 1.2).
            
        Returns
        -------
        AnalysisResults
            Results object containing all analysis data.
        """
        if atoms_list is not None:
            logger.info(f"Starting analysis of {len(atoms_list)} structures from provided list")
            num_structures = len(atoms_list)
            data_iterator = atoms_list
            is_db_mode = False
        elif structure_ids is not None:
            logger.info(f"Starting analysis of {len(structure_ids)} structures from database")
            num_structures = len(structure_ids)
            data_iterator = structure_ids
            is_db_mode = True
        else:
            raise ValueError("Either 'structure_ids' or 'atoms_list' must be provided.")
        
        # Set default metrics if not specified
        if metrics is None:
            metrics = ["force_stats", "energy_stats", "stress_stats"]
        
        # Initialize results containers
        struct_rows = []
        atom_rows = []
        results_cache = {}
        
        # Record metadata
        metadata = {
            "analysis_timestamp": datetime.now().isoformat(),
            "n_structures": num_structures,
            "n_calculators": self.evaluator.n_calculators if self.evaluator else 0,
            "calculator_names": self.evaluator.calculator_names if self.evaluator else [],
            "ref_calc_name": self.ref_calc_name,
            "metrics_used": metrics,
            "parameters": {
                "batch_size": batch_size,
                "spatial_k": spatial_k,
                "dbscan_eps": dbscan_eps,
                "dbscan_min_samples": dbscan_min_samples,
                "difficulty_weights": difficulty_weights,
                "check_geometry": check_geometry_sanity,
                "geometry_cutoff": geometry_cutoff
            }
        }
        
        # Process in batches
        for i in tqdm(range(0, num_structures, batch_size), desc="Processing batches"):
            if is_db_mode:
                batch_ids = data_iterator[i:i+batch_size]
                
                # Get structures with reference calculations
                batch_atoms = self.db.get_batch_atoms_with_calculation(
                    batch_ids, calculator=self.ref_calc_name
                )
            else:
                batch_atoms = data_iterator[i:i+batch_size]

            if not batch_atoms:
                logger.warning(f"No structures found for batch {i//batch_size + 1}")
                continue
            
            # Evaluate with models
            if self.evaluator:
                predictions_list = self.evaluator.evaluate(batch_atoms)
            else:
                predictions_list = [{}] * len(batch_atoms)
            
            # Process each structure
            for atoms, preds_dict in zip(batch_atoms, predictions_list):
                if 'structure_id' not in atoms.info:
                    logger.warning("Structure missing 'structure_id' in .info, skipping.")
                    continue

                struct_id = atoms.info['structure_id']
                n_atoms = len(atoms)
                
                # Get reference data
                ref_energy = atoms.info.get(self.ref_energy_key)
                ref_forces = atoms.arrays.get(self.ref_forces_key)
                ref_stress = atoms.info.get(self.ref_stress_key)
                
                if ref_forces is None:
                    logger.warning(
                        f"No reference forces for structure {struct_id} "
                        f"using key '{self.ref_forces_key}'"
                    )
                    continue
                
                # Initialize structure metrics
                struct_metrics = {
                    'structure_id': struct_id,
                    'n_atoms': n_atoms,
                    'formula': atoms.get_chemical_formula(),
                    'config_type': atoms.info.get('config_type', 'unknown')
                }
                
                # Check geometry if requested
                if check_geometry_sanity:
                    geom_check = check_geometry(atoms, cutoff=geometry_cutoff)
                    struct_metrics['geometry_valid'] = geom_check['is_valid']
                    struct_metrics['has_duplicates'] = geom_check['has_duplicates']
                    struct_metrics['has_close_atoms'] = geom_check['has_close_atoms']
                
                # Process predictions (handle ensemble)
                force_errors_by_model = []
                all_predictions = []
                
                for calc_name, preds in preds_dict.items():
                    if preds is None:
                        continue
                    
                    all_predictions.append(preds)
                    
                    # Calculate force errors
                    if 'forces' in preds:
                        force_error = preds['forces'] - ref_forces
                        force_errors_by_model.append(force_error)
                
                if not force_errors_by_model:
                    logger.warning(f"No valid predictions for structure {struct_id}")
                    continue
                
                # Use mean prediction for main metrics
                mean_force_error = np.mean(force_errors_by_model, axis=0)
                force_error_magnitudes = np.linalg.norm(mean_force_error, axis=1)
                
                # Calculate metrics
                for metric_name in metrics:
                    try:
                        if metric_name == "force_stats":
                            metric_results = self.metric_registry.calculate(
                                metric_name,
                                all_predictions[0]['forces'],
                                ref_forces
                            )
                            struct_metrics.update(metric_results)
                        
                        elif metric_name == "energy_stats" and ref_energy is not None:
                            metric_results = self.metric_registry.calculate(
                                metric_name,
                                all_predictions[0].get('energy', ref_energy),
                                ref_energy,
                                n_atoms=n_atoms
                            )
                            struct_metrics.update(metric_results)
                        
                        elif metric_name == "stress_stats" and ref_stress is not None:
                            metric_results = self.metric_registry.calculate(
                                metric_name,
                                all_predictions[0].get('stress', ref_stress),
                                ref_stress
                            )
                            struct_metrics.update(metric_results)
                        
                        else:
                            # Custom metric
                            if metric_name in self.metric_registry.list_metrics():
                                metric_results = self.metric_registry.calculate(
                                    metric_name,
                                    all_predictions[0].get('forces', ref_forces),
                                    ref_forces
                                )
                                struct_metrics.update(metric_results)
                    
                    except Exception as e:
                        logger.warning(f"Failed to calculate {metric_name} for structure {struct_id}: {e}")
                
                # Spatial analysis
                spatial_results = analyze_spatial_patterns(
                    atoms,
                    force_error_magnitudes,
                    k=spatial_k,
                    eps=dbscan_eps,
                    min_samples=dbscan_min_samples
                )
                
                # Extract key spatial metrics
                struct_metrics['morans_i_global_metric'] = spatial_results.get('morans_i_global_metric', 0.0)
                struct_metrics['morans_i_pvalue_metric'] = spatial_results.get('morans_i_pvalue_metric', 1.0)
                struct_metrics['n_error_clusters_metric'] = spatial_results.get('n_error_clusters_metric', 0)
                
                # Ensemble variance
                ensemble_var = calculate_ensemble_variance(all_predictions, n_atoms)
                struct_metrics.update(ensemble_var)
                
                # Difficulty score
                difficulty_scores = calculate_difficulty_score(
                    force_error_magnitudes,
                    spatial_results.get('morans_i_global_metric', 0.0),
                    spatial_results.get('cluster_labels', np.array([])),
                    ensemble_var.get('force_variance_metric', 0.0),
                    n_atoms,
                    weights_dict=difficulty_weights
                )
                struct_metrics.update(difficulty_scores)
                
                # Add to results
                struct_rows.append(struct_metrics)
                
                # Per-atom data
                for atom_idx in range(n_atoms):
                    atom_row = {
                        'structure_id': struct_id,
                        'atom_index': atom_idx,
                        'symbol': atoms.symbols[atom_idx],
                        'force_error_x': mean_force_error[atom_idx, 0],
                        'force_error_y': mean_force_error[atom_idx, 1],
                        'force_error_z': mean_force_error[atom_idx, 2],
                        'force_error_mag': force_error_magnitudes[atom_idx]
                    }
                    
                    # Add spatial data if available
                    if 'local_morans_i' in spatial_results:
                        atom_row['local_moran_i'] = spatial_results['local_morans_i'][atom_idx]
                    if 'cluster_labels' in spatial_results:
                        atom_row['dbscan_cluster'] = spatial_results['cluster_labels'][atom_idx]
                    
                    atom_rows.append(atom_row)
                
                # Cache results for plotting
                results_cache[struct_id] = {
                    'atoms': atoms,
                    'force_error_magnitudes': force_error_magnitudes,
                    'force_error_vectors': mean_force_error,
                    'spatial_results': spatial_results,
                    'predictions': all_predictions
                }
        
        # Create results object
        results = AnalysisResults(
            structure_metrics=pd.DataFrame(struct_rows),
            atom_metrics=pd.DataFrame(atom_rows),
            metadata=metadata,
            results_cache=results_cache
        )
        
        # Store for convenience methods
        self._last_results = results
        
        logger.info("Analysis complete")
        return results
    
    def save(self, output_dir: Union[str, Path], **kwargs) -> None:
        """Save the last analysis results.
        
        Parameters
        ----------
        output_dir : Union[str, Path]
            Directory to save results in.
        **kwargs
            Additional arguments for save_analysis_results.
        """
        if self._last_results is None:
            raise RuntimeError("No analysis results to save. Run analysis first.")
        
        save_analysis_results(self._last_results, output_dir, **kwargs)
    
    @classmethod
    def load(
        cls,
        input_dir: Union[str, Path],
        db_manager: Optional[DatabaseManager] = None,
        calculators: Optional[Any] = None
    ) -> 'ErrorAnalyser':
        """Load a previously saved analysis.
        
        Parameters
        ----------
        input_dir : Union[str, Path]
            Directory containing saved analysis.
        db_manager : Optional[DatabaseManager]
            Database manager instance.
        calculators : Optional[Any]
            Calculators for further analysis.
            
        Returns
        -------
        ErrorAnalyser
            Analyser instance with loaded results.
        """
        from ..utils import load_analysis_results
        
        # Create instance
        instance = cls(db_manager, calculators)
        
        # Load data
        loaded_data = load_analysis_results(input_dir)
        
        # Create results object
        instance._last_results = AnalysisResults(
            structure_metrics=loaded_data['structure_metrics'],
            atom_metrics=loaded_data['atom_metrics'],
            metadata=loaded_data.get('metadata', {}),
            results_cache=loaded_data.get('results_cache', {})
        )
        
        logger.info(f"Loaded analysis from {input_dir}")
        return instance
    
    @property
    def results(self) -> Optional[AnalysisResults]:
        """Get the last analysis results."""
        return self._last_results 