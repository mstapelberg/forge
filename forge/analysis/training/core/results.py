"""Structured results storage for forge analysis."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Container for analysis results with convenient access methods.
    
    Attributes
    ----------
    structure_metrics : pd.DataFrame
        Per-structure aggregated metrics with columns for each metric.
    atom_metrics : pd.DataFrame
        Per-atom detailed metrics including force errors and spatial info.
    metadata : Dict[str, Any]
        Analysis parameters, timestamps, and configuration.
    results_cache : Dict[int, Dict[str, Any]]
        Cached intermediate results for plotting and detailed analysis.
    """
    
    structure_metrics: pd.DataFrame
    atom_metrics: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    results_cache: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    def filter_by_score(
        self, 
        metric: str, 
        threshold: float,
        above: bool = True
    ) -> List[int]:
        """Filter structures by a metric threshold.
        
        Parameters
        ----------
        metric : str
            Name of the metric column to filter by.
        threshold : float
            Threshold value.
        above : bool, optional
            If True, return structures with metric >= threshold.
            If False, return structures with metric < threshold.
            
        Returns
        -------
        List[int]
            List of structure IDs meeting the criteria.
        """
        if metric not in self.structure_metrics.columns:
            raise ValueError(
                f"Metric '{metric}' not found. "
                f"Available: {list(self.structure_metrics.columns)}"
            )
        
        if above:
            mask = self.structure_metrics[metric] >= threshold
        else:
            mask = self.structure_metrics[metric] < threshold
            
        return self.structure_metrics[mask]['structure_id'].tolist()
    
    def get_difficult_structures(
        self, 
        top_n: int = 100,
        metric: str = "difficulty_metric"
    ) -> List[int]:
        """Get the most difficult structures based on composite score.
        
        Parameters
        ----------
        top_n : int, optional
            Number of structures to return.
        metric : str, optional
            Metric to sort by for difficulty ranking.
            
        Returns
        -------
        List[int]
            List of structure IDs ordered by difficulty (highest first).
        """
        if metric not in self.structure_metrics.columns:
            raise ValueError(
                f"Metric '{metric}' not found. "
                f"Available metrics: {list(self.structure_metrics.columns)}"
            )
        
        sorted_df = self.structure_metrics.sort_values(
            by=metric, ascending=False
        )
        return sorted_df.head(top_n)['structure_id'].tolist()
    
    def get_structure_report(self, structure_id: int) -> Dict[str, Any]:
        """Get detailed report for a single structure.
        
        Parameters
        ----------
        structure_id : int
            ID of the structure to report on.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing structure metrics, atom details,
            and analysis summary.
        """
        # Get structure metrics
        struct_mask = self.structure_metrics['structure_id'] == structure_id
        if not struct_mask.any():
            raise ValueError(f"Structure {structure_id} not found in results")
        
        struct_data = self.structure_metrics[struct_mask].iloc[0].to_dict()
        
        # Get atom metrics
        atom_mask = self.atom_metrics['structure_id'] == structure_id
        atom_data = self.atom_metrics[atom_mask]
        
        # Get cached results if available
        cached = self.results_cache.get(structure_id, {})
        
        return {
            'structure_id': structure_id,
            'structure_metrics': struct_data,
            'atom_summary': {
                'n_atoms': len(atom_data),
                'max_force_error': atom_data['force_error_mag'].max() if not atom_data.empty else 0,
                'mean_force_error': atom_data['force_error_mag'].mean() if not atom_data.empty else 0,
                'error_clusters': len(atom_data[atom_data['dbscan_cluster'] >= 0]['dbscan_cluster'].unique()) if 'dbscan_cluster' in atom_data else 0
            },
            'top_error_atoms': atom_data.nlargest(5, 'force_error_mag')[
                ['atom_index', 'symbol', 'force_error_mag']
            ].to_dict('records') if not atom_data.empty else [],
            'cached_data_available': bool(cached)
        }
    
    def export_for_visualization(
        self, 
        structure_ids: Optional[List[int]] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """Export structures with analysis data for visualization.
        
        Parameters
        ----------
        structure_ids : Optional[List[int]]
            Specific structures to export. If None, exports all.
        output_path : Optional[Path]
            Output file path. If None, uses 'analysis_results.json'.
            
        Returns
        -------
        Path
            Path to the exported file.
        """
        if structure_ids is None:
            structure_ids = self.structure_metrics['structure_id'].tolist()
        
        if output_path is None:
            output_path = Path("analysis_results.json")
        
        export_data = {
            'metadata': self.metadata,
            'structures': []
        }
        
        for sid in structure_ids:
            try:
                report = self.get_structure_report(sid)
                export_data['structures'].append(report)
            except ValueError:
                logger.warning(f"Skipping structure {sid} - not found in results")
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(export_data['structures'])} structures to {output_path}")
        return output_path
    
    def summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the analysis.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics including metric distributions and counts.
        """
        numeric_cols = self.structure_metrics.select_dtypes(
            include=[np.number]
        ).columns
        
        summary = {
            'n_structures': len(self.structure_metrics),
            'n_atoms_total': len(self.atom_metrics),
            'metrics': {}
        }
        
        for col in numeric_cols:
            if col != 'structure_id':
                summary['metrics'][col] = {
                    'mean': float(self.structure_metrics[col].mean()),
                    'std': float(self.structure_metrics[col].std()),
                    'min': float(self.structure_metrics[col].min()),
                    'max': float(self.structure_metrics[col].max()),
                    'q25': float(self.structure_metrics[col].quantile(0.25)),
                    'q50': float(self.structure_metrics[col].quantile(0.50)),
                    'q75': float(self.structure_metrics[col].quantile(0.75)),
                }
        
        return summary 