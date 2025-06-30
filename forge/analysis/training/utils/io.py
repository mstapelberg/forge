"""I/O utilities for saving analysis results and generating reports."""
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
from ase import Atoms
from ase.io import write as ase_write
import logging

logger = logging.getLogger(__name__)


def save_analysis_results(
    results: Any,  # AnalysisResults object
    output_dir: Union[str, Path],
    save_pickle: bool = True,
    save_csv: bool = True,
    save_json: bool = True
) -> None:
    """Save analysis results to multiple formats.
    
    Parameters
    ----------
    results : AnalysisResults
        The analysis results object to save.
    output_dir : Union[str, Path]
        Directory to save results in.
    save_pickle : bool
        Whether to save pickle files for full data.
    save_csv : bool
        Whether to save CSV files for dataframes.
    save_json : bool
        Whether to save JSON summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrames
    if save_csv:
        results.structure_metrics.to_csv(
            output_dir / "structure_metrics.csv", index=False
        )
        results.atom_metrics.to_csv(
            output_dir / "atom_metrics.csv", index=False
        )
        logger.info(f"Saved CSV files to {output_dir}")
    
    if save_pickle:
        results.structure_metrics.to_pickle(output_dir / "structure_metrics.pkl")
        results.atom_metrics.to_pickle(output_dir / "atom_metrics.pkl")
        
        # Save results cache
        with open(output_dir / "results_cache.pkl", "wb") as f:
            pickle.dump(results.results_cache, f)
        
        # Save metadata
        with open(output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(results.metadata, f)
        
        logger.info(f"Saved pickle files to {output_dir}")
    
    if save_json:
        # Create summary for JSON
        summary = {
            "metadata": results.metadata,
            "summary_statistics": results.summary_statistics(),
            "n_structures": len(results.structure_metrics),
            "n_atoms": len(results.atom_metrics)
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved JSON summary to {output_dir}")


def load_analysis_results(
    input_dir: Union[str, Path],
    load_pickle: bool = True
) -> Dict[str, Any]:
    """Load previously saved analysis results.
    
    Parameters
    ----------
    input_dir : Union[str, Path]
        Directory containing saved results.
    load_pickle : bool
        Whether to load from pickle (faster) or CSV.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded data.
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    results = {}
    
    if load_pickle and (input_dir / "structure_metrics.pkl").exists():
        results["structure_metrics"] = pd.read_pickle(
            input_dir / "structure_metrics.pkl"
        )
        results["atom_metrics"] = pd.read_pickle(
            input_dir / "atom_metrics.pkl"
        )
        
        with open(input_dir / "results_cache.pkl", "rb") as f:
            results["results_cache"] = pickle.load(f)
        
        if (input_dir / "metadata.pkl").exists():
            with open(input_dir / "metadata.pkl", "rb") as f:
                results["metadata"] = pickle.load(f)
        
        logger.info(f"Loaded results from pickle files in {input_dir}")
        
    else:
        # Load from CSV
        results["structure_metrics"] = pd.read_csv(
            input_dir / "structure_metrics.csv"
        )
        results["atom_metrics"] = pd.read_csv(
            input_dir / "atom_metrics.csv"
        )
        
        # Try to load summary JSON for metadata
        if (input_dir / "summary.json").exists():
            with open(input_dir / "summary.json", "r") as f:
                summary = json.load(f)
                results["metadata"] = summary.get("metadata", {})
        
        logger.info(f"Loaded results from CSV files in {input_dir}")
    
    return results


def export_structures_to_extxyz(
    atoms_list: List[Atoms],
    results: Any,  # AnalysisResults object
    output_path: Union[str, Path],
    include_metrics: bool = True
) -> None:
    """Export structures with analysis data to extended XYZ format.
    
    Parameters
    ----------
    atoms_list : List[Atoms]
        List of ASE Atoms objects to export.
    results : AnalysisResults
        Analysis results containing metrics to attach.
    output_path : Union[str, Path]
        Path for output .extxyz file.
    include_metrics : bool
        Whether to include analysis metrics in the file.
    """
    output_path = Path(output_path)
    
    atoms_to_write = []
    
    for atoms in atoms_list:
        atoms_copy = atoms.copy()
        structure_id = atoms.info.get("structure_id")
        
        if include_metrics and structure_id is not None:
            # Add structure-level metrics to info
            struct_mask = results.structure_metrics["structure_id"] == structure_id
            if struct_mask.any():
                struct_data = results.structure_metrics[struct_mask].iloc[0]
                
                # Add key metrics to info
                for key in ["difficulty_metric", "force_rmse_metric", 
                           "force_kurtosis_metric", "force_gini_metric"]:
                    if key in struct_data:
                        atoms_copy.info[key] = float(struct_data[key])
            
            # Add atom-level metrics as arrays
            atom_mask = results.atom_metrics["structure_id"] == structure_id
            if atom_mask.any():
                atom_data = results.atom_metrics[atom_mask]
                
                if "force_error_mag" in atom_data.columns:
                    atoms_copy.new_array(
                        "force_error_mag",
                        atom_data["force_error_mag"].values
                    )
                
                if "local_moran_i" in atom_data.columns:
                    atoms_copy.new_array(
                        "local_moran_i",
                        atom_data["local_moran_i"].values
                    )
                
                if "dbscan_cluster" in atom_data.columns:
                    atoms_copy.new_array(
                        "dbscan_cluster",
                        atom_data["dbscan_cluster"].values
                    )
        
        atoms_to_write.append(atoms_copy)
    
    # Write all structures
    ase_write(output_path, atoms_to_write, format="extxyz")
    logger.info(f"Exported {len(atoms_to_write)} structures to {output_path}")


def generate_markdown_report(
    results: Any,  # AnalysisResults object
    output_path: Optional[Union[str, Path]] = None,
    top_n: int = 10
) -> str:
    """Generate a markdown report of the analysis results.
    
    Parameters
    ----------
    results : AnalysisResults
        The analysis results.
    output_path : Optional[Union[str, Path]]
        If provided, saves the report to this path.
    top_n : int
        Number of top difficult structures to include.
        
    Returns
    -------
    str
        The markdown report as a string.
    """
    summary = results.summary_statistics()
    
    report_lines = [
        "# Force Field Analysis Report",
        "",
        "## Summary Statistics",
        "",
        f"- Total Structures: {summary['n_structures']}",
        f"- Total Atoms: {summary['n_atoms_total']}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|--------|------|-----|-----|-----|"
    ]
    
    # Add metric statistics
    for metric, stats in summary["metrics"].items():
        if metric != "structure_id":
            report_lines.append(
                f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                f"{stats['min']:.4f} | {stats['max']:.4f} |"
            )
    
    # Add top difficult structures
    report_lines.extend([
        "",
        f"## Top {top_n} Most Difficult Structures",
        "",
        "| Rank | Structure ID | Difficulty Score | Force RMSE | Kurtosis |",
        "|------|--------------|------------------|------------|----------|"
    ])
    
    difficult_ids = results.get_difficult_structures(top_n=top_n)
    for i, sid in enumerate(difficult_ids):
        struct_data = results.structure_metrics[
            results.structure_metrics["structure_id"] == sid
        ].iloc[0]
        
        report_lines.append(
            f"| {i+1} | {sid} | "
            f"{struct_data.get('difficulty_metric', 0):.3f} | "
            f"{struct_data.get('force_rmse_metric', 0):.4f} | "
            f"{struct_data.get('force_kurtosis_metric', 0):.2f} |"
        )
    
    report = "\n".join(report_lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.write_text(report)
        logger.info(f"Saved markdown report to {output_path}")
    
    return report


def export_for_visualization(
    results: Any,  # AnalysisResults object
    structure_ids: Optional[List[int]] = None,
    output_format: str = "json",
    output_path: Optional[Union[str, Path]] = None
) -> Union[Dict, str]:
    """Export results in formats suitable for visualization tools.
    
    Parameters
    ----------
    results : AnalysisResults
        The analysis results.
    structure_ids : Optional[List[int]]
        Specific structures to export. If None, exports all.
    output_format : str
        Format to export in ('json', 'csv', 'parquet').
    output_path : Optional[Union[str, Path]]
        Path to save the export.
        
    Returns
    -------
    Union[Dict, str]
        The exported data (dict for json, path string for files).
    """
    if structure_ids is None:
        structure_ids = results.structure_metrics["structure_id"].tolist()
    
    # Filter data
    struct_mask = results.structure_metrics["structure_id"].isin(structure_ids)
    atom_mask = results.atom_metrics["structure_id"].isin(structure_ids)
    
    filtered_struct = results.structure_metrics[struct_mask]
    filtered_atom = results.atom_metrics[atom_mask]
    
    if output_format == "json":
        export_data = {
            "structures": filtered_struct.to_dict("records"),
            "atoms": filtered_atom.to_dict("records"),
            "metadata": results.metadata
        }
        
        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Exported to JSON: {output_path}")
            return str(output_path)
        return export_data
    
    elif output_format == "csv":
        if output_path:
            output_path = Path(output_path)
            base_path = output_path.parent / output_path.stem
            
            struct_path = f"{base_path}_structures.csv"
            atom_path = f"{base_path}_atoms.csv"
            
            filtered_struct.to_csv(struct_path, index=False)
            filtered_atom.to_csv(atom_path, index=False)
            
            logger.info(f"Exported to CSV: {struct_path}, {atom_path}")
            return f"{struct_path}, {atom_path}"
        else:
            raise ValueError("output_path required for CSV export")
    
    elif output_format == "parquet":
        if output_path:
            output_path = Path(output_path)
            base_path = output_path.parent / output_path.stem
            
            struct_path = f"{base_path}_structures.parquet"
            atom_path = f"{base_path}_atoms.parquet"
            
            filtered_struct.to_parquet(struct_path)
            filtered_atom.to_parquet(atom_path)
            
            logger.info(f"Exported to Parquet: {struct_path}, {atom_path}")
            return f"{struct_path}, {atom_path}"
        else:
            raise ValueError("output_path required for Parquet export")
    
    else:
        raise ValueError(f"Unsupported format: {output_format}") 