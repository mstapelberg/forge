"""Base metric protocol for forge analysis system."""
from typing import Protocol, Dict, Any, Optional
import numpy as np


class Metric(Protocol):
    """Protocol defining the interface for all metrics.
    
    All metrics should implement a calculate method that takes predicted
    and reference values and returns a dictionary of results.
    """
    
    def calculate(
        self, 
        pred: np.ndarray, 
        ref: np.ndarray, 
        **kwargs: Any
    ) -> Dict[str, float]:
        """Calculate metric values.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted values from the model.
        ref : np.ndarray
            Reference/true values.
        **kwargs : Any
            Additional parameters specific to the metric.
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric values. Keys should end with '_metric'.
        """
        ...


class BaseMetric:
    """Base class for metric implementations with common functionality."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize metric.
        
        Parameters
        ----------
        name : Optional[str]
            Name of the metric. If None, uses class name.
        """
        self.name = name or self.__class__.__name__.lower()
    
    def validate_inputs(self, pred: np.ndarray, ref: np.ndarray) -> None:
        """Validate input arrays.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted values.
        ref : np.ndarray
            Reference values.
            
        Raises
        ------
        ValueError
            If inputs have incompatible shapes or types.
        """
        if not isinstance(pred, np.ndarray) or not isinstance(ref, np.ndarray):
            raise ValueError("Inputs must be numpy arrays")
        
        if pred.shape != ref.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} != ref {ref.shape}"
            )
    
    def _add_metric_suffix(self, results: Dict[str, float]) -> Dict[str, float]:
        """Ensure all result keys end with '_metric'.
        
        Parameters
        ----------
        results : Dict[str, float]
            Raw metric results.
            
        Returns
        -------
        Dict[str, float]
            Results with '_metric' suffix added to keys.
        """
        return {
            k if k.endswith('_metric') else f"{k}_metric": v 
            for k, v in results.items()
        } 