"""Metric registry for managing analysis metrics."""
from typing import Dict, Callable, Any, Optional, List
import numpy as np
import logging
from .base import Metric, BaseMetric

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Registry for managing analysis metrics.
    
    This registry allows registration of built-in and custom metrics,
    providing a centralized way to access and apply metrics during analysis.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._metrics: Dict[str, Callable] = {}
        self._metric_info: Dict[str, Dict[str, Any]] = {}
        
    def register(
        self,
        name: str,
        metric: Callable,
        description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> None:
        """Register a metric function.
        
        Parameters
        ----------
        name : str
            Name to register the metric under.
        metric : Callable
            Metric function or class that implements the Metric protocol.
        description : Optional[str]
            Description of what the metric calculates.
        params : Optional[Dict[str, Any]]
            Default parameters for the metric.
        overwrite : bool
            Whether to overwrite existing metric with same name.
            
        Raises
        ------
        ValueError
            If metric with name already exists and overwrite=False.
        TypeError
            If metric doesn't have required interface.
        """
        if name in self._metrics and not overwrite:
            raise ValueError(
                f"Metric '{name}' already registered. "
                "Set overwrite=True to replace."
            )
        
        # Validate metric has correct interface
        if hasattr(metric, 'calculate'):
            # It's a class implementing Metric protocol
            self._metrics[name] = metric
        elif callable(metric):
            # It's a function - wrap it to match interface
            self._metrics[name] = self._wrap_function(metric)
        else:
            raise TypeError(
                f"Metric must be callable or implement Metric protocol"
            )
        
        # Store metadata
        self._metric_info[name] = {
            'description': description or "No description provided",
            'params': params or {},
            'type': 'class' if hasattr(metric, 'calculate') else 'function'
        }
        
        logger.info(f"Registered metric: {name}")
    
    def _wrap_function(self, func: Callable) -> Metric:
        """Wrap a function to match the Metric protocol.
        
        Parameters
        ----------
        func : Callable
            Function taking (pred, ref, **kwargs) and returning dict.
            
        Returns
        -------
        Metric
            Wrapped function implementing Metric protocol.
        """
        class FunctionWrapper(BaseMetric):
            def calculate(self, pred: np.ndarray, ref: np.ndarray, **kwargs) -> Dict[str, float]:
                self.validate_inputs(pred, ref)
                results = func(pred, ref, **kwargs)
                
                # Ensure results is a dict
                if isinstance(results, (int, float)):
                    results = {func.__name__: float(results)}
                elif not isinstance(results, dict):
                    raise ValueError(
                        f"Metric function must return dict or scalar, "
                        f"got {type(results)}"
                    )
                
                return self._add_metric_suffix(results)
        
        return FunctionWrapper(name=func.__name__)
    
    def get(self, name: str) -> Callable:
        """Get a registered metric.
        
        Parameters
        ----------
        name : str
            Name of the metric.
            
        Returns
        -------
        Callable
            The metric function/class.
            
        Raises
        ------
        KeyError
            If metric not found.
        """
        if name not in self._metrics:
            available = list(self._metrics.keys())
            raise KeyError(
                f"Metric '{name}' not found. "
                f"Available metrics: {available}"
            )
        return self._metrics[name]
    
    def calculate(
        self,
        name: str,
        pred: np.ndarray,
        ref: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate a metric by name.
        
        Parameters
        ----------
        name : str
            Name of the metric.
        pred : np.ndarray
            Predicted values.
        ref : np.ndarray
            Reference values.
        **kwargs
            Additional parameters for the metric.
            
        Returns
        -------
        Dict[str, float]
            Metric results with keys ending in '_metric'.
        """
        metric = self.get(name)
        
        # Merge default params with provided kwargs
        default_params = self._metric_info[name].get('params', {})
        params = {**default_params, **kwargs}
        
        if hasattr(metric, 'calculate'):
            # It's a class instance
            if isinstance(metric, type):
                # Need to instantiate
                instance = metric()
                return instance.calculate(pred, ref, **params)
            else:
                # Already instantiated
                return metric.calculate(pred, ref, **params)
        else:
            # It's a wrapped function
            return metric().calculate(pred, ref, **params)
    
    def list_metrics(self) -> List[str]:
        """List all registered metric names.
        
        Returns
        -------
        List[str]
            Sorted list of metric names.
        """
        return sorted(list(self._metrics.keys()))
    
    def get_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about registered metrics.
        
        Parameters
        ----------
        name : Optional[str]
            Specific metric to get info for. If None, returns all.
            
        Returns
        -------
        Dict[str, Any]
            Metric information.
        """
        if name is not None:
            if name not in self._metric_info:
                raise KeyError(f"Metric '{name}' not found")
            return {name: self._metric_info[name]}
        return self._metric_info.copy()
    
    def clear(self) -> None:
        """Clear all registered metrics."""
        self._metrics.clear()
        self._metric_info.clear()
        logger.info("Cleared metric registry")


# Global registry instance
_global_registry = MetricRegistry()


def register_metric(
    name: str,
    metric: Optional[Callable] = None,
    **kwargs
) -> Callable:
    """Register a metric in the global registry.
    
    Can be used as a decorator or called directly.
    
    Parameters
    ----------
    name : str
        Name to register the metric under.
    metric : Optional[Callable]
        Metric to register. If None, returns decorator.
    **kwargs
        Additional arguments passed to registry.register().
        
    Returns
    -------
    Callable
        The metric (if called directly) or decorator.
    """
    def decorator(func: Callable) -> Callable:
        _global_registry.register(name, func, **kwargs)
        return func
    
    if metric is not None:
        _global_registry.register(name, metric, **kwargs)
        return metric
    
    return decorator


def get_metric(name: str) -> Callable:
    """Get a metric from the global registry."""
    return _global_registry.get(name)


def calculate_metric(name: str, pred: np.ndarray, ref: np.ndarray, **kwargs) -> Dict[str, float]:
    """Calculate a metric from the global registry."""
    return _global_registry.calculate(name, pred, ref, **kwargs)


def list_metrics() -> List[str]:
    """List all metrics in the global registry."""
    return _global_registry.list_metrics()


def get_registry() -> MetricRegistry:
    """Get the global metric registry instance."""
    return _global_registry 