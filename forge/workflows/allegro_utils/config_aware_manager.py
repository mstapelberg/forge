"""
Config-aware MetricsManager wrapper for NequIP.

This mirrors NequIP's MetricsManager interface but supports passing extra
batch fields (e.g., section_id) to metrics that need them.
"""

from typing import Dict, Any, Iterable

import torch
import torch.nn as nn


class ConfigAwareMetricsManager(nn.Module):
    """A drop-in replacement for NequIP's MetricsManager with extra inputs.

    The `metrics` argument is a list of dicts, each with keys:
      - name: metric name
      - field: data field name or a modifier config (already instantiated by Hydra)
      - metric: instantiated metric module (must have update/reset/compute)
      - coeff: optional scalar weight
      - extra_inputs: optional list of extra batch keys to forward to metric.update
    """

    def __init__(self, metrics: Iterable[Dict[str, Any]], **kwargs):
        super().__init__()
        self._metrics = nn.ModuleDict()
        self._field_map: Dict[str, Any] = {}
        self._coeff: Dict[str, float] = {}
        self._extras: Dict[str, Iterable[str]] = {}
        # Accept unused kwargs for compatibility with upstream configs (e.g., type_names)
        self._extra_kwargs: Dict[str, Any] = dict(kwargs)
        # Signal to NequIP Lightning that this manager returns a single weighted sum
        self.do_weighted_sum: bool = True
        # Cache last computed per-metric values for logging
        self._last_log: Dict[str, torch.Tensor] = {}
        # Default logging delimiter to mirror NequIP/Lightning usage
        self.logging_delimiter: str = "/"
        # Known prefixes some NequIP Lightning code expects without passing explicitly
        self._default_prefixes = ["train_loss_step"]

        for spec in metrics:
            name = spec["name"]
            metric = spec["metric"]
            field = spec["field"]
            coeff = float(spec.get("coeff", 1.0))
            extra_inputs = spec.get("extra_inputs", [])

            self._metrics[name] = metric
            self._field_map[name] = field
            self._coeff[name] = coeff
            self._extras[name] = list(extra_inputs) if extra_inputs else []

    @property
    def metrics(self) -> Dict[str, Dict[str, Any]]:
        """Expose a dict-like view compatible with NequIP callbacks.

        Returns a mapping from metric name to a small dict carrying at least
        the metric module and its coefficient. This mirrors the shape expected
        by callbacks like LossCoefficientMonitor that iterate `.metrics.items()`
        and read the `coeff` value.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name in self._metrics.keys():
            result[name] = {
                "metric": self._metrics[name],
                "coeff": self._coeff.get(name, 1.0),
                "field": self._field_map.get(name),
                "extra_inputs": self._extras.get(name, ()),
            }
        return result

    def keys(self):
        return self._metrics.keys()

    def values(self):
        return self._metrics.values()

    def items(self):
        return self._metrics.items()

    def __getitem__(self, name: str):
        return self._metrics[name]

    def reset(self):
        for m in self._metrics.values():
            if hasattr(m, "reset"):
                m.reset()

    def update(self, pred: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        for name, metric in self._metrics.items():
            field = self._field_map[name]
            coeff = self._coeff[name]
            extras = self._extras[name]

            # Resolve field/modifier
            if hasattr(field, "__call__") and not isinstance(field, str):
                # Per-atom modifier or similar callable
                y_pred = field(pred)
                y_ref = field(ref)
            else:
                y_pred = pred[field]
                y_ref = ref[field]

            # Build kwargs for extra inputs
            kw = {}
            for k in extras:
                if k in ref:
                    kw[k] = ref[k]
                elif k in pred:
                    kw[k] = pred[k]

            metric.update(y_pred, y_ref, **kw)

            # Prefer differentiable per-batch values if exposed by the metric
            val_for_backprop = None
            if hasattr(metric, "last_batch_value"):
                lbv = getattr(metric, "last_batch_value")
                if (lbv is not None) and isinstance(lbv, torch.Tensor) and lbv.requires_grad:
                    val_for_backprop = lbv
            # If not provided (or detached), make a differentiable fallback
            if val_for_backprop is None:
                metric_class = metric.__class__.__name__
                try:
                    if metric_class in {"MeanSquaredError", "MSE", "MeanSquaredErrorLoss"}:
                        diff = y_pred - y_ref
                        val_for_backprop = diff * diff  # elementwise squared error
                    elif metric_class in {"MeanAbsoluteError", "L1Loss", "MAE"}:
                        val_for_backprop = torch.abs(y_pred - y_ref)
                    else:
                        # Generic differentiable fallback: MSE
                        diff = y_pred - y_ref
                        val_for_backprop = diff * diff
                except Exception:
                    # If shapes/type don't match, use a small differentiable proxy
                    val_for_backprop = (y_pred - y_pred).sum() * 0.0

            # At this point we must have a differentiable tensor
            losses[name] = coeff * (val_for_backprop.mean() if getattr(val_for_backprop, "ndim", 0) > 0 else val_for_backprop)
        # Also build logging dict with unweighted per-metric values
        log: Dict[str, torch.Tensor] = {}
        for name, metric in self._metrics.items():
            v = None
            if hasattr(metric, "last_batch_value") and getattr(metric, "last_batch_value") is not None:
                v = metric.last_batch_value
            elif hasattr(metric, "compute"):
                try:
                    v = metric.compute()
                except Exception:
                    v = None
            if v is None:
                v = losses.get(name)
            if v is not None:
                log[name] = v.mean() if v.ndim > 0 else v
        # Weighted sum
        total = None
        for v in losses.values():
            total = v if total is None else (total + v)
        if total is not None:
            log["weighted_sum"] = total
        self._last_log = log
        return losses

    def __call__(self, pred: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        # Return a dict with per-metric values and a 'weighted_sum' key for logging & loss
        losses = self.update(pred, ref)
        if not losses:
            device = None
            if pred:
                try:
                    device = next(iter(pred.values())).device
                except Exception:
                    device = None
            self._last_log = {"weighted_sum": torch.tensor(0.0, device=device)}
            base = self._last_log
        else:
            # Ensure 'weighted_sum' present in last_log
            if "weighted_sum" not in self._last_log:
                total = None
                for v in losses.values():
                    total = v if total is None else (total + v)
                self._last_log["weighted_sum"] = total
            base = self._last_log

        # Build final mapping including optional prefixes
        final: Dict[str, torch.Tensor] = dict(base)
        key_prefix = kwargs.get("key_prefix") or kwargs.get("prefix")
        prefixes = list(self._default_prefixes)
        if key_prefix and key_prefix not in prefixes:
            prefixes.append(key_prefix)
        for pfx in prefixes:
            for k, v in base.items():
                final[f"{pfx}{self.logging_delimiter}{k}"] = v
        return final

    def compute(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute epoch-level metric values and weighted sum.

        This mirrors NequIP's MetricsManager.compute() behavior: per-metric
        aggregated values are obtained via each metric's own `compute()` when
        available and combined using this manager's coefficients. The returned
        dict also includes a `weighted_sum` entry and prefixed duplicates for
        logging, similar to `__call__`.

        Args:
            *args: Unused, accepted for API compatibility.
            **kwargs: May include `key_prefix` or `prefix` to control logging keys.

        Returns:
            Dict[str, torch.Tensor]: A mapping with per-metric values and
            `weighted_sum`, with optional prefixed keys added.
        """
        log: Dict[str, torch.Tensor] = {}
        device = None
        # Gather per-metric aggregates
        for name, metric in self._metrics.items():
            v = None
            if hasattr(metric, "compute"):
                try:
                    v = metric.compute()
                except Exception:
                    v = None
            if v is None and hasattr(metric, "last_batch_value") and getattr(metric, "last_batch_value") is not None:
                v = metric.last_batch_value
            if v is None:
                continue
            if device is None and isinstance(v, torch.Tensor):
                device = v.device
            log[name] = v.mean() if isinstance(v, torch.Tensor) and v.ndim > 0 else v

        # Weighted sum from aggregates
        total = None
        for name, val in log.items():
            coeff = self._coeff.get(name, 1.0)
            total = (val * coeff) if total is None else (total + val * coeff)
        if total is None:
            total = torch.tensor(0.0, device=device)
        log["weighted_sum"] = total

        self._last_log = log

        # Build final mapping including optional prefixes
        final: Dict[str, torch.Tensor] = dict(log)
        key_prefix = kwargs.get("key_prefix") or kwargs.get("prefix")
        prefixes = list(self._default_prefixes)
        if key_prefix and key_prefix not in prefixes:
            prefixes.append(key_prefix)
        for pfx in prefixes:
            for k, v in log.items():
                final[f"{pfx}{self.logging_delimiter}{k}"] = v
        return final


