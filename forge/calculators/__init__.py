"""Calculator abstractions and backends for ensemble calculations."""

from .interface import BaseEnsembleCalculator

# Conditional imports for backends
_available_backends = ['BaseEnsembleCalculator']

try:
    from .mace_backend import MACEBackend
    _available_backends.append('MACEBackend')
except ImportError:
    MACEBackend = None

try:
    from .allegro_backend import AllegroBackend
    _available_backends.append('AllegroBackend')
except ImportError:
    AllegroBackend = None

# Always import factory functions
from .factory import create_ensemble_calculator, get_supported_backends
_available_backends.extend(['create_ensemble_calculator', 'get_supported_backends'])

__all__ = _available_backends 