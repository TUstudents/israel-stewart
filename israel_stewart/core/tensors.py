"""
Tensor algebra and operations for relativistic hydrodynamics.

This module provides consolidated imports and backwards compatibility
for the modularized tensor framework.
"""

# Import all classes from modularized components
from .derivatives import CovariantDerivative, ProjectionOperator
from .four_vectors import FourVector
from .performance import (
    PerformanceMonitor,
    get_performance_monitor,
    monitor_performance,
    performance_report,
    reset_performance_stats,
)
from .stress_tensors import StressEnergyTensor, ViscousStressTensor
from .tensor_base import TensorField
from .tensor_utils import (
    DEFAULT_TOLERANCE,
    LOOSE_TOLERANCE,
    STRICT_TOLERANCE,
    PhysicsError,
    RelativisticError,
    TensorValidationError,
    convert_to_numpy,
    convert_to_sympy,
    ensure_compatible_types,
    is_numpy_array,
    is_sympy_matrix,
    optimized_einsum,
    validate_einsum_string,
    validate_index_compatibility,
    validate_tensor_dimensions,
)
from .transformations import CoordinateTransformation, LorentzTransformation

# Re-export everything for backwards compatibility
__all__ = [
    # Base tensor classes
    'TensorField', 'FourVector',

    # Physics tensor classes
    'StressEnergyTensor', 'ViscousStressTensor',

    # Operations and derivatives
    'CovariantDerivative', 'ProjectionOperator',

    # Transformations
    'LorentzTransformation', 'CoordinateTransformation',

    # Utilities and validation
    'is_numpy_array', 'is_sympy_matrix', 'ensure_compatible_types',
    'convert_to_sympy', 'convert_to_numpy', 'validate_tensor_dimensions',
    'validate_index_compatibility', 'validate_einsum_string', 'optimized_einsum',

    # Exception classes
    'TensorValidationError', 'PhysicsError', 'RelativisticError',

    # Constants
    'DEFAULT_TOLERANCE', 'STRICT_TOLERANCE', 'LOOSE_TOLERANCE',

    # Performance monitoring
    'PerformanceMonitor', 'get_performance_monitor', 'monitor_performance',
    'performance_report', 'reset_performance_stats'
]
