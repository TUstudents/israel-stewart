"""
Tensor utilities and common functions for relativistic hydrodynamics.

This module provides shared utilities, type guards, and import handling
used across the tensor algebra framework.
"""

import warnings
from typing import Any, TypeGuard

import numpy as np
import sympy as sp

# Handle opt_einsum import with fallback
try:
    import opt_einsum
    einsum = opt_einsum
    HAS_OPT_EINSUM = True
except ImportError:
    import numpy as np
    einsum = np  # Fallback to numpy for einsum operations
    HAS_OPT_EINSUM = False

# Forward reference for metrics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def is_numpy_array(obj: Any) -> TypeGuard[np.ndarray]:
    """Type guard for NumPy arrays."""
    return isinstance(obj, np.ndarray)


def is_sympy_matrix(obj: Any) -> TypeGuard[sp.Matrix]:
    """Type guard for SymPy matrices."""
    return isinstance(obj, sp.Matrix)


def ensure_compatible_types(obj1: Any, obj2: Any) -> bool:
    """Check if two objects have compatible types for tensor operations."""
    return (is_numpy_array(obj1) and is_numpy_array(obj2)) or \
           (is_sympy_matrix(obj1) and is_sympy_matrix(obj2))


def convert_to_sympy(components: np.ndarray | sp.Matrix) -> sp.Matrix:
    """Convert components to SymPy Matrix format."""
    if is_sympy_matrix(components):
        return components
    elif is_numpy_array(components):
        return sp.Matrix(components)
    else:
        raise TypeError(f"Cannot convert {type(components)} to SymPy Matrix")


def convert_to_numpy(components: np.ndarray | sp.Matrix) -> np.ndarray:
    """Convert components to NumPy array format."""
    if is_numpy_array(components):
        return components
    elif is_sympy_matrix(components):
        return np.array(components).astype(float)
    else:
        raise TypeError(f"Cannot convert {type(components)} to NumPy array")


def validate_tensor_dimensions(components: np.ndarray | sp.Matrix,
                              expected_shape: tuple[int, ...] | None = None) -> None:
    """
    Validate tensor component dimensions and values.

    Args:
        components: Tensor components to validate
        expected_shape: Expected shape tuple (None for any 4D shape)

    Raises:
        ValueError: If validation fails
    """
    if is_numpy_array(components):
        # Check for NaN/infinity
        if np.any(np.isnan(components)) or np.any(np.isinf(components)):
            raise ValueError("Tensor components contain NaN or infinite values")

        # Check dimensions are 4 (spacetime requirement)
        if any(dim != 4 for dim in components.shape):
            raise ValueError(f"All tensor dimensions must be 4 for spacetime, got shape {components.shape}")

        # Check specific shape if provided
        if expected_shape is not None and components.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {components.shape}")

        # Check for extremely large values that might cause numerical issues
        max_val = np.max(np.abs(components))
        if max_val > 1e100:
            warnings.warn(f"Tensor components are very large (max={max_val:.2e}), may cause numerical issues", stacklevel=2)

    elif is_sympy_matrix(components):
        # Check dimensions for SymPy
        if any(dim != 4 for dim in components.shape):
            raise ValueError(f"All tensor dimensions must be 4 for spacetime, got shape {components.shape}")

        if expected_shape is not None and components.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {components.shape}")
    else:
        raise TypeError(f"Components must be NumPy array or SymPy matrix, got {type(components)}")


def validate_index_compatibility(tensor1_indices: list[tuple[bool, str]],
                                tensor2_indices: list[tuple[bool, str]],
                                contraction_indices: tuple[int, int]) -> None:
    """
    Validate that tensor indices can be contracted.

    Args:
        tensor1_indices: First tensor's index list
        tensor2_indices: Second tensor's index list
        contraction_indices: Indices to contract (pos1, pos2)

    Raises:
        ValueError: If contraction is invalid
    """
    pos1, pos2 = contraction_indices

    if pos1 >= len(tensor1_indices):
        raise ValueError(f"Contraction index {pos1} out of range for first tensor (rank {len(tensor1_indices)})")
    if pos2 >= len(tensor2_indices):
        raise ValueError(f"Contraction index {pos2} out of range for second tensor (rank {len(tensor2_indices)})")

    # Get index types
    is_cov1, name1 = tensor1_indices[pos1]
    is_cov2, name2 = tensor2_indices[pos2]

    # Warn if contracting same type indices (should be one up, one down)
    if is_cov1 == is_cov2:
        warnings.warn(f"Contracting indices of same type: {'covariant' if is_cov1 else 'contravariant'} "
                     f"({name1}, {name2}). This may not be physically meaningful.", stacklevel=2)


def validate_einsum_string(einsum_str: str, rank1: int, rank2: int) -> None:
    """Validate Einstein summation string for correctness."""
    parts = einsum_str.split('->')
    if len(parts) != 2:
        raise ValueError(f"Invalid einsum string format: {einsum_str}")

    input_part, output_part = parts
    input_tensors = input_part.split(',')

    if len(input_tensors) != 2:
        raise ValueError(f"Expected 2 input tensors in einsum string: {einsum_str}")

    # Check input tensor ranks match
    if len(input_tensors[0]) != rank1:
        raise ValueError(f"First tensor rank mismatch: expected {rank1}, got {len(input_tensors[0])}")
    if len(input_tensors[1]) != rank2:
        raise ValueError(f"Second tensor rank mismatch: expected {rank2}, got {len(input_tensors[1])}")

    # Check for repeated indices (Einstein summation convention)
    all_input_indices = input_tensors[0] + input_tensors[1]
    output_indices = output_part

    # Find dummy (contracted) indices
    dummy_indices = []
    for idx in set(all_input_indices):
        if all_input_indices.count(idx) == 2:
            dummy_indices.append(idx)
        elif all_input_indices.count(idx) > 2:
            raise ValueError(f"Index '{idx}' appears more than twice in einsum string: {einsum_str}")

    # Check that all non-dummy indices appear in output
    free_indices = [idx for idx in all_input_indices if idx not in dummy_indices]
    expected_output = ''.join(sorted(set(free_indices), key=free_indices.index))

    if set(output_indices) != set(expected_output):
        warnings.warn(f"Output indices {output_indices} may not match expected {expected_output}", stacklevel=2)

    # Check for valid index characters (a-z)
    valid_indices = set('abcdefghijklmnopqrstuvwxyz')
    all_indices = set(all_input_indices + output_indices)
    if not all_indices.issubset(valid_indices):
        raise ValueError(f"Invalid index characters in einsum string: {all_indices - valid_indices}")


def optimized_einsum(einsum_str: str, *tensors) -> np.ndarray | sp.Matrix:
    """
    Perform optimized Einstein summation.

    Uses opt_einsum if available, falls back to numpy.einsum.

    Args:
        einsum_str: Einstein summation string
        *tensors: Input tensors to contract

    Returns:
        Contracted tensor result
    """
    if HAS_OPT_EINSUM and all(is_numpy_array(t) for t in tensors):
        # Use optimized contraction path if available
        if hasattr(einsum, 'contract'):
            return einsum.contract(einsum_str, *tensors)
        else:
            return einsum.einsum(einsum_str, *tensors)
    else:
        # Fallback to numpy einsum
        return np.einsum(einsum_str, *tensors)


# Exception classes for tensor operations
class TensorValidationError(Exception):
    """Exception for tensor validation failures."""
    pass


class PhysicsError(Exception):
    """Exception for physics-related errors."""
    pass


class RelativisticError(PhysicsError):
    """Exception for relativistic physics violations."""
    pass


# Common numerical tolerances
DEFAULT_TOLERANCE = 1e-10
STRICT_TOLERANCE = 1e-15
LOOSE_TOLERANCE = 1e-8

# Index naming conventions
SPACETIME_INDICES = ['t', 'x', 'y', 'z']
GREEK_INDICES = ['mu', 'nu', 'rho', 'sigma', 'tau', 'alpha', 'beta', 'gamma']
LATIN_INDICES = ['i', 'j', 'k', 'l', 'm', 'n']
