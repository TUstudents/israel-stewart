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
    return (is_numpy_array(obj1) and is_numpy_array(obj2)) or (
        is_sympy_matrix(obj1) and is_sympy_matrix(obj2)
    )


def convert_to_sympy(components: np.ndarray | sp.Matrix) -> sp.Matrix:
    """Convert components to SymPy Matrix format."""
    if is_sympy_matrix(components):
        return components
    elif is_numpy_array(components):
        return sp.Matrix(components)
    else:
        raise TypeError(f"Cannot convert {type(components)} to SymPy Matrix")


def convert_to_numpy(
    components: np.ndarray | sp.Matrix,
    dtype: np.dtype | type | str | None = None,
    warn_precision_loss: bool = True,
) -> np.ndarray:
    """
    Convert components to NumPy array format with intelligent type handling.

    Args:
        components: Components to convert
        dtype: Target NumPy dtype (None for automatic detection)
        warn_precision_loss: Whether to warn about potential precision loss

    Returns:
        NumPy array with appropriate dtype

    Raises:
        TypeError: If conversion is not possible
    """
    if is_numpy_array(components):
        if dtype is not None and components.dtype != dtype:
            return components.astype(dtype)
        return components
    elif is_sympy_matrix(components):
        # Try to evaluate symbolic expressions to numeric values
        try:
            # First try to evaluate symbolically to check for exact values
            evaluated = sp.Matrix(components)
            for i in range(components.rows):
                for j in range(components.cols):
                    evaluated[i, j] = sp.N(components[i, j])

            # Determine appropriate dtype first
            target_dtype: np.dtype[np.number]
            if dtype is None:
                # Check for complex values in the SymPy matrix
                has_complex = False
                try:
                    for i in range(components.rows):
                        for j in range(components.cols):
                            val = evaluated[i, j]
                            if hasattr(val, "is_real") and val.is_real is False:
                                has_complex = True
                                break
                            # Also check for I (imaginary unit)
                            if sp.I in val.free_symbols:
                                has_complex = True
                                break
                        if has_complex:
                            break
                except (AttributeError, TypeError):
                    # If we can't determine, check string representation for 'I'
                    str_repr = str(evaluated)
                    has_complex = "I" in str_repr

                target_dtype = np.dtype(np.complex128) if has_complex else np.dtype(np.float64)
            else:
                # Convert dtype to actual NumPy dtype
                if isinstance(dtype, np.dtype):
                    target_dtype = dtype
                else:
                    target_dtype = np.dtype(dtype)

            # Convert to NumPy array with proper dtype
            # Check for symbols first for warning
            has_symbols = any(
                components[i, j].free_symbols
                for i in range(components.rows)
                for j in range(components.cols)
            )

            # Always convert element by element for robustness
            result = np.zeros((components.rows, components.cols), dtype=target_dtype)

            for i in range(components.rows):
                for j in range(components.cols):
                    val = evaluated[i, j]
                    try:
                        if target_dtype == np.complex128:
                            result[i, j] = complex(val)
                        else:
                            result[i, j] = float(val)
                    except (TypeError, ValueError):
                        # For symbolic expressions that can't be converted,
                        # substitute with zero and let warning system handle it
                        if hasattr(val, "free_symbols") and val.free_symbols:
                            # Symbolic expression - substitute with 0
                            result[i, j] = 0.0
                        else:
                            # Real conversion error
                            raise TypeError(
                                f"Cannot convert SymPy expression {val} to numeric value"
                            ) from None

            # Warn about potential precision loss if requested
            if warn_precision_loss:
                # Check for symbolic expressions that couldn't be evaluated exactly
                has_symbols = any(
                    components[i, j].free_symbols
                    for i in range(components.rows)
                    for j in range(components.cols)
                )
                if has_symbols:
                    warnings.warn(
                        "Converting SymPy expressions with symbols to NumPy may result in "
                        "precision loss or approximation errors",
                        stacklevel=2,
                    )

                # Check for high precision numbers
                original_precision = float("inf")
                try:
                    for i in range(components.rows):
                        for j in range(components.cols):
                            val = components[i, j]
                            if hasattr(val, "precision") and val.precision:
                                original_precision = min(original_precision, val.precision)
                except (AttributeError, TypeError):
                    pass

                if (
                    original_precision != float("inf") and original_precision > 15
                ):  # float64 precision
                    warnings.warn(
                        f"Converting high-precision SymPy values (precision={original_precision}) "
                        f"to NumPy may result in precision loss",
                        stacklevel=2,
                    )

            return result

        except (AttributeError, TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot convert SymPy matrix to NumPy array: {e}. "
                "Matrix may contain unevaluable symbolic expressions."
            ) from e
    else:
        raise TypeError(f"Cannot convert {type(components)} to NumPy array")


def validate_tensor_dimensions(
    components: np.ndarray | sp.Matrix,
    expected_shape: tuple[int, ...] | None = None,
    tensor_rank: int | None = None,
) -> None:
    """
    Validate tensor component dimensions and values.

    Supports both single tensors and grid-based tensor fields. For grid-based tensors,
    only the trailing dimensions (corresponding to tensor indices) must be 4.

    Args:
        components: Tensor components to validate
        expected_shape: Expected shape tuple (None for flexible validation)
        tensor_rank: Rank of tensor (validates trailing dimensions only)

    Raises:
        ValueError: If validation fails
    """
    if is_numpy_array(components):
        # Check for NaN/infinity (skip for object arrays containing symbolic expressions)
        if components.dtype != object:
            if np.any(np.isnan(components)) or np.any(np.isinf(components)):
                raise ValueError("Tensor components contain NaN or infinite values")

        shape = components.shape

        if tensor_rank is not None:
            # Validate only trailing tensor dimensions for grid-based tensors
            if len(shape) < tensor_rank:
                raise ValueError(
                    f"Array has {len(shape)} dimensions, but tensor rank {tensor_rank} requires at least {tensor_rank} dimensions"
                )

            # Check that trailing dimensions (tensor indices) are all 4
            tensor_dims = shape[-tensor_rank:] if tensor_rank > 0 else ()
            if any(dim != 4 for dim in tensor_dims):
                raise ValueError(
                    f"Tensor index dimensions must be 4 for spacetime, "
                    f"got {tensor_dims} in shape {shape}"
                )
        else:
            # Legacy behavior: all dimensions must be 4 (for backward compatibility)
            if any(dim != 4 for dim in shape):
                raise ValueError(
                    f"All tensor dimensions must be 4 for spacetime, got shape {shape}"
                )

        # Check specific shape if provided
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {shape}")

        # Check for extremely large values that might cause numerical issues
        # (skip for object arrays containing symbolic expressions)
        if components.dtype != object:
            max_val = np.max(np.abs(components))
            if max_val > 1e100:
                warnings.warn(
                    f"Tensor components are very large (max={max_val:.2e}), may cause numerical issues",
                    stacklevel=2,
                )

    elif is_sympy_matrix(components):
        shape = components.shape

        if tensor_rank is not None:
            # For SymPy, handle different cases
            if len(shape) == 1 and tensor_rank == 1:
                # Vector case: should have 4 elements
                if shape[0] != 4:
                    raise ValueError(
                        f"SymPy vector must have 4 elements for spacetime, got shape {shape}"
                    )
            elif len(shape) == 2:
                # Matrix case: handle both 4x4 matrices and column vectors (4,1)
                if tensor_rank == 1 and shape == (4, 1):
                    # Column vector is acceptable for rank-1
                    pass
                elif tensor_rank == 2 and shape == (4, 4):
                    # 4x4 matrix is acceptable for rank-2
                    pass
                else:
                    raise ValueError(
                        f"SymPy tensor with rank {tensor_rank} has invalid shape {shape}"
                    )
            else:
                # Higher rank tensors: all dimensions must be 4
                if any(dim != 4 for dim in shape):
                    raise ValueError(
                        f"SymPy tensor dimensions must be 4 for spacetime, got shape {shape}"
                    )
        else:
            # Legacy behavior: all dimensions must be 4
            if any(dim != 4 for dim in shape):
                raise ValueError(
                    f"All tensor dimensions must be 4 for spacetime, got shape {shape}"
                )

        if expected_shape is not None and shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {shape}")
    else:
        raise TypeError(f"Components must be NumPy array or SymPy matrix, got {type(components)}")


def validate_index_compatibility(
    tensor1_indices: list[tuple[bool, str]],
    tensor2_indices: list[tuple[bool, str]],
    contraction_indices: tuple[int, int],
) -> None:
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
        raise ValueError(
            f"Contraction index {pos1} out of range for first tensor (rank {len(tensor1_indices)})"
        )
    if pos2 >= len(tensor2_indices):
        raise ValueError(
            f"Contraction index {pos2} out of range for second tensor (rank {len(tensor2_indices)})"
        )

    # Get index types
    is_cov1, name1 = tensor1_indices[pos1]
    is_cov2, name2 = tensor2_indices[pos2]

    # Warn if contracting same type indices (should be one up, one down)
    if is_cov1 == is_cov2:
        warnings.warn(
            f"Contracting indices of same type: {'covariant' if is_cov1 else 'contravariant'} "
            f"({name1}, {name2}). This may not be physically meaningful.",
            stacklevel=2,
        )


def validate_einsum_string(einsum_str: str, rank1: int, rank2: int) -> None:
    """Validate Einstein summation string for correctness."""
    parts = einsum_str.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid einsum string format: {einsum_str}")

    input_part, output_part = parts
    input_tensors = input_part.split(",")

    if len(input_tensors) != 2:
        raise ValueError(f"Expected 2 input tensors in einsum string: {einsum_str}")

    # Check input tensor ranks match
    if len(input_tensors[0]) != rank1:
        raise ValueError(
            f"First tensor rank mismatch: expected {rank1}, got {len(input_tensors[0])}"
        )
    if len(input_tensors[1]) != rank2:
        raise ValueError(
            f"Second tensor rank mismatch: expected {rank2}, got {len(input_tensors[1])}"
        )

    # Check for repeated indices (Einstein summation convention)
    all_input_indices = input_tensors[0] + input_tensors[1]
    output_indices = output_part

    # Find dummy (contracted) indices
    dummy_indices = []
    for idx in set(all_input_indices):
        if all_input_indices.count(idx) == 2:
            dummy_indices.append(idx)
        elif all_input_indices.count(idx) > 2:
            raise ValueError(
                f"Index '{idx}' appears more than twice in einsum string: {einsum_str}"
            )

    # Check that all non-dummy indices appear in output
    free_indices = [idx for idx in all_input_indices if idx not in dummy_indices]
    expected_output = "".join(sorted(set(free_indices), key=free_indices.index))

    if set(output_indices) != set(expected_output):
        warnings.warn(
            f"Output indices {output_indices} may not match expected {expected_output}",
            stacklevel=2,
        )

    # Check for valid index characters (a-z, A-Z)
    valid_indices = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    all_indices = set(all_input_indices + output_indices)
    if not all_indices.issubset(valid_indices):
        raise ValueError(
            f"Invalid index characters in einsum string: {all_indices - valid_indices}"
        )


def optimized_einsum(einsum_str: str, *tensors: np.ndarray | sp.Matrix) -> np.ndarray | sp.Matrix:
    """
    Perform optimized Einstein summation with intelligent backend selection.

    Uses opt_einsum for NumPy arrays when available, falls back to numpy.einsum.
    Handles SymPy matrices with automatic conversion.

    Args:
        einsum_str: Einstein summation string
        *tensors: Input tensors to contract

    Returns:
        Contracted tensor result

    Raises:
        ValueError: If einsum string is malformed
        TypeError: If tensor types are incompatible
    """
    if not tensors:
        raise ValueError("At least one tensor must be provided")

    # Check tensor type consistency
    all_numpy = all(is_numpy_array(t) for t in tensors)
    all_sympy = all(is_sympy_matrix(t) for t in tensors)

    if not (all_numpy or all_sympy):
        # Mixed types - convert all to NumPy for compatibility
        numpy_tensors = tuple(convert_to_numpy(t, warn_precision_loss=False) for t in tensors)
        all_numpy = True
        tensors = numpy_tensors

    if all_numpy:
        # Use optimal NumPy backend
        if HAS_OPT_EINSUM:
            # opt_einsum provides optimal contraction paths for complex expressions
            try:
                return opt_einsum.contract(einsum_str, *tensors, optimize=True)
            except (ImportError, AttributeError, ValueError):
                # Fallback if opt_einsum fails for any reason
                return np.einsum(einsum_str, *tensors)
        else:
            # Standard NumPy einsum
            return np.einsum(einsum_str, *tensors)

    elif all_sympy:
        # Handle SymPy tensors - limited einsum support
        try:
            # For simple cases, use SymPy's tensor operations
            if len(tensors) == 2:
                # Handle common contraction patterns manually
                tensor1, tensor2 = tensors

                # Parse einsum string for simple patterns
                parts = einsum_str.split("->")
                if len(parts) != 2:
                    raise ValueError(f"Invalid einsum format: {einsum_str}")

                input_part, output_part = parts
                input_specs = input_part.split(",")

                if len(input_specs) == 2:
                    spec1, spec2 = input_specs

                    # Simple matrix multiplication: 'ij,jk->ik'
                    if len(spec1) == 2 and len(spec2) == 2 and len(output_part) == 2:
                        if spec1[1] == spec2[0]:  # Contraction pattern
                            return tensor1 * tensor2

                    # Vector dot product: 'i,i->'
                    if len(spec1) == 1 and len(spec2) == 1 and spec1 == spec2 and output_part == "":
                        return tensor1.dot(tensor2)

                # Fallback: convert to NumPy, compute, then back to SymPy if needed
                warnings.warn(
                    f"Complex einsum '{einsum_str}' with SymPy tensors converted to NumPy",
                    stacklevel=2,
                )
                numpy_tensors = tuple(
                    convert_to_numpy(t, warn_precision_loss=False) for t in tensors
                )
                result = optimized_einsum(einsum_str, *numpy_tensors)
                return convert_to_sympy(result)
            else:
                # Multi-tensor SymPy contractions not directly supported
                warnings.warn(
                    "Multi-tensor einsum with SymPy matrices converted to NumPy",
                    stacklevel=2,
                )
                numpy_tensors = tuple(
                    convert_to_numpy(t, warn_precision_loss=False) for t in tensors
                )
                result = optimized_einsum(einsum_str, *numpy_tensors)
                return convert_to_sympy(result)

        except (AttributeError, ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to perform einsum '{einsum_str}' with SymPy tensors: {e}"
            ) from e

    else:
        # This shouldn't happen given our type checking above
        raise TypeError("Incompatible tensor types for einsum operation")


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
SPACETIME_INDICES = ["t", "x", "y", "z"]
GREEK_INDICES = ["mu", "nu", "rho", "sigma", "tau", "alpha", "beta", "gamma"]
LATIN_INDICES = ["i", "j", "k", "l", "m", "n"]
