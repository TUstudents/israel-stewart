"""
Base tensor field class for relativistic hydrodynamics.

This module provides the fundamental TensorField class with automatic
index management and core tensor operations.
"""

import warnings

# Forward reference for metrics
from typing import TYPE_CHECKING, Optional

import numpy as np
import sympy as sp

from .performance import monitor_performance

# Import utilities and performance monitoring
from .tensor_utils import (
    convert_to_sympy,
    is_numpy_array,
    is_sympy_matrix,
    optimized_einsum,
    validate_einsum_string,
    validate_index_compatibility,
)

if TYPE_CHECKING:
    from .metrics import MetricBase


class TensorField:
    """
    Base class for relativistic tensor fields with automatic index management.

    Supports both numerical (NumPy) and symbolic (SymPy) computations with
    automatic tracking of covariant and contravariant indices.
    """

    def __init__(
        self,
        components: np.ndarray | sp.Matrix,
        indices: str,
        metric: Optional["MetricBase"] = None,
    ):
        """
        Initialize tensor field.

        Args:
            components: Tensor components (numpy array or sympy matrix)
            indices: Index specification like 'mu nu' (contravariant) or '_mu _nu' (covariant)
            metric: Metric object for index raising/lowering operations
        """
        self.components = self._validate_components(components)
        self.indices = self._parse_indices(indices)
        self.metric = metric
        self.rank = len(self.indices)
        self._validate_tensor()

    def _validate_components(self, components: np.ndarray | sp.Matrix) -> np.ndarray | sp.Matrix:
        """Validate and standardize tensor components."""
        if isinstance(components, list | tuple):
            components = np.array(components)

        if is_numpy_array(components):
            # Check for numerical stability (skip for object arrays with symbolic expressions)
            if components.dtype != object:
                if np.any(np.isnan(components)) or np.any(np.isinf(components)):
                    raise ValueError("Tensor components contain NaN or infinite values")

            # For tensor fields on grids, only the trailing dimensions must be 4 (spacetime indices)
            # Allow shapes like (*grid_shape, 4), (*grid_shape, 4, 4), etc.
            if components.ndim == 0:
                raise ValueError("Tensor components cannot be scalar")

            # We need to know the tensor rank to validate properly, but we don't have indices parsed yet
            # For now, just check that the array has at least one dimension
            # More detailed validation will happen in _validate_tensor() after indices are parsed
            if components.size == 0:
                raise ValueError("Tensor components cannot be empty")

        elif is_sympy_matrix(components):
            # For SymPy matrices/vectors, handle different shapes
            shape = components.shape
            if len(shape) == 1:
                # Vector case: should have 4 elements
                if shape[0] != 4:
                    raise ValueError(
                        f"SymPy vector must have 4 elements for spacetime, got shape {shape}"
                    )
            elif len(shape) == 2:
                # Matrix case: both dimensions should be 4, or it could be a column vector (4,1)
                if shape == (4, 1):
                    # Column vector - this is acceptable for rank-1 tensors
                    pass
                elif all(dim == 4 for dim in shape):
                    # 4x4 matrix - acceptable for rank-2 tensors
                    pass
                else:
                    raise ValueError(
                        f"SymPy tensor dimensions must be 4 for spacetime, got shape {shape}"
                    )
            else:
                # Higher-rank tensors: all dimensions must be 4
                if any(dim != 4 for dim in shape):
                    raise ValueError(
                        f"SymPy tensor dimensions must be 4 for spacetime, got shape {shape}"
                    )
        else:
            raise TypeError(
                f"Components must be numpy array or sympy matrix, got {type(components)}"
            )

        return components

    def _parse_indices(self, indices: str) -> list[tuple[bool, str]]:
        """
        Parse index notation.

        Args:
            indices: String like '_mu nu _rho' where '_' indicates covariant

        Returns:
            List of (is_covariant, index_name) tuples
        """
        parsed = []
        for idx in indices.split():
            if idx.startswith("_"):
                parsed.append((True, idx[1:]))  # Covariant
            else:
                parsed.append((False, idx))  # Contravariant
        return parsed

    def _validate_tensor(self) -> None:
        """Validate tensor rank consistency and spacetime dimensions."""
        expected_rank = len(self.indices)

        if is_numpy_array(self.components):
            total_dims = self.components.ndim
            shape = self.components.shape

            # For grid-based tensors, the last `expected_rank` dimensions are tensor indices
            if total_dims < expected_rank:
                raise ValueError(
                    f"Tensor rank mismatch: indices specify rank {expected_rank}, "
                    f"but components have only {total_dims} dimensions"
                )

            # Check that the trailing dimensions (tensor indices) are all 4
            tensor_index_dims = shape[-expected_rank:] if expected_rank > 0 else ()
            if any(dim != 4 for dim in tensor_index_dims):
                raise ValueError(
                    f"Tensor index dimensions must be 4 for spacetime, "
                    f"got {tensor_index_dims} in shape {shape}"
                )

        elif is_sympy_matrix(self.components):
            shape = self.components.shape
            actual_rank = len(shape)

            # Special case: SymPy column vectors (4,1) should be treated as rank-1
            if shape == (4, 1) and expected_rank == 1:
                actual_rank = 1  # Override for column vectors
            elif actual_rank != expected_rank:
                raise ValueError(
                    f"SymPy tensor rank mismatch: indices specify rank {expected_rank}, "
                    f"components have rank {actual_rank}"
                )
        else:
            raise TypeError(f"Unsupported component type: {type(self.components)}")

        # Additional validation for specific ranks
        if expected_rank == 0:  # Scalar
            raise ValueError("Use scalar values directly, not TensorField for rank-0 tensors")
        elif expected_rank > 4:  # Reasonable upper limit for physics
            warnings.warn(
                f"High-rank tensor (rank {expected_rank}) may have performance issues",
                stacklevel=2,
            )

    def __str__(self) -> str:
        """String representation of tensor."""
        index_str = " ".join(f"{'_' if cov else ''}{name}" for cov, name in self.indices)
        return f"TensorField[{index_str}]"

    def __repr__(self) -> str:
        return f"TensorField(components={self.components.shape}, indices='{self._index_string()}')"

    def _index_string(self) -> str:
        """Convert indices back to string format."""
        return " ".join(f"{'_' if cov else ''}{name}" for cov, name in self.indices)

    def _create_subclass_instance(
        self, new_components: np.ndarray | sp.Matrix, index_str: str | None = None
    ) -> "TensorField":
        """
        Create new instance preserving subclass type with proper constructor signature.

        Args:
            new_components: New tensor components
            index_str: Index string (only used for TensorField base class)

        Returns:
            New instance of the same subclass type
        """
        if type(self) is TensorField:
            # Base class uses the standard constructor
            return TensorField(new_components, index_str or self._index_string(), self.metric)
        else:
            # For subclasses, use introspection to match constructor signature
            import inspect

            sig = inspect.signature(type(self).__init__)
            params = list(sig.parameters.keys())[1:]  # Skip 'self'

            if len(params) == 2:
                # Stress tensor classes: (components, metric)
                from typing import Any, cast

                constructor = cast(Any, type(self))
                return cast("TensorField", constructor(new_components, self.metric))
            elif len(params) == 3 and "is_covariant" in params:
                # FourVector: (components, is_covariant, metric)
                # For FourVector, determine covariance from the index_str if provided,
                # otherwise preserve from current indices
                if index_str and index_str.startswith("_"):
                    # Index string indicates covariant (starts with '_')
                    is_covariant = True
                elif index_str and not index_str.startswith("_"):
                    # Index string indicates contravariant (doesn't start with '_')
                    is_covariant = False
                else:
                    # Fallback to current indices
                    is_covariant = self.indices[0][0] if self.indices else False
                from typing import Any, cast

                constructor = cast(Any, type(self))
                return cast("TensorField", constructor(new_components, is_covariant, self.metric))
            else:
                # Fallback to standard constructor for other subclasses
                from typing import Any, cast

                constructor = cast(Any, type(self))
                return cast(
                    "TensorField",
                    constructor(new_components, index_str or self._index_string(), self.metric),
                )

    def copy(self) -> "TensorField":
        """Create a deep copy of the tensor."""
        if isinstance(self.components, np.ndarray):
            new_components = self.components.copy()
        else:
            new_components = self.components.copy()

        # Preserve subclass type (FourVector, StressEnergyTensor, etc.)
        return self._create_subclass_instance(new_components)

    @monitor_performance("tensor_transpose")
    def transpose(self, axis_order: tuple[int, ...] | None = None) -> "TensorField":
        """
        Transpose tensor indices.

        Args:
            axis_order: Order of axes for transposition

        Returns:
            Transposed tensor
        """
        if axis_order is None:
            # Default: reverse tensor indices only
            axis_order = tuple(range(self.rank - 1, -1, -1))

        if isinstance(self.components, np.ndarray):
            # For grid-based tensors, only transpose the tensor dimensions
            total_dims = self.components.ndim
            grid_dims = total_dims - self.rank

            # Build full axis order: keep grid dimensions in place, transpose tensor dimensions
            full_axis_order = list(range(grid_dims))  # Grid dimensions stay in place
            tensor_axis_mapping = [grid_dims + i for i in axis_order]  # Map to tensor dimensions
            full_axis_order.extend(tensor_axis_mapping)

            new_components = np.transpose(self.components, full_axis_order)
        else:
            # For SymPy matrices
            if self.rank == 1:
                # Vector transposition doesn't change anything for column vectors
                new_components = self.components.copy()
            elif self.rank == 2 and axis_order == (1, 0):
                new_components = self.components.T
            else:
                # For higher rank SymPy tensors, use SymPy Array operations for efficiency
                try:
                    import sympy.tensor.array as sp_array

                    # Convert to SymPy Array if not already
                    if hasattr(self.components, "rank") and self.components.rank() > 2:
                        # Already a SymPy Array
                        array_comp = self.components
                    else:
                        # Convert SymPy Matrix to Array
                        array_comp = sp_array.Array(self.components)

                    # Transpose using SymPy's permutedims
                    new_components = sp_array.permutedims(array_comp, axis_order)

                except ImportError:
                    # Fallback to NumPy conversion if SymPy Array not available
                    warnings.warn(
                        "SymPy higher-rank tensor transposition requires conversion to NumPy (SymPy Array not available)",
                        stacklevel=2,
                    )
                    temp_array = np.array(self.components).astype(complex)
                    transposed_array = np.transpose(temp_array, axis_order)
                    new_components = sp.Matrix(transposed_array)

        # Reorder indices accordingly
        new_indices = [self.indices[i] for i in axis_order]
        new_index_string = " ".join(f"{'_' if cov else ''}{name}" for cov, name in new_indices)

        # Preserve subclass type (FourVector, StressEnergyTensor, etc.)
        return self._create_subclass_instance(new_components, new_index_string)

    @monitor_performance("tensor_symmetrize")
    def symmetrize(self, indices_pair: tuple[int, int] | None = None) -> "TensorField":
        """
        Symmetrize tensor with respect to given indices.

        Args:
            indices_pair: Pair of indices to symmetrize (default: first two)

        Returns:
            Symmetrized tensor
        """
        if self.rank < 2:
            warnings.warn("Cannot symmetrize tensor with rank < 2", stacklevel=2)
            return self.copy()

        if indices_pair is None:
            indices_pair = (0, 1)

        i, j = indices_pair
        if i >= self.rank or j >= self.rank:
            raise ValueError(f"Index pair {indices_pair} out of range for rank-{self.rank} tensor")

        # Create transposition that swaps indices i and j
        axis_order = list(range(self.rank))
        axis_order[i], axis_order[j] = axis_order[j], axis_order[i]

        transposed = self.transpose(tuple(axis_order))
        symmetrized_components = 0.5 * (self.components + transposed.components)

        # Preserve subclass type (FourVector, StressEnergyTensor, etc.)
        return self._create_subclass_instance(symmetrized_components)

    @monitor_performance("tensor_antisymmetrize")
    def antisymmetrize(self, indices_pair: tuple[int, int] | None = None) -> "TensorField":
        """
        Antisymmetrize tensor with respect to given indices.

        Args:
            indices_pair: Pair of indices to antisymmetrize (default: first two)

        Returns:
            Antisymmetrized tensor
        """
        if self.rank < 2:
            warnings.warn("Cannot antisymmetrize tensor with rank < 2", stacklevel=2)
            return self.copy()

        if indices_pair is None:
            indices_pair = (0, 1)

        i, j = indices_pair
        if i >= self.rank or j >= self.rank:
            raise ValueError(f"Index pair {indices_pair} out of range for rank-{self.rank} tensor")

        # Create transposition that swaps indices i and j
        axis_order = list(range(self.rank))
        axis_order[i], axis_order[j] = axis_order[j], axis_order[i]

        transposed = self.transpose(tuple(axis_order))
        antisymmetrized_components = 0.5 * (self.components - transposed.components)

        # Preserve subclass type (FourVector, StressEnergyTensor, etc.)
        return self._create_subclass_instance(antisymmetrized_components)

    @monitor_performance("tensor_contract")
    def contract(
        self, other: "TensorField", self_index: int, other_index: int
    ) -> "TensorField | np.ndarray | float | complex":
        """
        Contract two tensors along specified indices.

        Args:
            other: Tensor to contract with
            self_index: Index position in self to contract
            other_index: Index position in other to contract

        Returns:
            Contracted tensor
        """
        if self_index >= self.rank or other_index >= other.rank:
            raise ValueError(
                f"Index out of range: self has rank {self.rank}, " f"other has rank {other.rank}"
            )

        # Validate index compatibility
        validate_index_compatibility(self.indices, other.indices, (self_index, other_index))

        # For grid-based tensors, we need to handle the grid dimensions
        if is_numpy_array(self.components) and is_numpy_array(other.components):
            self_total_dims = self.components.ndim
            other_total_dims = other.components.ndim
            self_grid_dims = self_total_dims - self.rank
            other_grid_dims = other_total_dims - other.rank
        else:
            # SymPy case - no grid dimensions
            self_total_dims = self.rank
            other_total_dims = other.rank
            self_grid_dims = 0
            other_grid_dims = 0

        # Check that grid shapes are compatible
        if self_grid_dims != other_grid_dims:
            raise ValueError(
                f"Grid dimension mismatch: self has {self_grid_dims} grid dims, "
                f"other has {other_grid_dims} grid dims"
            )

        # Build einsum string including grid dimensions
        grid_indices = [chr(65 + i) for i in range(self_grid_dims)]  # A, B, C, ... for grid
        self_tensor_indices = [chr(97 + i) for i in range(self.rank)]  # a, b, c, ... for tensor
        other_tensor_indices = [chr(97 + self.rank + i) for i in range(other.rank)]

        # Make contracted indices the same
        other_tensor_indices[other_index] = self_tensor_indices[self_index]

        # Result indices (grid + remaining tensor indices)
        result_tensor_indices = [
            idx for i, idx in enumerate(self_tensor_indices) if i != self_index
        ] + [idx for i, idx in enumerate(other_tensor_indices) if i != other_index]

        # Full index strings (grid + tensor)
        self_full_indices = grid_indices + self_tensor_indices
        other_full_indices = grid_indices + other_tensor_indices
        result_full_indices = grid_indices + result_tensor_indices

        einsum_str = (
            f"{''.join(self_full_indices)},{''.join(other_full_indices)}->"
            + f"{''.join(result_full_indices)}"
        )

        # Validate einsum string (pass total dimensions including grid)
        validate_einsum_string(einsum_str, self_total_dims, other_total_dims)

        # Perform contraction with type checking and optimization
        if is_numpy_array(self.components) and is_numpy_array(other.components):
            result_components = optimized_einsum(einsum_str, self.components, other.components)
        elif is_sympy_matrix(self.components) or is_sympy_matrix(other.components):
            # Use manual contraction for SymPy
            result_components = self._manual_contraction(other, self_index, other_index)
        else:
            raise TypeError(
                f"Unsupported tensor component types: {type(self.components)}, {type(other.components)}"
            )

        # Build result index string
        result_index_list = [self.indices[i] for i in range(self.rank) if i != self_index] + [
            other.indices[i] for i in range(other.rank) if i != other_index
        ]

        # Handle scalar result (rank 0)
        if len(result_index_list) == 0:
            # Return scalar result directly (not as TensorField)
            return result_components

        result_index_str = " ".join(
            f"{'_' if cov else ''}{name}" for cov, name in result_index_list
        )

        # For contract(), be conservative about subclass preservation
        # Only preserve subclass if result rank is compatible with subclass requirements
        result_rank = len(result_index_list)

        # Check if subclass can handle the result rank
        if type(self) is TensorField:
            # Base class can handle any rank
            return TensorField(result_components, result_index_str, self.metric)
        elif result_rank == self.rank:
            # Same rank - safe to preserve subclass
            return self._create_subclass_instance(result_components, result_index_str)
        else:
            # Different rank - use base TensorField to be safe
            # Subclasses may have rank-specific requirements
            return TensorField(result_components, result_index_str, self.metric)

    def _manual_contraction(
        self, other: "TensorField", self_index: int, other_index: int
    ) -> np.ndarray | sp.Matrix:
        """Manual tensor contraction for SymPy compatibility."""
        if not (is_sympy_matrix(self.components) or is_sympy_matrix(other.components)):
            raise ValueError("Manual contraction should only be used for SymPy tensors")

        # Convert to SymPy if needed
        self_comp = convert_to_sympy(self.components)
        other_comp = convert_to_sympy(other.components)

        # Handle different tensor ranks and contraction patterns
        if self.rank == 1 and other.rank == 1:  # Vector-vector contraction
            return sum(self_comp[i] * other_comp[i] for i in range(4))

        elif self.rank == 2 and other.rank == 1:  # Matrix-vector contraction
            if self_index == 0:  # Contract first index of matrix with vector
                return sp.Matrix(
                    [sum(self_comp[i, j] * other_comp[i] for i in range(4)) for j in range(4)]
                )
            elif self_index == 1:  # Contract second index of matrix with vector
                return sp.Matrix(
                    [sum(self_comp[i, j] * other_comp[j] for j in range(4)) for i in range(4)]
                )

        elif self.rank == 1 and other.rank == 2:  # Vector-matrix contraction
            if other_index == 0:  # Contract vector with first index of matrix
                return sp.Matrix(
                    [sum(self_comp[i] * other_comp[i, j] for i in range(4)) for j in range(4)]
                )
            elif other_index == 1:  # Contract vector with second index of matrix
                return sp.Matrix(
                    [sum(self_comp[i] * other_comp[j, i] for i in range(4)) for j in range(4)]
                )

        elif self.rank == 2 and other.rank == 2:  # Matrix-matrix contraction
            if self_index == 1 and other_index == 0:  # Standard matrix multiplication
                return self_comp * other_comp
            elif self_index == 0 and other_index == 0:  # Contract first indices
                return sp.Matrix(
                    [
                        [
                            sum(self_comp[k, i] * other_comp[k, j] for k in range(4))
                            for j in range(4)
                        ]
                        for i in range(4)
                    ]
                )
            elif self_index == 1 and other_index == 1:  # Contract second indices
                return sp.Matrix(
                    [
                        [
                            sum(self_comp[i, k] * other_comp[j, k] for k in range(4))
                            for j in range(4)
                        ]
                        for i in range(4)
                    ]
                )
            elif self_index == 0 and other_index == 1:  # Contract first with second
                return sp.Matrix(
                    [
                        [
                            sum(self_comp[k, i] * other_comp[j, k] for k in range(4))
                            for j in range(4)
                        ]
                        for i in range(4)
                    ]
                )

        # Additional common contraction patterns
        elif self.rank == 3 and other.rank == 1:  # Rank-3 tensor with vector
            if other_index == 0:
                # Contract rank-3 tensor T_{ijk} with vector V^i at specified self_index
                if self_index == 0:  # T_{ijk} V^i -> T'_{jk}
                    return sp.Matrix(
                        [
                            [
                                sum(self_comp[i, j, k] * other_comp[i] for i in range(4))
                                for k in range(4)
                            ]
                            for j in range(4)
                        ]
                    )
                elif self_index == 1:  # T_{ijk} V^j -> T'_{ik}
                    return sp.Matrix(
                        [
                            [
                                sum(self_comp[i, j, k] * other_comp[j] for j in range(4))
                                for k in range(4)
                            ]
                            for i in range(4)
                        ]
                    )
                elif self_index == 2:  # T_{ijk} V^k -> T'_{ij}
                    return sp.Matrix(
                        [
                            [
                                sum(self_comp[i, j, k] * other_comp[k] for k in range(4))
                                for j in range(4)
                            ]
                            for i in range(4)
                        ]
                    )

        elif self.rank == 1 and other.rank == 3:  # Vector with rank-3 tensor
            if self_index == 0:
                # Contract vector V_i with rank-3 tensor T^{ijk} at specified other_index
                if other_index == 0:  # V_i T^{ijk} -> T'_{jk}
                    return sp.Matrix(
                        [
                            [
                                sum(self_comp[i] * other_comp[i, j, k] for i in range(4))
                                for k in range(4)
                            ]
                            for j in range(4)
                        ]
                    )
                elif other_index == 1:  # V_j T^{ijk} -> T'_{ik}
                    return sp.Matrix(
                        [
                            [
                                sum(self_comp[j] * other_comp[i, j, k] for j in range(4))
                                for k in range(4)
                            ]
                            for i in range(4)
                        ]
                    )
                elif other_index == 2:  # V_k T^{ijk} -> T'_{ij}
                    return sp.Matrix(
                        [
                            [
                                sum(self_comp[k] * other_comp[i, j, k] for k in range(4))
                                for j in range(4)
                            ]
                            for i in range(4)
                        ]
                    )

        elif self.rank == 2 and other.rank == 3:  # Matrix with rank-3 tensor
            # Common case: Matrix A_{ij} with rank-3 tensor T^{klm}
            if self_index == 1 and other_index == 0:  # A_{ij} T^{jkl} -> R_{ikl}
                # This would result in a rank-3 tensor, which is complex to represent as SymPy Matrix
                # Use SymPy Array operations if available
                try:
                    import sympy.tensor.array as sp_array

                    self_array = sp_array.Array(self_comp)
                    other_array = sp_array.Array(other_comp)

                    # Perform contraction
                    axes = ([self_index], [other_index])
                    result = sp_array.tensorcontraction(
                        sp_array.tensorproduct(self_array, other_array), axes
                    )
                    return result

                except ImportError:
                    raise NotImplementedError(
                        "Matrix-rank3 tensor contraction requires SymPy Array support"
                    ) from None

        # Fall back to generic SymPy Array operations if available
        try:
            import sympy.tensor.array as sp_array

            # Convert both tensors to Arrays
            self_array = sp_array.Array(self_comp) if not hasattr(self_comp, "rank") else self_comp
            other_array = (
                sp_array.Array(other_comp) if not hasattr(other_comp, "rank") else other_comp
            )

            # Perform tensor contraction
            axes = ([self_index], [other_index])
            result = sp_array.tensorcontraction(
                sp_array.tensorproduct(self_array, other_array), axes
            )
            return result

        except ImportError:
            raise NotImplementedError(
                f"Manual contraction not implemented for ranks {self.rank} and {other.rank} "
                f"with indices {self_index} and {other_index}. "
                f"Install SymPy Array support for general tensor contractions."
            ) from None

    @monitor_performance("raise_index")
    def raise_index(self, index_pos: int) -> "TensorField":
        """
        Raise an index using the metric tensor.

        Args:
            index_pos: Position of index to raise

        Returns:
            Tensor with raised index
        """
        if self.metric is None:
            raise ValueError("Cannot raise index without metric tensor")

        if index_pos >= self.rank:
            raise ValueError(f"Index position {index_pos} out of range for rank-{self.rank} tensor")

        is_covariant, index_name = self.indices[index_pos]
        if not is_covariant:
            warnings.warn(f"Index {index_name} is already contravariant", stacklevel=2)
            return self.copy()

        # Contract with inverse metric
        metric_inverse = self.metric.inverse

        # Build contraction based on tensor rank and index position
        if self.rank == 1:  # Vector
            result_components = optimized_einsum("ij,j->i", metric_inverse, self.components)
        elif self.rank == 2:  # Matrix
            if index_pos == 0:
                result_components = optimized_einsum("ij,jk->ik", metric_inverse, self.components)
            else:
                result_components = optimized_einsum("ij,ki->kj", metric_inverse, self.components)
        elif self.rank == 3:  # Rank-3 tensor
            if index_pos == 0:
                result_components = optimized_einsum("ij,jkl->ikl", metric_inverse, self.components)
            elif index_pos == 1:
                result_components = optimized_einsum("ij,kij->kil", metric_inverse, self.components)
            else:  # index_pos == 2
                result_components = optimized_einsum("ij,kli->klj", metric_inverse, self.components)
        elif self.rank == 4:  # Rank-4 tensor
            # General einsum for rank-4 tensors
            indices = list("abcd")
            indices[index_pos] = "x"
            output_indices = list("abcd")
            output_indices[index_pos] = "y"
            einsum_str = f"xy,{''.join(indices)}->{''.join(output_indices)}"
            result_components = optimized_einsum(einsum_str, metric_inverse, self.components)
        else:
            raise NotImplementedError(f"Index raising not implemented for rank {self.rank} > 4")

        # Update indices
        new_indices: list[tuple[bool, str]] = self.indices.copy()
        new_indices[index_pos] = (False, index_name)  # Make contravariant
        new_index_str = " ".join(f"{'_' if cov else ''}{name}" for cov, name in new_indices)

        # Preserve subclass type (FourVector, StressEnergyTensor, etc.)
        return self._create_subclass_instance(result_components, new_index_str)

    @monitor_performance("lower_index")
    def lower_index(self, index_pos: int) -> "TensorField":
        """
        Lower an index using the metric tensor.

        Args:
            index_pos: Position of index to lower

        Returns:
            Tensor with lowered index
        """
        if self.metric is None:
            raise ValueError("Cannot lower index without metric tensor")

        if index_pos >= self.rank:
            raise ValueError(f"Index position {index_pos} out of range for rank-{self.rank} tensor")

        is_covariant, index_name = self.indices[index_pos]
        if is_covariant:
            warnings.warn(f"Index {index_name} is already covariant", stacklevel=2)
            return self.copy()

        # Contract with metric
        metric_components = self.metric.components

        # Build contraction based on tensor rank and index position
        if self.rank == 1:  # Vector
            result_components = optimized_einsum("ij,j->i", metric_components, self.components)
        elif self.rank == 2:  # Matrix
            if index_pos == 0:
                result_components = optimized_einsum(
                    "ij,jk->ik", metric_components, self.components
                )
            else:
                result_components = optimized_einsum(
                    "ij,ki->kj", metric_components, self.components
                )
        elif self.rank == 3:  # Rank-3 tensor
            if index_pos == 0:
                result_components = optimized_einsum(
                    "ij,jkl->ikl", metric_components, self.components
                )
            elif index_pos == 1:
                result_components = optimized_einsum(
                    "ij,kij->kil", metric_components, self.components
                )
            else:  # index_pos == 2
                result_components = optimized_einsum(
                    "ij,kli->klj", metric_components, self.components
                )
        elif self.rank == 4:  # Rank-4 tensor
            # General einsum for rank-4 tensors
            indices = list("abcd")
            indices[index_pos] = "x"
            output_indices = list("abcd")
            output_indices[index_pos] = "y"
            einsum_str = f"xy,{''.join(indices)}->{''.join(output_indices)}"
            result_components = optimized_einsum(einsum_str, metric_components, self.components)
        else:
            raise NotImplementedError(f"Index lowering not implemented for rank {self.rank} > 4")

        # Update indices
        new_indices: list[tuple[bool, str]] = self.indices.copy()
        new_indices[index_pos] = (True, index_name)  # Make covariant
        new_index_str = " ".join(f"{'_' if cov else ''}{name}" for cov, name in new_indices)

        # Preserve subclass type (FourVector, StressEnergyTensor, etc.)
        return self._create_subclass_instance(result_components, new_index_str)

    def trace(self, indices_pair: tuple[int, int] | None = None) -> float | sp.Expr:
        """
        Compute trace by contracting two indices.

        Args:
            indices_pair: Pair of indices to contract (default: first two)

        Returns:
            Scalar result of trace
        """
        if self.rank < 2:
            raise ValueError("Cannot take trace of tensor with rank < 2")

        if indices_pair is None:
            indices_pair = (0, 1)

        i, j = indices_pair
        if i >= self.rank or j >= self.rank:
            raise ValueError(f"Index pair {indices_pair} out of range for rank-{self.rank} tensor")

        # Validate indices can be contracted
        is_cov_i, name_i = self.indices[i]
        is_cov_j, name_j = self.indices[j]

        if is_cov_i == is_cov_j:
            warnings.warn(
                f"Taking trace of indices with same type: {'covariant' if is_cov_i else 'contravariant'}",
                stacklevel=2,
            )

        # Simple trace for rank-2 tensors
        if self.rank == 2:
            if is_numpy_array(self.components):
                return np.trace(self.components)
            else:
                return sum(self.components[k, k] for k in range(4))

        # For higher rank tensors, need general contraction
        if is_numpy_array(self.components):
            # For grid-based tensors, we need to map tensor indices to full array indices
            total_dims = self.components.ndim
            grid_dims = total_dims - self.rank

            # Map tensor indices to full array indices
            full_axis_i = grid_dims + i
            full_axis_j = grid_dims + j

            return np.trace(self.components, axis1=full_axis_i, axis2=full_axis_j)
        else:
            # SymPy implementation for higher rank
            if self.rank > 2:
                # Use pure SymPy operations for higher-rank tensors to preserve precision
                try:
                    import sympy.tensor.array as sp_array

                    # Convert to SymPy Array if needed
                    if hasattr(self.components, "rank") and self.components.rank() > 2:
                        array_comp = self.components
                    else:
                        array_comp = sp_array.Array(self.components)

                    # Compute trace by summing diagonal elements
                    # For a rank-n tensor, trace over indices i and j means:
                    # result[k1, k2, ..., k_{n-2}] = sum_m tensor[k1, ..., m, ..., m, ..., k_{n-2}]

                    shape = array_comp.shape
                    if shape[i] != shape[j]:
                        raise ValueError(
                            f"Cannot trace indices of different dimensions: {shape[i]} vs {shape[j]}"
                        )

                    # Create a symbolic sum over the diagonal
                    result: sp.Expr | int = 0
                    for m in range(shape[i]):
                        # Create index tuple with m at positions i and j
                        indices: list[slice | int] = [slice(None)] * len(shape)
                        indices[i] = m
                        indices[j] = m
                        result += array_comp[tuple(indices)]

                    return result

                except ImportError:
                    # Fallback to NumPy conversion if SymPy Array not available
                    warnings.warn(
                        "SymPy higher-rank tensor trace requires conversion to NumPy (SymPy Array not available)",
                        stacklevel=2,
                    )
                    temp_array = np.array(self.components).astype(complex)
                    result = np.trace(temp_array, axis1=i, axis2=j)
                    # Convert back to SymPy if the result is symbolic
                    if np.any(np.iscomplex(result)) or isinstance(result, complex):
                        return complex(result)
                    return float(result)
            else:
                # Manual trace for rank-3 and higher tensors without Array support
                if self.rank == 3:
                    # For 3rd rank tensor T_{ijk}, trace over indices (i,j) gives vector
                    # But this shouldn't happen since we check rank > 2, so fall through
                    pass

                raise NotImplementedError("SymPy trace for rank > 2 not implemented")
