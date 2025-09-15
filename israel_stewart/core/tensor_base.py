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
            # Check for numerical stability
            if np.any(np.isnan(components)) or np.any(np.isinf(components)):
                raise ValueError("Tensor components contain NaN or infinite values")
            # Check dimensions are 4 for spacetime
            if any(dim != 4 for dim in components.shape):
                raise ValueError(
                    f"All tensor dimensions must be 4 for spacetime, got shape {components.shape}"
                )
        elif is_sympy_matrix(components):
            # Check dimensions for SymPy
            if any(dim != 4 for dim in components.shape):
                raise ValueError(
                    f"All tensor dimensions must be 4 for spacetime, got shape {components.shape}"
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
        """Validate tensor rank consistency."""
        expected_rank = len(self.indices)

        if is_numpy_array(self.components):
            actual_rank = self.components.ndim
        elif is_sympy_matrix(self.components):
            actual_rank = len(self.components.shape)
        else:
            raise TypeError(f"Unsupported component type: {type(self.components)}")

        if actual_rank != expected_rank:
            raise ValueError(
                f"Tensor rank mismatch: indices specify rank {expected_rank}, "
                f"components have rank {actual_rank}"
            )

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

    def copy(self) -> "TensorField":
        """Create a deep copy of the tensor."""
        if isinstance(self.components, np.ndarray):
            new_components = self.components.copy()
        else:
            new_components = self.components.copy()
        return TensorField(new_components, self._index_string(), self.metric)

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
            # Default: reverse all indices
            axis_order = tuple(range(self.rank - 1, -1, -1))

        if isinstance(self.components, np.ndarray):
            new_components = np.transpose(self.components, axis_order)
        else:
            # For SymPy matrices, only support 2D transposition
            if self.rank == 2 and axis_order == (1, 0):
                new_components = self.components.T
            else:
                raise NotImplementedError("SymPy tensor transposition only supports 2D matrices")

        # Reorder indices accordingly
        new_indices = [self.indices[i] for i in axis_order]
        new_index_string = " ".join(f"{'_' if cov else ''}{name}" for cov, name in new_indices)

        return TensorField(new_components, new_index_string, self.metric)

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

        return TensorField(symmetrized_components, self._index_string(), self.metric)

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

        return TensorField(antisymmetrized_components, self._index_string(), self.metric)

    @monitor_performance("tensor_contract")
    def contract(self, other: "TensorField", self_index: int, other_index: int) -> "TensorField":
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

        # Build einsum string
        self_indices = [chr(97 + i) for i in range(self.rank)]  # a, b, c, ...
        other_indices = [chr(97 + self.rank + i) for i in range(other.rank)]

        # Make contracted indices the same
        other_indices[other_index] = self_indices[self_index]

        # Result indices (remove contracted ones)
        result_indices = [idx for i, idx in enumerate(self_indices) if i != self_index] + [
            idx for i, idx in enumerate(other_indices) if i != other_index
        ]

        einsum_str = (
            f"{''.join(self_indices)},{''.join(other_indices)}->" + f"{''.join(result_indices)}"
        )

        # Validate einsum string
        validate_einsum_string(einsum_str, self.rank, other.rank)

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
        result_index_str = " ".join(
            f"{'_' if cov else ''}{name}" for cov, name in result_index_list
        )

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
                    [sum(self_comp[j, i] * other_comp[i] for i in range(4)) for j in range(4)]
                )
            elif self_index == 1:  # Contract second index of matrix with vector
                return sp.Matrix(
                    [sum(self_comp[i, j] * other_comp[i] for i in range(4)) for j in range(4)]
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

        raise NotImplementedError(
            f"Manual contraction not implemented for ranks {self.rank} and {other.rank} "
            f"with indices {self_index} and {other_index}"
        )

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

        # Build contraction
        if self.rank == 1:  # Vector
            result_components = optimized_einsum("ij,j->i", metric_inverse, self.components)
        elif self.rank == 2:  # Matrix
            if index_pos == 0:
                result_components = optimized_einsum("ij,jk->ik", metric_inverse, self.components)
            else:
                result_components = optimized_einsum("ij,ki->kj", metric_inverse, self.components)
        else:
            raise NotImplementedError("Index raising not implemented for rank > 2")

        # Update indices
        new_indices = self.indices.copy()
        new_indices[index_pos] = (False, index_name)  # Make contravariant
        new_index_str = " ".join(f"{'_' if cov else ''}{name}" for cov, name in new_indices)

        return TensorField(result_components, new_index_str, self.metric)

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

        # Build contraction
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
        else:
            raise NotImplementedError("Index lowering not implemented for rank > 2")

        # Update indices
        new_indices = self.indices.copy()
        new_indices[index_pos] = (True, index_name)  # Make covariant
        new_index_str = " ".join(f"{'_' if cov else ''}{name}" for cov, name in new_indices)

        return TensorField(result_components, new_index_str, self.metric)

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
            trace_axes = (i, j)
            return np.trace(self.components, axis1=trace_axes[0], axis2=trace_axes[1])
        else:
            # SymPy implementation for higher rank
            raise NotImplementedError("SymPy trace for rank > 2 not implemented")
