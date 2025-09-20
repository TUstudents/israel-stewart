"""
Covariant derivatives and projection operators for relativistic hydrodynamics.

This module provides covariant differentiation in curved spacetime and
projection operators for fluid decomposition in Israel-Stewart formalism.
"""

import warnings
from functools import cached_property

# Forward reference for metrics
from typing import TYPE_CHECKING, Any

import numpy as np
import sympy as sp

from .four_vectors import FourVector
from .performance import monitor_performance

# Import base classes and utilities
from .tensor_base import TensorField
from .tensor_utils import (
    PhysicsError,
    is_sympy_array,
    is_sympy_matrix,
    is_sympy_type,
    optimized_einsum,
)

if TYPE_CHECKING:
    from .metrics import MetricBase


class CovariantDerivative:
    """
    Covariant derivative operator ∇_μ for tensor fields in curved spacetime.

    Handles derivatives of scalars, vectors, and higher-rank tensors using
    Christoffel symbols from the metric tensor.
    """

    def __init__(self, metric: "MetricBase"):
        """
        Initialize covariant derivative operator.

        Args:
            metric: Metric tensor providing Christoffel symbols
        """
        self.metric = metric

    @cached_property
    def christoffel_symbols(self) -> np.ndarray:
        """Get Christoffel symbols Γ^λ_μν from metric."""
        return self.metric.christoffel_symbols

    @monitor_performance("scalar_gradient")
    def scalar_gradient(
        self,
        scalar_field: np.ndarray | sp.Expr,
        coordinates: np.ndarray | list[np.ndarray],
    ) -> FourVector:
        """
        Compute covariant gradient ∇_μ φ = ∂_μ φ of scalar field.

        For scalars, covariant derivative equals ordinary partial derivative.

        Args:
            scalar_field: Scalar field φ(x^μ)
            coordinates: Coordinate grid [t, x, y, z]

        Returns:
            Gradient four-vector ∇_μ φ
        """
        if isinstance(scalar_field, sp.Expr):
            # Symbolic gradient
            gradient_components = []
            coord_symbols = [sp.Symbol(f"x{i}") for i in range(4)]

            for mu in range(4):
                gradient_components.append(sp.diff(scalar_field, coord_symbols[mu]))

            # Convert to proper format for FourVector (needs shape (4,) not (4,1))
            # Create a 1D SymPy Matrix that FourVector can handle properly
            try:
                import sympy.tensor.array as sp_array

                gradient_array = sp_array.Array(gradient_components)

                # Convert SymPy Array to flattened list for proper 1D handling
                try:
                    # Extract elements and create proper 1D structure
                    flat_components = [gradient_array[i] for i in range(4)]
                    return FourVector(flat_components, True, self.metric)
                except Exception:
                    # If conversion fails, use direct list approach
                    return FourVector(gradient_components, True, self.metric)

            except ImportError:
                # Fallback: use direct list for proper 1D format
                return FourVector(gradient_components, True, self.metric)

        else:
            # Numerical gradient using numpy
            if isinstance(coordinates, list):
                coord_arrays = coordinates
            else:
                coord_arrays = [coordinates]

            is_constant_field = False
            if isinstance(scalar_field, np.ndarray):
                if scalar_field.size <= 1:
                    is_constant_field = True
                elif scalar_field.dtype == object:
                    first = scalar_field.flat[0]
                    is_constant_field = all(element == first for element in scalar_field.flat)
                elif np.issubdtype(scalar_field.dtype, np.number):
                    is_constant_field = np.allclose(scalar_field, scalar_field.flat[0])
            elif np.isscalar(scalar_field):
                is_constant_field = True

            if is_constant_field:
                if isinstance(scalar_field, np.ndarray) and scalar_field.dtype == object:
                    zero_components = [sp.Integer(0)] * 4
                else:
                    dtype = getattr(scalar_field, "dtype", None)
                    zero_components = np.zeros(4, dtype=dtype if dtype is not None else float)
                return FourVector(zero_components, True, self.metric)

            gradient_arrays: list[np.ndarray] = []
            for mu in range(4):
                if scalar_field.ndim <= mu:
                    gradient_arrays.append(np.zeros_like(scalar_field))
                    continue

                axis_size = scalar_field.shape[mu]
                if axis_size < 2:
                    gradient_arrays.append(np.zeros_like(scalar_field))
                    continue

                edge_order = 2 if axis_size >= 3 else 1
                gradient_arrays.append(
                    np.gradient(scalar_field, coord_arrays[mu], axis=mu, edge_order=edge_order)
                )

            gradient_components = np.stack(gradient_arrays, axis=-1)
            return FourVector(gradient_components, True, self.metric)

    @monitor_performance("vector_divergence")
    def vector_divergence(
        self,
        vector_field: FourVector,
        coordinates: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute covariant divergence ∇_μ V^μ of a vector field across the entire grid.

        Optimized implementation that directly computes the trace of partial derivatives
        without creating large intermediate arrays.

        Returns a scalar field (grid) representing the divergence at each point.
        """
        christoffel = self.christoffel_symbols
        components = vector_field.components  # Shape: (..., 4)

        # 1. Compute partial derivative trace ∂_μ V^μ directly (much more efficient)
        partial_deriv_trace = np.zeros(components.shape[:-1])
        for mu in range(4):
            # Only compute diagonal terms we need for the trace
            # Check coordinate array size to determine appropriate edge_order
            coord_array = coordinates[mu]
            if hasattr(coord_array, 'ndim') and hasattr(coord_array, 'shape'):
                grid_size_mu = len(coord_array) if coord_array.ndim == 1 else coord_array.shape[mu]
            else:
                # Fallback for non-numpy array coordinates
                grid_size_mu = len(coord_array) if hasattr(coord_array, '__len__') else 1

            if grid_size_mu < 2:
                # Single point grids: derivative is undefined, contributes zero
                # This handles degenerate cases gracefully
                continue

            if grid_size_mu >= 3:
                # Use second-order accurate edges for grids with sufficient points
                partial_deriv_trace += np.gradient(
                    components[..., mu], coordinates[mu], axis=mu, edge_order=2
                )
            else:
                # Fall back to first-order for minimal grids (2 points)
                partial_deriv_trace += np.gradient(
                    components[..., mu], coordinates[mu], axis=mu, edge_order=1
                )

        # 2. Compute the Christoffel term Γ^μ_μλ V^λ, vectorized over the grid
        # christoffel shape: (4, 4, 4) -> extract trace Γ^μ_μλ
        # components shape: (..., 4)
        # result shape: (...)
        christoffel_term = optimized_einsum("iil,...l->...", christoffel, components)

        # 3. Return the sum as a scalar field
        result: np.ndarray = partial_deriv_trace + christoffel_term
        return result

    def material_derivative(
        self,
        tensor_field: TensorField,
        four_velocity: FourVector,
        coordinates: np.ndarray | list[np.ndarray],
    ) -> TensorField:
        """
        Compute material derivative D = u^μ ∇_μ along fluid flow.

        Optimized implementation that computes the full covariant derivative once
        and then contracts with the four-velocity, avoiding redundant calculations.

        Args:
            tensor_field: Tensor field to differentiate
            four_velocity: Fluid four-velocity u^μ
            coordinates: Coordinate grid

        Returns:
            Material derivative of tensor field
        """
        # Compute full covariant derivative ∇_μ T once (returns tensor with extra derivative index)
        full_cov_deriv = self.tensor_covariant_derivative(tensor_field, coordinates)

        # Contract with four-velocity: u^μ ∇_μ T
        # The derivative index is the last one in full_cov_deriv
        material_deriv_components = optimized_einsum(
            "m,...m->...", four_velocity.components, full_cov_deriv.components
        )

        return TensorField(material_deriv_components, tensor_field._index_string(), self.metric)

    def _vectorized_christoffel_contractions(
        self,
        tensor_components: np.ndarray,
        christoffel: np.ndarray,
        tensor_indices: list[tuple[bool, str]],
    ) -> np.ndarray:
        """
        Compute all Christoffel correction terms using vectorized operations.

        This method computes corrections for all tensor indices using proper
        einsum patterns that respect the mathematical structure of covariant derivatives.

        For covariant indices: -Γ^λ_μα T_...λ...
        For contravariant indices: +Γ^α_μλ T^...λ...

        Args:
            tensor_components: Original tensor components with shape (...grid..., ...tensor...)
            christoffel: Christoffel symbols Γ^λ_μν with shape (4, 4, 4)
            tensor_indices: List of (is_covariant, name) tuples describing tensor structure

        Returns:
            Combined correction array with shape (...grid..., ...tensor..., 4)
            where the last dimension is the derivative index μ
        """
        # Initialize correction array with proper shape
        correction = np.zeros(tensor_components.shape + (4,))

        # Get tensor rank from the indices list
        tensor_rank = len(tensor_indices)

        # Process each tensor index individually with correct einsum patterns
        for idx_pos, (is_covariant, _) in enumerate(tensor_indices):
            if is_covariant:
                # Covariant correction: -Γ^λ_μα T_...λ...
                correction += self._compute_covariant_correction(
                    tensor_components, christoffel, idx_pos, tensor_rank
                )
            else:
                # Contravariant correction: +Γ^α_μλ T^...λ...
                correction += self._compute_contravariant_correction(
                    tensor_components, christoffel, idx_pos, tensor_rank
                )

        return correction

    def _compute_covariant_correction(
        self,
        tensor_components: np.ndarray,
        christoffel: np.ndarray,
        covariant_idx_pos: int,
        tensor_rank: int,
    ) -> np.ndarray:
        """
        Compute covariant index correction: -Γ^λ_μα T_...λ...

        For covariant index α at position covariant_idx_pos, we need:
        -Γ^λ_μα T_...λ...

        Args:
            tensor_components: Tensor field with shape (...grid..., ...tensor...)
            christoffel: Christoffel symbols Γ^λ_μν with shape (4, 4, 4)
            covariant_idx_pos: Position of covariant index within tensor indices
            tensor_rank: Rank of the tensor (number of tensor indices)

        Returns:
            Correction array with shape (...grid..., ...tensor..., 4)
        """
        # Use einsum for efficient contraction
        # For covariant correction: -Γ^λ_μα T_...λ...
        # We contract λ from christoffel with the covariant_idx_pos-th tensor index

        # Build einsum pattern for the contraction
        ndim = tensor_components.ndim
        grid_dims = ndim - tensor_rank

        # Create subscript letters
        grid_letters = "ijklmnop"[:grid_dims] if grid_dims > 0 else ""
        tensor_letters = "abcdefgh"[:tensor_rank]

        # Use 'z' for the contracted index (λ) to avoid conflicts
        contracted_letter = "z"

        # Tensor subscript with contracted index replaced by contracted_letter
        tensor_letters_list = list(tensor_letters)
        tensor_letters_list[covariant_idx_pos] = contracted_letter
        tensor_subscript = grid_letters + "".join(tensor_letters_list)

        # Result subscript: original tensor + derivative index
        result_subscript = grid_letters + tensor_letters + "m"

        # Christoffel subscript: Γ^λ_μα where λ=z, μ=m, α=original_index
        original_index = tensor_letters[covariant_idx_pos]
        christoffel_subscript = f"{contracted_letter}m{original_index}"

        # Complete einsum pattern
        einsum_pattern = f"{tensor_subscript},{christoffel_subscript}->{result_subscript}"

        # Perform contraction with negative sign
        correction = -optimized_einsum(einsum_pattern, tensor_components, christoffel)

        return correction

    def _compute_contravariant_correction(
        self,
        tensor_components: np.ndarray,
        christoffel: np.ndarray,
        contravariant_idx_pos: int,
        tensor_rank: int,
    ) -> np.ndarray:
        """
        Compute contravariant index correction: +Γ^α_μλ T^...λ...

        For contravariant index α at position contravariant_idx_pos, we need:
        +Γ^α_μλ T^...λ...

        Args:
            tensor_components: Tensor field with shape (...grid..., ...tensor...)
            christoffel: Christoffel symbols Γ^λ_μν with shape (4, 4, 4)
            contravariant_idx_pos: Position of contravariant index within tensor indices
            tensor_rank: Rank of the tensor (number of tensor indices)

        Returns:
            Correction array with shape (...grid..., ...tensor..., 4)
        """
        # Use einsum for efficient contraction
        # For contravariant correction: +Γ^α_μλ T^...λ...
        # We contract λ from christoffel with the contravariant_idx_pos-th tensor index

        # Build einsum pattern for the contraction
        ndim = tensor_components.ndim
        grid_dims = ndim - tensor_rank

        # Create subscript letters
        grid_letters = "ijklmnop"[:grid_dims] if grid_dims > 0 else ""
        tensor_letters = "abcdefgh"[:tensor_rank]

        # Use 'z' for the contracted index (λ) to avoid conflicts
        contracted_letter = "z"

        # Tensor subscript with contracted index replaced by contracted_letter
        tensor_letters_list = list(tensor_letters)
        tensor_letters_list[contravariant_idx_pos] = contracted_letter
        tensor_subscript = grid_letters + "".join(tensor_letters_list)

        # Result subscript: original tensor + derivative index
        result_subscript = grid_letters + tensor_letters + "m"

        # Christoffel subscript: Γ^α_μλ where α=original_index, μ=m, λ=z
        original_index = tensor_letters[contravariant_idx_pos]
        christoffel_subscript = f"{original_index}m{contracted_letter}"

        # Complete einsum pattern
        einsum_pattern = f"{tensor_subscript},{christoffel_subscript}->{result_subscript}"

        # Perform contraction (positive sign)
        correction = optimized_einsum(einsum_pattern, tensor_components, christoffel)

        return correction

    @monitor_performance("tensor_covariant_derivative")
    def tensor_covariant_derivative(
        self,
        tensor_field: TensorField,
        coordinates: list[np.ndarray],
    ) -> TensorField:
        """
        Compute covariant derivative ∇_μ T^α...β... of a general tensor field.

        Optimized implementation using vectorized operations for significant performance
        improvement. Eliminates sequential index processing and intermediate array allocations.

        Returns a new tensor field with one additional covariant index (the derivative index).
        """
        # Handle SymPy tensors with dedicated path
        components = tensor_field.components

        if is_sympy_type(components):
            return self._symbolic_tensor_covariant_derivative(tensor_field, coordinates)

        if hasattr(components, "dtype") and components.dtype == object:
            return self._symbolic_tensor_covariant_derivative(tensor_field, coordinates)

        christoffel = self.christoffel_symbols

        # 1. Compute all partial derivatives ∂_μ T^..._... efficiently
        partial_derivatives = self._compute_all_partial_derivatives(components, coordinates)

        # 2. Compute all Christoffel corrections using vectorized operations
        # This replaces the sequential loop with optimized einsum operations
        christoffel_corrections = self._vectorized_christoffel_contractions(
            components, christoffel, tensor_field.indices
        )

        # 3. Combine partial derivatives and corrections in single operation
        # Avoids intermediate array allocation
        covariant_derivative = partial_derivatives + christoffel_corrections

        # 4. Build new index string efficiently
        new_indices = self._build_derivative_index_string(tensor_field)

        return TensorField(covariant_derivative, new_indices, self.metric)

    def _compute_all_partial_derivatives(
        self, tensor_components: np.ndarray, coordinates: list[np.ndarray]
    ) -> np.ndarray:
        """
        Compute all partial derivatives ∂_μ T efficiently using vectorized operations.

        For tensor fields on a grid, the first 4 dimensions of tensor_components
        correspond to the spacetime grid (t,x,y,z), and remaining dimensions
        are tensor component indices.

        Returns array with shape (..., 4) where last dimension is derivative index.
        """
        # Pre-allocate result array
        result_shape = tensor_components.shape + (4,)
        partial_derivatives = np.zeros(result_shape, dtype=tensor_components.dtype)

        # Compute derivatives along each coordinate direction
        # mu=0,1,2,3 corresponds to coordinates t,x,y,z respectively
        for mu in range(4):
            # Determine appropriate edge_order based on grid size
            grid_size_mu = tensor_components.shape[mu]
            if grid_size_mu < 2:
                continue
            if grid_size_mu >= 3:
                # Take gradient along grid axis mu with respect to coordinate mu
                partial_derivatives[..., mu] = np.gradient(
                    tensor_components, coordinates[mu], axis=mu, edge_order=2
                )
            else:
                # Take gradient along grid axis mu with respect to coordinate mu
                partial_derivatives[..., mu] = np.gradient(
                    tensor_components, coordinates[mu], axis=mu, edge_order=1
                )

        return partial_derivatives

    def _build_derivative_index_string(self, tensor_field: TensorField) -> str:
        """Build index string for tensor with additional covariant derivative index."""
        if hasattr(tensor_field, "add_covariant_index"):
            # Use proper method if available
            return str(tensor_field.add_covariant_index("d"))
        else:
            # Fallback to string manipulation with better formatting
            current_indices = tensor_field._index_string()
            return current_indices + " _d" if current_indices else "_d"

    def _symbolic_tensor_covariant_derivative(
        self, tensor_field: TensorField, coordinates: list[np.ndarray]
    ) -> TensorField:
        """
        Compute covariant derivative for symbolic (SymPy) tensor fields.

        Uses SymPy-specific operations to preserve symbolic precision when possible.
        """
        try:
            import sympy.tensor.array as sp_array

            # Check if we can use SymPy Array operations
            if hasattr(tensor_field.components, "is_Matrix") and tensor_field.components.is_Matrix:
                return self._sympy_array_covariant_derivative(tensor_field, coordinates)
            else:
                # Fall back for non-matrix SymPy objects
                return self._fallback_tensor_covariant_derivative(tensor_field, coordinates)

        except ImportError:
            # SymPy Array not available - use fallback
            warnings.warn(
                "SymPy Array not available for symbolic covariant derivative, "
                "using fallback implementation which may convert to numerical",
                UserWarning,
                stacklevel=3,
            )
            return self._fallback_tensor_covariant_derivative(tensor_field, coordinates)

    def _sympy_array_covariant_derivative(
        self, tensor_field: TensorField, coordinates: list[np.ndarray]
    ) -> TensorField:
        """
        Compute covariant derivative using SymPy Array operations for symbolic precision.

        This method uses SymPy's tensor array framework to maintain symbolic expressions
        throughout the calculation without numerical conversion.
        """
        import sympy as sp
        import sympy.tensor.array as sp_array

        # Convert SymPy Matrix to Array if needed
        if hasattr(tensor_field.components, "is_Matrix") and tensor_field.components.is_Matrix:
            tensor_array = sp_array.Array(tensor_field.components)
        else:
            tensor_array = tensor_field.components

        # Get symbolic Christoffel symbols
        christoffel = self.christoffel_symbols
        if hasattr(christoffel, "is_Matrix") and christoffel.is_Matrix:
            christoffel_array = sp_array.Array(christoffel)
        elif hasattr(christoffel, "dtype") and christoffel.dtype == object:
            # Convert NumPy object array with SymPy expressions to SymPy Array
            christoffel_array = sp_array.Array(christoffel)
        else:
            # Numerical Christoffel - convert to SymPy for symbolic computation
            christoffel_array = sp_array.Array(sp.Matrix(christoffel))

        # For symbolic computation, we need coordinate symbols
        coord_symbols = [sp.Symbol(f"x_{i}") for i in range(4)]

        # Compute symbolic partial derivatives
        partial_derivatives = []
        for mu in range(4):
            # Symbolic differentiation with respect to coordinate μ
            partial_deriv = sp_array.derive_by_array(tensor_array, coord_symbols[mu])
            partial_derivatives.append(partial_deriv)

        # Stack partial derivatives
        partial_derivatives_array = sp_array.Array(partial_derivatives)

        # Compute Christoffel corrections using SymPy Array operations
        corrections = []
        for mu in range(4):  # For each derivative direction
            correction_mu = sp.zeros(*tensor_array.shape)

            # Add corrections for each tensor index
            for i, (is_cov, _) in enumerate(tensor_field.indices):
                if is_cov:
                    # Covariant correction: -Γ^λ_μα T_...λ...
                    correction_mu += self._sympy_covariant_index_correction(
                        tensor_array, christoffel_array, i, mu
                    )
                else:
                    # Contravariant correction: +Γ^α_μλ T^...λ...
                    correction_mu += self._sympy_contravariant_index_correction(
                        tensor_array, christoffel_array, i, mu
                    )

            corrections.append(correction_mu)

        # Combine partial derivatives and corrections
        corrections_array = sp_array.Array(corrections)
        covariant_derivative = partial_derivatives_array + corrections_array

        # Build new index string
        new_indices = self._build_derivative_index_string(tensor_field)

        # Convert result back to appropriate SymPy type
        if tensor_field.rank == 1:
            # Result is rank-2: convert to Matrix
            result_matrix = sp.Matrix(covariant_derivative)
            return TensorField(result_matrix, new_indices, self.metric)
        else:
            # Higher rank: keep as Array
            return TensorField(covariant_derivative, new_indices, self.metric)

    def _sympy_covariant_index_correction(
        self, tensor_array: "sp.Array", christoffel_array: "sp.Array", index_pos: int, mu: int
    ) -> sp.Basic:
        """Compute covariant index correction for SymPy tensors: -Γ^λ_μα T_...λ..."""
        import sympy.tensor.array as sp_array

        # For covariant index correction: -Γ^λ_μα T_...λ...
        # Contract Christoffel's first index with tensor's index_pos
        result = sp_array.tensorcontraction(
            sp_array.tensorproduct(christoffel_array, tensor_array),
            (0, len(christoffel_array.shape) + index_pos),
        )

        # Extract the μ,α component and apply negative sign
        return -result[mu, index_pos]

    def _sympy_contravariant_index_correction(
        self, tensor_array: "sp.Array", christoffel_array: "sp.Array", index_pos: int, mu: int
    ) -> sp.Basic:
        """Compute contravariant index correction for SymPy tensors: +Γ^α_μλ T^...λ..."""
        import sympy.tensor.array as sp_array

        # For contravariant index correction: +Γ^α_μλ T^...λ...
        # Contract Christoffel's third index with tensor's index_pos
        result = sp_array.tensorcontraction(
            sp_array.tensorproduct(christoffel_array, tensor_array),
            (2, len(christoffel_array.shape) + index_pos),
        )

        # Extract the α,μ component (positive sign)
        return result[index_pos, mu]

    def _fallback_tensor_covariant_derivative(
        self, tensor_field: TensorField, coordinates: list[np.ndarray]
    ) -> TensorField:
        """
        Fallback implementation using original sequential approach.

        Used for SymPy tensors or when vectorized approach encounters issues.
        """
        christoffel = self.christoffel_symbols
        components = tensor_field.components

        # Original implementation for compatibility
        partial_derivatives = np.stack(
            [self._partial_derivative(components, mu, coordinates) for mu in range(4)], axis=-1
        )

        correction = np.zeros_like(partial_derivatives)

        # Sequential processing for compatibility
        for i, (is_cov, _) in enumerate(tensor_field.indices):
            if is_cov:
                # Covariant index correction: -Γ^λ_μi T_...λ...
                correction -= self._contract_christoffel_covariant(components, christoffel, i)
            else:
                # Contravariant index correction: +Γ^i_μλ T^...λ...
                correction += self._contract_christoffel(components, christoffel, i)

        covariant_derivative = partial_derivatives + correction
        new_indices = self._build_derivative_index_string(tensor_field)

        return TensorField(covariant_derivative, new_indices, self.metric)

    def _partial_derivative(
        self,
        tensor_components: np.ndarray,
        coord_index: int,
        coordinates: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute partial derivative ∂_μ T along a specified coordinate axis.

        Args:
            tensor_components: Tensor field values on grid
            coord_index: Which coordinate to differentiate (0=t, 1=x, 2=y, 3=z)
            coordinates: List of coordinate arrays [t_coords, x_coords, y_coords, z_coords]

        Returns:
            Partial derivative along the specified coordinate direction
        """
        # Determine appropriate edge_order based on grid size
        grid_size = tensor_components.shape[coord_index]
        if grid_size < 2:
            return np.zeros_like(tensor_components)
        if grid_size >= 3:
            # Take gradient along grid axis coord_index with respect to coordinate coord_index
            result: np.ndarray = np.gradient(
                tensor_components, coordinates[coord_index], axis=coord_index, edge_order=2
            )
        else:
            # Take gradient along grid axis coord_index with respect to coordinate coord_index
            result: np.ndarray = np.gradient(
                tensor_components, coordinates[coord_index], axis=coord_index, edge_order=1
            )
        return result

    def _contract_christoffel(
        self,
        tensor_components: np.ndarray,
        christoffel: np.ndarray,
        tensor_index_pos: int,
    ) -> np.ndarray:
        """Computes the contravariant correction term: Γ^α_μλ T^...λ..."""
        # For grid-based tensors, need to account for grid dimensions
        # tensor_index_pos is relative to tensor indices, not absolute array position
        ndim = tensor_components.ndim

        # Determine tensor rank from the last dimensions that are size 4
        tensor_rank = 0
        for i in range(ndim - 1, -1, -1):
            if tensor_components.shape[i] == 4:
                tensor_rank += 1
            else:
                break

        # Calculate absolute index position in the array
        grid_dims = ndim - tensor_rank
        absolute_index_pos = grid_dims + tensor_index_pos

        # Move the contracted axis to the end
        moved_tensor = np.moveaxis(tensor_components, absolute_index_pos, -1)
        # Contract with Christoffel symbols
        # Γ^α_μλ T^...λ -> result^...α_μ
        # (a, m, l) * (..., l) -> (..., a, m)
        correction = np.tensordot(moved_tensor, christoffel, axes=([-1], [2]))
        # Move the new axes to their correct positions
        return np.moveaxis(correction, [-2, -1], [absolute_index_pos, tensor_components.ndim])

    def _contract_christoffel_covariant(
        self,
        tensor_components: np.ndarray,
        christoffel: np.ndarray,
        tensor_index_pos: int,
    ) -> np.ndarray:
        """Computes the covariant correction term: Γ^λ_μα T_...λ..."""
        # For grid-based tensors, need to account for grid dimensions
        # tensor_index_pos is relative to tensor indices, not absolute array position
        ndim = tensor_components.ndim

        # Determine tensor rank from the last dimensions that are size 4
        tensor_rank = 0
        for i in range(ndim - 1, -1, -1):
            if tensor_components.shape[i] == 4:
                tensor_rank += 1
            else:
                break

        # Calculate absolute index position in the array
        grid_dims = ndim - tensor_rank
        absolute_index_pos = grid_dims + tensor_index_pos

        # Move the contracted axis to the end
        moved_tensor = np.moveaxis(tensor_components, absolute_index_pos, -1)
        # Contract with Christoffel symbols
        # Γ^λ_μα T_...λ -> result_...α_μ
        # (l, m, a) * (..., l) -> (..., m, a)
        correction = np.tensordot(moved_tensor, christoffel, axes=([-1], [0]))
        # Move the new axes to their correct positions
        return np.moveaxis(correction, [-2, -1], [tensor_components.ndim, absolute_index_pos])

    def laplacian(
        self,
        scalar_field: np.ndarray | sp.Expr,
        coordinates: np.ndarray | list[np.ndarray],
    ) -> float | sp.Expr:
        """
        Compute covariant Laplacian ∇^μ ∇_μ φ = g^μν ∇_μ ∇_ν φ.

        Args:
            scalar_field: Scalar field φ
            coordinates: Coordinate grid

        Returns:
            Laplacian scalar
        """
        # First compute gradient
        gradient = self.scalar_gradient(scalar_field, coordinates)

        # Then compute divergence of gradient (raise index first)
        gradient_contravariant = gradient.raise_index(0)

        return self.vector_divergence(gradient_contravariant, coordinates)

    def lie_derivative(
        self,
        tensor_field: TensorField,
        vector_field: FourVector,
        coordinates: np.ndarray | list[np.ndarray],
    ) -> TensorField:
        """
        Compute Lie derivative L_V T of tensor along vector field.

        **WARNING**: This implementation is incomplete! It only includes the
        material derivative term V^μ ∇_μ T. The full Lie derivative requires
        additional terms involving the derivatives of the vector field.

        Args:
            tensor_field: Tensor to differentiate
            vector_field: Vector field V^μ
            coordinates: Coordinate grid

        Returns:
            Incomplete Lie derivative (material derivative part only)

        Raises:
            UserWarning: Always warns about incomplete implementation
        """
        warnings.warn(
            "lie_derivative is incomplete - only material derivative V^μ ∇_μ T implemented. "
            "Missing tensor index contraction terms (∇_μ V^ν) for proper Lie derivative. "
            "Use with caution for physics calculations.",
            UserWarning,
            stacklevel=2,
        )

        # Lie derivative: L_V T = V^μ ∇_μ T + (∇_μ V^ν) T terms
        # Only the first term is implemented:
        material_part = self.material_derivative(tensor_field, vector_field, coordinates)

        # TODO: Add remaining terms for complete Lie derivative:
        # For scalars: complete (material derivative is sufficient)
        # For vectors: + (∇_μ V^ν) T_ν terms
        # For tensors: + appropriate index contraction terms

        return material_part


class ProjectionOperator:
    """
    Projection operators for fluid decomposition in relativistic hydrodynamics.

    Decomposes tensors into components parallel and perpendicular to the
    fluid four-velocity, essential for Israel-Stewart formalism.
    """

    def __init__(self, four_velocity: FourVector, metric: "MetricBase"):
        """
        Initialize projection operators.

        Args:
            four_velocity: Fluid four-velocity u^μ (assumed normalized)
            metric: Spacetime metric tensor
        """
        self.u = four_velocity
        self.metric = metric

        # Pre-compute covariant and contravariant forms for reuse
        if four_velocity.indices[0][0]:  # Stored vector is covariant
            self.u_cov = four_velocity
            self.u_contra = four_velocity.raise_index(0)
        else:
            self.u_contra = four_velocity
            self.u_cov = four_velocity.lower_index(0)

        if self.u_contra.indices[0][0]:
            self.u_contra = self.u_contra.raise_index(0)
        if not self.u_cov.indices[0][0]:
            self.u_cov = self.u_cov.lower_index(0)

        # Validate four-velocity normalization with correct signature handling
        mag_sq = self.u_contra.dot(self.u_contra)
        signature = getattr(metric, "signature", (-1, 1, 1, 1))

        if signature[0] < 0:  # Mostly-plus signature (-,+,+,+)
            expected_norm = -1.0
            if abs(mag_sq - expected_norm) > 1e-10:
                try:
                    normalized = self.u_contra.normalize()
                    self.u_contra = normalized
                    self.u_cov = normalized.lower_index(0)
                    mag_sq = self.u_contra.dot(self.u_contra)
                except PhysicsError:
                    warnings.warn(
                        f"Four-velocity should be normalized: u^μ u_μ = -1, got {mag_sq:.2e}",
                        stacklevel=2,
                    )
        else:  # Mostly-minus signature (+,-,-,-)
            expected_norm = 1.0
            if abs(mag_sq - expected_norm) > 1e-10:
                try:
                    normalized = self.u_contra.normalize()
                    self.u_contra = normalized
                    self.u_cov = normalized.lower_index(0)
                    mag_sq = self.u_contra.dot(self.u_contra)
                except PhysicsError:
                    warnings.warn(
                        f"Four-velocity should be normalized: u^μ u_μ = +1, got {mag_sq:.2e}",
                        stacklevel=2,
                    )

    @staticmethod
    def _to_sympy_column(components: Any) -> sp.Matrix:
        if is_sympy_array(components):
            return sp.Matrix([components[i] for i in range(4)])
        if isinstance(components, np.ndarray):
            flat = np.asarray(components).reshape(-1)
            return sp.Matrix(flat)
        return sp.Matrix(components)

    def parallel_projector(self) -> TensorField:
        """
        Compute parallel projection tensor P^μν = u^μ u^ν.

        Projects onto direction parallel to four-velocity.

        Returns:
            Parallel projection tensor
        """
        if isinstance(self.u_contra.components, np.ndarray):
            u_outer = optimized_einsum(
                "...m,...n->...mn", self.u_contra.components, self.u_contra.components
            )
            return TensorField(u_outer, "mu nu", self.metric)

        # SymPy path
        u_vec = self._to_sympy_column(self.u_contra.components)
        return TensorField(u_vec * u_vec.T, "mu nu", self.metric)

    @monitor_performance("perpendicular_projector")
    def perpendicular_projector(self) -> TensorField:
        """
        Compute perpendicular projection tensor Δ^μν = g^μν + σ u^μ u^ν.

        Projects onto 3-space orthogonal to four-velocity using σ = -sign(u · u)
        so the construction adapts automatically to both mostly-plus and
        mostly-minus metric signatures.

        Returns:
            Perpendicular projection tensor Δ^μν
        """
        # Get inverse metric g^μν
        g_inverse = self.metric.inverse

        if isinstance(self.u_contra.components, np.ndarray):
            u_contra = self.u_contra.components
            u_cov = self.u_cov.components

            u_dot_u = optimized_einsum("...m,...m->...", u_contra, u_cov)

            if np.any(np.isclose(u_dot_u, 0.0)):
                raise ValueError("Four-velocity must be timelike for perpendicular projection")

            sigma = -np.sign(u_dot_u)
            sigma = np.asarray(sigma, dtype=u_contra.dtype)
            if np.ndim(sigma) == 0:
                sigma_value = float(sigma)
                if np.isclose(sigma_value, 0.0):
                    raise ValueError("Cannot determine projector signature for null four-velocity")
            else:
                if np.any(np.isclose(sigma, 0.0)):
                    raise ValueError("Cannot determine projector signature for null four-velocity")

            u_outer = optimized_einsum("...m,...n->...mn", u_contra, u_contra)

            if np.ndim(sigma) == 0:
                delta = g_inverse + sigma * u_outer
            else:
                delta = g_inverse + sigma[..., np.newaxis, np.newaxis] * u_outer

            return TensorField(delta, "mu nu", self.metric)

        # SymPy path; treat components as column vectors where necessary
        u_vec = self._to_sympy_column(self.u_contra.components)
        u_cov_vec = self._to_sympy_column(self.u_cov.components)

        u_dot_u = sum(u_vec[i] * u_cov_vec[i] for i in range(4))
        if u_dot_u == 0:
            raise ValueError("Four-velocity must be timelike for perpendicular projection")

        sigma = -sp.sign(u_dot_u)
        if sigma == 0:
            raise ValueError("Cannot determine projector signature for null four-velocity")

        g_inverse_matrix = sp.Matrix(g_inverse)
        u_outer = u_vec * u_vec.T
        delta = g_inverse_matrix + sigma * u_outer

        return TensorField(delta, "mu nu", self.metric)

    def project_vector_parallel(self, vector: FourVector) -> FourVector:
        """
        Project vector parallel to four-velocity: V_∥^μ = (u · V / u · u) u^μ.

        The sign convention depends on metric signature:
        - Mostly plus (-,+,+,+): u · u = -c², so V_∥^μ = -(u · V) u^μ
        - Mostly minus (+,-,-,-): u · u = +c², so V_∥^μ = +(u · V) u^μ

        Args:
            vector: Input four-vector

        Returns:
            Parallel component of vector
        """
        # Convert vector to contravariant form for consistent operations
        if vector.indices[0][0]:
            vector_contra = vector.raise_index(0)
        else:
            vector_contra = vector

        if isinstance(self.u_contra.components, np.ndarray):
            u_cov = self.u_cov.components
            u_contra = self.u_contra.components
            v_contra = vector_contra.components
            u_dot_v = optimized_einsum("...m,...m->...", u_cov, v_contra)
            u_dot_u = optimized_einsum("...m,...m->...", u_cov, u_contra)

            if np.any(np.isclose(u_dot_u, 0.0)):
                raise ValueError("Four-velocity must be timelike for projection")

            coeff = u_dot_v / u_dot_u
            coeff_array = np.asarray(coeff)
            if coeff_array.ndim == 0:
                parallel = coeff_array * u_contra
            else:
                parallel = coeff_array[..., np.newaxis] * u_contra
            result = FourVector(parallel, False, self.metric)
            return result.lower_index(0) if vector.indices[0][0] else result

        # SymPy path
        u_cov_vec = self._to_sympy_column(self.u_cov.components)
        u_contra_vec = self._to_sympy_column(self.u_contra.components)
        v_contra_vec = self._to_sympy_column(vector_contra.components)

        u_dot_v = sum(u_cov_vec[i] * v_contra_vec[i] for i in range(4))
        u_dot_u = sum(u_cov_vec[i] * u_contra_vec[i] for i in range(4))

        if u_dot_u == 0:
            raise ValueError("Four-velocity must be timelike for projection")

        parallel = (u_dot_v / u_dot_u) * u_contra_vec
        result = FourVector(parallel, False, self.metric)
        return result.lower_index(0) if vector.indices[0][0] else result

    @monitor_performance("project_vector_perpendicular")
    def project_vector_perpendicular(self, vector: FourVector) -> FourVector:
        """
        Project vector perpendicular to four-velocity: V_⊥^μ = Δ^μν V_ν.

        Args:
            vector: Input four-vector

        Returns:
            Perpendicular component of vector
        """
        delta = self.perpendicular_projector()
        vector_contra = vector.raise_index(0) if vector.indices[0][0] else vector
        vector_cov = vector_contra.lower_index(0)

        if isinstance(vector_contra.components, np.ndarray):
            perp_components = optimized_einsum(
                "...mn,...n->...m", delta.components, vector_cov.components
            )
            result = FourVector(perp_components, False, self.metric)
            return result.lower_index(0) if vector.indices[0][0] else result

        # SymPy path
        perp_components = delta.components * self._to_sympy_column(vector_cov.components)
        result = FourVector(perp_components, False, self.metric)
        return result.lower_index(0) if vector.indices[0][0] else result

    def project_tensor_spatial(self, tensor: TensorField) -> TensorField:
        """
        Project tensor into spatial hypersurface: T_spatial^μν = Δ^μα Δ^νβ T_αβ.

        Args:
            tensor: Input rank-2 tensor

        Returns:
            Spatially projected tensor
        """
        if tensor.rank != 2:
            raise ValueError("Spatial projection only implemented for rank-2 tensors")

        delta = self.perpendicular_projector()
        tensor_lowered = tensor.lower_index(0).lower_index(1)

        # Double projection: T_spatial^μν = Δ^μα Δ^νβ T_αβ
        if isinstance(tensor.components, np.ndarray):
            spatial = optimized_einsum(
                "ma,nb,ab->mn",
                delta.components,
                delta.components,
                tensor_lowered.components,
            )
        else:
            # Proper SymPy tensor contraction: Δ^μα T_αβ Δ^βν
            spatial = delta.components @ tensor_lowered.components @ delta.components.T

        return TensorField(spatial, "mu nu", self.metric)

    def extract_scalar_density(self, tensor: TensorField) -> float | sp.Expr:
        """
        Extract scalar density from tensor: ρ = u_μ u_ν T^μν.

        Args:
            tensor: Input rank-2 tensor

        Returns:
            Scalar density
        """
        if tensor.rank != 2:
            raise ValueError("Scalar extraction only works for rank-2 tensors")

        u_cov = self.u.lower_index(0)

        # Contract: ρ = u_μ u_ν T^μν
        if isinstance(tensor.components, np.ndarray):
            scalar = optimized_einsum(
                "i,j,ij->", u_cov.components, u_cov.components, tensor.components
            )
        else:
            scalar = 0
            for mu in range(4):
                for nu in range(4):
                    scalar += (
                        u_cov.components[mu] * u_cov.components[nu] * tensor.components[mu, nu]
                    )

        return scalar

    def extract_vector_density(self, tensor: TensorField) -> FourVector:
        """
        Extract vector density from tensor: j^μ = -Δ^μν u_α T^α_ν.

        Args:
            tensor: Input rank-2 tensor

        Returns:
            Vector density (spatial)
        """
        if tensor.rank != 2:
            raise ValueError("Vector extraction only works for rank-2 tensors")

        delta = self.perpendicular_projector()
        u_cov = self.u.lower_index(0)
        tensor_mixed = tensor.lower_index(1)  # T^α_ν

        # Contract: j^μ = -Δ^μν u_α T^α_ν
        if isinstance(tensor.components, np.ndarray):
            vector = -optimized_einsum(
                "mn,a,an->m",
                delta.components,
                u_cov.components,
                tensor_mixed.components,
            )
        else:
            vector = sp.zeros(4, 1)
            for mu in range(4):
                for nu in range(4):
                    for alpha in range(4):
                        vector[mu] -= (
                            delta.components[mu, nu]
                            * u_cov.components[alpha]
                            * tensor_mixed.components[alpha, nu]
                        )

        return FourVector(vector, False, self.metric)

    def decompose_tensor(
        self, tensor: TensorField
    ) -> dict[str, float | sp.Expr | FourVector | TensorField]:
        """
        Complete 3+1 decomposition of rank-2 tensor.

        Decomposes T^μν into:
        - Scalar density: ρ = u_μ u_ν T^μν
        - Vector density: j^μ = -Δ^μν u_α T^α_ν
        - Pressure tensor: P^μν = Δ^μα Δ^νβ T_αβ

        Args:
            tensor: Rank-2 tensor to decompose

        Returns:
            Dictionary with decomposed components
        """
        if tensor.rank != 2:
            raise ValueError("Tensor decomposition only works for rank-2 tensors")

        return {
            "scalar_density": self.extract_scalar_density(tensor),
            "vector_density": self.extract_vector_density(tensor),
            "pressure_tensor": self.project_tensor_spatial(tensor),
        }

    def fluid_frame_components(
        self, tensor: TensorField
    ) -> dict[str, float | np.ndarray | sp.Matrix]:
        """
        Extract components in fluid rest frame.

        Args:
            tensor: Rank-2 tensor

        Returns:
            Dictionary with frame components
        """
        if tensor.rank != 2:
            raise ValueError("Frame extraction only works for rank-2 tensors")

        # Energy density: T^00 in fluid frame
        energy_density = self.extract_scalar_density(tensor)

        # Momentum density: T^0i in fluid frame (should be zero for perfect fluid)
        momentum_density = self.extract_vector_density(tensor)

        # Stress tensor: T^ij in fluid frame
        stress_tensor = self.project_tensor_spatial(tensor)

        return {
            "energy_density": energy_density,
            "momentum_density": momentum_density.spatial_components,
            "stress_tensor": stress_tensor.components[1:4, 1:4],  # 3x3 spatial part
        }
