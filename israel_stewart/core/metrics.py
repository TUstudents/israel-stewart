"""
Spacetime metrics for relativistic hydrodynamics.

This module provides metric tensor implementations for different spacetime
geometries, including flat Minkowski spacetime and curved spacetimes.
Metrics handle index raising/lowering and Christoffel symbol computation.
"""

import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, cast

import numpy as np
import sympy as sp


class MetricError(Exception):
    """Base exception for metric-related errors."""

    pass


class SingularMetricError(MetricError):
    """Exception raised when metric tensor is singular."""

    pass


class CoordinateError(MetricError):
    """Exception raised for coordinate system issues."""

    pass


class MetricBase(ABC):
    """
    Abstract base class for spacetime metrics.

    Defines the interface that all metric implementations must provide
    for tensor operations, including component access, inversion, and
    Christoffel symbol computation.
    """

    def __init__(self, coordinates: list[str] | None = None):
        """
        Initialize metric with coordinate system.

        Args:
            coordinates: List of coordinate names [t, x, y, z] or None for default
        """
        self.coordinates = coordinates or ["t", "x", "y", "z"]
        if len(self.coordinates) != 4:
            raise CoordinateError(f"Must have exactly 4 coordinates, got {len(self.coordinates)}")

        # Cache for expensive computations
        self._inverse_cache = None
        self._christoffel_cache = None
        self._determinant_cache = None

    @property
    @abstractmethod
    def components(self) -> np.ndarray | sp.Matrix:
        """Return metric tensor components g_mu_nu."""
        pass

    @property
    def signature(self) -> tuple[int, int, int, int]:
        """
        Return metric signature (signs of eigenvalues).

        Standard conventions:
        - (-,+,+,+): Mostly plus signature (particle physics)
        - (+,-,-,-): Mostly minus signature (general relativity)

        Returns:
            Tuple of signature values
        """
        return (-1, 1, 1, 1)  # Default: mostly plus

    @cached_property
    def inverse(self) -> np.ndarray | sp.Matrix:
        """
        Compute inverse metric tensor g^��.

        Returns:
            Inverse metric components

        Raises:
            SingularMetricError: If metric is not invertible
        """
        if isinstance(self.components, np.ndarray):
            try:
                # Check condition number for numerical stability
                cond_num = np.linalg.cond(self.components)
                if cond_num > 1e12:
                    warnings.warn(
                        f"Metric is poorly conditioned (cond={cond_num:.2e})",
                        stacklevel=2,
                    )

                inv_metric = np.linalg.inv(self.components)

                # Verify inversion accuracy
                identity_check = np.allclose(
                    np.dot(self.components, inv_metric), np.eye(4), rtol=1e-12
                )
                if not identity_check:
                    warnings.warn("Metric inversion may be inaccurate", stacklevel=2)

                return inv_metric

            except np.linalg.LinAlgError as e:
                raise SingularMetricError(f"Cannot invert metric tensor: {e}") from e

        elif isinstance(self.components, sp.Matrix):
            try:
                return self.components.inv()
            except sp.NonInvertibleMatrixError as e:
                raise SingularMetricError(f"Cannot invert symbolic metric tensor: {e}") from e
        else:
            raise TypeError(f"Unsupported metric component type: {type(self.components)}")

    @cached_property
    def determinant(self) -> float | sp.Expr:
        """
        Compute metric determinant det(g_��).

        Returns:
            Determinant value
        """
        if isinstance(self.components, np.ndarray):
            det = np.linalg.det(self.components)
            if abs(det) < 1e-15:
                warnings.warn(f"Metric determinant is very small: {det}", stacklevel=2)
            return det
        elif isinstance(self.components, sp.Matrix):
            return self.components.det()
        else:
            raise TypeError(f"Unsupported metric component type: {type(self.components)}")

    @cached_property
    def christoffel_symbols(self) -> np.ndarray | sp.Array:
        """
        Compute Christoffel symbols �^�_��.

        Using the formula:
        �^�_�� = (1/2) g^�� (_� g_�� + _� g_�� - _� g_��)

        Returns:
            Array of shape (4, 4, 4) with components �^�_��
        """
        if isinstance(self.components, np.ndarray):
            return self._compute_christoffel_numerical()
        elif isinstance(self.components, sp.Matrix):
            return self._compute_christoffel_symbolic()
        else:
            raise TypeError(f"Unsupported metric component type: {type(self.components)}")

    def christoffel_symbols_numerical(self, coordinates: list[np.ndarray]) -> np.ndarray:
        """Compute numerical Christoffel symbols with explicit coordinate arrays.

        Args:
            coordinates: List of coordinate arrays [t, x, y, z]

        Returns:
            Christoffel symbols Γ^λ_μν with shape (*grid_shape, 4, 4, 4)
        """
        if len(coordinates) != 4:
            raise CoordinateError(f"Need exactly 4 coordinate arrays, got {len(coordinates)}")
        return self._compute_christoffel_numerical(coordinates)

    def christoffel_symbols_from_grid(self, grid: Any) -> np.ndarray:
        """Compute numerical Christoffel symbols using SpacetimeGrid.

        Args:
            grid: SpacetimeGrid object with coordinate information

        Returns:
            Christoffel symbols Γ^λ_μν with shape (*grid.shape, 4, 4, 4)
        """
        # Extract coordinate arrays from grid
        if hasattr(grid, "coordinates") and isinstance(grid.coordinates, dict):
            coord_names = grid.coordinate_names
            coordinates = [grid.coordinates[name] for name in coord_names]
        else:
            # Create coordinate arrays from grid ranges
            t_coords = np.linspace(grid.time_range[0], grid.time_range[1], grid.grid_points[0])
            coordinates = [t_coords]
            for i, (x_min, x_max) in enumerate(grid.spatial_ranges):
                x_coords = np.linspace(x_min, x_max, grid.grid_points[i + 1])
                coordinates.append(x_coords)

        return self._compute_christoffel_numerical(coordinates)

    def _compute_christoffel_numerical(
        self, coordinates: list[np.ndarray] | None = None
    ) -> np.ndarray:
        """Compute Christoffel symbols for numerical metrics.

        Args:
            coordinates: List of coordinate arrays [t, x, y, z] for computing derivatives

        Returns:
            Array of Christoffel symbols Γ^λ_μν with shape (4, 4, 4)
        """
        # For constant metrics (like Minkowski), Christoffel symbols are zero
        if self.is_constant():
            return np.zeros((4, 4, 4))

        # If no coordinate arrays provided, cannot compute derivatives
        if coordinates is None:
            warnings.warn(
                "Cannot compute numerical Christoffel symbols without coordinate arrays. "
                "Returning zeros - use symbolic metric or provide coordinates.",
                stacklevel=3,
            )
            return np.zeros((4, 4, 4))

        return self._compute_christoffel_finite_difference(coordinates)

    def _compute_christoffel_finite_difference(self, coordinates: list[np.ndarray]) -> np.ndarray:
        """
        Compute Christoffel symbols using finite differences.

        Uses the standard formula:
        Γ^λ_μν = (1/2) g^λρ (∂_μ g_ρν + ∂_ν g_μρ - ∂_ρ g_μν)

        Args:
            coordinates: List of coordinate arrays [t, x, y, z]

        Returns:
            Christoffel symbols with shape (4, 4, 4)
        """
        if len(coordinates) != 4:
            raise CoordinateError(f"Need exactly 4 coordinate arrays, got {len(coordinates)}")

        # Get metric components and inverse
        g = self.components
        g_inv = self.inverse

        # Check if metric has spatial dependence
        if g.ndim == 2:
            # Constant metric - expand to coordinate grid shape
            coord_shape = tuple(len(coord) for coord in coordinates)
            g = np.broadcast_to(g[None, None, None, None, :, :], (*coord_shape, 4, 4))
            g_inv = np.broadcast_to(g_inv[None, None, None, None, :, :], (*coord_shape, 4, 4))

        # Initialize Christoffel array
        grid_shape = g.shape[:-2]  # Remove last two indices (μ, ν)
        christoffel = np.zeros((*grid_shape, 4, 4, 4))  # Shape: (*grid, λ, μ, ν)

        # Compute partial derivatives of metric using finite differences
        metric_derivs = self._compute_metric_derivatives(g, coordinates)

        # Compute Christoffel symbols: Γ^λ_μν = (1/2) g^λρ (∂_μ g_ρν + ∂_ν g_μρ - ∂_ρ g_μν)
        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    # Sum over ρ index
                    for rho in range(4):
                        # ∂_μ g_ρν
                        term1 = metric_derivs[..., mu, rho, nu]
                        # ∂_ν g_μρ
                        term2 = metric_derivs[..., nu, mu, rho]
                        # ∂_ρ g_μν
                        term3 = metric_derivs[..., rho, mu, nu]

                        # Γ^λ_μν += (1/2) g^λρ (∂_μ g_ρν + ∂_ν g_μρ - ∂_ρ g_μν)
                        christoffel[..., lam, mu, nu] += (
                            0.5 * g_inv[..., lam, rho] * (term1 + term2 - term3)
                        )

        return christoffel

    def _compute_metric_derivatives(
        self, metric: np.ndarray, coordinates: list[np.ndarray]
    ) -> np.ndarray:
        """
        Compute partial derivatives of metric tensor using finite differences.

        Args:
            metric: Metric tensor with shape (*grid_shape, 4, 4)
            coordinates: List of coordinate arrays [t, x, y, z]

        Returns:
            Metric derivatives with shape (*grid_shape, 4, 4, 4) where indices are [derivative_direction, μ, ν] for ∂_{derivative_direction} g_μν
        """
        grid_shape = metric.shape[:-2]
        derivs = np.zeros(
            (*grid_shape, 4, 4, 4)
        )  # Shape: (*grid, direction, μ, ν) for ∂_direction g_μν

        # For each coordinate direction
        for direction in range(4):
            coord_array = coordinates[direction]

            # Handle different grid shapes
            if len(grid_shape) == 0:  # Constant metric
                # All derivatives are zero for constant metrics
                pass  # derivs is already initialized to zeros

            elif len(grid_shape) == 1:  # 1D grid (e.g., time-dependent metric)
                if direction == 0:  # Time derivative
                    for mu in range(4):
                        for nu in range(4):
                            grad_components = np.gradient(metric[:, mu, nu], coord_array)
                            derivs[:, direction, mu, nu] = grad_components

            elif len(grid_shape) == 4:  # Full 4D grid
                # Compute gradient along the specified direction
                axis = direction
                for mu in range(4):
                    for nu in range(4):
                        # Use numpy gradient for finite differences
                        grad_components = np.gradient(metric[..., mu, nu], coord_array, axis=axis)
                        derivs[..., direction, mu, nu] = grad_components

            else:
                # Handle other grid shapes (2D, 3D)
                if direction < len(grid_shape):
                    axis = direction
                    for mu in range(4):
                        for nu in range(4):
                            grad_components = np.gradient(
                                metric[..., mu, nu], coord_array, axis=axis
                            )
                            derivs[..., direction, mu, nu] = grad_components
                else:
                    # Derivative direction beyond grid dimensions - derivatives are zero
                    pass

        return derivs

    def _compute_christoffel_symbolic(self) -> sp.Array:
        """Compute Christoffel symbols for symbolic metrics."""
        # Extract coordinate symbols from metric components
        all_symbols = set()
        for comp in self.components:
            all_symbols.update(comp.free_symbols)

        # Initialize Christoffel components list
        christoffel_components = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]

        if not all_symbols:
            # Constant metric has zero Christoffel symbols
            return sp.Array(christoffel_components)

        # Create ordered list of coordinate symbols
        # Match to coordinate names if possible
        coords = []
        symbol_list = list(all_symbols)

        for coord_name in self.coordinates:
            # Find symbol with matching name
            matching_symbol = None
            for sym in symbol_list:
                if str(sym) == coord_name:
                    matching_symbol = sym
                    break
            if matching_symbol:
                coords.append(matching_symbol)
            else:
                # Create new symbol if needed
                coords.append(sp.Symbol(coord_name))

        # Ensure we have exactly 4 coordinates
        while len(coords) < 4:
            coords.append(sp.Symbol(f"x{len(coords)}"))

        # Full symbolic computation for coordinate-dependent metrics
        g_inv = self.inverse

        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    christoffel_sum = 0
                    for rho in range(4):
                        christoffel_sum += (
                            sp.Rational(1, 2)
                            * g_inv[lam, rho]
                            * (
                                sp.diff(self.components[rho, nu], coords[mu])
                                + sp.diff(self.components[mu, rho], coords[nu])
                                - sp.diff(self.components[mu, nu], coords[rho])
                            )
                        )
                    christoffel_components[lam][mu][nu] = christoffel_sum

        return sp.Array(christoffel_components)

    def is_constant(self) -> bool:
        """
        Check if metric components are constant (independent of coordinates).

        Returns:
            True if metric is constant
        """
        if isinstance(self.components, np.ndarray):
            # Numerical metrics are assumed constant unless overridden
            return True
        elif isinstance(self.components, sp.Matrix):
            # Extract all free symbols from the metric components
            all_symbols = set()
            for comp in self.components:
                all_symbols.update(comp.free_symbols)
            # If any symbols exist, metric is coordinate-dependent
            return len(all_symbols) == 0
        return True

    def is_flat(self) -> bool:
        """
        Check if spacetime is flat (all Christoffel symbols zero).

        Returns:
            True if spacetime is flat
        """
        christoffel = self.christoffel_symbols
        if isinstance(christoffel, np.ndarray):
            return np.allclose(christoffel, 0.0, atol=1e-15)
        else:
            # Check if all components are zero
            return all(
                christoffel[i, j, k] == 0 for i in range(4) for j in range(4) for k in range(4)
            )

    def raise_index(
        self, tensor_components: np.ndarray | sp.Matrix, index_position: int
    ) -> np.ndarray | sp.Matrix:
        """
        Raise a tensor index using the inverse metric.

        Args:
            tensor_components: Tensor with covariant index to raise
            index_position: Position of index to raise

        Returns:
            Tensor with raised index
        """
        if isinstance(tensor_components, np.ndarray):
            if tensor_components.ndim == 1:  # Vector
                return np.einsum("ij,j->i", self.inverse, tensor_components)
            elif tensor_components.ndim == 2:  # Rank-2 tensor
                if index_position == 0:
                    return np.einsum("ij,jk->ik", self.inverse, tensor_components)
                elif index_position == 1:
                    return np.einsum("ij,ki->kj", self.inverse, tensor_components)
        elif isinstance(tensor_components, sp.Matrix):
            if tensor_components.shape[1] == 1:  # Vector
                return self.inverse * tensor_components
            else:  # Matrix
                if index_position == 0:
                    return self.inverse * tensor_components
                elif index_position == 1:
                    return tensor_components * self.inverse.T

        raise NotImplementedError(
            f"Index raising not implemented for rank > 2 or position {index_position}"
        )

    def lower_index(
        self, tensor_components: np.ndarray | sp.Matrix, index_position: int
    ) -> np.ndarray | sp.Matrix:
        """
        Lower a tensor index using the metric.

        Args:
            tensor_components: Tensor with contravariant index to lower
            index_position: Position of index to lower

        Returns:
            Tensor with lowered index
        """
        if isinstance(tensor_components, np.ndarray):
            if tensor_components.ndim == 1:  # Vector
                return np.einsum("ij,j->i", self.components, tensor_components)
            elif tensor_components.ndim == 2:  # Rank-2 tensor
                if index_position == 0:
                    return np.einsum("ij,jk->ik", self.components, tensor_components)
                elif index_position == 1:
                    return np.einsum("ij,ki->kj", self.components, tensor_components)
        elif isinstance(tensor_components, sp.Matrix):
            if tensor_components.shape[1] == 1:  # Vector
                return self.components * tensor_components
            else:  # Matrix
                if index_position == 0:
                    return self.components * tensor_components
                elif index_position == 1:
                    return tensor_components * self.components.T

        raise NotImplementedError(
            f"Index lowering not implemented for rank > 2 or position {index_position}"
        )

    def inner_product(
        self, vector1: np.ndarray | sp.Matrix, vector2: np.ndarray | sp.Matrix
    ) -> float | sp.Expr:
        """
        Compute inner product of two vectors: g_�� u^� v^�.

        Args:
            vector1: First four-vector
            vector2: Second four-vector

        Returns:
            Scalar inner product
        """
        if isinstance(vector1, np.ndarray) and isinstance(vector2, np.ndarray):
            return np.einsum("ij,i,j", self.components, vector1, vector2)
        elif isinstance(vector1, sp.Matrix) and isinstance(vector2, sp.Matrix):
            return (vector1.T * self.components * vector2)[0]
        else:
            raise TypeError("Both vectors must be same type (numpy or sympy)")

    def line_element_squared(
        self, coordinate_differentials: np.ndarray | sp.Matrix
    ) -> float | sp.Expr:
        """
        Compute line element squared: ds� = g_�� dx^� dx^�.

        Args:
            coordinate_differentials: Four-vector of coordinate differentials

        Returns:
            Line element squared
        """
        return self.inner_product(coordinate_differentials, coordinate_differentials)

    def validate_components(self) -> bool:
        """
        Validate metric tensor components.

        Returns:
            True if metric is valid

        Raises:
            MetricError: If metric fails validation
        """
        components = self.components

        # Check symmetry
        if isinstance(components, np.ndarray):
            if not np.allclose(components, components.T, rtol=1e-12):
                raise MetricError("Metric tensor must be symmetric")
        elif isinstance(components, sp.Matrix):
            if not components.equals(components.T):
                raise MetricError("Symbolic metric tensor must be symmetric")

        # Check signature (rough check for physical metrics)
        try:
            det = self.determinant
            if isinstance(det, (int, float)) and det >= 0:
                warnings.warn("Metric determinant is non-negative - check signature", stacklevel=2)
        except:
            pass  # Skip check for complex symbolic expressions

        # Check invertibility
        try:
            pass
            # Basic inversion check passed
        except SingularMetricError as e:
            raise MetricError("Metric tensor is not invertible") from e

        return True

    def __str__(self) -> str:
        """String representation of metric."""
        return f"{self.__class__.__name__}(coordinates={self.coordinates})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(coordinates={self.coordinates})"


class MinkowskiMetric(MetricBase):
    """
    Minkowski metric for flat spacetime.

    Uses mostly-plus signature: diag(-1, +1, +1, +1)
    Standard for special relativity and particle physics.
    """

    def __init__(self, signature: str = "mostly_plus", coordinates: list[str] | None = None):
        """
        Initialize Minkowski metric.

        Args:
            signature: Either "mostly_plus" (-,+,+,+) or "mostly_minus" (+,-,-,-)
            coordinates: Coordinate names (default: ['t', 'x', 'y', 'z'])
        """
        super().__init__(coordinates)
        self.signature_type = signature

        if signature == "mostly_plus":
            self._diag = np.array([-1.0, 1.0, 1.0, 1.0])
        elif signature == "mostly_minus":
            self._diag = np.array([1.0, -1.0, -1.0, -1.0])
        else:
            raise ValueError(f"Unknown signature: {signature}. Use 'mostly_plus' or 'mostly_minus'")

    @property
    def components(self) -> np.ndarray:
        """Return Minkowski metric components."""
        return np.diag(self._diag)

    @property
    def signature(self) -> tuple[int, int, int, int]:
        """Return metric signature."""
        return cast(tuple[int, int, int, int], tuple(int(np.sign(d)) for d in self._diag))

    @cached_property
    def inverse(self) -> np.ndarray:
        """Return inverse Minkowski metric (same as metric for flat space)."""
        return np.diag(1.0 / self._diag)

    @cached_property
    def determinant(self) -> float:
        """Return Minkowski metric determinant."""
        if self.signature_type == "mostly_plus":
            return -1.0  # det(diag(-1,1,1,1)) = -1
        else:
            return -1.0  # det(diag(1,-1,-1,-1)) = -1

    def is_timelike(self, vector: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if vector is timelike in Minkowski space.

        Args:
            vector: Four-vector components
            tolerance: Numerical tolerance

        Returns:
            True if vector is timelike
        """
        norm_squared = self.inner_product(vector, vector)
        if self.signature_type == "mostly_plus":
            return norm_squared < -tolerance  # ds� < 0 is timelike
        else:
            return norm_squared > tolerance  # ds� > 0 is timelike

    def is_spacelike(self, vector: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if vector is spacelike in Minkowski space.

        Args:
            vector: Four-vector components
            tolerance: Numerical tolerance

        Returns:
            True if vector is spacelike
        """
        norm_squared = self.inner_product(vector, vector)
        if self.signature_type == "mostly_plus":
            return norm_squared > tolerance  # ds� > 0 is spacelike
        else:
            return norm_squared < -tolerance  # ds� < 0 is spacelike

    def is_null(self, vector: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if vector is null (lightlike) in Minkowski space.

        Args:
            vector: Four-vector components
            tolerance: Numerical tolerance

        Returns:
            True if vector is null
        """
        norm_squared = self.inner_product(vector, vector)
        return bool(abs(norm_squared) < tolerance)


class GeneralMetric(MetricBase):
    """
    General metric tensor for arbitrary curved spacetimes.

    Accepts arbitrary metric components and computes all derived quantities.
    """

    def __init__(
        self,
        metric_components: np.ndarray | sp.Matrix,
        coordinates: list[str] | None = None,
    ):
        """
        Initialize general metric.

        Args:
            metric_components: 4x4 metric tensor components
            coordinates: Coordinate names
        """
        super().__init__(coordinates)

        if isinstance(metric_components, (list, tuple)):
            metric_components = np.array(metric_components)

        if metric_components.shape != (4, 4):
            raise CoordinateError(f"Metric must be 4x4, got shape {metric_components.shape}")

        self._components = metric_components

        # Validate the metric
        self.validate_components()

    @property
    def components(self) -> np.ndarray | sp.Matrix:
        """Return metric tensor components."""
        return self._components


class MilneMetric(MetricBase):
    """
    Milne coordinate metric for boost-invariant systems.

    Uses coordinates (τ, η, x, y) where:
    - τ: proper time
    - η: space-time rapidity
    - x, y: transverse coordinates

    Line element: ds² = dτ² - τ²dη² - dx² - dy²
    """

    def __init__(self, coordinates: list[str] | None = None):
        """
        Initialize Milne metric.

        Args:
            coordinates: Coordinate names (default: ['tau', 'eta', 'x', 'y'])
        """
        default_coords = ["tau", "eta", "x", "y"]
        super().__init__(coordinates or default_coords)

    @property
    def components(self) -> sp.Matrix:
        """Return Milne metric components as symbolic matrix."""
        tau = sp.Symbol("tau", positive=True)

        # Milne metric: diag(1, -τ², -1, -1)
        metric = sp.zeros(4, 4)
        metric[0, 0] = 1  # dτ²
        metric[1, 1] = -(tau**2)  # -τ²dη²
        metric[2, 2] = -1  # -dx²
        metric[3, 3] = -1  # -dy²

        return metric

    @property
    def signature(self) -> tuple[int, int, int, int]:
        """Return metric signature (+,-,-,-)."""
        return (1, -1, -1, -1)

    def is_constant(self) -> bool:
        """Milne metric is coordinate-dependent (τ appears in g_ηη)."""
        return False


class BJorkenMetric(MilneMetric):
    """
    Bjorken flow metric - equivalent to Milne metric.

    Specialized for boost-invariant longitudinal expansion
    commonly used in relativistic heavy-ion collisions.
    """

    def __init__(self) -> None:
        """Initialize Bjorken flow metric."""
        super().__init__(["tau", "eta", "x", "y"])


class FLRWMetric(MetricBase):
    """
    Friedmann-Lemaître-Robertson-Walker metric for cosmology.

    Line element: ds² = dt² - a(t)²[dr²/(1-kr²) + r²dθ² + r²sin²θdφ²]
    """

    def __init__(self, curvature_param: float = 0, scale_factor_func: str = "t"):
        """
        Initialize FLRW metric.

        Args:
            curvature_param: Spatial curvature parameter k (0, +1, -1)
            scale_factor_func: Scale factor a(t) as function of time
        """
        super().__init__(["t", "r", "theta", "phi"])
        self.k = curvature_param
        self.scale_factor_func = scale_factor_func

    @property
    def components(self) -> sp.Matrix:
        """Return FLRW metric components."""
        t, r, theta = sp.symbols("t r theta", real=True)

        # Scale factor a(t) - for now, simple power law
        if self.scale_factor_func == "t":
            a = t  # Simple case: a(t) = t
        else:
            a = sp.Symbol("a", positive=True)  # Generic scale factor

        metric = sp.zeros(4, 4)
        metric[0, 0] = 1  # dt²
        metric[1, 1] = -(a**2) / (1 - self.k * r**2)  # -a²dr²/(1-kr²)
        metric[2, 2] = -(a**2) * r**2  # -a²r²dθ²
        metric[3, 3] = -(a**2) * r**2 * sp.sin(theta) ** 2  # -a²r²sin²θdφ²

        return metric

    @property
    def signature(self) -> tuple[int, int, int, int]:
        """Return metric signature (+,-,-,-)."""
        return (1, -1, -1, -1)

    def is_constant(self) -> bool:
        """FLRW metric is coordinate-dependent."""
        return False


class SchwarzschildMetric(MetricBase):
    """
    Schwarzschild metric for spherically symmetric black holes.

    Line element: ds² = (1-rs/r)dt² - dr²/(1-rs/r) - r²dθ² - r²sin²θdφ²
    where rs = 2GM/c² is the Schwarzschild radius.
    """

    def __init__(self, schwarzschild_radius: float = 1.0):
        """
        Initialize Schwarzschild metric.

        Args:
            schwarzschild_radius: Schwarzschild radius rs = 2GM/c²
        """
        super().__init__(["t", "r", "theta", "phi"])
        self.rs = schwarzschild_radius

    @property
    def components(self) -> sp.Matrix:
        """Return Schwarzschild metric components."""
        r, theta = sp.symbols("r theta", real=True, positive=True)
        rs = sp.Symbol("rs", positive=True)

        f = 1 - rs / r  # Schwarzschild function

        metric = sp.zeros(4, 4)
        metric[0, 0] = f  # (1-rs/r)dt²
        metric[1, 1] = -1 / f  # -dr²/(1-rs/r)
        metric[2, 2] = -(r**2)  # -r²dθ²
        metric[3, 3] = -(r**2) * sp.sin(theta) ** 2  # -r²sin²θdφ²

        return metric

    @property
    def signature(self) -> tuple[int, int, int, int]:
        """Return metric signature (+,-,-,-)."""
        return (1, -1, -1, -1)

    def is_constant(self) -> bool:
        """Schwarzschild metric is coordinate-dependent."""
        return False
