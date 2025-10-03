"""
Pure 3D spatial grid for hydrodynamic evolution.

This module provides a clean separation between spatial discretization and
time evolution for relativistic hydrodynamics. Unlike SpacetimeGrid, this
class represents only the 3D spatial domain, with time treated as an
evolution parameter rather than a grid dimension.

Key Design:
-----------
- Grid points: (nx, ny, nz) - pure 3D spatial
- Coordinates: {x, y, z} or {r, θ, φ} or {ρ, φ, z}
- No time dimension in storage or operations
- Optimized for FFT-based spectral methods with periodic boundaries
- Supports curved spatial geometries via metric tensor

Usage:
------
    grid = SpaceGrid(
        coordinate_system='cartesian',
        spatial_ranges=[(0.0, 2*np.pi)] * 3,
        grid_points=(64, 64, 64),
        boundary_conditions='periodic'
    )

    # Pure 3D field operations
    field = np.zeros(grid.shape)  # (64, 64, 64)
    grad_x = grid.gradient(field, axis=0)
"""

import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import numpy as np
import sympy as sp

from .performance import monitor_performance

if TYPE_CHECKING:
    from .metrics import MetricBase


class SpaceGrid:
    """
    Pure 3D spatial grid for hydrodynamic evolution.

    Manages spatial coordinate systems, grid spacing, and boundary conditions
    for relativistic fluid dynamics. Time is treated as an evolution parameter,
    not a grid dimension.

    Attributes:
        coordinate_system: Coordinate system type ('cartesian', 'spherical', 'cylindrical')
        spatial_ranges: Domain extents [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        grid_points: Grid dimensions (nx, ny, nz)
        shape: Alias for grid_points
        ndim: Number of dimensions (always 3)
        nx, ny, nz: Individual grid dimensions
        dx, dy, dz: Grid spacing in each direction
        spatial_spacing: Tuple of (dx, dy, dz)
        coordinates: Dictionary of 1D coordinate arrays
        metric: Optional spatial metric tensor for curved space
        boundary_conditions: Boundary condition type affecting grid spacing
    """

    def __init__(
        self,
        coordinate_system: str,
        spatial_ranges: list[tuple[float, float]],
        grid_points: tuple[int, int, int],
        metric: Optional["MetricBase"] = None,
        boundary_conditions: Literal["periodic", "dirichlet", "neumann"] = "periodic",
    ):
        """
        Initialize pure 3D spatial grid.

        Args:
            coordinate_system: Coordinate system type
                - 'cartesian': (x, y, z)
                - 'spherical': (r, θ, φ)
                - 'cylindrical': (ρ, φ, z)
            spatial_ranges: Domain extents for each coordinate
            grid_points: Number of grid points (nx, ny, nz)
            metric: Optional metric tensor for curved spatial geometry
            boundary_conditions: Boundary condition type:
                - 'periodic': dx = L/N (excludes endpoint, FFT-compatible)
                - 'dirichlet': dx = L/(N-1) (includes endpoints)
                - 'neumann': dx = L/(N-1) (includes endpoints)

        Raises:
            ValueError: If parameters are invalid or inconsistent
        """
        self.coordinate_system = coordinate_system
        self.spatial_ranges = spatial_ranges
        self.grid_points = grid_points
        self.metric = metric
        self.boundary_conditions = boundary_conditions

        # Validate all inputs
        self._validate_grid_parameters()

        # Grid dimensions
        self.shape = grid_points
        self.ndim = 3
        self.nx, self.ny, self.nz = grid_points
        self.total_points = int(np.prod(grid_points))

        # Compute spatial spacing
        self.dx, self.dy, self.dz = self._compute_spacing()
        self.spatial_spacing = (self.dx, self.dy, self.dz)

        # Create coordinate arrays
        self.coordinates = self._create_coordinate_arrays()

    def _validate_grid_parameters(self) -> None:
        """Validate grid initialization parameters."""
        # Validate coordinate system
        valid_systems = ["cartesian", "spherical", "cylindrical"]
        if self.coordinate_system not in valid_systems:
            raise ValueError(
                f"Coordinate system must be one of {valid_systems}, got '{self.coordinate_system}'"
            )

        # Validate spatial ranges
        if len(self.spatial_ranges) != 3:
            raise ValueError(
                f"Must provide exactly 3 spatial coordinate ranges, got {len(self.spatial_ranges)}"
            )

        for i, (r_min, r_max) in enumerate(self.spatial_ranges):
            if r_max <= r_min:
                raise ValueError(
                    f"Spatial range {i} is invalid: max ({r_max}) must be > min ({r_min})"
                )

        # Validate grid points
        if len(self.grid_points) != 3:
            raise ValueError(
                f"Must provide exactly 3 grid point counts (nx, ny, nz), got {len(self.grid_points)}"
            )

        if any(n < 1 for n in self.grid_points):
            raise ValueError(
                f"All grid dimensions must have at least 1 point, got {self.grid_points}"
            )

        # Validate boundary conditions
        valid_bc = ["periodic", "dirichlet", "neumann"]
        if self.boundary_conditions not in valid_bc:
            raise ValueError(
                f"Boundary conditions must be one of {valid_bc}, got '{self.boundary_conditions}'"
            )

        # Coordinate system specific validation
        self._validate_coordinate_system_constraints()

    def _validate_coordinate_system_constraints(self) -> None:
        """Validate coordinate system specific constraints."""
        if self.coordinate_system == "spherical":
            # Validate r ≥ 0
            r_range = self.spatial_ranges[0]
            if r_range[0] < 0:
                raise ValueError(
                    f"Spherical coordinate r must be non-negative, got range {r_range}"
                )

            # Validate θ ∈ [0, π]
            theta_range = self.spatial_ranges[1]
            if theta_range[0] < 0 or theta_range[1] > np.pi:
                warnings.warn(
                    f"Spherical coordinate θ range {theta_range} extends outside [0, π]. "
                    "This may cause issues with volume elements and derivatives.",
                    UserWarning,
                    stacklevel=3,
                )

            # Validate φ range
            phi_range = self.spatial_ranges[2]
            if phi_range[1] - phi_range[0] > 2 * np.pi:
                warnings.warn(
                    f"Spherical coordinate φ range spans more than 2π: {phi_range}. "
                    "This may cause issues with periodic boundary conditions.",
                    UserWarning,
                    stacklevel=3,
                )

        elif self.coordinate_system == "cylindrical":
            # Validate ρ ≥ 0
            rho_range = self.spatial_ranges[0]
            if rho_range[0] < 0:
                raise ValueError(
                    f"Cylindrical coordinate ρ must be non-negative, got range {rho_range}"
                )

            # Validate φ range
            phi_range = self.spatial_ranges[1]
            if phi_range[1] - phi_range[0] > 2 * np.pi:
                warnings.warn(
                    f"Cylindrical coordinate φ range spans more than 2π: {phi_range}. "
                    "This may cause issues with periodic boundary conditions.",
                    UserWarning,
                    stacklevel=3,
                )

    def _compute_spacing(self) -> tuple[float, float, float]:
        """
        Compute grid spacing for each spatial direction.

        Returns:
            Tuple of (dx, dy, dz) spacing values

        Notes:
            - Periodic boundaries: dx = L/N (excludes endpoint for FFT compatibility)
            - Dirichlet/Neumann: dx = L/(N-1) (includes both endpoints)
        """
        spacing = []
        for (r_min, r_max), n in zip(self.spatial_ranges, self.grid_points, strict=True):
            if n == 1:
                # Degenerate axis
                spacing.append(0.0)
            elif self.boundary_conditions == "periodic":
                # Periodic: L/N spacing
                spacing.append((r_max - r_min) / n)
            else:
                # Dirichlet/Neumann: L/(N-1) spacing
                spacing.append((r_max - r_min) / (n - 1))

        return cast(tuple[float, float, float], tuple(spacing))

    def _create_coordinate_arrays(self) -> dict[str, np.ndarray]:
        """
        Create 1D coordinate arrays for each spatial direction.

        Returns:
            Dictionary mapping coordinate names to 1D arrays

        Notes:
            - Cartesian: {'x', 'y', 'z'}
            - Spherical: {'r', 'theta', 'phi'}
            - Cylindrical: {'rho', 'phi', 'z'}
        """
        coords: dict[str, np.ndarray] = {}

        # Get coordinate names for this system
        coord_names = self._get_coordinate_names()

        for i, name in enumerate(coord_names):
            r_min, r_max = self.spatial_ranges[i]
            n_points = self.grid_points[i]

            if self.boundary_conditions == "periodic":
                # Periodic: exclude endpoint for FFT compatibility
                # Grid points: [r_min, r_min + dx, ..., r_min + (N-1)*dx]
                extent = r_max - r_min
                dx = extent / n_points
                coords[name] = r_min + np.arange(n_points, dtype=np.float64) * dx
            else:
                # Dirichlet/Neumann: include both endpoints
                coords[name] = np.linspace(r_min, r_max, n_points, dtype=np.float64)

        return coords

    def _get_coordinate_names(self) -> list[str]:
        """Get coordinate names for the current system."""
        if self.coordinate_system == "cartesian":
            return ["x", "y", "z"]
        elif self.coordinate_system == "spherical":
            return ["r", "theta", "phi"]
        elif self.coordinate_system == "cylindrical":
            return ["rho", "phi", "z"]
        else:
            raise ValueError(f"Unknown coordinate system: {self.coordinate_system}")

    @property
    def coordinate_names(self) -> list[str]:
        """Get list of coordinate names for this grid."""
        return self._get_coordinate_names()

    def meshgrid(self, indexing: Literal["ij", "xy"] = "ij") -> tuple[np.ndarray, ...]:
        """
        Create 3D coordinate meshgrids.

        Args:
            indexing: Indexing convention
                - 'ij': Matrix indexing (default)
                - 'xy': Cartesian indexing

        Returns:
            Tuple of 3 3D arrays with shape (nx, ny, nz) for 'ij' or (ny, nx, nz) for 'xy'

        Example:
            >>> grid = SpaceGrid("cartesian", [(0, 1)] * 3, (8, 8, 8))
            >>> X, Y, Z = grid.meshgrid()
            >>> X.shape
            (8, 8, 8)
        """
        coord_arrays = [self.coordinates[name] for name in self.coordinate_names]
        return np.meshgrid(*coord_arrays, indexing=indexing)

    @monitor_performance("space_gradient")
    def gradient(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Compute spatial gradient along specified axis using finite differences.

        Args:
            field: Scalar field with shape (nx, ny, nz)
            axis: Spatial axis (0=first, 1=second, 2=third coordinate)

        Returns:
            Gradient array with same shape as input

        Raises:
            ValueError: If field shape doesn't match grid or axis is invalid

        Notes:
            - Uses second-order accurate centered differences for interior points
            - Edge accuracy depends on available points (first or second order)
            - Assumes uniform grid spacing (warns if non-uniform)
        """
        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid shape {self.shape}")

        if axis < 0 or axis >= 3:
            raise ValueError(f"Axis must be 0, 1, or 2 for 3D grid, got {axis}")

        # Get coordinate array and spacing
        coord_name = self.coordinate_names[axis]
        coord_array = self.coordinates[coord_name]
        n_points = len(coord_array)

        # Handle degenerate axis
        if n_points < 2:
            return np.zeros_like(field)

        # Check for uniform spacing
        spacing = coord_array[1] - coord_array[0]
        if n_points > 2:
            max_spacing_diff = np.max(np.diff(coord_array)) - np.min(np.diff(coord_array))
            if max_spacing_diff > 1e-10 * abs(spacing):
                warnings.warn(
                    f"Non-uniform grid spacing detected in {coord_name} direction. "
                    "Gradient calculation assumes uniform spacing and may be inaccurate.",
                    UserWarning,
                    stacklevel=2,
                )

        # Compute gradient using numpy's gradient function
        edge_order = 2 if n_points >= 3 else 1
        return cast(np.ndarray, np.gradient(field, spacing, axis=axis, edge_order=edge_order))

    @monitor_performance("space_divergence")
    def divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute divergence of 3D vector field.

        Args:
            vector_field: Vector field with shape (nx, ny, nz, 3)

        Returns:
            Divergence field with shape (nx, ny, nz)

        Raises:
            ValueError: If vector field shape is incorrect

        Notes:
            - Uses covariant divergence if metric is provided
            - Falls back to flat-space divergence for Cartesian coordinates
            - Warns if using flat-space approximation with non-Cartesian coordinates
        """
        expected_shape = (*self.shape, 3)
        if vector_field.shape != expected_shape:
            raise ValueError(
                f"Vector field shape {vector_field.shape} doesn't match expected {expected_shape}"
            )

        # Check for degenerate axes
        coord_lengths = [len(self.coordinates[name]) for name in self.coordinate_names]

        if self.metric is not None:
            # Covariant divergence: ∇_i V^i = ∂_i V^i + Γ^i_{ij} V^j
            divergence = np.zeros(self.shape)

            # Partial derivative contributions
            for i in range(3):
                if coord_lengths[i] < 2:
                    continue
                divergence += self.gradient(vector_field[..., i], axis=i)

            # Christoffel symbol contributions
            christoffel = self.metric.christoffel_symbols

            # Handle numerical Christoffel symbols
            if hasattr(christoffel, "shape") and isinstance(christoffel, np.ndarray):
                # Trace: Γ^i_{ij} V^j (sum over i, j)
                from .tensor_utils import optimized_einsum

                i_indices = np.arange(3)
                gamma_trace = christoffel[i_indices, i_indices, :]  # Shape: (3, 3)
                christoffel_correction = optimized_einsum("ij,...j->...", gamma_trace, vector_field)
                divergence += christoffel_correction

            # Handle symbolic Christoffel symbols
            elif hasattr(christoffel, "__getitem__"):
                for i in range(3):
                    for j in range(3):
                        gamma_trace = christoffel[i, i, j]
                        divergence += gamma_trace * vector_field[..., j]

            return divergence

        else:
            # Flat-space divergence
            if self.coordinate_system != "cartesian":
                warnings.warn(
                    f"Using flat-space divergence for {self.coordinate_system} coordinates "
                    "without metric tensor. This may produce incorrect results. "
                    "Consider providing a metric tensor.",
                    UserWarning,
                    stacklevel=2,
                )

            divergence = np.zeros(self.shape)
            for i in range(3):
                if coord_lengths[i] < 2:
                    continue
                divergence += self.gradient(vector_field[..., i], axis=i)

            return divergence

    @monitor_performance("space_laplacian")
    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of scalar field.

        Args:
            field: Scalar field with shape (nx, ny, nz)

        Returns:
            Laplacian field with shape (nx, ny, nz)

        Raises:
            ValueError: If field shape doesn't match grid

        Notes:
            - Uses covariant Laplacian if metric is provided: ∇² φ = ∇_i ∇^i φ
            - Falls back to flat-space Laplacian for Cartesian coordinates
            - Warns if using flat-space approximation with non-Cartesian coordinates
        """
        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid shape {self.shape}")

        coord_lengths = [len(self.coordinates[name]) for name in self.coordinate_names]

        if self.metric is not None:
            # Covariant Laplacian: ∇² φ = ∇_i ∇^i φ = ∇_i (g^ij ∇_j φ)

            # First compute gradient (covariant): ∇_i φ
            gradient_field = np.zeros((*self.shape, 3))
            for i in range(3):
                if coord_lengths[i] < 2:
                    continue
                gradient_field[..., i] = self.gradient(field, axis=i)

            # Raise index: ∇^i φ = g^ij ∇_j φ
            from .tensor_utils import optimized_einsum

            # Get spatial part of metric inverse (3×3)
            metric_inv_spatial = self.metric.inverse[1:, 1:]  # Assuming time is index 0
            contravariant_gradient = optimized_einsum(
                "ij,...j->...i", metric_inv_spatial, gradient_field
            )

            # Compute divergence of contravariant gradient
            return self.divergence(contravariant_gradient)

        else:
            # Flat-space Laplacian
            if self.coordinate_system != "cartesian":
                warnings.warn(
                    f"Using flat-space Laplacian for {self.coordinate_system} coordinates "
                    "without metric tensor. This may produce incorrect results. "
                    "Consider providing a metric tensor.",
                    UserWarning,
                    stacklevel=2,
                )

            laplacian = np.zeros_like(field)
            for axis in range(3):
                if coord_lengths[axis] < 2:
                    continue

                coord_name = self.coordinate_names[axis]
                coord_array = self.coordinates[coord_name]
                spacing = coord_array[1] - coord_array[0]

                # Second derivative using numpy gradient
                edge_order = 2 if len(coord_array) >= 3 else 1
                first_deriv = np.gradient(field, spacing, axis=axis, edge_order=edge_order)
                second_deriv = np.gradient(first_deriv, spacing, axis=axis, edge_order=edge_order)
                laplacian += second_deriv

            return laplacian

    def coordinate_at_index(self, indices: tuple[int, int, int]) -> np.ndarray:
        """
        Get coordinate values at specified grid indices.

        Args:
            indices: Grid indices (ix, iy, iz)

        Returns:
            Coordinate values [x, y, z] (or corresponding coordinate system)

        Raises:
            IndexError: If indices are out of bounds

        Example:
            >>> grid = SpaceGrid("cartesian", [(0, 1)] * 3, (8, 8, 8))
            >>> coords = grid.coordinate_at_index((4, 4, 4))
            >>> coords
            array([0.5, 0.5, 0.5])
        """
        coords = np.zeros(3)
        for i, (idx, name) in enumerate(zip(indices, self.coordinate_names, strict=True)):
            if idx < 0 or idx >= self.grid_points[i]:
                raise IndexError(
                    f"Index {idx} out of bounds for axis {i} with size {self.grid_points[i]}"
                )
            coords[i] = self.coordinates[name][idx]
        return coords

    def index_from_coordinate(self, coords: np.ndarray) -> tuple[int, int, int]:
        """
        Find nearest grid indices for given coordinate values.

        Args:
            coords: Coordinate values [x, y, z] (or corresponding coordinate system)

        Returns:
            Nearest grid indices (ix, iy, iz)

        Example:
            >>> grid = SpaceGrid("cartesian", [(0, 1)] * 3, (8, 8, 8))
            >>> indices = grid.index_from_coordinate(np.array([0.5, 0.5, 0.5]))
            >>> indices
            (4, 4, 4)
        """
        indices = []
        for _i, (coord_val, name) in enumerate(zip(coords, self.coordinate_names, strict=True)):
            coord_array = self.coordinates[name]
            idx = int(np.argmin(np.abs(coord_array - coord_val)))
            indices.append(idx)
        return cast(tuple[int, int, int], tuple(indices))

    def interpolate(self, field: np.ndarray, coords: np.ndarray, method: str = "linear") -> float:
        """
        Interpolate field value at arbitrary coordinates.

        Args:
            field: Field values on grid with shape (nx, ny, nz)
            coords: Target coordinates [x, y, z]
            method: Interpolation method ('linear', 'nearest', 'cubic')

        Returns:
            Interpolated field value

        Raises:
            ValueError: If field shape doesn't match grid
        """
        from scipy.interpolate import RegularGridInterpolator

        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid shape {self.shape}")

        # Create interpolator
        coord_arrays = [self.coordinates[name] for name in self.coordinate_names]
        interpolator = RegularGridInterpolator(
            coord_arrays, field, method=method, bounds_error=False, fill_value=None
        )

        # Interpolate at target coordinates
        return float(interpolator(coords))

    def __str__(self) -> str:
        """String representation of grid."""
        return (
            f"SpaceGrid({self.coordinate_system}, "
            f"{self.nx}×{self.ny}×{self.nz}, "
            f"{self.boundary_conditions})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"SpaceGrid(\n"
            f"  coordinate_system='{self.coordinate_system}',\n"
            f"  spatial_ranges={self.spatial_ranges},\n"
            f"  grid_points={self.grid_points},\n"
            f"  boundary_conditions='{self.boundary_conditions}'\n"
            f")"
        )
