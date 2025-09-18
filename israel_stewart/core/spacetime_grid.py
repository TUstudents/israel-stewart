"""
Spacetime grid management for Israel-Stewart hydrodynamics simulations.

This module provides coordinate grids, boundary conditions, and spatial
discretization utilities for relativistic hydrodynamics calculations.
"""

# Forward reference for metrics
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from .performance import monitor_performance

# Import tensor framework components
if TYPE_CHECKING:
    from .derivatives import CovariantDerivative
    from .four_vectors import FourVector
    from .metrics import MetricBase


class SpacetimeGrid:
    """
    Spacetime coordinate grid for hydrodynamics simulations.

    Manages coordinate systems, grid spacing, and boundary conditions
    for relativistic fluid dynamics calculations.
    """

    def __init__(
        self,
        coordinate_system: str,
        time_range: tuple[float, float],
        spatial_ranges: list[tuple[float, float]],
        grid_points: tuple[int, int, int, int],
        metric: Optional["MetricBase"] = None,
    ):
        """
        Initialize spacetime grid.

        Args:
            coordinate_system: 'cartesian', 'spherical', 'cylindrical', 'milne'
            time_range: (t_min, t_max) for time coordinate
            spatial_ranges: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            grid_points: (Nt, Nx, Ny, Nz) number of grid points
            metric: Spacetime metric tensor
        """
        self.coordinate_system = coordinate_system
        self.time_range = time_range
        self.spatial_ranges = spatial_ranges
        self.grid_points = grid_points
        self.metric = metric

        # Validate inputs
        self._validate_grid_parameters()

        # Create coordinate arrays
        self.coordinates = self._create_coordinate_arrays()

        # Grid properties
        self.shape = grid_points
        self.ndim = 4
        self.total_points = np.prod(grid_points)

        # Compute grid spacing
        self.dt = (time_range[1] - time_range[0]) / (grid_points[0] - 1)
        self.spatial_spacing = [
            (r[1] - r[0]) / (n - 1) for r, n in zip(spatial_ranges, grid_points[1:], strict=False)
        ]

    def _validate_grid_parameters(self) -> None:
        """Validate grid initialization parameters."""
        valid_systems = ["cartesian", "spherical", "cylindrical", "milne"]
        if self.coordinate_system not in valid_systems:
            raise ValueError(f"Coordinate system must be one of {valid_systems}")

        if len(self.spatial_ranges) != 3:
            raise ValueError("Must provide exactly 3 spatial coordinate ranges")

        if len(self.grid_points) != 4:
            raise ValueError("Must provide exactly 4 grid point counts (Nt, Nx, Ny, Nz)")

        if any(n < 2 for n in self.grid_points):
            raise ValueError("All grid dimensions must have at least 2 points")

        # Validate coordinate system specific constraints
        self._validate_coordinate_system_constraints()

    def _validate_coordinate_system_constraints(self) -> None:
        """Validate coordinate system specific constraints."""
        if self.coordinate_system == "spherical":
            # Check that theta range is [0, π]
            theta_range = self.spatial_ranges[1]  # theta is second spatial coordinate
            if theta_range[0] < 0 or theta_range[1] > np.pi:
                warnings.warn(
                    f"Spherical coordinates theta range {theta_range} extends outside [0, π]. "
                    "This may cause issues with volume elements and derivatives.",
                    stacklevel=3,
                )

            # Check that phi range is reasonable
            phi_range = self.spatial_ranges[2]  # phi is third spatial coordinate
            if phi_range[1] - phi_range[0] > 2 * np.pi:
                warnings.warn(
                    f"Spherical coordinates phi range spans more than 2π: {phi_range}. "
                    "This may cause issues with periodic boundary conditions.",
                    stacklevel=3,
                )

        elif self.coordinate_system == "cylindrical":
            # Check that rho is non-negative
            rho_range = self.spatial_ranges[0]  # rho is first spatial coordinate
            if rho_range[0] < 0:
                warnings.warn(
                    f"Cylindrical coordinates rho range {rho_range} includes negative values. "
                    "Negative radial coordinates may cause issues.",
                    stacklevel=3,
                )

        elif self.coordinate_system == "milne":
            # Check that tau is positive
            tau_range = self.time_range
            if tau_range[0] <= 0:
                warnings.warn(
                    f"Milne coordinates tau range {tau_range} includes non-positive values. "
                    "Proper time must be positive in Milne coordinates.",
                    stacklevel=3,
                )

    def _create_coordinate_arrays(self) -> dict[str, np.ndarray]:
        """Create coordinate arrays for the grid."""
        coords = {}

        # Time coordinate - use appropriate name for coordinate system
        if self.coordinate_system == "milne":
            time_name = "tau"
        else:
            time_name = "t"
        coords[time_name] = np.linspace(self.time_range[0], self.time_range[1], self.grid_points[0])

        # Spatial coordinates depend on coordinate system
        if self.coordinate_system == "cartesian":
            coord_names = ["x", "y", "z"]
        elif self.coordinate_system == "spherical":
            coord_names = ["r", "theta", "phi"]
        elif self.coordinate_system == "cylindrical":
            coord_names = ["rho", "phi", "z"]
        elif self.coordinate_system == "milne":
            coord_names = ["eta", "x", "y"]  # tau is time-like

        for i, name in enumerate(coord_names):
            range_i = self.spatial_ranges[i]
            n_points = self.grid_points[i + 1]
            coords[name] = np.linspace(range_i[0], range_i[1], n_points)

        return coords

    def _create_spectral_coordinates(self) -> None:
        """
        Create coordinate arrays optimized for spectral methods.

        Replaces the default coordinate arrays with ones that use proper
        spectral spacing: dx = L/N instead of L/(N-1). This ensures
        periodicity and FFT compatibility.
        """
        coords = {}

        # Time coordinate - keep standard spacing for time evolution
        if self.coordinate_system == "milne":
            time_name = "tau"
        else:
            time_name = "t"
        coords[time_name] = np.linspace(self.time_range[0], self.time_range[1], self.grid_points[0])

        # Spatial coordinates with spectral spacing
        if self.coordinate_system == "cartesian":
            coord_names = ["x", "y", "z"]
        elif self.coordinate_system == "spherical":
            coord_names = ["r", "theta", "phi"]
        elif self.coordinate_system == "cylindrical":
            coord_names = ["rho", "phi", "z"]
        elif self.coordinate_system == "milne":
            coord_names = ["eta", "x", "y"]

        for i, name in enumerate(coord_names):
            range_i = self.spatial_ranges[i]
            n_points = self.grid_points[i + 1]
            # Spectral spacing: dx = L/N, points at [0, dx, 2*dx, ..., (N-1)*dx]
            extent = range_i[1] - range_i[0]
            dx = extent / n_points
            coords[name] = range_i[0] + np.arange(n_points, dtype=np.float64) * dx

        # Update coordinates and recompute spacing
        self.coordinates = coords
        self.spatial_spacing = [
            (r[1] - r[0]) / n
            for r, n in zip(self.spatial_ranges, self.grid_points[1:], strict=False)
        ]

    @property
    def coordinate_names(self) -> list[str]:
        """Get list of coordinate names."""
        if self.coordinate_system == "cartesian":
            return ["t", "x", "y", "z"]
        elif self.coordinate_system == "spherical":
            return ["t", "r", "theta", "phi"]
        elif self.coordinate_system == "cylindrical":
            return ["t", "rho", "phi", "z"]
        elif self.coordinate_system == "milne":
            return ["tau", "eta", "x", "y"]
        else:
            raise ValueError(f"Unknown coordinate system: {self.coordinate_system}")

    def meshgrid(self, indexing: str = "ij") -> tuple[np.ndarray, ...]:
        """
        Create coordinate meshgrids for the full 4D spacetime.

        Args:
            indexing: 'ij' for matrix indexing or 'xy' for Cartesian indexing

        Returns:
            Tuple of 4D coordinate arrays
        """
        coord_arrays = [self.coordinates[name] for name in self.coordinate_names]
        return np.meshgrid(*coord_arrays, indexing=indexing)  # type: ignore[arg-type]

    def coordinate_at_index(self, indices: tuple[int, int, int, int]) -> np.ndarray:
        """
        Get coordinate values at given grid indices.

        Args:
            indices: (it, ix, iy, iz) grid indices

        Returns:
            Coordinate values [t, x, y, z]
        """
        coords = np.zeros(4)
        coord_names = self.coordinate_names

        for i, (idx, name) in enumerate(zip(indices, coord_names, strict=False)):
            coords[i] = self.coordinates[name][idx]

        return coords

    def index_from_coordinate(self, coords: np.ndarray) -> tuple[int, int, int, int]:
        """
        Find nearest grid indices for given coordinates.

        Args:
            coords: Coordinate values [t, x, y, z]

        Returns:
            Nearest grid indices (it, ix, iy, iz)
        """
        indices = []
        coord_names = self.coordinate_names

        for _i, (coord_val, name) in enumerate(zip(coords, coord_names, strict=False)):
            coord_array = self.coordinates[name]
            idx = np.argmin(np.abs(coord_array - coord_val))
            indices.append(idx)

        return cast(tuple[int, int, int, int], tuple(indices))

    @monitor_performance("grid_gradient")
    def gradient(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Compute gradient along specified axis using finite differences.

        Assumes uniform grid spacing. For non-uniform grids, this method
        may produce inaccurate results.

        Args:
            field: Field values on grid
            axis: Axis along which to compute gradient (0=t, 1=x, 2=y, 3=z)

        Returns:
            Gradient array
        """
        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid shape {self.shape}")

        if axis < 0 or axis >= 4:
            raise ValueError(f"Axis must be in range [0, 3], got {axis}")

        coord_name = self.coordinate_names[axis]
        coord_array = self.coordinates[coord_name]

        # Check for uniform spacing and warn if not
        spacing = coord_array[1] - coord_array[0]
        if len(coord_array) > 2:
            max_spacing_diff = np.max(np.diff(coord_array)) - np.min(np.diff(coord_array))
            if max_spacing_diff > 1e-10 * spacing:
                warnings.warn(
                    f"Non-uniform grid spacing detected in {coord_name} direction. "
                    "Gradient calculation assumes uniform spacing and may be inaccurate.",
                    stacklevel=2,
                )

        return np.gradient(field, spacing, axis=axis)  # type: ignore[no-any-return]

    def divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Compute divergence of vector field.

        Uses covariant divergence if metric is available, otherwise falls back
        to flat-space divergence with a warning for non-Cartesian coordinates.

        Args:
            vector_field: Vector field with shape (*grid.shape, 4)

        Returns:
            Divergence field
        """
        expected_shape = (*self.shape, 4)
        if vector_field.shape != expected_shape:
            raise ValueError(
                f"Vector field shape {vector_field.shape} doesn't match expected {expected_shape}"
            )

        # Use covariant divergence if metric is available
        if self.metric is not None:
            # Import here to avoid circular imports
            from .derivatives import CovariantDerivative
            from .four_vectors import FourVector

            # Create coordinate arrays for covariant derivative
            coord_arrays = [self.coordinates[name] for name in self.coordinate_names]

            # Create CovariantDerivative instance
            covariant_deriv = CovariantDerivative(self.metric)

            # Convert to FourVector format and compute covariant divergence
            four_vector = FourVector(vector_field, is_covariant=False, metric=self.metric)
            return covariant_deriv.vector_divergence(four_vector, coord_arrays)  # type: ignore[no-any-return]
        else:
            # Flat-space approximation - warn for non-Cartesian coordinates
            if self.coordinate_system != "cartesian":
                warnings.warn(
                    f"Using flat-space divergence approximation for {self.coordinate_system} coordinates. "
                    "This may produce incorrect results. Consider providing a metric tensor.",
                    stacklevel=2,
                )

            # Simple flat-space divergence
            divergence = np.zeros(self.shape)
            for mu in range(4):
                grad_component = self.gradient(vector_field[..., mu], axis=mu)
                divergence += grad_component

            return divergence

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of scalar field.

        Uses covariant Laplacian if metric is available, otherwise falls back
        to flat-space Laplacian with a warning for non-Cartesian coordinates.

        Args:
            field: Scalar field on grid

        Returns:
            Laplacian field
        """
        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid shape {self.shape}")

        # Use covariant Laplacian if metric is available
        if self.metric is not None:
            # Import here to avoid circular imports
            from .derivatives import CovariantDerivative

            # Create coordinate arrays for covariant derivative
            coord_arrays = [self.coordinates[name] for name in self.coordinate_names]

            # Create CovariantDerivative instance
            covariant_deriv = CovariantDerivative(self.metric)

            # Compute covariant Laplacian: ∇² φ = ∇_μ ∇^μ φ
            # First compute gradient
            gradient_field = np.zeros((*self.shape, 4))
            for mu in range(4):
                gradient_field[..., mu] = self.gradient(field, axis=mu)

            # Then compute divergence of gradient
            return self.divergence(gradient_field)
        else:
            # Flat-space approximation - warn for non-Cartesian coordinates
            if self.coordinate_system != "cartesian":
                warnings.warn(
                    f"Using flat-space Laplacian approximation for {self.coordinate_system} coordinates. "
                    "This may produce incorrect results. Consider providing a metric tensor.",
                    stacklevel=2,
                )

            # Simple flat-space Laplacian (spatial only)
            laplacian = np.zeros_like(field)
            for axis in range(1, 4):  # Exclude time coordinate
                coord_name = self.coordinate_names[axis]
                spacing = self.coordinates[coord_name][1] - self.coordinates[coord_name][0]

                # Second derivative using central differences
                second_deriv = np.gradient(
                    np.gradient(field, spacing, axis=axis), spacing, axis=axis
                )
                laplacian += second_deriv

            return laplacian

    def interpolate(self, field: np.ndarray, coords: np.ndarray, method: str = "linear") -> float:
        """
        Interpolate field value at arbitrary coordinates.

        Args:
            field: Field values on grid
            coords: Target coordinates [t, x, y, z]
            method: Interpolation method ('linear', 'nearest', 'cubic')

        Returns:
            Interpolated field value
        """
        from scipy.interpolate import RegularGridInterpolator

        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} doesn't match grid shape {self.shape}")

        # Create interpolator
        coord_arrays = [self.coordinates[name] for name in self.coordinate_names]
        interpolator = RegularGridInterpolator(coord_arrays, field, method=method)

        return float(interpolator(coords))

    def apply_boundary_conditions(
        self, field: np.ndarray, boundary_conditions: dict[str, str | tuple[str, float]]
    ) -> np.ndarray:
        """
        Apply boundary conditions to field.

        Args:
            field: Field to apply boundary conditions to
            boundary_conditions: Dict mapping boundary names to condition types
                                 - 'periodic', 'reflecting', 'absorbing' (strings)
                                 - ('fixed', value) for fixed boundaries with specified value

        Returns:
            Field with boundary conditions applied
        """
        field_bc = field.copy()

        for boundary, condition in boundary_conditions.items():
            if condition == "periodic":
                field_bc = self._apply_periodic_bc(field_bc, boundary)
            elif condition == "reflecting":
                field_bc = self._apply_reflecting_bc(field_bc, boundary)
            elif condition == "absorbing":
                field_bc = self._apply_absorbing_bc(field_bc, boundary)
            elif isinstance(condition, tuple) and condition[0] == "fixed":
                # Handle fixed boundary with specified value
                value = condition[1] if len(condition) > 1 else 0.0
                field_bc = self._apply_fixed_bc(field_bc, boundary, value)
            elif condition == "fixed":
                # Handle fixed boundary with default value (0.0)
                field_bc = self._apply_fixed_bc(field_bc, boundary, 0.0)
            else:
                raise ValueError(f"Unknown boundary condition: {condition}")

        return field_bc

    def _apply_periodic_bc(self, field: np.ndarray, boundary: str) -> np.ndarray:
        """Apply periodic boundary conditions.

        Args:
            field: Field array to apply boundary conditions to
            boundary: Boundary specification (e.g., 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')

        Returns:
            Field with periodic boundary conditions applied
        """
        field_bc = field.copy()

        # Parse boundary specification
        parts = boundary.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid boundary specification: {boundary}")

        coord_name, side = parts

        # Map coordinate names to axis indices
        # Note: Coordinate ordering is (t, spatial1, spatial2, spatial3)
        coord_to_axis = {
            "t": 0,
            "time": 0,
            "tau": 0,
            "x": 1,
            "y": 2,
            "z": 3,
            "r": 1,  # spherical: (t, r, theta, phi)
            "theta": 2,
            "phi": 3,  # Fixed: phi is at index 3 in spherical coordinates
            "rho": 1,  # cylindrical: (t, rho, phi, z)
            "eta": 1,  # milne: (tau, eta, x, y)
        }

        if coord_name not in coord_to_axis:
            raise ValueError(f"Unknown coordinate: {coord_name}")

        axis = coord_to_axis[coord_name]

        # Apply periodic boundary condition
        if side == "min":
            # Set minimum boundary to match maximum boundary
            if axis == 0:  # time
                field_bc[0, ...] = field_bc[-1, ...]
            elif axis == 1:  # x
                field_bc[:, 0, ...] = field_bc[:, -1, ...]
            elif axis == 2:  # y
                field_bc[:, :, 0, :] = field_bc[:, :, -1, :]
            elif axis == 3:  # z
                field_bc[:, :, :, 0] = field_bc[:, :, :, -1]
        elif side == "max":
            # Set maximum boundary to match minimum boundary
            if axis == 0:  # time
                field_bc[-1, ...] = field_bc[0, ...]
            elif axis == 1:  # x
                field_bc[:, -1, ...] = field_bc[:, 0, ...]
            elif axis == 2:  # y
                field_bc[:, :, -1, :] = field_bc[:, :, 0, :]
            elif axis == 3:  # z
                field_bc[:, :, :, -1] = field_bc[:, :, :, 0]
        else:
            raise ValueError(f"Invalid boundary side: {side}")

        return field_bc

    def apply_periodic_bc_all_spatial(self, field: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary conditions to all spatial dimensions.

        Convenience method for spectral methods that require periodic
        boundary conditions in all spatial directions.

        Args:
            field: Field array to apply boundary conditions to

        Returns:
            Field with periodic boundary conditions applied to all spatial dimensions
        """
        field_bc = field.copy()

        # Get spatial coordinate names
        if self.coordinate_system == "cartesian":
            spatial_coords = ["x", "y", "z"]
        elif self.coordinate_system == "spherical":
            spatial_coords = ["r", "theta", "phi"]
        elif self.coordinate_system == "cylindrical":
            spatial_coords = ["rho", "phi", "z"]
        elif self.coordinate_system == "milne":
            spatial_coords = ["eta", "x", "y"]
        else:
            raise ValueError(f"Unknown coordinate system: {self.coordinate_system}")

        # Apply periodic boundary conditions to all spatial directions
        for coord in spatial_coords:
            field_bc = self._apply_periodic_bc(field_bc, f"{coord}_min")
            field_bc = self._apply_periodic_bc(field_bc, f"{coord}_max")

        return field_bc

    def _apply_reflecting_bc(self, field: np.ndarray, boundary: str) -> np.ndarray:
        """Apply reflecting boundary conditions.

        Reflecting boundaries mirror field values across the boundary,
        appropriate for problems with symmetry.

        Args:
            field: Field array to apply boundary conditions to
            boundary: Boundary specification (e.g., 'x_min', 'x_max')

        Returns:
            Field with reflecting boundary conditions applied
        """
        field_bc = field.copy()

        # Parse boundary specification
        parts = boundary.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid boundary specification: {boundary}")

        coord_name, side = parts

        # Use same coordinate mapping as periodic BC
        coord_to_axis = {
            "t": 0,
            "time": 0,
            "tau": 0,
            "x": 1,
            "y": 2,
            "z": 3,
            "r": 1,
            "theta": 2,
            "phi": 3,
            "rho": 1,
            "eta": 1,
        }

        if coord_name not in coord_to_axis:
            raise ValueError(f"Unknown coordinate: {coord_name}")

        axis = coord_to_axis[coord_name]

        # Apply reflecting boundary condition
        if side == "min":
            # Reflect from interior points to boundary
            if axis == 0:  # time
                field_bc[0, ...] = field_bc[1, ...]
            elif axis == 1:  # x
                field_bc[:, 0, ...] = field_bc[:, 1, ...]
            elif axis == 2:  # y
                field_bc[:, :, 0, :] = field_bc[:, :, 1, :]
            elif axis == 3:  # z
                field_bc[:, :, :, 0] = field_bc[:, :, :, 1]
        elif side == "max":
            # Reflect from interior points to boundary
            if axis == 0:  # time
                field_bc[-1, ...] = field_bc[-2, ...]
            elif axis == 1:  # x
                field_bc[:, -1, ...] = field_bc[:, -2, ...]
            elif axis == 2:  # y
                field_bc[:, :, -1, :] = field_bc[:, :, -2, :]
            elif axis == 3:  # z
                field_bc[:, :, :, -1] = field_bc[:, :, :, -2]
        else:
            raise ValueError(f"Invalid boundary side: {side}")

        return field_bc

    def _apply_absorbing_bc(self, field: np.ndarray, boundary: str) -> np.ndarray:
        """Apply absorbing boundary conditions.

        Absorbing boundaries set the field to zero at the boundary,
        preventing reflection and allowing waves to "exit" the domain.

        Args:
            field: Field array to apply boundary conditions to
            boundary: Boundary specification (e.g., 'x_min', 'x_max')

        Returns:
            Field with absorbing boundary conditions applied
        """
        field_bc = field.copy()

        # Parse boundary specification
        parts = boundary.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid boundary specification: {boundary}")

        coord_name, side = parts

        # Use same coordinate mapping as periodic BC
        coord_to_axis = {
            "t": 0,
            "time": 0,
            "tau": 0,
            "x": 1,
            "y": 2,
            "z": 3,
            "r": 1,
            "theta": 2,
            "phi": 3,
            "rho": 1,
            "eta": 1,
        }

        if coord_name not in coord_to_axis:
            raise ValueError(f"Unknown coordinate: {coord_name}")

        axis = coord_to_axis[coord_name]

        # Apply absorbing boundary condition (set to zero)
        if side == "min":
            if axis == 0:  # time
                field_bc[0, ...] = 0.0
            elif axis == 1:  # x
                field_bc[:, 0, ...] = 0.0
            elif axis == 2:  # y
                field_bc[:, :, 0, :] = 0.0
            elif axis == 3:  # z
                field_bc[:, :, :, 0] = 0.0
        elif side == "max":
            if axis == 0:  # time
                field_bc[-1, ...] = 0.0
            elif axis == 1:  # x
                field_bc[:, -1, ...] = 0.0
            elif axis == 2:  # y
                field_bc[:, :, -1, :] = 0.0
            elif axis == 3:  # z
                field_bc[:, :, :, -1] = 0.0
        else:
            raise ValueError(f"Invalid boundary side: {side}")

        return field_bc

    def _apply_fixed_bc(self, field: np.ndarray, boundary: str, value: float = 0.0) -> np.ndarray:
        """Apply fixed value boundary conditions.

        Fixed boundaries set the field to a specified constant value at the boundary.

        Args:
            field: Field array to apply boundary conditions to
            boundary: Boundary specification (e.g., 'x_min', 'x_max')
            value: Fixed value to set at boundary (default: 0.0)

        Returns:
            Field with fixed boundary conditions applied
        """
        field_bc = field.copy()

        # Parse boundary specification
        parts = boundary.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid boundary specification: {boundary}")

        coord_name, side = parts

        # Use same coordinate mapping as periodic BC
        coord_to_axis = {
            "t": 0,
            "time": 0,
            "tau": 0,
            "x": 1,
            "y": 2,
            "z": 3,
            "r": 1,
            "theta": 2,
            "phi": 3,
            "rho": 1,
            "eta": 1,
        }

        if coord_name not in coord_to_axis:
            raise ValueError(f"Unknown coordinate: {coord_name}")

        axis = coord_to_axis[coord_name]

        # Apply fixed boundary condition
        if side == "min":
            if axis == 0:  # time
                field_bc[0, ...] = value
            elif axis == 1:  # x
                field_bc[:, 0, ...] = value
            elif axis == 2:  # y
                field_bc[:, :, 0, :] = value
            elif axis == 3:  # z
                field_bc[:, :, :, 0] = value
        elif side == "max":
            if axis == 0:  # time
                field_bc[-1, ...] = value
            elif axis == 1:  # x
                field_bc[:, -1, ...] = value
            elif axis == 2:  # y
                field_bc[:, :, -1, :] = value
            elif axis == 3:  # z
                field_bc[:, :, :, -1] = value
        else:
            raise ValueError(f"Invalid boundary side: {side}")

        return field_bc

    def volume_element(self) -> np.ndarray:
        """
        Compute volume element √(-g) for the grid.

        Returns:
            Volume element at each grid point
        """
        if self.metric is None:
            # Flat spacetime volume element
            if self.coordinate_system == "cartesian":
                return np.ones(self.shape)
            elif self.coordinate_system == "spherical":
                # d⁴x = dt dr r dθ r sin(θ) dφ = r² sin(θ) dt dr dθ dφ
                r_mesh, theta_mesh = np.meshgrid(
                    self.coordinates["r"], self.coordinates["theta"], indexing="ij"
                )
                volume_elem = r_mesh**2 * np.sin(theta_mesh)
                # Broadcast to full 4D grid
                return np.broadcast_to(volume_elem[None, :, :, None], self.shape)
            else:
                # Other coordinate systems - simplified
                return np.ones(self.shape)
        else:
            # General coordinate system with metric
            metric_determinant = self.metric.determinant
            return np.sqrt(-metric_determinant)  # type: ignore[no-any-return]

    def coordinate_transformation_jacobian(self, target_system: str) -> np.ndarray:
        """
        Compute Jacobian for coordinate transformation.

        Args:
            target_system: Target coordinate system

        Returns:
            Jacobian matrix ∂x'^μ/∂x^ν

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        raise NotImplementedError(
            "Coordinate transformation Jacobian computation is not yet implemented. "
            "This would require implementing analytical transformations between "
            f"'{self.coordinate_system}' and '{target_system}' coordinate systems."
        )

    def create_subgrid(
        self, time_slice: slice | None = None, spatial_slices: list[slice] | None = None
    ) -> "SpacetimeGrid":
        """
        Create subgrid from current grid.

        Args:
            time_slice: Slice for time dimension
            spatial_slices: List of slices for spatial dimensions

        Returns:
            New SpacetimeGrid covering subregion

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        raise NotImplementedError(
            "Subgrid creation is not yet implemented. This would involve: "
            "1) Extracting coordinate ranges from slices, "
            "2) Computing new grid parameters, "
            "3) Creating a new SpacetimeGrid instance with the subregion parameters. "
            "This is useful for adaptive mesh refinement and domain decomposition."
        )

    def __str__(self) -> str:
        return (
            f"SpacetimeGrid({self.coordinate_system}, "
            f"shape={self.shape}, "
            f"total_points={self.total_points})"
        )

    def __repr__(self) -> str:
        return (
            f"SpacetimeGrid(coordinate_system='{self.coordinate_system}', "
            f"time_range={self.time_range}, "
            f"spatial_ranges={self.spatial_ranges}, "
            f"grid_points={self.grid_points})"
        )


class AdaptiveMeshRefinement:
    """
    Adaptive mesh refinement for spacetime grids.

    Dynamically refines grid resolution in regions with large gradients
    or other refinement criteria.
    """

    def __init__(self, base_grid: SpacetimeGrid):
        """
        Initialize AMR system.

        Args:
            base_grid: Base spacetime grid
        """
        self.base_grid = base_grid
        self.refined_regions: list[dict[str, Any]] = []
        self.refinement_criteria: list[Callable] = []

    def add_refinement_criterion(self, criterion: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Add refinement criterion function.

        Args:
            criterion: Function that takes field array and returns refinement mask
        """
        self.refinement_criteria.append(criterion)

    def refine_grid(self, field: np.ndarray, refinement_factor: int = 2) -> dict[str, Any]:
        """
        Refine grid based on field gradients and criteria.

        Args:
            field: Field to analyze for refinement
            refinement_factor: Factor by which to refine grid spacing

        Returns:
            Dictionary with refined grid information

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        raise NotImplementedError(
            "Adaptive mesh refinement is not yet implemented. This would involve: "
            "1) Analyzing field gradients and applying refinement criteria, "
            "2) Identifying regions requiring refinement, "
            "3) Creating refined subgrids with increased resolution, "
            "4) Managing communication between different refinement levels. "
            "Consider using existing AMR libraries like BoxLib or Chombo."
        )


# Utility functions for common grid operations
def create_cartesian_grid(
    time_range: tuple[float, float],
    spatial_extent: float,
    grid_points: tuple[int, int, int, int],
) -> SpacetimeGrid:
    """
    Create symmetric Cartesian grid.

    Args:
        time_range: (t_min, t_max)
        spatial_extent: Spatial size (symmetric around origin)
        grid_points: (Nt, Nx, Ny, Nz)

    Returns:
        Cartesian SpacetimeGrid
    """
    spatial_ranges = [(-spatial_extent / 2, spatial_extent / 2) for _ in range(3)]
    return SpacetimeGrid("cartesian", time_range, spatial_ranges, grid_points)


def create_milne_grid(
    tau_range: tuple[float, float],
    eta_range: tuple[float, float],
    transverse_extent: float,
    grid_points: tuple[int, int, int, int],
) -> SpacetimeGrid:
    """
    Create Milne coordinate grid for boost-invariant systems.

    Args:
        tau_range: (τ_min, τ_max) proper time range
        eta_range: (η_min, η_max) rapidity range
        transverse_extent: Transverse size
        grid_points: (Nτ, Nη, Nx, Ny)

    Returns:
        Milne SpacetimeGrid
    """
    spatial_ranges = [
        eta_range,
        (-transverse_extent / 2, transverse_extent / 2),
        (-transverse_extent / 2, transverse_extent / 2),
    ]
    return SpacetimeGrid("milne", tau_range, spatial_ranges, grid_points)


def create_spectral_cartesian_grid(
    time_range: tuple[float, float],
    spatial_extent: float,
    grid_points: tuple[int, int, int, int],
) -> SpacetimeGrid:
    """
    Create Cartesian grid optimized for spectral methods.

    Uses proper spacing for periodic boundary conditions: dx = L/N instead of L/(N-1).
    This ensures that the grid points are [0, dx, 2*dx, ..., (N-1)*dx] which is
    periodic on [0, L) and compatible with FFT-based spectral methods.

    Args:
        time_range: (t_min, t_max)
        spatial_extent: Spatial size (symmetric around origin)
        grid_points: (Nt, Nx, Ny, Nz)

    Returns:
        Cartesian SpacetimeGrid optimized for spectral methods
    """
    # For spectral methods, we need periodic grids with spacing L/N
    # Override the coordinate creation to use proper spectral spacing
    spatial_ranges = [(-spatial_extent / 2, spatial_extent / 2) for _ in range(3)]
    grid = SpacetimeGrid("cartesian", time_range, spatial_ranges, grid_points)

    # Override coordinates for spectral compatibility
    grid._create_spectral_coordinates()

    return grid
