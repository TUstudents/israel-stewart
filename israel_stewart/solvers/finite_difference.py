"""
Finite difference spatial discretization for Israel-Stewart hydrodynamics.

This module implements high-order finite difference schemes for spatial derivatives
in the conservation laws and Israel-Stewart evolution equations. Includes both
conservative and non-conservative forms with proper boundary treatment.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sparse

from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MetricBase
from ..core.performance import monitor_performance
from ..core.spacetime_grid import SpacetimeGrid


class FiniteDifferenceScheme(ABC):
    """
    Abstract base class for finite difference spatial discretization schemes.

    Defines the interface for computing spatial derivatives needed in
    relativistic hydrodynamics conservation laws.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        boundary_conditions: str = "periodic",
        stencil_width: int = 3,
    ):
        """
        Initialize finite difference scheme.

        Args:
            grid: Spacetime grid for discretization
            metric: Spacetime metric for covariant operations
            boundary_conditions: Type of boundary conditions ('periodic', 'outflow', 'reflecting')
            stencil_width: Number of points in finite difference stencil
        """
        self.grid = grid
        self.metric = metric
        self.boundary_conditions = boundary_conditions
        self.stencil_width = stencil_width

        # Grid spacing
        self.dx = grid.spatial_spacing
        self.coordinates = grid.coordinate_system

        # Precompute finite difference coefficients
        self.derivative_coefficients = self._compute_derivative_coefficients()

        # Boundary handling
        self.ghost_points = stencil_width // 2

    @abstractmethod
    def compute_spatial_derivatives(
        self,
        fields: ISFieldConfiguration,
        variable: str,
        derivative_order: int = 1,
    ) -> dict[str, np.ndarray]:
        """
        Compute spatial derivatives of field variables.

        Args:
            fields: Field configuration
            variable: Variable name ('rho', 'pressure', 'u_mu', etc.)
            derivative_order: Order of derivative (1 or 2)

        Returns:
            Dictionary of spatial derivatives
        """
        pass

    @abstractmethod
    def compute_divergence(
        self,
        tensor_field: np.ndarray,
        component_indices: tuple[int, ...],
    ) -> np.ndarray:
        """
        Compute covariant divergence of tensor field.

        Args:
            tensor_field: Tensor field to differentiate
            component_indices: Indices specifying which component to compute

        Returns:
            Divergence of tensor field
        """
        pass

    def _compute_derivative_coefficients(self) -> dict[int, dict[int, np.ndarray]]:
        """Compute finite difference coefficients for various orders and stencils."""
        coefficients = {}

        # First derivative coefficients
        if self.stencil_width == 3:
            # Second-order central difference
            coefficients[1] = {
                0: np.array([-0.5, 0.0, 0.5]),  # Central points
                -1: np.array([-1.5, 2.0, -0.5]),  # Left boundary (forward)
                1: np.array([0.5, -2.0, 1.5]),  # Right boundary (backward)
            }
        elif self.stencil_width == 5:
            # Fourth-order central difference
            coefficients[1] = {
                0: np.array([1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12]),  # Central
                -2: np.array([-25 / 12, 4.0, -3.0, 4 / 3, -1 / 4]),  # Left boundary
                -1: np.array([-1 / 4, -5 / 6, 1.5, -1 / 2, 1 / 12]),  # Near left
                1: np.array([-1 / 12, 1 / 2, -1.5, 5 / 6, 1 / 4]),  # Near right
                2: np.array([1 / 4, -4 / 3, 3.0, -4.0, 25 / 12]),  # Right boundary
            }
        else:
            # Default to second-order
            coefficients[1] = {
                0: np.array([-0.5, 0.0, 0.5]),
                -1: np.array([-1.5, 2.0, -0.5]),
                1: np.array([0.5, -2.0, 1.5]),
            }

        # Second derivative coefficients
        if self.stencil_width == 3:
            coefficients[2] = {
                0: np.array([1.0, -2.0, 1.0]),  # Central
                -1: np.array([2.0, -5.0, 4.0, -1.0]),  # Left boundary
                1: np.array([-1.0, 4.0, -5.0, 2.0]),  # Right boundary
            }
        elif self.stencil_width == 5:
            coefficients[2] = {
                0: np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),  # Central
                -2: np.array([15 / 4, -77 / 6, 107 / 6, -13.0, 61 / 12, -5 / 6]),  # Left
                -1: np.array([5 / 6, -5 / 4, -1 / 3, 7 / 6, -1 / 2, 1 / 12]),  # Near left
                1: np.array([1 / 12, -1 / 2, 7 / 6, -1 / 3, -5 / 4, 5 / 6]),  # Near right
                2: np.array([-5 / 6, 61 / 12, -13.0, 107 / 6, -77 / 6, 15 / 4]),  # Right
            }
        else:
            coefficients[2] = {
                0: np.array([1.0, -2.0, 1.0]),
                -1: np.array([2.0, -5.0, 4.0, -1.0]),
                1: np.array([-1.0, 4.0, -5.0, 2.0]),
            }

        return coefficients

    def _apply_boundary_conditions(
        self,
        field: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """Apply boundary conditions by extending field with ghost points."""
        if self.boundary_conditions == "periodic":
            return self._apply_periodic_bc(field, axis)
        elif self.boundary_conditions == "outflow":
            return self._apply_outflow_bc(field, axis)
        elif self.boundary_conditions == "reflecting":
            return self._apply_reflecting_bc(field, axis)
        else:
            warnings.warn(f"Unknown boundary condition: {self.boundary_conditions}")
            return field

    def _apply_periodic_bc(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Apply periodic boundary conditions."""
        # Add ghost points by wrapping around
        left_ghost = np.take(field, range(-self.ghost_points, 0), axis=axis)
        right_ghost = np.take(field, range(self.ghost_points), axis=axis)

        extended_field = np.concatenate([left_ghost, field, right_ghost], axis=axis)
        return extended_field

    def _apply_outflow_bc(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Apply outflow (zero gradient) boundary conditions."""
        # Extend by repeating boundary values
        left_ghost = np.repeat(np.take(field, [0], axis=axis), self.ghost_points, axis=axis)
        right_ghost = np.repeat(np.take(field, [-1], axis=axis), self.ghost_points, axis=axis)

        extended_field = np.concatenate([left_ghost, field, right_ghost], axis=axis)
        return extended_field

    def _apply_reflecting_bc(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Apply reflecting boundary conditions."""
        # Mirror field values across boundaries
        left_ghost = np.flip(np.take(field, range(1, self.ghost_points + 1), axis=axis), axis=axis)
        right_ghost = np.flip(
            np.take(field, range(-self.ghost_points - 1, -1), axis=axis), axis=axis
        )

        extended_field = np.concatenate([left_ghost, field, right_ghost], axis=axis)
        return extended_field


class ConservativeFiniteDifference(FiniteDifferenceScheme):
    """
    Conservative finite difference scheme for hydrodynamics.

    Implements conservative discretization that preserves conservation laws
    at the discrete level. Essential for shock-capturing and long-time stability.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        boundary_conditions: str = "periodic",
        order: int = 2,
        flux_limiter: str = "none",
    ):
        """
        Initialize conservative finite difference scheme.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            boundary_conditions: Boundary condition type
            order: Order of accuracy (2, 4, or 6)
            flux_limiter: Flux limiting for shock capturing
        """
        stencil_width = order + 1
        super().__init__(grid, metric, boundary_conditions, stencil_width)

        self.order = order
        self.flux_limiter = flux_limiter

    @monitor_performance("conservative_spatial_derivatives")
    def compute_spatial_derivatives(
        self,
        fields: ISFieldConfiguration,
        variable: str,
        derivative_order: int = 1,
    ) -> dict[str, np.ndarray]:
        """
        Compute spatial derivatives using conservative form.

        For conservation laws: du/dt + div(F(u)) = S(u)
        We discretize as: du/dt + (F_{i+1/2} - F_{i-1/2})/dx = S(u)
        """
        # Get field to differentiate
        field_data = getattr(fields, variable)
        if field_data is None:
            return {}

        derivatives = {}

        # Compute derivatives in each spatial direction
        for dim in range(3):  # 3 spatial dimensions
            axis = dim + 1  # Skip time dimension

            if derivative_order == 1:
                # First derivative using conservative differences
                deriv = self._compute_conservative_derivative(field_data, axis, dim)
                derivatives[f"d{variable}_dx{dim}"] = deriv

            elif derivative_order == 2:
                # Second derivative for diffusion terms
                deriv = self._compute_second_derivative(field_data, axis, dim)
                derivatives[f"d2{variable}_dx{dim}2"] = deriv

        return derivatives

    def _compute_conservative_derivative(
        self, field: np.ndarray, axis: int, spatial_dim: int
    ) -> np.ndarray:
        """Compute conservative first derivative."""
        # Apply boundary conditions
        extended_field = self._apply_boundary_conditions(field, axis)

        # Compute flux differences
        if self.order == 2:
            # Second-order conservative scheme
            flux_left = self._compute_numerical_flux(extended_field, axis, -0.5)
            flux_right = self._compute_numerical_flux(extended_field, axis, 0.5)

            derivative = (flux_right - flux_left) / self.dx[spatial_dim]

        elif self.order == 4:
            # Fourth-order conservative scheme
            flux_left_far = self._compute_numerical_flux(extended_field, axis, -1.5)
            flux_left = self._compute_numerical_flux(extended_field, axis, -0.5)
            flux_right = self._compute_numerical_flux(extended_field, axis, 0.5)
            flux_right_far = self._compute_numerical_flux(extended_field, axis, 1.5)

            derivative = (-flux_right_far + 8 * flux_right - 8 * flux_left + flux_left_far) / (
                12 * self.dx[spatial_dim]
            )

        else:
            # Default to second-order
            flux_left = self._compute_numerical_flux(extended_field, axis, -0.5)
            flux_right = self._compute_numerical_flux(extended_field, axis, 0.5)
            derivative = (flux_right - flux_left) / self.dx[spatial_dim]

        # Remove ghost points
        slices = [slice(None)] * field.ndim
        slices[axis] = slice(self.ghost_points, -self.ghost_points)

        return derivative[tuple(slices)]

    def _compute_numerical_flux(
        self, extended_field: np.ndarray, axis: int, offset: float
    ) -> np.ndarray:
        """
        Compute numerical flux at cell interfaces.

        For now, we use simple averaging. In practice, you'd implement
        more sophisticated schemes like Lax-Friedrichs, Roe, or HLLC.
        """
        # Simple average for demonstration
        if offset == 0.5:
            # Right interface: F_{i+1/2} = (F_i + F_{i+1})/2
            left_slice = [slice(None)] * extended_field.ndim
            right_slice = [slice(None)] * extended_field.ndim

            left_slice[axis] = slice(self.ghost_points, -self.ghost_points - 1)
            right_slice[axis] = slice(self.ghost_points + 1, -self.ghost_points)

            flux = 0.5 * (extended_field[tuple(left_slice)] + extended_field[tuple(right_slice)])

        elif offset == -0.5:
            # Left interface: F_{i-1/2} = (F_{i-1} + F_i)/2
            left_slice = [slice(None)] * extended_field.ndim
            right_slice = [slice(None)] * extended_field.ndim

            left_slice[axis] = slice(self.ghost_points - 1, -self.ghost_points - 2)
            right_slice[axis] = slice(self.ghost_points, -self.ghost_points - 1)

            flux = 0.5 * (extended_field[tuple(left_slice)] + extended_field[tuple(right_slice)])

        else:
            # For higher-order schemes, implement more complex stencils
            warnings.warn(f"Flux computation for offset {offset} not implemented")
            return np.zeros_like(extended_field)

        return flux

    def _compute_second_derivative(
        self, field: np.ndarray, axis: int, spatial_dim: int
    ) -> np.ndarray:
        """Compute second derivative for diffusion terms."""
        extended_field = self._apply_boundary_conditions(field, axis)

        # Use standard finite difference for second derivatives
        coeffs = self.derivative_coefficients[2][0]  # Central difference

        derivative = np.zeros_like(field)

        for i, coeff in enumerate(coeffs):
            offset = i - len(coeffs) // 2
            shifted_slice = [slice(None)] * field.ndim
            shifted_slice[axis] = slice(
                self.ghost_points + offset, -self.ghost_points + offset if offset < 0 else None
            )
            derivative += coeff * extended_field[tuple(shifted_slice)]

        derivative /= self.dx[spatial_dim] ** 2

        return derivative

    @monitor_performance("conservative_divergence")
    def compute_divergence(
        self,
        tensor_field: np.ndarray,
        component_indices: tuple[int, ...],
    ) -> np.ndarray:
        """
        Compute covariant divergence: div_mu T^{mu nu}.

        In conservative form, this becomes a sum of conservative derivatives
        plus Christoffel symbol contributions.
        """
        divergence = np.zeros(tensor_field.shape[: -len(component_indices)])

        # Loop over spatial dimensions
        for dim in range(3):  # 3 spatial dimensions
            axis = dim + 1  # Skip time dimension

            # Extract the appropriate tensor component
            component_slice = [slice(None)] * (tensor_field.ndim - len(component_indices))
            component_slice.extend([component_indices[0], dim])  # T^mu_i component

            component_field = tensor_field[tuple(component_slice)]

            # Compute conservative derivative
            deriv = self._compute_conservative_derivative(component_field, axis, dim)
            divergence += deriv

        # Add Christoffel symbol contributions
        christoffel_terms = self._compute_christoffel_contributions(tensor_field, component_indices)
        divergence += christoffel_terms

        return divergence

    def _compute_christoffel_contributions(
        self,
        tensor_field: np.ndarray,
        component_indices: tuple[int, ...],
    ) -> np.ndarray:
        """Compute Christoffel symbol contributions to covariant divergence."""
        # For now, return zeros (valid in Cartesian coordinates)
        # In curved spacetime, this would compute Gamma^mu_alpha_beta T^alpha_beta terms
        return np.zeros(tensor_field.shape[: -len(component_indices)])


class UpwindFiniteDifference(FiniteDifferenceScheme):
    """
    Upwind finite difference scheme for hyperbolic equations.

    Uses characteristic information to determine upwind direction,
    providing better stability for advection-dominated problems.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        boundary_conditions: str = "periodic",
        order: int = 1,
        characteristic_speeds: Callable | None = None,
    ):
        """
        Initialize upwind finite difference scheme.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            boundary_conditions: Boundary condition type
            order: Order of accuracy (1, 2, or 3)
            characteristic_speeds: Function to compute characteristic velocities
        """
        super().__init__(grid, metric, boundary_conditions, order + 2)

        self.order = order
        self.characteristic_speeds = characteristic_speeds or self._default_speeds

    def _default_speeds(self, fields: ISFieldConfiguration) -> dict[str, np.ndarray]:
        """Default characteristic speeds (sound speed + fluid velocity)."""
        # Speed of sound (approximately 1/sqrt(3) for ultrarelativistic fluid)
        cs = np.full_like(fields.temperature, 1.0 / np.sqrt(3.0))

        # Extract spatial components of four-velocity
        u_spatial = fields.u_mu[..., 1 : self.grid.spatial_dimensions + 1]

        speeds = {}
        for dim in range(self.grid.spatial_dimensions):
            # Characteristic speeds: u +- cs
            speeds[f"lambda_plus_{dim}"] = u_spatial[..., dim] + cs
            speeds[f"lambda_minus_{dim}"] = u_spatial[..., dim] - cs

        return speeds

    @monitor_performance("upwind_spatial_derivatives")
    def compute_spatial_derivatives(
        self,
        fields: ISFieldConfiguration,
        variable: str,
        derivative_order: int = 1,
    ) -> dict[str, np.ndarray]:
        """Compute spatial derivatives using upwind differencing."""
        field_data = getattr(fields, variable)
        if field_data is None:
            return {}

        # Get characteristic speeds
        speeds = self.characteristic_speeds(fields)

        derivatives = {}

        for dim in range(self.grid.spatial_dimensions):
            axis = dim + 1

            if derivative_order == 1:
                # Choose upwind or downwind based on characteristic speed
                speed_plus = speeds.get(f"lambda_plus_{dim}", np.zeros_like(field_data))
                speed_minus = speeds.get(f"lambda_minus_{dim}", np.zeros_like(field_data))

                # Compute both upwind and downwind derivatives
                deriv_upwind = self._compute_upwind_derivative(field_data, axis, dim, "upwind")
                deriv_downwind = self._compute_upwind_derivative(field_data, axis, dim, "downwind")

                # Combine based on characteristic speeds
                deriv = np.where(speed_plus >= 0, deriv_upwind, deriv_downwind)

                derivatives[f"d{variable}_dx{dim}_upwind"] = deriv

        return derivatives

    def _compute_upwind_derivative(
        self,
        field: np.ndarray,
        axis: int,
        spatial_dim: int,
        direction: str,
    ) -> np.ndarray:
        """Compute upwind derivative in specified direction."""
        extended_field = self._apply_boundary_conditions(field, axis)

        if self.order == 1:
            if direction == "upwind":
                # Backward difference: (u_i - u_{i-1})/dx
                coeffs = np.array([1.0, -1.0])
                offsets = [0, -1]
            else:  # downwind
                # Forward difference: (u_{i+1} - u_i)/dx
                coeffs = np.array([-1.0, 1.0])
                offsets = [0, 1]

        elif self.order == 2:
            if direction == "upwind":
                # Second-order upwind: (3u_i - 4u_{i-1} + u_{i-2})/(2dx)
                coeffs = np.array([1.5, -2.0, 0.5])
                offsets = [0, -1, -2]
            else:  # downwind
                # Second-order downwind: (-u_{i+2} + 4u_{i+1} - 3u_i)/(2dx)
                coeffs = np.array([-1.5, 2.0, -0.5])
                offsets = [0, 1, 2]
        else:
            # Default to first-order
            if direction == "upwind":
                coeffs = np.array([1.0, -1.0])
                offsets = [0, -1]
            else:
                coeffs = np.array([-1.0, 1.0])
                offsets = [0, 1]

        # Apply finite difference stencil
        derivative = np.zeros_like(field)

        for coeff, offset in zip(coeffs, offsets):
            shifted_slice = [slice(None)] * field.ndim
            shifted_slice[axis] = slice(
                self.ghost_points + offset, -self.ghost_points + offset if offset < 0 else None
            )
            derivative += coeff * extended_field[tuple(shifted_slice)]

        derivative /= self.dx[spatial_dim]

        return derivative

    def compute_divergence(
        self,
        tensor_field: np.ndarray,
        component_indices: tuple[int, ...],
    ) -> np.ndarray:
        """Compute divergence using upwind differences."""
        # For simplicity, delegate to conservative scheme
        # In practice, you'd implement characteristic-based upwinding
        conservative_scheme = ConservativeFiniteDifference(
            self.grid, self.metric, self.boundary_conditions, order=2
        )
        return conservative_scheme.compute_divergence(tensor_field, component_indices)


class WENOFiniteDifference(FiniteDifferenceScheme):
    """
    Weighted Essentially Non-Oscillatory (WENO) finite difference scheme.

    High-order scheme with shock-capturing capability. Automatically
    switches between high-order stencils in smooth regions and
    lower-order stencils near discontinuities.
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        boundary_conditions: str = "periodic",
        weno_order: int = 5,
        epsilon: float = 1e-6,
    ):
        """
        Initialize WENO finite difference scheme.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            boundary_conditions: Boundary condition type
            weno_order: WENO order (3, 5, or 7)
            epsilon: Small parameter for WENO weights
        """
        super().__init__(grid, metric, boundary_conditions, weno_order + 2)

        self.weno_order = weno_order
        self.epsilon = epsilon

        # WENO coefficients
        self.weno_coeffs = self._setup_weno_coefficients()

    def _setup_weno_coefficients(self) -> dict[str, Any]:
        """Setup WENO reconstruction coefficients."""
        if self.weno_order == 5:
            # WENO5 coefficients
            return {
                "optimal_weights": np.array([0.1, 0.6, 0.3]),
                "stencil_coeffs": [
                    np.array([1 / 3, -7 / 6, 11 / 6]),  # S0: smoothest stencil
                    np.array([-1 / 6, 5 / 6, 1 / 3]),  # S1: middle stencil
                    np.array([1 / 3, 5 / 6, -1 / 6]),  # S2: least smooth stencil
                ],
                "smoothness_coeffs": [
                    [13 / 12, 1 / 4],  # beta0 coefficients
                    [13 / 12, 1 / 4],  # beta1 coefficients
                    [13 / 12, 1 / 4],  # beta2 coefficients
                ],
            }
        else:
            # Default to simpler coefficients
            return {
                "optimal_weights": np.array([0.5, 0.5]),
                "stencil_coeffs": [
                    np.array([0.5, 0.5]),
                    np.array([0.5, 0.5]),
                ],
                "smoothness_coeffs": [[1, 1], [1, 1]],
            }

    @monitor_performance("weno_spatial_derivatives")
    def compute_spatial_derivatives(
        self,
        fields: ISFieldConfiguration,
        variable: str,
        derivative_order: int = 1,
    ) -> dict[str, np.ndarray]:
        """Compute spatial derivatives using WENO reconstruction."""
        field_data = getattr(fields, variable)
        if field_data is None:
            return {}

        derivatives = {}

        for dim in range(self.grid.spatial_dimensions):
            axis = dim + 1

            if derivative_order == 1:
                # WENO reconstruction for first derivative
                deriv = self._compute_weno_derivative(field_data, axis, dim)
                derivatives[f"d{variable}_dx{dim}_weno"] = deriv

        return derivatives

    def _compute_weno_derivative(
        self,
        field: np.ndarray,
        axis: int,
        spatial_dim: int,
    ) -> np.ndarray:
        """Compute derivative using WENO reconstruction."""
        extended_field = self._apply_boundary_conditions(field, axis)

        # Get WENO coefficients
        optimal_weights = self.weno_coeffs["optimal_weights"]
        stencil_coeffs = self.weno_coeffs["stencil_coeffs"]

        # Initialize result
        derivative = np.zeros_like(field)

        # Compute WENO weights and reconstruction
        num_stencils = len(stencil_coeffs)

        for i in range(num_stencils):
            # Compute smoothness indicator
            beta = self._compute_smoothness_indicator(extended_field, axis, i)

            # Compute WENO weight
            alpha = optimal_weights[i] / (self.epsilon + beta) ** 2

            # Apply stencil
            coeffs = stencil_coeffs[i]
            stencil_result = np.zeros_like(field)

            for j, coeff in enumerate(coeffs):
                offset = j - len(coeffs) // 2
                shifted_slice = [slice(None)] * field.ndim
                shifted_slice[axis] = slice(
                    self.ghost_points + offset, -self.ghost_points + offset if offset < 0 else None
                )
                stencil_result += coeff * extended_field[tuple(shifted_slice)]

            derivative += alpha * stencil_result

        # Normalize by sum of weights
        weight_sum = np.sum(
            [
                optimal_weights[i]
                / (self.epsilon + self._compute_smoothness_indicator(extended_field, axis, i)) ** 2
                for i in range(num_stencils)
            ],
            axis=0,
        )

        derivative /= weight_sum * self.dx[spatial_dim]

        return derivative

    def _compute_smoothness_indicator(
        self,
        extended_field: np.ndarray,
        axis: int,
        stencil_index: int,
    ) -> np.ndarray:
        """Compute smoothness indicator for WENO weights."""
        # Simplified smoothness indicator based on local variation
        # In practice, this would be more sophisticated

        # Get local stencil points
        stencil_size = len(self.weno_coeffs["stencil_coeffs"][stencil_index])
        start_offset = stencil_index - stencil_size // 2

        # Compute local variation
        variation = np.zeros(
            extended_field.shape[:-1] if axis == extended_field.ndim - 1 else extended_field.shape
        )

        for i in range(stencil_size - 1):
            offset1 = start_offset + i
            offset2 = start_offset + i + 1

            slice1 = [slice(None)] * extended_field.ndim
            slice2 = [slice(None)] * extended_field.ndim

            slice1[axis] = slice(
                self.ghost_points + offset1, -self.ghost_points + offset1 if offset1 < 0 else None
            )
            slice2[axis] = slice(
                self.ghost_points + offset2, -self.ghost_points + offset2 if offset2 < 0 else None
            )

            diff = extended_field[tuple(slice2)] - extended_field[tuple(slice1)]
            variation += diff**2

        return variation + self.epsilon  # Avoid division by zero

    def compute_divergence(
        self,
        tensor_field: np.ndarray,
        component_indices: tuple[int, ...],
    ) -> np.ndarray:
        """Compute divergence using WENO reconstruction."""
        # Use conservative scheme as base, apply WENO for reconstruction
        conservative_scheme = ConservativeFiniteDifference(
            self.grid, self.metric, self.boundary_conditions, order=self.weno_order
        )
        return conservative_scheme.compute_divergence(tensor_field, component_indices)


# Factory function for creating finite difference schemes
def create_finite_difference_solver(
    scheme_type: str,
    grid: SpacetimeGrid,
    metric: MetricBase,
    **kwargs: Any,
) -> FiniteDifferenceScheme:
    """
    Factory function to create finite difference schemes.

    Args:
        scheme_type: Type of scheme ('conservative', 'upwind', 'weno')
        grid: Spacetime grid
        metric: Spacetime metric
        **kwargs: Additional scheme-specific parameters

    Returns:
        Configured finite difference scheme

    Raises:
        ValueError: If scheme_type is not recognized
    """
    if scheme_type.lower() == "conservative":
        return ConservativeFiniteDifference(grid, metric, **kwargs)
    elif scheme_type.lower() == "upwind":
        return UpwindFiniteDifference(grid, metric, **kwargs)
    elif scheme_type.lower() == "weno":
        return WENOFiniteDifference(grid, metric, **kwargs)
    else:
        raise ValueError(
            f"Unknown finite difference scheme: {scheme_type}. "
            f"Available types: 'conservative', 'upwind', 'weno'"
        )
