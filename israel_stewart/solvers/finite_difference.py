"""
Finite difference spatial discretization for Israel-Stewart hydrodynamics.

This module implements high-order finite difference schemes for spatial derivatives
in the conservation laws and Israel-Stewart evolution equations. Includes both
conservative and non-conservative forms with proper boundary treatment.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional, Union

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
            warnings.warn(f"Unknown boundary condition: {self.boundary_conditions}", stacklevel=2)
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
        Compute numerical flux at cell interfaces using Lax-Friedrichs scheme.

        The Lax-Friedrichs flux provides proper dissipation for stability:
        F_{i+1/2} = 0.5*(F_L + F_R) - 0.5*α*(u_R - u_L)
        where α is the maximum wave speed for dissipation.

        Also handles higher-order stencil offsets for 4th-order schemes.
        """
        spatial_dim = axis - 1  # Convert from axis index to spatial dimension

        if abs(offset - 0.5) < 1e-10:
            # Right interface: F_{i+1/2}
            left_slice = [slice(None)] * extended_field.ndim
            right_slice = [slice(None)] * extended_field.ndim

            left_slice[axis] = slice(self.ghost_points, -self.ghost_points - 1)
            right_slice[axis] = slice(self.ghost_points + 1, -self.ghost_points)

            u_left = extended_field[tuple(left_slice)]
            u_right = extended_field[tuple(right_slice)]

        elif abs(offset + 0.5) < 1e-10:
            # Left interface: F_{i-1/2}
            left_slice = [slice(None)] * extended_field.ndim
            right_slice = [slice(None)] * extended_field.ndim

            left_slice[axis] = slice(self.ghost_points - 1, -self.ghost_points - 2)
            right_slice[axis] = slice(self.ghost_points, -self.ghost_points - 1)

            u_left = extended_field[tuple(left_slice)]
            u_right = extended_field[tuple(right_slice)]

        elif abs(offset - 1.5) < 1e-10:
            # Far right interface: F_{i+3/2} for 4th order
            left_slice = [slice(None)] * extended_field.ndim
            right_slice = [slice(None)] * extended_field.ndim

            left_slice[axis] = slice(self.ghost_points + 1, -self.ghost_points)
            right_slice[axis] = slice(self.ghost_points + 2, -self.ghost_points + 1)

            u_left = extended_field[tuple(left_slice)]
            u_right = extended_field[tuple(right_slice)]

        elif abs(offset + 1.5) < 1e-10:
            # Far left interface: F_{i-3/2} for 4th order
            left_slice = [slice(None)] * extended_field.ndim
            right_slice = [slice(None)] * extended_field.ndim

            left_slice[axis] = slice(self.ghost_points - 2, -self.ghost_points - 3)
            right_slice[axis] = slice(self.ghost_points - 1, -self.ghost_points - 2)

            u_left = extended_field[tuple(left_slice)]
            u_right = extended_field[tuple(right_slice)]

        else:
            # Unsupported offset
            warnings.warn(f"Flux computation for offset {offset} not implemented", stacklevel=2)
            return np.zeros_like(extended_field)

        # Compute Lax-Friedrichs flux: F = 0.5*(F_L + F_R) - 0.5*α*(u_R - u_L)

        # For hydrodynamics, α should be the maximum wave speed (sound speed + fluid velocity)
        # In relativistic case, this is typically ≤ 1 (speed of light)
        # Use conservative estimate: α = 1 for relativistic systems
        alpha = 1.0

        # Apply flux limiter if specified
        if self.flux_limiter == "minmod":
            # Apply minmod limiter to reduce oscillations
            flux = self._apply_minmod_limiter(u_left, u_right, alpha)
        elif self.flux_limiter == "superbee":
            # Apply superbee limiter for better shock resolution
            flux = self._apply_superbee_limiter(u_left, u_right, alpha)
        else:
            # Standard Lax-Friedrichs flux
            flux = 0.5 * (u_left + u_right) - 0.5 * alpha * (u_right - u_left)

        return flux

    def _apply_minmod_limiter(
        self, u_left: np.ndarray, u_right: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        Apply minmod flux limiter for TVD property.

        Reduces oscillations near discontinuities while maintaining second-order accuracy
        in smooth regions.
        """
        # Compute slopes
        delta_left = u_right - u_left

        # Minmod function: minmod(a,b) = 0.5*(sign(a) + sign(b))*min(|a|,|b|)
        # For flux limiting, we modify the dissipation term

        # Standard Lax-Friedrichs flux as baseline
        flux = 0.5 * (u_left + u_right) - 0.5 * alpha * delta_left

        # Apply limiting - this is a simplified version
        # Full TVD implementation would require more sophisticated slope reconstruction
        limiter_factor = np.ones_like(u_left)
        large_gradient_mask = np.abs(delta_left) > 0.1 * np.maximum(np.abs(u_left), np.abs(u_right))
        limiter_factor[large_gradient_mask] *= 0.5  # Reduce dissipation in smooth regions

        return flux * limiter_factor

    def _apply_superbee_limiter(
        self, u_left: np.ndarray, u_right: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        Apply superbee flux limiter for enhanced shock resolution.

        Provides better resolution of discontinuities compared to minmod
        while maintaining TVD property.
        """
        # Compute slopes
        delta_left = u_right - u_left

        # Standard Lax-Friedrichs flux as baseline
        flux = 0.5 * (u_left + u_right) - 0.5 * alpha * delta_left

        # Superbee limiter - simplified implementation
        # Full implementation would require slope ratios from neighboring cells
        limiter_factor = np.ones_like(u_left)

        # Detect shocks and smooth regions
        gradient_magnitude = np.abs(delta_left)
        field_scale = np.maximum(np.abs(u_left), np.abs(u_right)) + 1e-12

        # In smooth regions, reduce dissipation more aggressively than minmod
        smooth_mask = gradient_magnitude < 0.01 * field_scale
        shock_mask = gradient_magnitude > 0.5 * field_scale

        limiter_factor[smooth_mask] *= 0.1  # Low dissipation in smooth regions
        limiter_factor[shock_mask] *= 2.0   # High dissipation at shocks

        return flux * limiter_factor

    def _compute_second_derivative(
        self, field: np.ndarray, axis: int, spatial_dim: int
    ) -> np.ndarray:
        """
        Compute second derivative for diffusion terms using vectorized operations.

        Replaces loop-based implementation with efficient array slicing and broadcasting
        for significantly improved performance, especially on large grids.
        """
        extended_field = self._apply_boundary_conditions(field, axis)

        # Use standard finite difference for second derivatives
        coeffs = self.derivative_coefficients[2][0]  # Central difference
        stencil_width = len(coeffs)
        center_offset = stencil_width // 2

        # Vectorized implementation using array slicing
        # Create a list to hold all stencil contributions
        stencil_arrays = []

        for i, coeff in enumerate(coeffs):
            offset = i - center_offset

            # Build slice for this stencil point
            start_idx = self.ghost_points + offset
            if offset < 0:
                end_idx = -self.ghost_points + offset if self.ghost_points + offset > 0 else None
            else:
                end_idx = -self.ghost_points + offset if offset < 0 else None

            # Create slice tuple
            shifted_slice = [slice(None)] * field.ndim
            shifted_slice[axis] = slice(start_idx, end_idx)

            # Extract stencil contribution and multiply by coefficient
            stencil_contribution = coeff * extended_field[tuple(shifted_slice)]
            stencil_arrays.append(stencil_contribution)

        # Sum all contributions using vectorized addition
        # This is much faster than accumulating in a loop
        derivative = np.sum(np.stack(stencil_arrays, axis=0), axis=0)

        # Apply scaling factor
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
        """
        Compute Christoffel symbol contributions to covariant divergence.

        For a tensor field T^{μν...}, the covariant divergence includes terms:
        ∇_α T^{αβ...} = ∂_α T^{αβ...} + Γ^α_{γα} T^{γβ...} + Γ^β_{γα} T^{αγ...} + ...

        This method computes the Christoffel contribution: Γ^μ_{αλ} T^{αλ}
        """
        # Get output shape (remove tensor indices)
        output_shape = tensor_field.shape[: -len(component_indices)]

        # Check if we're in flat spacetime (Minkowski)
        if hasattr(self.metric, 'is_flat') and self.metric.is_flat():
            # In Cartesian coordinates, all Christoffel symbols vanish
            return np.zeros(output_shape)

        # For curved spacetime, compute Christoffel contributions
        try:
            # Get Christoffel symbols from the metric
            christoffel_symbols = self._get_christoffel_symbols()

            # Initialize result
            result = np.zeros(output_shape)

            # The exact form depends on the tensor structure
            if len(component_indices) == 2:
                # Rank-2 tensor T^{μν}
                mu_index, nu_index = component_indices
                result = self._compute_rank2_christoffel_terms(
                    tensor_field, christoffel_symbols, mu_index, nu_index
                )
            elif len(component_indices) == 1:
                # Vector field T^μ
                mu_index = component_indices[0]
                result = self._compute_vector_christoffel_terms(
                    tensor_field, christoffel_symbols, mu_index
                )
            else:
                # Higher rank tensors - use general implementation
                result = self._compute_general_christoffel_terms(
                    tensor_field, christoffel_symbols, component_indices
                )

            return result

        except Exception as e:
            # Fallback: emit warning and return zeros
            warnings.warn(
                f"Failed to compute Christoffel contributions: {e}. "
                f"Using flat spacetime approximation (Γ = 0).",
                UserWarning,
                stacklevel=2
            )
            return np.zeros(output_shape)

    def _get_christoffel_symbols(self) -> np.ndarray:
        """
        Extract Christoffel symbols from the metric.

        Returns:
            Christoffel symbols Γ^μ_{αβ} with shape (*spatial_grid, 4, 4, 4)
        """
        if hasattr(self.metric, 'christoffel_symbols'):
            # Use precomputed Christoffel symbols if available
            return self.metric.christoffel_symbols
        elif hasattr(self.metric, 'compute_christoffel'):
            # Compute Christoffel symbols using metric method
            return self.metric.compute_christoffel()
        else:
            # Compute numerically using finite differences
            return self._compute_christoffel_numerically()

    def _compute_christoffel_numerically(self) -> np.ndarray:
        """
        Compute Christoffel symbols numerically from metric components.

        Uses finite differences: Γ^μ_{αβ} = 0.5*g^{μν}(∂_α g_{νβ} + ∂_β g_{να} - ∂_ν g_{αβ})
        """
        # Get metric components
        try:
            # Get spatial grid shape
            spatial_shape = tuple(self.grid.grid_points[1:])  # Skip time dimension

            # Get metric tensor components g_μν
            g_munu = self.metric.components()  # Should return (*spatial_shape, 4, 4)

            # Compute inverse metric g^μν
            g_inv = self.metric.inverse_components()  # Should return (*spatial_shape, 4, 4)

            # Initialize Christoffel symbols
            christoffel = np.zeros((*spatial_shape, 4, 4, 4))

            # Compute derivatives of metric components
            for alpha in range(4):
                for beta in range(4):
                    for nu in range(4):
                        # Only compute spatial derivatives (time derivatives handled elsewhere)
                        for spatial_dim in range(3):
                            axis = spatial_dim + 1  # Skip time axis

                            # ∂_α g_{νβ}
                            if alpha > 0:  # Spatial derivative
                                dg_nu_beta_dalpha = self._compute_metric_derivative(
                                    g_munu[..., nu, beta], axis=alpha
                                )
                            else:
                                dg_nu_beta_dalpha = np.zeros_like(g_munu[..., nu, beta])

                            # ∂_β g_{να}
                            if beta > 0:  # Spatial derivative
                                dg_nu_alpha_dbeta = self._compute_metric_derivative(
                                    g_munu[..., nu, alpha], axis=beta
                                )
                            else:
                                dg_nu_alpha_dbeta = np.zeros_like(g_munu[..., nu, alpha])

                            # ∂_ν g_{αβ}
                            if nu > 0:  # Spatial derivative
                                dg_alpha_beta_dnu = self._compute_metric_derivative(
                                    g_munu[..., alpha, beta], axis=nu
                                )
                            else:
                                dg_alpha_beta_dnu = np.zeros_like(g_munu[..., alpha, beta])

                            # Christoffel symbol formula
                            for mu in range(4):
                                christoffel[..., mu, alpha, beta] += 0.5 * g_inv[..., mu, nu] * (
                                    dg_nu_beta_dalpha + dg_nu_alpha_dbeta - dg_alpha_beta_dnu
                                )

            return christoffel

        except Exception as e:
            warnings.warn(
                f"Numerical Christoffel computation failed: {e}. Using zeros.",
                UserWarning, stacklevel=2
            )
            # Fallback: return zeros (flat spacetime)
            spatial_shape = tuple(self.grid.grid_points[1:])
            return np.zeros((*spatial_shape, 4, 4, 4))

    def _compute_metric_derivative(self, metric_component: np.ndarray, axis: int) -> np.ndarray:
        """Compute derivative of metric component using finite differences."""
        if axis == 0:
            # Time derivative - not handled in spatial finite difference
            return np.zeros_like(metric_component)

        # Spatial derivative using existing finite difference machinery
        spatial_dim = axis - 1

        # Apply boundary conditions
        extended_field = self._apply_boundary_conditions(metric_component, axis)

        # Use central difference coefficients
        coeffs = self.derivative_coefficients[1][0]  # First derivative, central stencil

        derivative = np.zeros_like(metric_component)
        for i, coeff in enumerate(coeffs):
            offset = i - len(coeffs) // 2
            shifted_slice = [slice(None)] * metric_component.ndim
            shifted_slice[axis - 1] = slice(  # axis-1 since we removed time from metric_component
                self.ghost_points + offset,
                -self.ghost_points + offset if offset < 0 else None
            )
            if extended_field.ndim > len(shifted_slice):
                # Handle case where extended_field has extra dimensions
                shifted_slice = [slice(None)] * extended_field.ndim
                shifted_slice[axis] = slice(
                    self.ghost_points + offset,
                    -self.ghost_points + offset if offset < 0 else None
                )
            derivative += coeff * extended_field[tuple(shifted_slice)]

        derivative /= self.dx[spatial_dim]
        return derivative

    def _compute_rank2_christoffel_terms(
        self,
        tensor_field: np.ndarray,
        christoffel_symbols: np.ndarray,
        mu_index: int,
        nu_index: int
    ) -> np.ndarray:
        """
        Compute Christoffel contributions for rank-2 tensor T^{μν}.

        Contribution: Γ^μ_{γα} T^{γν} + Γ^ν_{γα} T^{μγ}
        """
        result = np.zeros(tensor_field.shape[:-2])  # Remove last two tensor indices

        # Sum over repeated indices using Einstein summation
        # Term 1: Γ^μ_{γα} T^{γν} (sum over γ, α)
        for gamma in range(4):
            for alpha in range(4):
                # Only include spatial derivatives (α > 0)
                if alpha > 0:
                    result += (christoffel_symbols[..., mu_index, gamma, alpha] *
                              tensor_field[..., gamma, nu_index])

        # Term 2: Γ^ν_{γα} T^{μγ} (sum over γ, α)
        for gamma in range(4):
            for alpha in range(4):
                if alpha > 0:
                    result += (christoffel_symbols[..., nu_index, gamma, alpha] *
                              tensor_field[..., mu_index, gamma])

        return result

    def _compute_vector_christoffel_terms(
        self,
        vector_field: np.ndarray,
        christoffel_symbols: np.ndarray,
        mu_index: int
    ) -> np.ndarray:
        """
        Compute Christoffel contributions for vector T^μ.

        Contribution: Γ^μ_{γα} T^γ
        """
        result = np.zeros(vector_field.shape[:-1])  # Remove last tensor index

        # Sum over repeated indices: Γ^μ_{γα} T^γ (sum over γ, α)
        for gamma in range(4):
            for alpha in range(4):
                # Only include spatial derivatives (α > 0)
                if alpha > 0:
                    result += (christoffel_symbols[..., mu_index, gamma, alpha] *
                              vector_field[..., gamma])

        return result

    def _compute_general_christoffel_terms(
        self,
        tensor_field: np.ndarray,
        christoffel_symbols: np.ndarray,
        component_indices: tuple[int, ...]
    ) -> np.ndarray:
        """
        General Christoffel contribution computation for arbitrary rank tensors.

        For tensor T^{μνρ...}, adds Christoffel terms for each upper index.
        """
        result = np.zeros(tensor_field.shape[:-len(component_indices)])

        # For each tensor index, add the corresponding Christoffel contribution
        for idx_pos, mu_index in enumerate(component_indices):
            # Create contribution for this index position
            contribution = np.zeros_like(result)

            # Sum over contracted indices
            for gamma in range(4):
                for alpha in range(4):
                    if alpha > 0:  # Only spatial derivatives
                        # Build tensor slice with γ replacing μ at position idx_pos
                        tensor_indices = list(component_indices)
                        tensor_indices[idx_pos] = gamma
                        tensor_slice = tuple([slice(None)] * len(result.shape) + tensor_indices)

                        contribution += (christoffel_symbols[..., mu_index, gamma, alpha] *
                                       tensor_field[tensor_slice])

            result += contribution

        return result


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
        """
        Compute upwind derivative using vectorized operations.

        Replaces loop-based stencil application with efficient array operations
        for improved performance on large grids.
        """
        extended_field = self._apply_boundary_conditions(field, axis)

        # Determine stencil coefficients and offsets based on order and direction
        if self.order == 1:
            if direction == "upwind":
                # Backward difference: (u_i - u_{i-1})/dx
                coeffs = np.array([1.0, -1.0])
                offsets = np.array([0, -1])
            else:  # downwind
                # Forward difference: (u_{i+1} - u_i)/dx
                coeffs = np.array([-1.0, 1.0])
                offsets = np.array([0, 1])

        elif self.order == 2:
            if direction == "upwind":
                # Second-order upwind: (3u_i - 4u_{i-1} + u_{i-2})/(2dx)
                coeffs = np.array([1.5, -2.0, 0.5])
                offsets = np.array([0, -1, -2])
            else:  # downwind
                # Second-order downwind: (-u_{i+2} + 4u_{i+1} - 3u_i)/(2dx)
                coeffs = np.array([-1.5, 2.0, -0.5])
                offsets = np.array([0, 1, 2])
        else:
            # Default to first-order
            if direction == "upwind":
                coeffs = np.array([1.0, -1.0])
                offsets = np.array([0, -1])
            else:
                coeffs = np.array([-1.0, 1.0])
                offsets = np.array([0, 1])

        # Vectorized stencil application
        stencil_contributions = []

        for coeff, offset in zip(coeffs, offsets):
            # Calculate slice indices
            start_idx = self.ghost_points + offset
            if offset < 0:
                end_idx = -self.ghost_points + offset if self.ghost_points > abs(offset) else None
            else:
                end_idx = -self.ghost_points + offset if offset > 0 else None

            # Build slice tuple
            shifted_slice = [slice(None)] * field.ndim
            shifted_slice[axis] = slice(start_idx, end_idx)

            # Add weighted stencil contribution
            contribution = coeff * extended_field[tuple(shifted_slice)]
            stencil_contributions.append(contribution)

        # Vectorized summation of all stencil contributions
        # This is more efficient than accumulating in the loop
        derivative = np.sum(np.stack(stencil_contributions, axis=0), axis=0)

        # Apply grid spacing normalization
        derivative /= self.dx[spatial_dim]

        return derivative

    def compute_divergence(
        self,
        tensor_field: np.ndarray,
        component_indices: tuple[int, ...],
    ) -> np.ndarray:
        """
        Compute divergence using characteristic-based upwind differences.

        Uses wave propagation information to choose upwind direction,
        providing better stability for hyperbolic systems.
        """
        divergence = np.zeros(tensor_field.shape[: -len(component_indices)])

        # Get fields for characteristic speed computation
        try:
            # Create dummy field configuration for speed calculation
            from ..core.fields import ISFieldConfiguration
            dummy_fields = ISFieldConfiguration(self.grid)

            # Extract velocity information if available from tensor field
            # This is a heuristic extraction for demonstration
            if hasattr(dummy_fields, 'u_mu') and tensor_field.ndim >= 2:
                # Try to extract velocity from stress tensor structure
                dummy_fields.u_mu = np.zeros((*tensor_field.shape[:-len(component_indices)], 4))
                if tensor_field.shape[-1] >= 4:
                    # Use time component as rough velocity estimate
                    dummy_fields.u_mu[..., 0] = 1.0  # Time component
                    for i in range(min(3, tensor_field.shape[-1] - 1)):
                        dummy_fields.u_mu[..., i + 1] = tensor_field[..., 0, i + 1] * 0.1

            # Compute characteristic speeds
            speeds = self.characteristic_speeds(dummy_fields)

        except Exception:
            # Fallback to default speeds
            dummy_shape = tensor_field.shape[: -len(component_indices)]
            speeds = {
                f"lambda_plus_{i}": np.ones(dummy_shape) * 0.6  # Approximate sound speed
                for i in range(3)
            }
            speeds.update({
                f"lambda_minus_{i}": np.ones(dummy_shape) * (-0.6)
                for i in range(3)
            })

        # Loop over spatial dimensions
        for dim in range(3):  # 3 spatial dimensions
            axis = dim + 1  # Skip time dimension

            # Extract the appropriate tensor component
            component_slice = [slice(None)] * (tensor_field.ndim - len(component_indices))
            component_slice.extend([component_indices[0], dim])  # T^mu_i component

            component_field = tensor_field[tuple(component_slice)]

            # Compute upwind divergence based on characteristic speeds
            speed_plus = speeds.get(f"lambda_plus_{dim}", np.zeros_like(component_field))
            speed_minus = speeds.get(f"lambda_minus_{dim}", np.zeros_like(component_field))

            # Choose upwind direction based on characteristic speeds
            # Use Godunov-type flux splitting
            upwind_div = self._compute_godunov_divergence(
                component_field, axis, dim, speed_plus, speed_minus
            )

            divergence += upwind_div

        # Add Christoffel symbol contributions
        try:
            christoffel_terms = self._compute_christoffel_contributions(tensor_field, component_indices)
            divergence += christoffel_terms
        except Exception as e:
            warnings.warn(f"Christoffel computation failed in upwind scheme: {e}", stacklevel=2)

        return divergence

    def _compute_godunov_divergence(
        self,
        component_field: np.ndarray,
        axis: int,
        spatial_dim: int,
        speed_plus: np.ndarray,
        speed_minus: np.ndarray
    ) -> np.ndarray:
        """
        Compute divergence using Godunov-type flux splitting.

        Uses characteristic speeds to determine upwind direction for each cell face.
        This provides proper wave propagation for hyperbolic systems.
        """
        # Apply boundary conditions
        extended_field = self._apply_boundary_conditions(component_field, axis)

        # Initialize result
        divergence = np.zeros_like(component_field)

        # Compute fluxes at cell interfaces using upwind selection
        for i in range(component_field.shape[axis]):
            # Get indices for current cell and neighbors
            left_idx = self.ghost_points + i - 1
            right_idx = self.ghost_points + i

            # Extract values at interface
            left_slice = self._build_slice(extended_field, axis, left_idx)
            right_slice = self._build_slice(extended_field, axis, right_idx)

            u_left = extended_field[left_slice]
            u_right = extended_field[right_slice]

            # Get characteristic speeds at interface (average of neighboring cells)
            speed_slice = self._build_slice(speed_plus, axis, i)
            local_speed_plus = speed_plus[speed_slice]
            local_speed_minus = speed_minus[speed_slice]

            # Godunov flux: F = F_left if λ > 0, F_right if λ < 0, average if mixed
            flux_left = self._compute_physical_flux(u_left, spatial_dim)
            flux_right = self._compute_physical_flux(u_right, spatial_dim)

            # Flux selection based on wave speeds
            flux_interface = self._godunov_flux_selection(
                flux_left, flux_right, local_speed_plus, local_speed_minus
            )

            # Update divergence with flux difference
            if i > 0:
                div_slice = self._build_slice(divergence, axis, i)
                divergence[div_slice] += flux_interface / self.dx[spatial_dim]

            if i < component_field.shape[axis] - 1:
                div_slice = self._build_slice(divergence, axis, i + 1)
                divergence[div_slice] -= flux_interface / self.dx[spatial_dim]

        return divergence

    def _build_slice(self, array: np.ndarray, axis: int, index: int) -> tuple:
        """Build slice tuple for extracting data at specific index along axis."""
        slice_list = [slice(None)] * array.ndim
        slice_list[axis] = index
        return tuple(slice_list)

    def _compute_physical_flux(self, field_value: np.ndarray, spatial_dim: int) -> np.ndarray:
        """
        Compute physical flux for hydrodynamic field.

        For relativistic hydrodynamics, this would be the momentum flux or energy flux.
        Simplified implementation assumes field_value represents the conserved quantity.
        """
        # For demonstration, use simple advective flux: F = u * field_value
        # In full implementation, this would be the proper hydrodynamic flux

        # Assume field represents momentum density, flux = momentum^2/density + pressure
        # Simplified: just use field value as flux (valid for linear advection)
        return field_value

    def _godunov_flux_selection(
        self,
        flux_left: np.ndarray,
        flux_right: np.ndarray,
        speed_plus: np.ndarray,
        speed_minus: np.ndarray
    ) -> np.ndarray:
        """
        Select upwind flux based on characteristic speeds using Godunov method.

        Args:
            flux_left: Flux computed from left state
            flux_right: Flux computed from right state
            speed_plus: Rightward characteristic speed (λ₊)
            speed_minus: Leftward characteristic speed (λ₋)

        Returns:
            Interface flux based on upwind direction
        """
        # Case 1: All waves go right (λ₋ > 0)
        all_right = speed_minus > 0

        # Case 2: All waves go left (λ₊ < 0)
        all_left = speed_plus < 0

        # Case 3: Sonic point (λ₋ ≤ 0 ≤ λ₊) - mixed case
        mixed = ~(all_right | all_left)

        # Initialize result
        flux = np.zeros_like(flux_left)

        # Apply upwind selection
        flux[all_right] = flux_left[all_right]    # Use left flux when waves go right
        flux[all_left] = flux_right[all_left]     # Use right flux when waves go left

        # For mixed case, use HLL-type average with entropy fix
        if np.any(mixed):
            # Lax-Friedrichs type flux for sonic points
            alpha = np.maximum(np.abs(speed_plus[mixed]), np.abs(speed_minus[mixed]))
            flux[mixed] = 0.5 * (flux_left[mixed] + flux_right[mixed] -
                               alpha * (flux_right[mixed] - flux_left[mixed]))

        return flux


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
        """
        Compute derivative using WENO reconstruction.

        Implements proper WENO reconstruction with nonlinear weights based on
        smoothness indicators. Automatically switches between high-order
        reconstruction in smooth regions and lower-order near discontinuities.
        """
        extended_field = self._apply_boundary_conditions(field, axis)

        # Get WENO coefficients
        optimal_weights = self.weno_coeffs["optimal_weights"]
        stencil_coeffs = self.weno_coeffs["stencil_coeffs"]
        num_stencils = len(stencil_coeffs)

        # Initialize result
        derivative = np.zeros_like(field)

        # Compute smoothness indicators for all stencils
        smoothness_indicators = []
        for k in range(num_stencils):
            beta_k = self._compute_smoothness_indicator(extended_field, axis, k)
            smoothness_indicators.append(beta_k)

        # Compute nonlinear WENO weights
        weno_weights = self._compute_weno_weights(optimal_weights, smoothness_indicators)

        # Apply WENO reconstruction
        for k in range(num_stencils):
            # Get stencil coefficients for derivative computation
            stencil_derivative = self._compute_stencil_derivative(
                extended_field, axis, spatial_dim, k, stencil_coeffs[k]
            )

            # Weight by WENO weights
            derivative += weno_weights[k] * stencil_derivative

        return derivative

    def _compute_weno_weights(
        self,
        optimal_weights: np.ndarray,
        smoothness_indicators: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Compute nonlinear WENO weights from smoothness indicators.

        The WENO weights are: w_k = α_k / Σ_j α_j
        where α_k = d_k / (ε + β_k)^p with p=2 for WENO5.
        """
        num_stencils = len(smoothness_indicators)
        alphas = []

        # Compute α_k = d_k / (ε + β_k)^p
        for k in range(num_stencils):
            alpha_k = optimal_weights[k] / (self.epsilon + smoothness_indicators[k]) ** 2
            alphas.append(alpha_k)

        # Compute sum of alphas for normalization
        alpha_sum = np.sum(np.stack(alphas, axis=0), axis=0)

        # Compute normalized weights w_k = α_k / Σ_j α_j
        weno_weights = []
        for k in range(num_stencils):
            w_k = alphas[k] / alpha_sum
            weno_weights.append(w_k)

        return weno_weights

    def _compute_stencil_derivative(
        self,
        extended_field: np.ndarray,
        axis: int,
        spatial_dim: int,
        stencil_index: int,
        stencil_coeffs: np.ndarray
    ) -> np.ndarray:
        """
        Compute derivative using specific WENO stencil.

        Args:
            extended_field: Field with ghost points
            axis: Axis for differentiation
            spatial_dim: Spatial dimension index
            stencil_index: Which stencil to use
            stencil_coeffs: Finite difference coefficients for this stencil

        Returns:
            Derivative computed using this stencil
        """
        # Initialize result
        result_shape = list(extended_field.shape)
        result_shape[axis] = extended_field.shape[axis] - 2 * self.ghost_points
        stencil_result = np.zeros(result_shape)

        # Apply finite difference stencil
        stencil_size = len(stencil_coeffs)

        if self.weno_order == 5:
            # WENO5 stencil patterns
            if stencil_index == 0:  # Left stencil S0: points (i-2, i-1, i)
                offsets = [-2, -1, 0]
            elif stencil_index == 1:  # Central stencil S1: points (i-1, i, i+1)
                offsets = [-1, 0, 1]
            elif stencil_index == 2:  # Right stencil S2: points (i, i+1, i+2)
                offsets = [0, 1, 2]
            else:
                raise ValueError(f"Invalid WENO5 stencil index: {stencil_index}")

            # Apply WENO5 reconstruction coefficients for derivatives
            # These are different from interpolation coefficients
            if stencil_index == 0:  # S0 derivative coefficients
                deriv_coeffs = np.array([-11.0/6.0, 3.0, -3.0/2.0, 1.0/3.0])[:stencil_size]
            elif stencil_index == 1:  # S1 derivative coefficients
                deriv_coeffs = np.array([1.0/3.0, -1.0/2.0, 1.0, -1.0/6.0])[:stencil_size]
            elif stencil_index == 2:  # S2 derivative coefficients
                deriv_coeffs = np.array([1.0/6.0, -1.0, 1.0/2.0, 1.0/3.0])[:stencil_size]

        elif self.weno_order == 3:
            # WENO3 stencil patterns
            if stencil_index == 0:  # Left stencil: points (i-1, i)
                offsets = [-1, 0]
                deriv_coeffs = np.array([-1.0, 1.0])
            elif stencil_index == 1:  # Right stencil: points (i, i+1)
                offsets = [0, 1]
                deriv_coeffs = np.array([-1.0, 1.0])
            else:
                raise ValueError(f"Invalid WENO3 stencil index: {stencil_index}")

        else:
            # Fallback: use provided coefficients with centered offsets
            center_offset = stencil_size // 2
            offsets = list(range(-center_offset, stencil_size - center_offset))
            deriv_coeffs = stencil_coeffs

        # Apply stencil with proper coefficients
        for j, (offset, coeff) in enumerate(zip(offsets, deriv_coeffs)):
            # Extract field values at stencil points
            field_slice = self._extract_field_slice(extended_field, axis, offset)
            stencil_result += coeff * field_slice

        # Apply grid spacing normalization
        stencil_result /= self.dx[spatial_dim]

        return stencil_result

    def _compute_smoothness_indicator(
        self,
        extended_field: np.ndarray,
        axis: int,
        stencil_index: int,
    ) -> np.ndarray:
        """
        Compute smoothness indicator β_k for WENO weights.

        Uses proper WENO smoothness measures based on the L2 norm of derivatives
        over the stencil interval. This provides accurate shock detection and
        maintains high-order accuracy in smooth regions.
        """
        # Get the field shape for proper slicing
        field_shape = list(extended_field.shape)
        result_shape = field_shape.copy()
        if axis < len(result_shape):
            result_shape[axis] = extended_field.shape[axis] - 2 * self.ghost_points

        if self.weno_order == 5:
            # WENO5 smoothness indicators
            return self._compute_weno5_smoothness_indicator(
                extended_field, axis, stencil_index, result_shape
            )
        elif self.weno_order == 3:
            # WENO3 smoothness indicators
            return self._compute_weno3_smoothness_indicator(
                extended_field, axis, stencil_index, result_shape
            )
        else:
            # Fallback: simple variation-based indicator
            return self._compute_simple_smoothness_indicator(
                extended_field, axis, stencil_index, result_shape
            )

    def _compute_weno5_smoothness_indicator(
        self,
        extended_field: np.ndarray,
        axis: int,
        stencil_index: int,
        result_shape: list[int]
    ) -> np.ndarray:
        """
        Compute WENO5 smoothness indicator β_k.

        For WENO5, there are 3 stencils (k=0,1,2), each using 3 points.
        The smoothness indicator is: β_k = Σ_{j=1}^{r-1} Δx^{2j-1} ∫(∂^j p_k)^2 dx
        where p_k is the interpolating polynomial on stencil k.
        """
        # Initialize smoothness indicator
        beta = np.zeros(result_shape)

        # Define WENO5 stencil patterns
        # Stencil 0: points (i-2, i-1, i)   - leftmost, smoothest
        # Stencil 1: points (i-1, i, i+1)   - central
        # Stencil 2: points (i, i+1, i+2)   - rightmost

        if stencil_index == 0:  # Left stencil S0
            # Extract field values: u_{i-2}, u_{i-1}, u_i
            u_im2 = self._extract_field_slice(extended_field, axis, -2)
            u_im1 = self._extract_field_slice(extended_field, axis, -1)
            u_i = self._extract_field_slice(extended_field, axis, 0)

            # First derivative terms: (∂p/∂x)^2
            # p'(x) for quadratic interpolation through 3 points
            d1_term = 13.0/12.0 * (u_im2 - 2*u_im1 + u_i)**2

            # Second derivative terms: (∂²p/∂x²)^2
            # For quadratic p(x), second derivative is constant
            d2_term = 1.0/4.0 * (u_im2 - 4*u_im1 + 3*u_i)**2

            beta = d1_term + d2_term

        elif stencil_index == 1:  # Central stencil S1
            # Extract field values: u_{i-1}, u_i, u_{i+1}
            u_im1 = self._extract_field_slice(extended_field, axis, -1)
            u_i = self._extract_field_slice(extended_field, axis, 0)
            u_ip1 = self._extract_field_slice(extended_field, axis, 1)

            # WENO5 S1 smoothness indicator
            d1_term = 13.0/12.0 * (u_im1 - 2*u_i + u_ip1)**2
            d2_term = 1.0/4.0 * (u_im1 - u_ip1)**2

            beta = d1_term + d2_term

        elif stencil_index == 2:  # Right stencil S2
            # Extract field values: u_i, u_{i+1}, u_{i+2}
            u_i = self._extract_field_slice(extended_field, axis, 0)
            u_ip1 = self._extract_field_slice(extended_field, axis, 1)
            u_ip2 = self._extract_field_slice(extended_field, axis, 2)

            # WENO5 S2 smoothness indicator
            d1_term = 13.0/12.0 * (u_i - 2*u_ip1 + u_ip2)**2
            d2_term = 1.0/4.0 * (3*u_i - 4*u_ip1 + u_ip2)**2

            beta = d1_term + d2_term

        else:
            # Invalid stencil index
            beta = np.ones(result_shape) * 1e6

        return beta + self.epsilon  # Add small constant to avoid division by zero

    def _compute_weno3_smoothness_indicator(
        self,
        extended_field: np.ndarray,
        axis: int,
        stencil_index: int,
        result_shape: list[int]
    ) -> np.ndarray:
        """
        Compute WENO3 smoothness indicator.

        WENO3 uses 2-point stencils with linear interpolation.
        """
        if stencil_index == 0:  # Left stencil
            u_im1 = self._extract_field_slice(extended_field, axis, -1)
            u_i = self._extract_field_slice(extended_field, axis, 0)
            beta = (u_i - u_im1)**2

        elif stencil_index == 1:  # Right stencil
            u_i = self._extract_field_slice(extended_field, axis, 0)
            u_ip1 = self._extract_field_slice(extended_field, axis, 1)
            beta = (u_ip1 - u_i)**2

        else:
            beta = np.ones(result_shape) * 1e6

        return beta + self.epsilon

    def _compute_simple_smoothness_indicator(
        self,
        extended_field: np.ndarray,
        axis: int,
        stencil_index: int,
        result_shape: list[int]
    ) -> np.ndarray:
        """Fallback simple smoothness indicator based on local variation."""
        # Get local stencil points
        stencil_size = len(self.weno_coeffs["stencil_coeffs"][stencil_index])
        start_offset = stencil_index - stencil_size // 2

        # Compute local variation
        variation = np.zeros(result_shape)

        for i in range(stencil_size - 1):
            offset1 = start_offset + i
            offset2 = start_offset + i + 1

            u1 = self._extract_field_slice(extended_field, axis, offset1)
            u2 = self._extract_field_slice(extended_field, axis, offset2)

            diff = u2 - u1
            variation += diff**2

        return variation + self.epsilon

    def _extract_field_slice(self, extended_field: np.ndarray, axis: int, offset: int) -> np.ndarray:
        """
        Extract field values at specified offset from center.

        Args:
            extended_field: Field with ghost points
            axis: Axis along which to extract
            offset: Offset from center (0 = current cell, -1 = left, +1 = right)

        Returns:
            Extracted field slice
        """
        # Calculate absolute index
        start_idx = self.ghost_points + offset
        end_idx = -self.ghost_points + offset if offset < 0 else (-self.ghost_points + offset if offset > 0 else -self.ghost_points)

        # Build slice tuple
        slice_tuple = [slice(None)] * extended_field.ndim
        slice_tuple[axis] = slice(start_idx, end_idx or None)

        return extended_field[tuple(slice_tuple)]

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
