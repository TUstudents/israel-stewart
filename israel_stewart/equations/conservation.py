"""
Energy-momentum and particle number conservation laws for Israel-Stewart hydrodynamics.

This module implements the conservation equations:
- Energy-momentum conservation: ∂_μ T^μν = 0
- Particle number conservation: ∂_μ N^μ = 0

The stress-energy tensor includes perfect fluid, bulk viscosity, shear stress,
and heat flux contributions according to the Israel-Stewart formalism.
"""

import warnings
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..core.derivatives import CovariantDerivative
from ..core.fields import ISFieldConfiguration
from ..core.performance import monitor_performance

# Import core tensor framework
from ..core.tensor_utils import optimized_einsum

# Forward references
if TYPE_CHECKING:
    # from .coefficients import TransportCoefficients  # TODO: Implement this class
    pass


class ConservationLaws:
    """
    Implements ∂_μ T^μν = 0 and ∂_μ N^μ = 0

    Provides methods to construct the full Israel-Stewart stress-energy tensor
    and compute conservation law equations for hydrodynamic evolution.
    """

    def __init__(
        self,
        fields: ISFieldConfiguration,
        coefficients: Any = None,  # TODO: Replace with TransportCoefficients
    ):
        """
        Initialize conservation laws.

        Args:
            fields: ISFieldConfiguration containing all hydrodynamic variables
            coefficients: Transport coefficients (can be None for now)
        """
        self.fields = fields
        self.coeffs = coefficients

        # Initialize covariant derivative operator
        if self.fields.grid.metric is not None:
            self.covariant_derivative = CovariantDerivative(self.fields.grid.metric)
        else:
            # Use Minkowski metric for covariant derivatives
            from ..core.metrics import MinkowskiMetric

            minkowski = MinkowskiMetric()
            self.covariant_derivative = CovariantDerivative(minkowski)

    @monitor_performance("stress_energy_tensor")
    def stress_energy_tensor(self) -> np.ndarray:
        """
        Construct T^μν including all Israel-Stewart corrections:
        T^μν = ρu^μu^ν + (p+Π)Δ^μν + π^μν + q^μu^ν + q^νu^μ

        Returns:
            Stress-energy tensor with shape (*grid.shape, 4, 4)
        """
        f = self.fields
        grid_shape = f.grid.shape

        # Initialize total stress-energy tensor
        T_total = np.zeros((*grid_shape, 4, 4))

        # Perfect fluid part: ρu^μu^ν
        T_perfect = optimized_einsum("...,...i,...j->...ij", f.rho, f.u_mu, f.u_mu)

        # Pressure term with projector Δ^μν = g^μν + u^μu^ν/c²
        Delta = self._spatial_projector()
        pressure_total = f.pressure + f.Pi  # p + bulk viscosity Π
        T_pressure = optimized_einsum("...,...ij->...ij", pressure_total, Delta)

        # Shear stress contribution π^μν
        T_shear = f.pi_munu.copy()

        # Heat flux contribution: q^μu^ν + q^νu^μ (symmetric)
        T_heat_1 = optimized_einsum("...i,...j->...ij", f.q_mu, f.u_mu)
        T_heat_2 = optimized_einsum("...j,...i->...ij", f.q_mu, f.u_mu)
        T_heat = T_heat_1 + T_heat_2

        # Combine all contributions
        T_total = T_perfect + T_pressure + T_shear + T_heat

        result: np.ndarray = T_total
        return result

    @monitor_performance("divergence_T")
    def divergence_T(self) -> np.ndarray:
        """
        Compute ∂_μ T^μν using covariant derivatives.

        For each ν, computes ∇_μ T^μν = ∂_μ T^μν + Γ^μ_μλ T^λν + Γ^ν_μλ T^μλ

        Returns:
            Divergence with shape (*grid.shape, 4) - one component for each ν
        """
        T = self.stress_energy_tensor()
        grid_shape = self.fields.grid.shape
        div_T = np.zeros((*grid_shape, 4))

        # Get coordinate arrays for numerical derivatives
        coords = self._get_coordinate_arrays()

        # For each component ν of the conservation equation
        for nu in range(4):
            div_component = np.zeros_like(T[..., 0, 0])

            # Sum over contracted index μ: ∂_μ T^μν
            for mu in range(4):
                # Extract T^μν component
                T_mu_nu = T[..., mu, nu]

                # Compute partial derivative ∂_μ T^μν (using FFT for periodic spatial derivatives)
                partial_deriv = self._efficient_partial_derivative(T_mu_nu, mu, coords)
                div_component += partial_deriv

                # Add Christoffel symbol corrections if metric is not Minkowski
                try:
                    christoffel = self.covariant_derivative.christoffel_symbols

                    # Connection term: Γ^μ_μλ T^λν
                    for lam in range(4):
                        connection_1 = christoffel[mu, mu, lam] * T[..., lam, nu]
                        div_component += connection_1

                    # Connection term: Γ^ν_μλ T^μλ
                    for lam in range(4):
                        connection_2 = christoffel[nu, mu, lam] * T[..., mu, lam]
                        div_component += connection_2
                except (TypeError, AttributeError):
                    # Skip Christoffel corrections for Minkowski metric
                    pass

            div_T[..., nu] = div_component

        return div_T

    @monitor_performance("evolution_equations")
    def evolution_equations(self) -> dict[str, np.ndarray]:
        """
        Return RHS of evolution equations from conservation laws.

        From ∂_μ T^μν = 0, extract:
        - ∂_t ρ = -∂_i T^0i  (energy conservation, ν=0)
        - ∂_t (ρu^j) = -∂_i T^ij  (momentum conservation, ν=1,2,3)

        Returns:
            Dictionary with evolution equations:
            - 'drho_dt': Energy density time derivative
            - 'dmom_dt': Momentum density time derivatives (3-vector)
        """
        div_T = self.divergence_T()

        # Energy conservation: ∂_t ρ = -∇·T^0i = -div_T[0]
        drho_dt = -div_T[..., 0]

        # Momentum conservation: ∂_t (ρu^j) = -∇·T^ij = -div_T[j] for j=1,2,3
        dmom_dt = -div_T[..., 1:4]

        return {"drho_dt": drho_dt, "dmom_dt": dmom_dt}

    def _spatial_projector(self) -> np.ndarray:
        """
        Compute spatial projector Δ^μν = g^μν + u^μu^ν/c².

        For Minkowski metric: Δ^μν = diag(-1,1,1,1) + u^μu^ν

        Returns:
            Projector tensor with shape (*grid.shape, 4, 4)
        """
        grid_shape = self.fields.grid.shape
        u = self.fields.u_mu

        # Get metric tensor (inverse)
        if self.fields.grid.metric is None:
            # Minkowski metric: g^μν = diag(-1, 1, 1, 1)
            g_inv = np.zeros((*grid_shape, 4, 4))
            g_inv[..., 0, 0] = -1.0
            g_inv[..., 1, 1] = 1.0
            g_inv[..., 2, 2] = 1.0
            g_inv[..., 3, 3] = 1.0
        else:
            # General metric
            metric_components = self.fields.grid.metric.inverse
            broadcasted = np.broadcast_to(metric_components, (*grid_shape, 4, 4))
            g_inv = broadcasted.copy().astype(np.float64).reshape((*grid_shape, 4, 4))

        # Four-velocity outer product: u^μu^ν
        u_outer = optimized_einsum("...i,...j->...ij", u, u)

        # Spatial projector: Δ^μν = g^μν + u^μu^ν (note sign convention)
        Delta = g_inv + u_outer

        return Delta  # type: ignore[no-any-return]

    def _get_coordinate_arrays(self) -> list:
        """
        Get coordinate arrays for numerical derivatives.

        Returns:
            List of coordinate arrays [t, x, y, z]
        """
        grid = self.fields.grid

        # For SpacetimeGrid, use the coordinates attribute
        if hasattr(grid, "coordinates") and isinstance(grid.coordinates, dict):
            # Extract coordinate arrays in order [t, x, y, z]
            coord_names = grid.coordinate_names
            return [grid.coordinates[name] for name in coord_names]
        else:
            # Construct coordinate arrays from grid ranges
            t_coords = np.linspace(grid.time_range[0], grid.time_range[1], grid.grid_points[0])

            coord_arrays = [t_coords]
            for i, (x_min, x_max) in enumerate(grid.spatial_ranges):
                x_coords = np.linspace(x_min, x_max, grid.grid_points[i + 1])
                coord_arrays.append(x_coords)

            return coord_arrays

    def _partial_derivative(self, field: np.ndarray, direction: int, coords: list) -> np.ndarray:
        """
        Compute partial derivative ∂_μ field using finite differences.

        Args:
            field: Field to differentiate with shape (*grid.shape,)
            direction: Direction index (0=time, 1,2,3=spatial)
            coords: Coordinate arrays

        Returns:
            Partial derivative ∂_μ field
        """
        if direction >= len(coords):
            raise ValueError(f"Direction {direction} exceeds coordinate dimensions")

        # Use numpy gradient for finite differences
        # np.gradient returns derivatives w.r.t. each axis
        gradients = np.gradient(field, *coords)

        return gradients[direction]  # type: ignore[no-any-return]

    def _spectral_derivative(self, field: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute partial derivative using FFT for periodic boundaries.

        Much faster than numpy.gradient for periodic domains.

        Args:
            field: Field to differentiate with shape (*grid.shape,)
            direction: Direction index (0=time, 1,2,3=spatial)

        Returns:
            Partial derivative ∂_μ field
        """
        # For time derivatives (direction=0), fall back to finite differences
        if direction == 0:
            coords = self._get_coordinate_arrays()
            return self._partial_derivative(field, direction, coords)

        # For spatial derivatives (direction=1,2,3), use FFT
        # Map direction to spatial axis (1->0, 2->1, 3->2)
        spatial_axis = direction - 1

        # Get spatial grid spacing
        grid = self.fields.grid
        if spatial_axis == 0:
            dx = grid.spatial_spacing[0]
            nx = grid.grid_points[1]  # grid_points = (nt, nx, ny, nz)
        elif spatial_axis == 1:
            dx = grid.spatial_spacing[1]
            nx = grid.grid_points[2]
        elif spatial_axis == 2:
            dx = grid.spatial_spacing[2]
            nx = grid.grid_points[3]
        else:
            raise ValueError(f"Invalid spatial direction: {direction}")

        # Compute wavenumbers
        k = 2 * np.pi * np.fft.fftfreq(nx, dx)

        # Determine correct axis for FFT based on field dimensionality
        # If field has 3 dimensions, it's a spatial slice (t, x, y, z) -> (x, y, z)
        # If field has 4 dimensions, it's full spacetime (t, x, y, z)
        if field.ndim == 3:
            # Field is a spatial slice, axis mapping is direct
            fft_axis = spatial_axis
        elif field.ndim == 4:
            # Field is full spacetime, skip time axis
            fft_axis = spatial_axis + 1
        else:
            raise ValueError(f"Unexpected field dimensionality: {field.ndim}")

        # Create wavenumber array with proper broadcasting
        k_shape = [1] * field.ndim
        k_shape[fft_axis] = nx
        k_reshaped = k.reshape(k_shape)

        # FFT -> multiply by ik -> IFFT (use scipy.fft for better performance)
        try:
            import scipy.fft
            field_fft = scipy.fft.fft(field, axis=fft_axis, workers=-1)
            deriv_fft = 1j * k_reshaped * field_fft
            derivative = scipy.fft.ifft(deriv_fft, axis=fft_axis, workers=-1).real
        except ImportError:
            # Fall back to numpy.fft if scipy not available
            field_fft = np.fft.fft(field, axis=fft_axis)
            deriv_fft = 1j * k_reshaped * field_fft
            derivative = np.fft.ifft(deriv_fft, axis=fft_axis).real

        return derivative

    def _efficient_partial_derivative(self, field: np.ndarray, direction: int, coords: list) -> np.ndarray:
        """
        Compute partial derivative with automatic method selection.

        Uses FFT for periodic spatial derivatives, finite differences otherwise.
        """
        # Check if we have periodic boundary conditions and spatial derivative
        grid = self.fields.grid
        if (hasattr(grid, 'boundary_conditions') and
            grid.boundary_conditions == 'periodic' and
            direction > 0):  # Spatial direction
            return self._spectral_derivative(field, direction)
        else:
            return self._partial_derivative(field, direction, coords)

    def _covariant_div(self, tensor_component: np.ndarray, index: int) -> np.ndarray:
        """
        Compute covariant divergence of tensor component.

        Args:
            tensor_component: Component to differentiate
            index: Contracted index

        Returns:
            Covariant divergence
        """
        # Get coordinate arrays
        coords = self._get_coordinate_arrays()

        # Partial derivative
        partial = self._efficient_partial_derivative(tensor_component, index, coords)

        # Add connection terms if not flat spacetime
        try:
            christoffel = self.covariant_derivative.christoffel_symbols
            # Connection correction: Γ^μ_μλ T^λ
            for lam in range(4):
                if np.any(christoffel[index, index, lam] != 0):
                    connection_term = christoffel[index, index, lam] * tensor_component
                    partial += connection_term
        except (TypeError, AttributeError):
            # Skip Christoffel corrections for Minkowski metric
            pass

        return partial

    def particle_number_conservation(self) -> np.ndarray:
        """
        Compute particle number conservation ∂_μ N^μ = ∂_μ (n u^μ) = 0.

        Returns:
            Particle number conservation equation: ∂_t n + ∇·(n v) = 0
        """
        f = self.fields

        # Particle current: N^μ = n u^μ
        N_mu = f.n[..., np.newaxis] * f.u_mu

        # Compute divergence ∂_μ N^μ
        coords = self._get_coordinate_arrays()
        div_N = np.zeros_like(N_mu[..., 0])

        for mu in range(4):
            partial = self._efficient_partial_derivative(N_mu[..., mu], mu, coords)
            div_N += partial

            # Add Christoffel corrections if needed
            try:
                christoffel = self.covariant_derivative.christoffel_symbols
                for lam in range(4):
                    connection = christoffel[mu, mu, lam] * N_mu[..., lam]
                    div_N += connection
            except (TypeError, AttributeError):
                # Skip Christoffel corrections for Minkowski metric
                pass

        return div_N

    def validate_conservation(self, tolerance: float = 1e-10) -> dict[str, bool]:
        """
        Validate that conservation laws are satisfied.

        Args:
            tolerance: Numerical tolerance for conservation check

        Returns:
            Dictionary of validation results
        """
        validation = {}

        # Check energy-momentum conservation
        div_T = self.divergence_T()
        energy_conserved = np.allclose(div_T[..., 0], 0.0, atol=tolerance)
        momentum_conserved = np.allclose(div_T[..., 1:4], 0.0, atol=tolerance)

        validation["energy_momentum_conserved"] = energy_conserved and momentum_conserved

        # Check particle number conservation
        div_N = self.particle_number_conservation()
        particle_conserved = np.allclose(div_N, 0.0, atol=tolerance)
        validation["particle_number_conserved"] = particle_conserved

        # Overall validation
        validation["all_conserved"] = all(validation.values())

        if not validation["all_conserved"]:
            warnings.warn("Conservation laws not satisfied within tolerance", stacklevel=2)

        return validation

    def __str__(self) -> str:
        return f"ConservationLaws(grid_shape={self.fields.grid.shape})"

    def __repr__(self) -> str:
        return f"ConservationLaws(fields={self.fields!r}, " f"coefficients={self.coeffs!r})"
