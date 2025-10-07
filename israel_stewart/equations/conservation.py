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
        spectral_solver: Optional[Any] = None,  # Allow passing in a spectral solver
    ):
        """
        Initialize conservation laws.

        Args:
            fields: ISFieldConfiguration containing all hydrodynamic variables
            coefficients: Transport coefficients (can be None for now)
            spectral_solver: Optional spectral solver for computing derivatives.
        """
        self.fields = fields
        self.coeffs = coefficients
        self.spectral_solver = spectral_solver

        # Ensure metric is always available
        self.metric = self.fields.grid.metric
        if self.metric is None:
            from ..core.metrics import MinkowskiMetric

            self.metric = MinkowskiMetric()

        # Initialize covariant derivative operator
        self.covariant_derivative = CovariantDerivative(self.metric)

    @monitor_performance("stress_energy_tensor")
    def stress_energy_tensor(self) -> np.ndarray:
        """
        Construct T^μν including all Israel-Stewart corrections:
        T^μν = (ε+p)u^μu^ν + p g^μν + ΠΔ^μν + π^μν + q^μu^ν + q^νu^μ

        Returns:
            Stress-energy tensor with shape (*grid.shape, 4, 4)
        """
        f = self.fields
        grid_shape = f.grid.shape

        # Get metric tensor g^μν (inverse metric)
        g_inv = self.metric.inverse
        if g_inv.ndim == 2:
            g_inv = np.broadcast_to(g_inv, (*grid_shape, 4, 4))

        # Standard perfect fluid tensor: T_pf^μν = (ε+p)u^μu^ν + p g^μν
        enthalpy = f.rho + f.pressure
        u_outer = optimized_einsum("...,...i,...j->...ij", enthalpy, f.u_mu, f.u_mu)
        p_metric = optimized_einsum("...,...ij->...ij", f.pressure, g_inv)
        T_perfect = u_outer + p_metric

        # Viscous corrections
        Delta = self._spatial_projector()
        T_bulk = optimized_einsum("...,...ij->...ij", f.Pi, Delta)  # Π Δ^μν
        T_shear = f.pi_munu.copy()  # π^μν

        # Heat flux contribution: q^μu^ν + q^νu^μ (symmetric)
        T_heat_1 = optimized_einsum("...i,...j->...ij", f.q_mu, f.u_mu)
        T_heat_2 = optimized_einsum("...j,...i->...ij", f.q_mu, f.u_mu)
        T_heat = T_heat_1 + T_heat_2

        # Combine all contributions
        T_total = T_perfect + T_bulk + T_shear + T_heat
        result: np.ndarray = T_total
        return result

    @monitor_performance("divergence_T")
    def divergence_T(self) -> np.ndarray:
        """
        Compute ∂_μ T^μν using covariant derivatives.

        For each ν, computes ∇_μ T^μν = ∂_μ T^μν + Γ^μ_μλ T^λν + Γ^ν_μλ T^μλ

        For SpaceGrid (3+1D formalism):
        - Only computes spatial divergence ∂_i T^iν (i=1,2,3)
        - Time derivative ∂_0 T^0ν must be computed separately via evolution_equations()

        For SpacetimeGrid (legacy 4D):
        - Computes full 4D divergence ∂_μ T^μν (μ=0,1,2,3)

        Returns:
            Divergence with shape (*grid.shape, 4) - one component for each ν
        """
        T = self.stress_energy_tensor()
        grid_shape = self.fields.grid.shape
        div_T = np.zeros((*grid_shape, 4))

        # Get coordinate arrays for numerical derivatives
        coords = self._get_coordinate_arrays()

        # Detect grid type: SpaceGrid has 3 coords, SpacetimeGrid has 4
        is_spacegrid = len(coords) == 3

        # For SpaceGrid: only compute spatial divergence (μ runs over spatial indices only)
        # For SpacetimeGrid: compute full 4D divergence (μ runs over 0,1,2,3)
        mu_start = 1 if is_spacegrid else 0  # Skip time index for SpaceGrid
        mu_range = range(mu_start, 4)

        # For each component ν of the conservation equation
        for nu in range(4):
            div_component = np.zeros_like(T[..., 0, 0])

            # Sum over contracted index μ: ∂_μ T^μν
            for mu in mu_range:
                # Extract T^μν component
                T_mu_nu = T[..., mu, nu]

                # Map tensor index to coordinate index
                # SpaceGrid: tensor μ=1,2,3 → coord 0,1,2
                # SpacetimeGrid: tensor μ=0,1,2,3 → coord 0,1,2,3
                coord_idx = mu - 1 if is_spacegrid else mu

                # Compute partial derivative ∂_μ T^μν
                partial_deriv = self._partial_derivative(T_mu_nu, coord_idx, coords)
                div_component += partial_deriv

                # Add Christoffel symbol corrections if metric is not flat
                if self.metric and not self.metric.is_flat():
                    christoffel = self.covariant_derivative.christoffel_symbols

                    # Connection term: Γ^μ_μλ T^λν
                    for lam in range(4):
                        connection_1 = christoffel[mu, mu, lam] * T[..., lam, nu]
                        div_component += connection_1

                    # Connection term: Γ^ν_μλ T^μλ
                    for lam in range(4):
                        connection_2 = christoffel[nu, mu, lam] * T[..., mu, lam]
                        div_component += connection_2

            div_T[..., nu] = div_component

        return div_T

    @monitor_performance("evolution_equations")
    def evolution_equations(self) -> dict[str, np.ndarray]:
        """
        Return RHS of evolution equations from conservation laws.

        From ∂_μ T^μν = 0, extract:
        - ∂_t ρ = -∂_i T^0i  (energy conservation, ν=0)
        - ∂_t (ρu^j) = -∂_i T^ij  (momentum conservation, ν=1,2,3)

        For 3+1D time evolution, we compute ONLY spatial divergence, not full 4D divergence.
        The conservation equation ∂_0 T^0ν + ∂_i T^iν = 0 rearranges to:
        ∂_t T^0ν = -∂_i T^iν (spatial divergence determines time evolution)

        Returns:
            Dictionary with evolution equations:
            - 'drho_dt': Energy density time derivative
            - 'dmom_dt': Momentum density time derivatives (3-vector)
        """
        T = self.stress_energy_tensor()
        grid_shape = self.fields.grid.shape

        # Get coordinate arrays for spatial derivatives only
        coords = self._get_coordinate_arrays()

        # Detect grid type to correctly map tensor indices to coordinate indices
        # SpaceGrid: coords = [x, y, z] (3 elements, indices 0,1,2)
        # SpacetimeGrid: coords = [t, x, y, z] (4 elements, indices 0,1,2,3)
        is_spacegrid = len(coords) == 3

        # For 3+1D evolution: ∂_t T^0ν = -∂_i T^iν (only spatial divergence)
        drho_dt = np.zeros(grid_shape)
        dmom_dt = np.zeros((*grid_shape, 3))

        # Use spectral divergence for periodic boundaries for efficiency and accuracy
        if self.spectral_solver is not None and self.fields.grid.boundary_conditions == "periodic":
            # Energy conservation: ∂_t ρ = -∂_i T^i0
            energy_flux_vector = T[..., 1:4, 0]  # Vector (T^10, T^20, T^30)
            drho_dt = -self.spectral_solver.spatial_divergence(energy_flux_vector)

            # Momentum conservation: ∂_t(ρu^j) = -∂_i T^ij
            for j in range(1, 4):  # j = 1, 2, 3 (momentum tensor components)
                momentum_flux_vector = T[..., 1:4, j]  # Vector (T^1j, T^2j, T^3j)
                dmom_dt[..., j - 1] = -self.spectral_solver.spatial_divergence(
                    momentum_flux_vector
                )
        else:
            # Fallback to grid-based vectorized derivatives for non-spectral solvers
            # Use grid.divergence() for better performance and consistency

            # Energy conservation: ∂_t ρ = -∂_i T^i0
            energy_flux_vector = T[..., 1:4, 0]  # Vector (T^10, T^20, T^30)
            drho_dt = -self.fields.grid.divergence(energy_flux_vector, order=2)

            # Momentum conservation: ∂_t(ρu^j) = -∂_i T^ij
            for j in range(1, 4):  # j = 1, 2, 3 (momentum tensor components)
                momentum_flux_vector = T[..., 1:4, j]  # Vector (T^1j, T^2j, T^3j)
                dmom_dt[..., j - 1] = -self.fields.grid.divergence(
                    momentum_flux_vector, order=2
                )

        # Add Christoffel symbol corrections if metric is not flat
        if self.metric and not self.metric.is_flat():
            christoffel = self.covariant_derivative.christoffel_symbols

            # Energy: connection terms for spatial divergence only
            for i in range(1, 4):
                for lam in range(4):
                    drho_dt -= christoffel[i, i, lam] * T[..., lam, 0]
                    drho_dt -= christoffel[0, i, lam] * T[..., i, lam]

            # Momentum: connection terms for spatial divergence only
            for j in range(1, 4):
                for i in range(1, 4):
                    for lam in range(4):
                        dmom_dt[..., j - 1] -= christoffel[i, i, lam] * T[..., lam, j]
                        dmom_dt[..., j - 1] -= christoffel[j, i, lam] * T[..., i, lam]

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

        CRITICAL: Must use grid's coordinate arrays to respect boundary_conditions.
        Both SpaceGrid and SpacetimeGrid create coordinates with proper spacing:
        - periodic: dx = L/N (excludes endpoint)
        - dirichlet/neumann: dx = L/(N-1) (includes endpoint)

        Returns:
            List of coordinate arrays
            - SpaceGrid: [x, y, z] (3D spatial)
            - SpacetimeGrid: [t, x, y, z] (4D spacetime)
        """
        grid = self.fields.grid

        # Always use grid's coordinate arrays (respects boundary_conditions)
        if hasattr(grid, "coordinates") and isinstance(grid.coordinates, dict):
            # Extract coordinate arrays using grid's coordinate names
            coord_names = grid.coordinate_names
            return [grid.coordinates[name] for name in coord_names]
        else:
            # Fallback should never be reached for properly initialized grids
            # If it is, something is wrong with grid initialization
            raise ValueError(
                "Grid must have 'coordinates' attribute (SpaceGrid or SpacetimeGrid required). "
                "Cannot reconstruct coordinates safely without knowing boundary_conditions."
            )

    def _partial_derivative(self, field: np.ndarray, direction: int, coords: list) -> np.ndarray:
        """
        Compute partial derivative ∂_μ field using the most appropriate method.

        Hierarchy:
        1. Spectral solver (if available and periodic BC)
        2. Grid gradient method (robust finite differences with proper BC)
        3. np.gradient fallback (legacy, avoid if possible)
        """
        # Use spectral solver if available (preferred for accuracy and consistency)
        if self.spectral_solver is not None:
            # The spectral solver operates on pure 3D spatial fields.
            # The `direction` here corresponds to the spatial axes (0,1,2 for x,y,z).
            if field.ndim != 3:
                # Pad or slice field if it doesn't match the expected 3D shape
                # This is a temporary workaround. A better solution is to ensure
                # all fields passed to this function are purely spatial.
                warnings.warn(
                    f"Field with shape {field.shape} is not a pure 3D spatial field. "
                    "Attempting to use spectral derivative on a slice.",
                    UserWarning,
                    stacklevel=2,
                )
                if field.ndim > 3:
                    field = field[..., 0]  # Example: take first component

            # Ensure direction is within the spatial dimensions
            if direction < 3:
                return self.spectral_solver.spatial_derivative(field, direction=direction)

        # Use grid gradient method (respects boundary conditions)
        if hasattr(self.fields.grid, 'gradient') and field.ndim == 3:
            if direction >= len(coords):
                raise ValueError(f"Direction {direction} exceeds coordinate dimensions")
            return self.fields.grid.gradient(field, axis=direction, order=2)

        # Legacy fallback to np.gradient (does not respect periodic BC properly)
        if direction >= len(coords):
            raise ValueError(f"Direction {direction} exceeds coordinate dimensions")

        spacing = coords[direction]
        gradient_dir = np.gradient(field, spacing, axis=direction)

        return gradient_dir

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
        partial = self._partial_derivative(tensor_component, index, coords)

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

        For SpaceGrid (3+1D formalism):
        - Only computes spatial divergence ∂_i N^i (i=1,2,3)

        For SpacetimeGrid (legacy 4D):
        - Computes full 4D divergence ∂_μ N^μ (μ=0,1,2,3)

        Returns:
            Particle number conservation equation: ∂_t n + ∇·(n v) = 0
        """
        f = self.fields

        # Particle current: N^μ = n u^μ
        N_mu = f.n[..., np.newaxis] * f.u_mu

        # Compute divergence ∂_μ N^μ
        coords = self._get_coordinate_arrays()
        div_N = np.zeros_like(N_mu[..., 0])

        # Detect grid type: SpaceGrid has 3 coords, SpacetimeGrid has 4
        is_spacegrid = len(coords) == 3

        # For SpaceGrid: only compute spatial divergence (μ runs over spatial indices only)
        # For SpacetimeGrid: compute full 4D divergence (μ runs over 0,1,2,3)
        mu_start = 1 if is_spacegrid else 0  # Skip time index for SpaceGrid
        mu_range = range(mu_start, 4)

        for mu in mu_range:
            # Map tensor index to coordinate index
            # SpaceGrid: tensor μ=1,2,3 → coord 0,1,2
            # SpacetimeGrid: tensor μ=0,1,2,3 → coord 0,1,2,3
            coord_idx = mu - 1 if is_spacegrid else mu

            partial = self._partial_derivative(N_mu[..., mu], coord_idx, coords)
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
