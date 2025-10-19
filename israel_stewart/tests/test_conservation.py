"""
Unit tests for Conservation Laws implementation.

Tests the ConservationLaws class for Israel-Stewart hydrodynamics, including:
- Stress-energy tensor construction
- Covariant divergence computation
- Evolution equation extraction
- Particle number conservation
- Validation methods
"""

from unittest.mock import Mock

import numpy as np
import pytest

from israel_stewart.core import ISFieldConfiguration
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations import ConservationLaws


class TestConservationLawsInitialization:
    """Test ConservationLaws initialization and setup."""

    def test_init_with_minkowski_metric(self) -> None:
        """Test initialization with Minkowski metric."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0)] * 3,
            grid_points=(4, 4, 4),
            boundary_conditions="periodic",
        )
        fields = ISFieldConfiguration(grid)

        conservation = ConservationLaws(fields)

        assert conservation.fields is fields
        assert conservation.coeffs is None
        assert conservation.covariant_derivative is not None

    def test_init_with_transport_coefficients(self) -> None:
        """Test initialization with transport coefficients."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0)] * 3,
            grid_points=(4, 4, 4),
            boundary_conditions="periodic",
        )
        fields = ISFieldConfiguration(grid)

        # Mock transport coefficients
        mock_coeffs = Mock()
        conservation = ConservationLaws(fields, mock_coeffs)

        assert conservation.coeffs is mock_coeffs

    def test_init_with_general_metric(self) -> None:
        """Test initialization with general metric."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0)] * 3,
            grid_points=(4, 4, 4),
            boundary_conditions="periodic",
        )
        metric = MinkowskiMetric()
        grid.metric = metric
        fields = ISFieldConfiguration(grid)

        conservation = ConservationLaws(fields)

        assert conservation.covariant_derivative.metric is metric


class TestStressEnergyTensor:
    """Test stress-energy tensor construction."""

    @pytest.fixture
    def simple_fields(self) -> ISFieldConfiguration:
        """Create simple field configuration for testing."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0)] * 3,
            grid_points=(3, 3, 3),
            boundary_conditions="periodic",
        )
        fields = ISFieldConfiguration(grid)

        # Initialize with simple values
        fields.rho[:] = 1.0  # Energy density
        fields.pressure[:] = 0.3  # Pressure
        fields.Pi[:] = 0.1  # Bulk viscosity
        fields.u_mu[..., 0] = 1.0  # Rest frame

        # Add some shear stress and particle diffusion current
        fields.pi_munu[..., 1, 2] = 0.05
        fields.pi_munu[..., 2, 1] = 0.05  # Symmetry
        fields.V_mu[..., 1] = 0.02

        return fields

    def test_perfect_fluid_contribution(self, simple_fields: ISFieldConfiguration) -> None:
        """Test perfect fluid part of stress-energy tensor."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # Check shape
        expected_shape = (*simple_fields.grid.shape, 4, 4)
        assert T.shape == expected_shape

        # Check T^00 component (energy density contribution)
        # T^00 = ρ u^0 u^0 + (p+Π)Δ^00 + π^00 + 2q^0u^0
        # For rest frame: u^0 = 1, Δ^00 = 0 (spatial projector)
        # T^00 = ρ * 1 * 1 + (p+Π) * 0 + π^00 + 0 = ρ
        expected_T00 = 1.0  # Just energy density ρ
        np.testing.assert_allclose(T[..., 0, 0], expected_T00, rtol=1e-12)

    def test_pressure_contribution(self, simple_fields: ISFieldConfiguration) -> None:
        """Test pressure and bulk viscosity contribution."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # Check T^11 component (should include pressure + bulk viscosity)
        # T^11 = ρ u^1 u^1 + (p+Π)g^11 + π^11 + q^1u^1 + q^1u^1
        # For rest frame: u^1 = 0, g^11 = 1
        # T^11 = 0 + (p+Π) + π^11 + 0
        expected_T11 = 0.4 + 0.0  # (p+Π) + π^11
        np.testing.assert_allclose(T[..., 1, 1], expected_T11, rtol=1e-12)

    def test_shear_stress_contribution(self, simple_fields: ISFieldConfiguration) -> None:
        """Test shear stress contribution with IReD sign convention."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # Check off-diagonal components with shear stress
        # IReD sign convention: T = ... + π^μν (PLUS sign for dissipative terms)
        # For rest frame: T^12 = π^12 = 0.05
        expected_T12 = 0.05  # +π^12 (IReD convention)
        np.testing.assert_allclose(T[..., 1, 2], expected_T12, rtol=1e-12)
        np.testing.assert_allclose(T[..., 2, 1], expected_T12, rtol=1e-12)

    def test_diffusion_current_contribution(self, simple_fields: ISFieldConfiguration) -> None:
        """Test that diffusion current does NOT appear in stress-energy tensor (Landau frame)."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # LANDAU FRAME: V^μ appears in J^μ = n u^μ + V^μ, NOT in T^μν!
        # T^01 = ρ u^0 u^1 + (p+Π)Δ^01 - π^01
        # For rest frame: u^0 = 1, u^1 = 0, Δ^01 = 0, π^01 = 0
        # T^01 = 0 + 0 - 0 = 0
        # The V^1 = 0.02 field value should NOT contribute to T^μν
        expected_T01 = 0.0  # No contribution from V^μ in Landau frame
        np.testing.assert_allclose(T[..., 0, 1], expected_T01, atol=1e-12)

    def test_tensor_symmetry(self, simple_fields: ISFieldConfiguration) -> None:
        """Test that stress-energy tensor is symmetric."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # Check symmetry T^μν = T^νμ
        for mu in range(4):
            for nu in range(4):
                np.testing.assert_allclose(T[..., mu, nu], T[..., nu, mu], rtol=1e-12)


class TestSpatialProjector:
    """Test spatial projector computation."""

    def test_minkowski_projector(self) -> None:
        """Test spatial projector in Minkowski spacetime."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (3, 3, 3), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        fields.u_mu[..., 0] = 1.0  # Rest frame

        conservation = ConservationLaws(fields)
        Delta = conservation._spatial_projector()

        # Check shape
        expected_shape = (*grid.shape, 4, 4)
        assert Delta.shape == expected_shape

        # For rest frame in Minkowski: Δ^μν = g^μν + u^μu^ν
        # Δ^00 = -1 + 1*1 = 0
        # Δ^11 = 1 + 0*0 = 1
        # Δ^01 = 0 + 1*0 = 0
        np.testing.assert_allclose(Delta[..., 0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(Delta[..., 1, 1], 1.0, rtol=1e-12)
        np.testing.assert_allclose(Delta[..., 0, 1], 0.0, atol=1e-12)

    def test_projector_with_moving_fluid(self) -> None:
        """Test spatial projector with moving fluid."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (2, 2, 2), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)

        # Set fluid moving in x-direction
        gamma = 2.0
        v_x = np.sqrt(1 - 1 / gamma**2)
        fields.u_mu[..., 0] = gamma
        fields.u_mu[..., 1] = gamma * v_x

        conservation = ConservationLaws(fields)
        Delta = conservation._spatial_projector()

        # For Minkowski metric in mostly-plus signature: g = diag(-1, 1, 1, 1)
        # To check orthogonality u^μ Δ_μν = 0, we need to lower the first index
        # u_μ = g_μν u^ν, so u_0 = -u^0, u_i = u^i for i=1,2,3
        u_lower = fields.u_mu.copy()
        u_lower[..., 0] *= -1  # Lower time component

        # Check orthogonality: u_μ Δ^μν = 0
        u_contract = np.einsum("...i,...ij->...j", u_lower, Delta)
        np.testing.assert_allclose(u_contract, 0.0, atol=1e-10)


class TestDivergenceComputation:
    """Test covariant divergence computation."""

    @pytest.fixture
    def uniform_fields(self) -> ISFieldConfiguration:
        """Create uniform field configuration."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (4, 4, 4), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)

        # Uniform fields (should give zero divergence)
        fields.rho[:] = 1.0
        fields.pressure[:] = 0.3
        fields.u_mu[..., 0] = 1.0

        return fields

    @pytest.fixture
    def gradient_fields(self) -> ISFieldConfiguration:
        """Create field configuration with gradients."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 4.0)] * 3,
            grid_points=(6, 6, 6),
            boundary_conditions="periodic",
        )
        fields = ISFieldConfiguration(grid)

        # Create coordinate meshes
        x_mesh, y_mesh, z_mesh = grid.meshgrid()

        # Add spatial gradients
        fields.rho[:] = 1.0 + 0.1 * x_mesh
        fields.pressure[:] = 0.3 + 0.05 * y_mesh
        fields.u_mu[..., 0] = 1.0

        return fields

    def test_divergence_uniform_fields(self, uniform_fields: ISFieldConfiguration) -> None:
        """Test divergence with uniform fields should be zero."""
        conservation = ConservationLaws(uniform_fields)
        div_T = conservation.divergence_T()

        # Check shape
        expected_shape = (*uniform_fields.grid.shape, 4)
        assert div_T.shape == expected_shape

        # Uniform fields should have zero divergence
        np.testing.assert_allclose(div_T, 0.0, atol=1e-15)

    def test_divergence_with_gradients(self, gradient_fields: ISFieldConfiguration) -> None:
        """Test divergence with spatial gradients."""
        conservation = ConservationLaws(gradient_fields)
        div_T = conservation.divergence_T()

        # Should have non-zero divergence due to gradients
        assert np.max(np.abs(div_T)) > 1e-15

        # For SpaceGrid (3+1D): Only compute spatial divergence ∂_i T^iν
        # Energy component (ν=0): ∂_i T^i0 = 0 in rest frame (no energy flux)
        # Momentum components (ν=1,2,3): ∂_i T^ij = ∂_j p ≠ 0 (pressure gradient)
        # Check that momentum components respond to pressure gradient
        assert np.max(np.abs(div_T[..., 2])) > 0  # y-component has pressure gradient

    def test_coordinate_array_handling(self) -> None:
        """Test coordinate array construction."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (3, 3, 3), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        conservation = ConservationLaws(fields)

        coords = conservation._get_coordinate_arrays()

        assert len(coords) == 3  # [x, y, z] for SpaceGrid
        assert all(isinstance(c, np.ndarray) for c in coords)
        assert coords[0].shape == (3,)  # x coordinates

    def test_partial_derivative_computation(self) -> None:
        """Test partial derivative computation."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (5, 5, 5), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        conservation = ConservationLaws(fields)

        # Create test field with known gradient
        test_field = np.ones((5, 5, 5))  # 3D for SpaceGrid
        coords = conservation._get_coordinate_arrays()

        # Should return zero for uniform field
        deriv = conservation._partial_derivative(test_field, 0, coords)
        np.testing.assert_allclose(deriv, 0.0, atol=1e-15)


class TestEvolutionEquations:
    """Test evolution equation extraction."""

    def test_evolution_with_uniform_fields(self) -> None:
        """Test evolution equations with uniform fields."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (4, 4, 4), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.pressure[:] = 0.3
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        evolution = conservation.evolution_equations()

        # Check keys
        assert "drho_dt" in evolution
        assert "dmom_dt" in evolution

        # Check shapes
        assert evolution["drho_dt"].shape == fields.grid.shape
        assert evolution["dmom_dt"].shape == (*fields.grid.shape, 3)

        # Uniform fields should have zero time derivatives
        np.testing.assert_allclose(evolution["drho_dt"], 0.0, atol=1e-15)
        np.testing.assert_allclose(evolution["dmom_dt"], 0.0, atol=1e-15)

    def test_evolution_with_pressure_gradient(self) -> None:
        """Test evolution with pressure gradients."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (6, 6, 6), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)

        # Create pressure gradient
        x_mesh, y_mesh, z_mesh = grid.meshgrid()
        fields.rho[:] = 1.0
        fields.pressure[:] = (0.3 + 0.1 * x_mesh).reshape(
            fields.pressure.shape
        )  # Pressure gradient in x
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        evolution = conservation.evolution_equations()

        # Should have non-zero momentum evolution due to pressure gradient
        assert np.max(np.abs(evolution["dmom_dt"])) > 0


class TestParticleConservation:
    """Test particle number conservation."""

    def test_particle_conservation_uniform(self) -> None:
        """Test particle conservation with uniform density."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (4, 4, 4), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        fields.n[:] = 0.5  # Uniform particle density
        fields.u_mu[..., 0] = 1.0  # Rest frame

        conservation = ConservationLaws(fields)
        div_N = conservation.particle_number_conservation()

        # Uniform density should give zero divergence
        np.testing.assert_allclose(div_N, 0.0, atol=1e-15)

    def test_particle_conservation_with_gradient(self) -> None:
        """Test particle conservation with density gradient."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (5, 5, 5), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)

        # Create particle density gradient
        x_mesh, y_mesh, z_mesh = grid.meshgrid()
        fields.n[:] = (0.5 + 0.1 * x_mesh).reshape(fields.n.shape)
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        div_N = conservation.particle_number_conservation()

        # Test that the function executes without errors
        # (Numerical divergence may be very small due to discretization)
        assert div_N.shape == fields.grid.shape
        assert np.isfinite(div_N).all()


class TestConservationValidation:
    """Test conservation law validation."""

    def test_validation_perfect_conservation(self) -> None:
        """Test validation with perfectly conserved quantities."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (3, 3, 3), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.pressure[:] = 0.3
        fields.n[:] = 0.5
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        validation = conservation.validate_conservation()

        assert validation["energy_momentum_conserved"] is True
        assert validation["particle_number_conserved"] is True
        assert validation["all_conserved"] is True

    def test_validation_violated_conservation(self) -> None:
        """Test validation with violated conservation."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (5, 5, 5), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)

        # Add gradients to violate conservation
        x_mesh, y_mesh, z_mesh = grid.meshgrid()
        fields.rho[:] = (1.0 + 0.2 * x_mesh).reshape(fields.rho.shape)
        fields.pressure[:] = (0.3 + 0.1 * y_mesh).reshape(fields.pressure.shape)
        fields.n[:] = (0.5 + 0.1 * x_mesh).reshape(fields.n.shape)
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        validation = conservation.validate_conservation(tolerance=1e-10)

        assert validation["energy_momentum_conserved"] is False
        assert validation["all_conserved"] is False

    def test_validation_custom_tolerance(self) -> None:
        """Test validation with custom tolerance."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (4, 4, 4), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)

        # Create large gradients that definitely violate conservation
        x_mesh, y_mesh, z_mesh = grid.meshgrid()
        fields.rho[:] = (1.0 + 0.5 * x_mesh).reshape(
            fields.rho.shape
        )  # Large gradient for reliable detection
        fields.pressure[:] = (0.3 + 0.2 * y_mesh).reshape(fields.pressure.shape)
        fields.n[:] = 0.5
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)

        # Test that validation works with different tolerances
        validation_result = conservation.validate_conservation(tolerance=1e-3)
        assert isinstance(validation_result, dict)
        assert "energy_momentum_conserved" in validation_result
        assert "all_conserved" in validation_result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_grid(self) -> None:
        """Test with minimal grid size."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (2, 2, 2), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.pressure[:] = 0.3
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)

        # Should not raise errors
        T = conservation.stress_energy_tensor()
        # SpaceGrid: 3D spatial shape (2, 2, 2) + tensor indices (4, 4)
        assert T.shape == (2, 2, 2, 4, 4)

    def test_christoffel_symbol_handling(self) -> None:
        """Test handling of Christoffel symbols."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (3, 3, 3), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)

        # Should handle Christoffel symbol computation gracefully
        div_T = conservation.divergence_T()
        # SpaceGrid: 3D spatial shape (3, 3, 3) + divergence components (4)
        assert div_T.shape == (3, 3, 3, 4)

    def test_string_representations(self) -> None:
        """Test string representations."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (3, 3, 3), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        conservation = ConservationLaws(fields)

        str_repr = str(conservation)
        assert "ConservationLaws" in str_repr
        assert "grid_shape" in str_repr

        repr_str = repr(conservation)
        assert "ConservationLaws" in repr_str
        assert "fields=" in repr_str


class TestExpansionScalar:
    """Test expansion scalar θ = ∇_μ u^μ computation for known analytical flows."""

    def test_expansion_uniform_cartesian(self) -> None:
        """Test θ = 0 for static uniform fluid in Cartesian coordinates."""
        from israel_stewart.core.metrics import MinkowskiMetric

        grid = SpaceGrid(
            "cartesian",
            [(0.0, 2.0)] * 3,
            (8, 8, 8),
            boundary_conditions="periodic",
            metric=MinkowskiMetric(),
        )
        fields = ISFieldConfiguration(grid)

        # Static uniform fluid: u^μ = (1, 0, 0, 0) everywhere
        fields.rho[:] = 1.0
        fields.pressure[:] = 0.3
        fields.u_mu[..., 0] = 1.0  # Rest frame

        # Create relaxation equations to access expansion computation
        from israel_stewart.core.fields import TransportCoefficients
        from israel_stewart.equations.relaxation import ISRelaxationEquations

        coeffs = TransportCoefficients(shear_viscosity=0.1, shear_relaxation_time=0.5)
        relaxation = ISRelaxationEquations(grid, grid.metric, coeffs)

        # Compute expansion scalar
        theta = relaxation._compute_expansion_scalar(fields.u_mu)

        # Expected: θ = 0 (no expansion in static uniform fluid)
        np.testing.assert_allclose(theta, 0.0, atol=1e-12)

    def test_expansion_linear_velocity_gradient(self) -> None:
        """Verify expansion computation matches analytical formula in flat space."""
        # This is a simpler verification test: for uniform expansion in flat space
        # θ = ∇_μ u^μ = ∂_i u^i (no Christoffel terms in Minkowski)
        from israel_stewart.core.metrics import MinkowskiMetric

        grid = SpaceGrid(
            "cartesian",
            [(0.0, 2.0)] * 3,
            (16, 16, 16),
            boundary_conditions="periodic",
            metric=MinkowskiMetric(),
        )
        fields = ISFieldConfiguration(grid)

        # Test case: u^i varies linearly in one direction
        # u^x = 0.05 * x, u^y = 0, u^z = 0
        # Then ∂_x u^x = 0.05, so θ = 0.05
        X, Y, Z = grid.meshgrid()
        velocity_gradient = 0.05

        fields.u_mu[..., 0] = 1.0  # Time component
        fields.u_mu[..., 1] = velocity_gradient * (X - 1.0)  # Centered around x=1
        fields.u_mu[..., 2] = 0.0
        fields.u_mu[..., 3] = 0.0

        fields.rho[:] = 1.0
        fields.pressure[:] = 0.3

        # Create relaxation equations
        from israel_stewart.core.fields import TransportCoefficients
        from israel_stewart.equations.relaxation import ISRelaxationEquations

        coeffs = TransportCoefficients(shear_viscosity=0.1, shear_relaxation_time=0.5)
        relaxation = ISRelaxationEquations(grid, grid.metric, coeffs)

        # Compute expansion scalar
        theta = relaxation._compute_expansion_scalar(fields.u_mu)

        # Expected: θ = ∂_x u^x = velocity_gradient (uniform across grid)
        # Check interior points (away from boundaries where numerical derivatives may be less accurate)
        theta_interior = theta[4:-4, 4:-4, 4:-4]
        expected = velocity_gradient

        np.testing.assert_allclose(np.mean(theta_interior), expected, rtol=0.15)


class TestIntegrationWithGrid:
    """Test integration with different grid types."""

    def test_cartesian_grid_integration(self) -> None:
        """Test with Cartesian coordinates."""
        grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (4, 4, 4), boundary_conditions="periodic")
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        coords = conservation._get_coordinate_arrays()

        assert len(coords) == 3  # SpaceGrid has 3D spatial coordinates
        assert grid.coordinate_names == ["x", "y", "z"]

    @pytest.mark.skip(
        reason="Milne coordinates require SpacetimeGrid, not yet updated for SpaceGrid"
    )
    def test_milne_grid_integration(self) -> None:
        """Test with Milne coordinates (skipped - needs SpacetimeGrid support)."""
        pass


@pytest.mark.parametrize("grid_size", [(3, 3, 3), (4, 5, 6)])
def test_different_grid_sizes(grid_size: tuple[int, int, int]) -> None:
    """Test with different grid sizes."""
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2.0)] * 3,
        grid_points=grid_size,
        boundary_conditions="periodic",
    )
    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 0.3
    fields.u_mu[..., 0] = 1.0

    conservation = ConservationLaws(fields)
    T = conservation.stress_energy_tensor()

    expected_shape = (*grid_size, 4, 4)
    assert T.shape == expected_shape


@pytest.mark.parametrize("rho,pressure", [(1.0, 0.3), (2.5, 0.8), (0.1, 0.03)])
def test_different_thermodynamic_states(rho: float, pressure: float) -> None:
    """Test with different thermodynamic conditions."""
    grid = SpaceGrid("cartesian", [(0.0, 2.0)] * 3, (3, 3, 3), boundary_conditions="periodic")
    fields = ISFieldConfiguration(grid)
    fields.rho[:] = rho
    fields.pressure[:] = pressure
    fields.u_mu[..., 0] = 1.0

    conservation = ConservationLaws(fields)
    T = conservation.stress_energy_tensor()

    # Check that T^00 includes energy density contribution
    # For perfect fluid at rest: T^00 = ρ (spatial projector Δ^00 = 0)
    # T^11 = pressure (spatial projector Δ^11 = 1)
    expected_T00 = rho
    expected_T11 = pressure
    np.testing.assert_allclose(T[..., 0, 0], expected_T00, rtol=1e-12)
    np.testing.assert_allclose(T[..., 1, 1], expected_T11, rtol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
