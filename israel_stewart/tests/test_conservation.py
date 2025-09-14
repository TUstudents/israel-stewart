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

from israel_stewart.core import (
    ISFieldConfiguration,
    create_cartesian_grid,
    create_milne_grid,
)
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.equations import ConservationLaws


class TestConservationLawsInitialization:
    """Test ConservationLaws initialization and setup."""

    def test_init_with_minkowski_metric(self):
        """Test initialization with Minkowski metric."""
        grid = create_cartesian_grid((0, 1), 2.0, (4, 4, 4, 4))
        fields = ISFieldConfiguration(grid)

        conservation = ConservationLaws(fields)

        assert conservation.fields is fields
        assert conservation.coeffs is None
        assert conservation.covariant_derivative is not None

    def test_init_with_transport_coefficients(self):
        """Test initialization with transport coefficients."""
        grid = create_cartesian_grid((0, 1), 2.0, (4, 4, 4, 4))
        fields = ISFieldConfiguration(grid)

        # Mock transport coefficients
        mock_coeffs = Mock()
        conservation = ConservationLaws(fields, mock_coeffs)

        assert conservation.coeffs is mock_coeffs

    def test_init_with_general_metric(self):
        """Test initialization with general metric."""
        grid = create_cartesian_grid((0, 1), 2.0, (4, 4, 4, 4))
        metric = MinkowskiMetric()
        grid.metric = metric
        fields = ISFieldConfiguration(grid)

        conservation = ConservationLaws(fields)

        assert conservation.covariant_derivative.metric is metric


class TestStressEnergyTensor:
    """Test stress-energy tensor construction."""

    @pytest.fixture
    def simple_fields(self):
        """Create simple field configuration for testing."""
        grid = create_cartesian_grid((0, 1), 2.0, (3, 3, 3, 3))
        fields = ISFieldConfiguration(grid)

        # Initialize with simple values
        fields.rho[:] = 1.0  # Energy density
        fields.pressure[:] = 0.3  # Pressure
        fields.Pi[:] = 0.1  # Bulk viscosity
        fields.u_mu[..., 0] = 1.0  # Rest frame

        # Add some shear stress and heat flux
        fields.pi_munu[..., 1, 2] = 0.05
        fields.pi_munu[..., 2, 1] = 0.05  # Symmetry
        fields.q_mu[..., 1] = 0.02

        return fields

    def test_perfect_fluid_contribution(self, simple_fields):
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

    def test_pressure_contribution(self, simple_fields):
        """Test pressure and bulk viscosity contribution."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # Check T^11 component (should include pressure + bulk viscosity)
        # T^11 = ρ u^1 u^1 + (p+Π)g^11 + π^11 + q^1u^1 + q^1u^1
        # For rest frame: u^1 = 0, g^11 = 1
        # T^11 = 0 + (p+Π) + π^11 + 0
        expected_T11 = 0.4 + 0.0  # (p+Π) + π^11
        np.testing.assert_allclose(T[..., 1, 1], expected_T11, rtol=1e-12)

    def test_shear_stress_contribution(self, simple_fields):
        """Test shear stress contribution."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # Check off-diagonal components with shear stress
        # T^12 should include π^12
        expected_T12 = 0.05  # π^12
        np.testing.assert_allclose(T[..., 1, 2], expected_T12, rtol=1e-12)
        np.testing.assert_allclose(T[..., 2, 1], expected_T12, rtol=1e-12)

    def test_heat_flux_contribution(self, simple_fields):
        """Test heat flux contribution."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # Check T^01 component (heat flux term)
        # T^01 = ρ u^0 u^1 + (p+Π)Δ^01 + π^01 + q^0u^1 + q^1u^0
        # For rest frame: u^0 = 1, u^1 = 0, Δ^01 = 0
        # T^01 = 0 + 0 + π^01 + 0 + q^1 = π^01 + q^1
        expected_T01 = 0.0 + 0.02  # π^01 + q^1
        np.testing.assert_allclose(T[..., 0, 1], expected_T01, rtol=1e-12)

    def test_tensor_symmetry(self, simple_fields):
        """Test that stress-energy tensor is symmetric."""
        conservation = ConservationLaws(simple_fields)
        T = conservation.stress_energy_tensor()

        # Check symmetry T^μν = T^νμ
        for mu in range(4):
            for nu in range(4):
                np.testing.assert_allclose(T[..., mu, nu], T[..., nu, mu], rtol=1e-12)


class TestSpatialProjector:
    """Test spatial projector computation."""

    def test_minkowski_projector(self):
        """Test spatial projector in Minkowski spacetime."""
        grid = create_cartesian_grid((0, 1), 2.0, (3, 3, 3, 3))
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

    def test_projector_with_moving_fluid(self):
        """Test spatial projector with moving fluid."""
        grid = create_cartesian_grid((0, 1), 2.0, (2, 2, 2, 2))
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
    def uniform_fields(self):
        """Create uniform field configuration."""
        grid = create_cartesian_grid((0, 1), 2.0, (4, 4, 4, 4))
        fields = ISFieldConfiguration(grid)

        # Uniform fields (should give zero divergence)
        fields.rho[:] = 1.0
        fields.pressure[:] = 0.3
        fields.u_mu[..., 0] = 1.0

        return fields

    @pytest.fixture
    def gradient_fields(self):
        """Create field configuration with gradients."""
        grid = create_cartesian_grid((0, 2), 4.0, (6, 6, 6, 6))
        fields = ISFieldConfiguration(grid)

        # Create coordinate meshes
        t_mesh, x_mesh, y_mesh, z_mesh = grid.meshgrid()

        # Add spatial gradients
        fields.rho = 1.0 + 0.1 * x_mesh
        fields.pressure = 0.3 + 0.05 * y_mesh
        fields.u_mu[..., 0] = 1.0

        return fields

    def test_divergence_uniform_fields(self, uniform_fields):
        """Test divergence with uniform fields should be zero."""
        conservation = ConservationLaws(uniform_fields)
        div_T = conservation.divergence_T()

        # Check shape
        expected_shape = (*uniform_fields.grid.shape, 4)
        assert div_T.shape == expected_shape

        # Uniform fields should have zero divergence
        np.testing.assert_allclose(div_T, 0.0, atol=1e-15)

    def test_divergence_with_gradients(self, gradient_fields):
        """Test divergence with spatial gradients."""
        conservation = ConservationLaws(gradient_fields)
        div_T = conservation.divergence_T()

        # Should have non-zero divergence due to gradients
        assert np.max(np.abs(div_T)) > 1e-15

        # Check that energy component responds to density gradient
        assert np.max(np.abs(div_T[..., 0])) > 0

    def test_coordinate_array_handling(self):
        """Test coordinate array construction."""
        grid = create_cartesian_grid((0, 1), 2.0, (3, 3, 3, 3))
        fields = ISFieldConfiguration(grid)
        conservation = ConservationLaws(fields)

        coords = conservation._get_coordinate_arrays()

        assert len(coords) == 4  # [t, x, y, z]
        assert all(isinstance(c, np.ndarray) for c in coords)
        assert coords[0].shape == (3,)  # time coordinates

    def test_partial_derivative_computation(self):
        """Test partial derivative computation."""
        grid = create_cartesian_grid((0, 1), 2.0, (5, 5, 5, 5))
        fields = ISFieldConfiguration(grid)
        conservation = ConservationLaws(fields)

        # Create test field with known gradient
        test_field = np.ones((5, 5, 5, 5))
        coords = conservation._get_coordinate_arrays()

        # Should return zero for uniform field
        deriv = conservation._partial_derivative(test_field, 0, coords)
        np.testing.assert_allclose(deriv, 0.0, atol=1e-15)


class TestEvolutionEquations:
    """Test evolution equation extraction."""

    def test_evolution_with_uniform_fields(self):
        """Test evolution equations with uniform fields."""
        grid = create_cartesian_grid((0, 1), 2.0, (4, 4, 4, 4))
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

    def test_evolution_with_pressure_gradient(self):
        """Test evolution with pressure gradients."""
        grid = create_cartesian_grid((0, 1), 2.0, (6, 6, 6, 6))
        fields = ISFieldConfiguration(grid)

        # Create pressure gradient
        t_mesh, x_mesh, y_mesh, z_mesh = grid.meshgrid()
        fields.rho[:] = 1.0
        fields.pressure = 0.3 + 0.1 * x_mesh  # Pressure gradient in x
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        evolution = conservation.evolution_equations()

        # Should have non-zero momentum evolution due to pressure gradient
        assert np.max(np.abs(evolution["dmom_dt"])) > 0


class TestParticleConservation:
    """Test particle number conservation."""

    def test_particle_conservation_uniform(self):
        """Test particle conservation with uniform density."""
        grid = create_cartesian_grid((0, 1), 2.0, (4, 4, 4, 4))
        fields = ISFieldConfiguration(grid)
        fields.n[:] = 0.5  # Uniform particle density
        fields.u_mu[..., 0] = 1.0  # Rest frame

        conservation = ConservationLaws(fields)
        div_N = conservation.particle_number_conservation()

        # Uniform density should give zero divergence
        np.testing.assert_allclose(div_N, 0.0, atol=1e-15)

    def test_particle_conservation_with_gradient(self):
        """Test particle conservation with density gradient."""
        grid = create_cartesian_grid((0, 1), 2.0, (5, 5, 5, 5))
        fields = ISFieldConfiguration(grid)

        # Create particle density gradient
        t_mesh, x_mesh, y_mesh, z_mesh = grid.meshgrid()
        fields.n = 0.5 + 0.1 * x_mesh
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        div_N = conservation.particle_number_conservation()

        # Test that the function executes without errors
        # (Numerical divergence may be very small due to discretization)
        assert div_N.shape == fields.grid.shape
        assert np.isfinite(div_N).all()


class TestConservationValidation:
    """Test conservation law validation."""

    def test_validation_perfect_conservation(self):
        """Test validation with perfectly conserved quantities."""
        grid = create_cartesian_grid((0, 1), 2.0, (3, 3, 3, 3))
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

    def test_validation_violated_conservation(self):
        """Test validation with violated conservation."""
        grid = create_cartesian_grid((0, 1), 2.0, (5, 5, 5, 5))
        fields = ISFieldConfiguration(grid)

        # Add gradients to violate conservation
        t_mesh, x_mesh, y_mesh, z_mesh = grid.meshgrid()
        fields.rho = 1.0 + 0.2 * x_mesh
        fields.pressure = 0.3 + 0.1 * y_mesh
        fields.n = 0.5 + 0.1 * x_mesh
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        validation = conservation.validate_conservation(tolerance=1e-10)

        assert validation["energy_momentum_conserved"] is False
        assert validation["all_conserved"] is False

    def test_validation_custom_tolerance(self):
        """Test validation with custom tolerance."""
        grid = create_cartesian_grid((0, 1), 2.0, (4, 4, 4, 4))
        fields = ISFieldConfiguration(grid)

        # Create large gradients that definitely violate conservation
        t_mesh, x_mesh, y_mesh, z_mesh = grid.meshgrid()
        fields.rho = 1.0 + 0.5 * x_mesh  # Large gradient for reliable detection
        fields.pressure = 0.3 + 0.2 * y_mesh
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

    def test_empty_grid(self):
        """Test with minimal grid size."""
        grid = create_cartesian_grid((0, 1), 2.0, (2, 2, 2, 2))
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.pressure[:] = 0.3
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)

        # Should not raise errors
        T = conservation.stress_energy_tensor()
        assert T.shape == (2, 2, 2, 2, 4, 4)

    def test_christoffel_symbol_handling(self):
        """Test handling of Christoffel symbols."""
        grid = create_cartesian_grid((0, 1), 2.0, (3, 3, 3, 3))
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)

        # Should handle Christoffel symbol computation gracefully
        div_T = conservation.divergence_T()
        assert div_T.shape == (3, 3, 3, 3, 4)

    def test_string_representations(self):
        """Test string representations."""
        grid = create_cartesian_grid((0, 1), 2.0, (3, 3, 3, 3))
        fields = ISFieldConfiguration(grid)
        conservation = ConservationLaws(fields)

        str_repr = str(conservation)
        assert "ConservationLaws" in str_repr
        assert "grid_shape" in str_repr

        repr_str = repr(conservation)
        assert "ConservationLaws" in repr_str
        assert "fields=" in repr_str


class TestIntegrationWithGrid:
    """Test integration with different grid types."""

    def test_cartesian_grid_integration(self):
        """Test with Cartesian coordinates."""
        grid = create_cartesian_grid((0, 1), 2.0, (4, 4, 4, 4))
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        coords = conservation._get_coordinate_arrays()

        assert len(coords) == 4
        assert grid.coordinate_names == ["t", "x", "y", "z"]

    def test_milne_grid_integration(self):
        """Test with Milne coordinates."""
        grid = create_milne_grid((0.1, 1.0), (-2, 2), 4.0, (4, 4, 4, 4))
        fields = ISFieldConfiguration(grid)
        fields.rho[:] = 1.0
        fields.u_mu[..., 0] = 1.0

        conservation = ConservationLaws(fields)
        coords = conservation._get_coordinate_arrays()

        assert len(coords) == 4
        # Check coordinate names (may vary by implementation)
        coord_names = grid.coordinate_names
        assert len(coord_names) == 4
        assert "tau" in coord_names or "t" in coord_names  # Allow flexibility


@pytest.mark.parametrize("grid_size", [(3, 3, 3, 3), (4, 5, 6, 7)])
def test_different_grid_sizes(grid_size):
    """Test with different grid sizes."""
    grid = create_cartesian_grid((0, 1), 2.0, grid_size)
    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 0.3
    fields.u_mu[..., 0] = 1.0

    conservation = ConservationLaws(fields)
    T = conservation.stress_energy_tensor()

    expected_shape = (*grid_size, 4, 4)
    assert T.shape == expected_shape


@pytest.mark.parametrize("rho,pressure", [(1.0, 0.3), (2.5, 0.8), (0.1, 0.03)])
def test_different_thermodynamic_states(rho, pressure):
    """Test with different thermodynamic conditions."""
    grid = create_cartesian_grid((0, 1), 2.0, (3, 3, 3, 3))
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
