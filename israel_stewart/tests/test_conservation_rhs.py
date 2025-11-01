"""
Tier 1: RHS Consistency Tests for Conservation Laws.

These tests verify that evolution_equations() is self-consistent with
the stress-energy tensor construction at t=0 (before any time evolution).

This is the most fundamental test of conservation law correctness:
- No time integration errors
- Tests the formulation directly
- Should be exact to machine precision
- Works for ALL integration methods

Why this approach works:
The conservation laws state: ∂_t Q = -∇·F
Where Q is a conserved density and F is its flux.

At t=0, we can compute both sides independently:
- LHS: evolution_equations() computes ∂_t Q directly
- RHS: Construct T^μν, then compute -∇·T

These must match exactly if the implementation is correct.

This avoids the time-integration mismatch that causes the pointwise
tests to fail for momentum (evolving dissipative fluxes).
"""

import numpy as np
import pytest

from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations import ConservationLaws
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


class TestRHSConsistency:
    """Test that RHS of conservation equations is self-consistent at t=0."""

    @pytest.fixture
    def test_grid(self):
        """Create test grid with periodic boundaries for spectral methods."""
        return SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0, 2 * np.pi)] * 3,
            grid_points=(16, 16, 16),
            boundary_conditions="periodic",
        )

    @pytest.fixture
    def test_fields(self, test_grid):
        """Create fields with gradients to drive non-trivial evolution."""
        fields = ISFieldConfiguration(test_grid)

        X, Y, Z = test_grid.meshgrid()
        fields.rho[:] = 1.0 + 0.05 * np.sin(X)
        fields.n[:] = 0.5 + 0.02 * np.sin(X)
        fields.pressure[:] = fields.rho / 3.0  # Radiation EOS

        # Add velocity and dissipative fluxes
        fields.u_mu[..., 0] = 1.0
        fields.u_mu[..., 1] = 0.05 * np.sin(X)
        fields.Pi[:] = 0.01 * np.sin(X)
        fields.pi_munu[..., 1, 2] = 0.01 * np.sin(X)
        fields.V_mu[..., 1] = 0.02 * np.sin(X)

        fields.apply_constraints()
        return fields

    @pytest.fixture
    def test_coeffs(self):
        """Transport coefficients within Israel-Stewart regime."""
        return TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            diffusion_coefficient=0.05,
            shear_relaxation_time=0.1,  # |τω| = 0.80 < 1
            bulk_relaxation_time=0.05,
            diffusion_relaxation_time=0.1,
        )

    @pytest.mark.parametrize("integration_method", ["split_step", "spectral_imex", "rk4"])
    def test_energy_rhs_consistency_at_t0(
        self, test_grid, test_fields, test_coeffs, integration_method
    ):
        """
        Test that dρ/dt from evolution_equations() matches -∇·T^{i0} at t=0.

        This tests the energy conservation law formulation without any
        time integration complications.
        """
        solver = SpectralISHydrodynamics(test_grid, test_fields, test_coeffs)
        conservation = ConservationLaws(test_fields, test_coeffs, solver.spectral)

        # Compute LHS: dρ/dt from evolution_equations()
        evolution_rhs = conservation.evolution_equations()
        drho_dt_evolution = evolution_rhs.get("drho_dt", np.zeros_like(test_fields.rho))

        # Compute RHS: -∇·T^{i0} from stress tensor
        T = conservation.stress_energy_tensor()
        energy_flux = T[..., 1:4, 0]  # T^{i0} for i=1,2,3
        div_energy_flux = solver.spectral.spatial_divergence(energy_flux)
        drho_dt_from_T = -div_energy_flux

        # These should match EXACTLY at t=0 (no time integration)
        max_diff = np.max(np.abs(drho_dt_evolution - drho_dt_from_T))
        typical_scale = np.max(np.abs(drho_dt_evolution)) + 1e-15
        relative_diff = max_diff / typical_scale

        # Should be exact to machine precision
        assert relative_diff < 1e-12, (
            f"Energy RHS mismatch for {integration_method}: "
            f"{relative_diff:.2e} (should be < 1e-12)"
        )

    @pytest.mark.parametrize("integration_method", ["split_step", "spectral_imex", "rk4"])
    def test_momentum_rhs_consistency_at_t0(
        self, test_grid, test_fields, test_coeffs, integration_method
    ):
        """
        Test that d(ρu^i)/dt from evolution_equations() matches -∇·T^{ij} at t=0.

        This is the critical test for momentum conservation - should be exact
        at t=0 regardless of integration method.
        """
        solver = SpectralISHydrodynamics(test_grid, test_fields, test_coeffs)
        conservation = ConservationLaws(test_fields, test_coeffs, solver.spectral)

        # Compute LHS: d(ρu^i)/dt from evolution_equations()
        evolution_rhs = conservation.evolution_equations()
        dmom_dt_evolution = evolution_rhs.get("dmom_dt", np.zeros_like(test_fields.u_mu[..., 1:4]))

        # Compute RHS: -∇·T^{ij} from stress tensor (for each j=1,2,3)
        T = conservation.stress_energy_tensor()
        dmom_dt_from_T = np.zeros_like(dmom_dt_evolution)

        for j in range(1, 4):  # j = 1, 2, 3 (spatial momentum components)
            momentum_flux = T[..., 1:4, j]  # T^{ij} for i=1,2,3
            div_momentum_flux = solver.spectral.spatial_divergence(momentum_flux)
            dmom_dt_from_T[..., j - 1] = -div_momentum_flux

        # Check each momentum component
        for j in range(3):
            max_diff = np.max(np.abs(dmom_dt_evolution[..., j] - dmom_dt_from_T[..., j]))
            typical_scale = np.max(np.abs(dmom_dt_evolution[..., j])) + 1e-15
            relative_diff = max_diff / typical_scale

            # Should be exact to machine precision
            assert relative_diff < 1e-12, (
                f"Momentum-{j+1} RHS mismatch for {integration_method}: "
                f"{relative_diff:.2e} (should be < 1e-12)"
            )

    @pytest.mark.parametrize("integration_method", ["split_step", "spectral_imex", "rk4"])
    def test_particle_rhs_consistency_at_t0(
        self, test_grid, test_fields, test_coeffs, integration_method
    ):
        """
        Test that dn/dt from evolution_equations() matches -∇·N^i at t=0.

        Tests particle number conservation with diffusion current.
        """
        solver = SpectralISHydrodynamics(test_grid, test_fields, test_coeffs)
        conservation = ConservationLaws(test_fields, test_coeffs, solver.spectral)

        # Compute LHS: dn/dt from evolution_equations()
        evolution_rhs = conservation.evolution_equations()
        dn_dt_evolution = evolution_rhs.get("dn_dt", np.zeros_like(test_fields.n))

        # Compute RHS: -∇·N^i where N^i = n u^i + V^i (Landau frame)
        n_flux = test_fields.n[..., np.newaxis] * test_fields.u_mu[..., 1:4]
        diffusion_flux = test_fields.V_mu[..., 1:4]
        total_flux = n_flux + diffusion_flux
        div_particle_flux = solver.spectral.spatial_divergence(total_flux)
        dn_dt_from_N = -div_particle_flux

        # These should match EXACTLY at t=0
        max_diff = np.max(np.abs(dn_dt_evolution - dn_dt_from_N))
        typical_scale = np.max(np.abs(dn_dt_evolution)) + 1e-15
        relative_diff = max_diff / typical_scale

        # Should be exact to machine precision
        assert relative_diff < 1e-12, (
            f"Particle RHS mismatch for {integration_method}: "
            f"{relative_diff:.2e} (should be < 1e-12)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
