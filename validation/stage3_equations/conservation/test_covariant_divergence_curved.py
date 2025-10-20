#!/usr/bin/env -S uv run python
"""
Test covariant divergence in curved spacetime.

This test validates the critical bug fix in conservation.py where connection terms
from Christoffel symbols were incorrectly summed. For Bjorken/Milne metric, the
correct covariant divergence must include:

    ∇_i T^iν = ∂_i T^iν + Γ^i_{iλ}T^λν + Γ^ν_{iλ}T^iλ

This is Stage 3A.4 from the validation plan.
"""

import numpy as np
import pytest

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MilneMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations.conservation import ConservationLaws


def test_milne_metric_connection_terms():
    """
    Test that connection terms from Christoffel symbols are correctly computed
    in Milne (Bjorken) coordinates.

    For Milne metric with proper time τ, the non-zero Christoffel symbols are:
        Γ^τ_ηη = τ
        Γ^η_τη = Γ^η_ητ = 1/τ

    For uniform Bjorken flow, the stress-energy tensor has:
        T^00 = ρ (energy density)
        T^11 = T^22 = T^33 = p (pressure, assuming isotropy in transverse)

    The connection correction for energy equation (ν=0) should be:
        Σ_i (Γ^i_{iλ}T^λ0 + Γ^0_{iλ}T^iλ) = Γ^η_ητ·T^00 + Γ^0_ηη·T^ηη
                                             = (1/τ)·ρ + τ·p_η

    For radiation fluid (p = ρ/3) in Bjorken flow, this gives -(ρ+p)/τ as expected.
    """
    # Setup: uniform Bjorken flow at τ = 0.6 fm/c
    tau = 0.6
    metric = MilneMetric(tau_value=tau)

    # Create grid with Milne metric
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(4, 4, 4),
        boundary_conditions="periodic",
        metric=metric,
    )

    # Initialize fields for radiation fluid
    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 0.105  # Energy density
    fields.pressure[:] = 0.035  # p = ρ/3
    fields.u_mu[..., 0] = 1.0  # Rest frame

    # Zero dissipative fields
    fields.Pi[:] = 0.0
    fields.pi_munu[:] = 0.0
    fields.V_mu[:] = 0.0

    # Compute evolution equations
    coeffs = TransportCoefficients(shear_viscosity=0.1, bulk_viscosity=0.05)
    conservation = ConservationLaws(fields, coeffs)
    rhs = conservation.evolution_equations()

    # Check energy equation: ∂_τ ρ = -(ρ+p)/τ
    # For Bjorken flow: drho/dt = -(ρ+p)/τ
    rho_mean = np.mean(fields.rho)
    p_mean = np.mean(fields.pressure)
    expected_drho_dt = -(rho_mean + p_mean) / tau

    drho_dt_mean = np.mean(rhs["drho_dt"])

    # Verify connection terms give correct result
    assert np.allclose(
        drho_dt_mean, expected_drho_dt, rtol=1e-10
    ), f"Energy equation: expected {expected_drho_dt:.6f}, got {drho_dt_mean:.6f}"

    print("✓ Milne metric connection terms correct:")
    print(f"  τ = {tau:.2f} fm/c")
    print(f"  ρ = {rho_mean:.6f}")
    print(f"  p = {p_mean:.6f}")
    print(f"  drho/dt = {drho_dt_mean:.6f} (expected {expected_drho_dt:.6f})")


def test_flat_space_no_connection():
    """
    Test that in flat Minkowski spacetime, connection terms are zero.

    This verifies that the Christoffel symbol corrections only apply in
    curved spacetime and don't affect flat space calculations.
    """
    # Setup: flat Minkowski space
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(4, 4, 4),
        boundary_conditions="periodic",
        # No metric → defaults to flat Minkowski
    )

    # Uniform fields (no spatial gradients)
    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0
    fields.Pi[:] = 0.0
    fields.pi_munu[:] = 0.0
    fields.V_mu[:] = 0.0

    # Compute evolution
    coeffs = TransportCoefficients(shear_viscosity=0.1, bulk_viscosity=0.05)
    conservation = ConservationLaws(fields, coeffs)
    rhs = conservation.evolution_equations()

    # In flat space with uniform fields, all derivatives should be zero
    assert np.allclose(
        rhs["drho_dt"], 0.0, atol=1e-14
    ), "Flat space with uniform fields should have drho/dt = 0"

    assert np.allclose(
        rhs["dmom_dt"], 0.0, atol=1e-14
    ), "Flat space with uniform fields should have dmom/dt = 0"

    print("✓ Flat Minkowski space: no spurious connection terms")


def test_connection_term_structure():
    """
    Test that connection terms have the correct mathematical structure.

    Verifies that the implementation matches:
        connection_energy = Σ_i Σ_λ (Γ^i_{iλ}T^λ0 + Γ^0_{iλ}T^iλ)
        connection_momentum_j = Σ_i Σ_λ (Γ^i_{iλ}T^λj + Γ^j_{iλ}T^iλ)

    This is the bug that was fixed - previously the summations were incorrect.
    """
    tau = 1.0
    metric = MilneMetric(tau_value=tau)

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(4, 4, 4),
        boundary_conditions="periodic",
        metric=metric,
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0
    fields.Pi[:] = 0.0
    fields.pi_munu[:] = 0.0
    fields.V_mu[:] = 0.0

    # Manually compute connection terms
    T = fields.compute_stress_energy_tensor()
    gamma = metric.christoffel_symbols

    # Energy equation connection
    connection_energy_manual = 0.0
    for i in range(1, 4):  # Spatial indices
        for lam in range(4):  # All indices
            connection_energy_manual += gamma[i, i, lam] * np.mean(T[..., lam, 0])
            connection_energy_manual += gamma[0, i, lam] * np.mean(T[..., i, lam])

    # Get RHS from conservation laws
    coeffs = TransportCoefficients(shear_viscosity=0.0, bulk_viscosity=0.0)
    conservation = ConservationLaws(fields, coeffs)
    rhs = conservation.evolution_equations()

    # For uniform fields with no gradients, ∂_i T^iν = 0
    # So: drho/dt = -connection_correction
    drho_dt = np.mean(rhs["drho_dt"])

    # Verify structure
    assert np.allclose(
        drho_dt, -connection_energy_manual, rtol=1e-10
    ), f"Connection term structure incorrect: expected {-connection_energy_manual:.6f}, got {drho_dt:.6f}"

    print("✓ Connection term mathematical structure verified")
    print(f"  Manual calculation: {connection_energy_manual:.6f}")
    print(f"  Implementation gives: drho/dt = {drho_dt:.6f}")


if __name__ == "__main__":
    test_milne_metric_connection_terms()
    test_flat_space_no_connection()
    test_connection_term_structure()
    print()
    print("=" * 80)
    print("✓✓✓ ALL COVARIANT DIVERGENCE TESTS PASSED ✓✓✓")
    print("=" * 80)
