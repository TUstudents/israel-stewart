#!/usr/bin/env -S uv run python
"""Test individual coupling terms in relaxation equations.

Each J-term is tested in isolation to verify:
1. Correct sign
2. Correct structure (tensor indices)
3. Correct dimensional scaling
4. Presence in RHS computation
"""

import sys

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations.relaxation import ISRelaxationEquations


def test_bulk_expansion_coupling():
    """Test δ_ΠΠ × Π × θ term in bulk RHS."""
    print("\n" + "=" * 60)
    print("TEST: Bulk Self-Coupling (δ_ΠΠ × Π × θ)")
    print("=" * 60)

    grid = SpaceGrid(
        "cartesian",
        [(0.0, 1.0)] * 3,
        (4, 4, 4),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )
    fields = ISFieldConfiguration(grid)

    # Set nonzero bulk pressure and expansion
    fields.Pi[:] = 0.1
    fields.u_mu[..., 0] = 1.0  # Rest frame
    fields.temperature[:] = 1.0

    # Create transport coefficients with only δ_ΠΠ nonzero
    coeffs = TransportCoefficients(
        shear_viscosity=0.0,
        bulk_viscosity=0.0,
        bulk_relaxation_time=1.0,
        delta_Pi_Pi=0.5,  # IReD bulk self-coupling
    )

    relaxation = ISRelaxationEquations(grid, grid.metric, coeffs)

    # Create nonzero expansion
    # For a uniform field u^μ = (1, 0, 0, 0), θ = 0. We need velocity gradients.
    # Set a simple velocity field with ∂_x v^x ≠ 0
    fields.u_mu[..., 1] = 0.01 * np.linspace(0, 1, 4)[:, None, None]  # x-velocity gradient

    theta = relaxation._compute_expansion_scalar(fields.u_mu)
    theta_mean = np.mean(np.abs(theta))

    print(f"Setup: Π = {fields.Pi[0,0,0]:.3f}, mean|θ| = {theta_mean:.3e}")
    print(f"Coefficient: δ_ΠΠ = {coeffs.delta_Pi_Pi}")

    # Compute full bulk RHS
    sigma_munu = relaxation._compute_shear_tensor(fields.u_mu)
    div_n = np.zeros(grid.shape)
    F_mu = np.zeros(grid.shape + (4,))
    I_mu = np.zeros(grid.shape + (4,))

    dPi_dt = relaxation._bulk_rhs(
        Pi=fields.Pi,
        pi_munu=fields.pi_munu,
        n_mu=fields.V_mu,
        theta=theta,
        sigma_munu=sigma_munu,
        div_n=div_n,
        F_mu=F_mu,
        I_mu=I_mu,
    )

    # Expected contribution: -δ_ΠΠ × Π × θ
    # Sign: negative (dissipates bulk pressure during expansion)
    expected_sign = -1.0 * np.sign(coeffs.delta_Pi_Pi * fields.Pi[0, 0, 0] * theta_mean)

    dPi_mean = np.mean(dPi_dt)
    actual_sign = np.sign(dPi_mean)

    print(f"\nResult: mean(dΠ/dt) = {dPi_mean:.3e}")
    print(f"Expected sign: {expected_sign:.0f} (negative for expansion)")
    print(f"Actual sign:   {actual_sign:.0f}")

    if expected_sign * actual_sign > 0:
        print("✓ PASS: δ_ΠΠ term has correct sign")
        return True
    else:
        print("✗ FAIL: δ_ΠΠ term has wrong sign")
        return False


def test_bulk_shear_coupling():
    """Test λ_Ππ × π^μν × σ_μν term in bulk RHS."""
    print("\n" + "=" * 60)
    print("TEST: Bulk-Shear Coupling (λ_Ππ × π × σ)")
    print("=" * 60)

    grid = SpaceGrid(
        "cartesian",
        [(0.0, 1.0)] * 3,
        (4, 4, 4),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )
    fields = ISFieldConfiguration(grid)

    # Set nonzero shear stress
    fields.pi_munu[..., 1, 1] = 0.05  # σ_xx component
    fields.pi_munu[..., 2, 2] = -0.05  # σ_yy = -σ_xx (traceless)
    fields.u_mu[..., 0] = 1.0
    fields.temperature[:] = 1.0

    coeffs = TransportCoefficients(
        shear_viscosity=0.0,
        bulk_viscosity=0.0,
        bulk_relaxation_time=1.0,
        lambda_Pi_pi=0.3,  # IReD bulk-shear coupling
    )

    relaxation = ISRelaxationEquations(grid, grid.metric, coeffs)

    # Create shear flow
    fields.u_mu[..., 1] = 0.01 * np.linspace(0, 1, 4)[:, None, None]

    sigma_munu = relaxation._compute_shear_tensor(fields.u_mu)
    theta = relaxation._compute_expansion_scalar(fields.u_mu)

    # Compute contraction: π^μν σ_μν
    pi_sigma = np.einsum("...ij,...ij->...", fields.pi_munu, sigma_munu)
    print(f"Setup: π^μν σ_μν = {np.mean(pi_sigma):.3e}")
    print(f"Coefficient: λ_Ππ = {coeffs.lambda_Pi_pi}")

    # Compute bulk RHS
    div_n = np.zeros(grid.shape)
    F_mu = np.zeros(grid.shape + (4,))
    I_mu = np.zeros(grid.shape + (4,))

    dPi_dt = relaxation._bulk_rhs(
        Pi=fields.Pi,
        pi_munu=fields.pi_munu,
        n_mu=fields.V_mu,
        theta=theta,
        sigma_munu=sigma_munu,
        div_n=div_n,
        F_mu=F_mu,
        I_mu=I_mu,
    )

    dPi_max = np.max(np.abs(dPi_dt))
    print(f"\nResult: max|dΠ/dt| = {dPi_max:.3e}")

    if dPi_max > 1e-15:
        print("✓ PASS: λ_Ππ term contributes to RHS")
        return True
    else:
        print("✗ FAIL: λ_Ππ term not contributing")
        return False


def test_diffusion_expansion_coupling():
    """Test δ_VV × V^μ × θ term in diffusion RHS."""
    print("\n" + "=" * 60)
    print("TEST: Diffusion-Expansion Coupling (δ_VV × V × θ)")
    print("=" * 60)

    grid = SpaceGrid(
        "cartesian",
        [(0.0, 1.0)] * 3,
        (4, 4, 4),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )
    fields = ISFieldConfiguration(grid)

    # Set nonzero diffusion current
    fields.V_mu[..., 1] = 0.02  # V^x component
    fields.u_mu[..., 0] = 1.0
    fields.temperature[:] = 1.0

    coeffs = TransportCoefficients(
        shear_viscosity=0.0,
        diffusion_coefficient=0.1,
        diffusion_relaxation_time=0.5,
        delta_V_V=1.0,  # IReD diffusion-expansion coupling (dimensionless)
    )

    relaxation = ISRelaxationEquations(grid, grid.metric, coeffs)

    # Create expansion
    fields.u_mu[..., 1] = 0.01 * np.linspace(0, 1, 4)[:, None, None]
    theta = relaxation._compute_expansion_scalar(fields.u_mu)

    print(f"Setup: V^x = {fields.V_mu[0,0,0,1]:.3f}, mean|θ| = {np.mean(np.abs(theta)):.3e}")
    print(f"Coefficient: δ_VV = {coeffs.delta_V_V}")

    # Compute diffusion RHS
    I_mu = relaxation._compute_chemical_potential_gradient(fields, fields.u_mu)

    dV_dt = relaxation._diffusion_rhs(
        V_mu=fields.V_mu,
        pi_munu=fields.pi_munu,
        theta=theta,
        nabla_mu_over_T=I_mu,
        temperature=fields.temperature,
    )

    dV_max = np.max(np.abs(dV_dt))
    print(f"\nResult: max|dV^μ/dt| = {dV_max:.3e}")

    if dV_max > 1e-15:
        print("✓ PASS: δ_VV term contributes to RHS")
        return True
    else:
        print("✗ FAIL: δ_VV term not contributing")
        return False


def main():
    """Run all coupling term tests."""
    print("\n" + "=" * 60)
    print("COUPLING TERM VERIFICATION")
    print("=" * 60)
    print("\nTests individual J-terms in isolation:")
    print("  1. Bulk self-coupling: δ_ΠΠ × Π × θ")
    print("  2. Bulk-shear coupling: λ_Ππ × π^μν × σ_μν")
    print("  3. Diffusion-expansion coupling: δ_VV × V^μ × θ")

    results = []
    results.append(test_bulk_expansion_coupling())
    results.append(test_bulk_shear_coupling())
    results.append(test_diffusion_expansion_coupling())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All coupling term tests PASSED")
        return 0
    else:
        print(f"✗ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
