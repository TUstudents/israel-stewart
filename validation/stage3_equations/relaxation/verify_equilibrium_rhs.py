#!/usr/bin/env -S uv run python
"""Verify equilibrium RHS = 0 for all relaxation equations.

At equilibrium (uniform fields, no gradients):
- Expansion: θ = 0
- Shear: σ^μν = 0
- Gradients: ∇P = 0, ∇(μ/T) = 0
- Divergence: ∇·V = 0

Expected: All RHS = 0 (equilibrium preserved)
"""

import sys

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations.relaxation import ISRelaxationEquations


def setup_equilibrium_state():
    """Create equilibrium field configuration."""
    grid = SpaceGrid(
        "cartesian",
        [(0.0, 2.0)] * 3,
        (8, 8, 8),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )

    fields = ISFieldConfiguration(grid)

    # Equilibrium: uniform density, pressure, rest frame
    fields.rho[:] = 1.0
    fields.pressure[:] = 0.3
    fields.temperature[:] = 1.0
    fields.u_mu[..., 0] = 1.0  # Rest frame: u^t = 1

    # Zero dissipative fields (equilibrium)
    fields.Pi[:] = 0.0
    fields.pi_munu[:] = 0.0
    fields.V_mu[:] = 0.0

    coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        diffusion_coefficient=0.02,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
        diffusion_relaxation_time=0.4,
    )

    relaxation = ISRelaxationEquations(grid, grid.metric, coeffs)

    return fields, relaxation, coeffs


def verify_bulk_rhs_equilibrium():
    """Test bulk viscous pressure RHS = 0 at equilibrium."""
    print("\n" + "=" * 60)
    print("TEST 1: Bulk RHS at Equilibrium")
    print("=" * 60)

    fields, relaxation, coeffs = setup_equilibrium_state()

    # Compute all required quantities
    theta = relaxation._compute_expansion_scalar(fields.u_mu)
    sigma_munu = relaxation._compute_shear_tensor(fields.u_mu)
    div_n = relaxation._compute_diffusion_divergence(fields.V_mu)
    F_mu = relaxation._compute_pressure_gradient(fields, fields.u_mu)
    I_mu = relaxation._compute_chemical_potential_gradient(fields, fields.u_mu)

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

    # Check results
    max_error = np.max(np.abs(dPi_dt))
    theta_max = np.max(np.abs(theta))
    div_n_max = np.max(np.abs(div_n))
    F_mu_max = np.max(np.abs(F_mu))
    I_mu_max = np.max(np.abs(I_mu))

    print(f"θ (expansion):         max |θ| = {theta_max:.3e}")
    print(f"∇·V (div diffusion):   max |∇·V| = {div_n_max:.3e}")
    print(f"F^μ (pressure grad):   max |F^μ| = {F_mu_max:.3e}")
    print(f"I^μ (chem pot grad):   max |I^μ| = {I_mu_max:.3e}")
    print(f"\nBulk RHS:              max |dΠ/dt| = {max_error:.3e}")

    threshold = 1e-14
    if max_error < threshold:
        print(f"✓ PASS: Bulk RHS < {threshold:.0e} at equilibrium")
        return True
    else:
        print(f"✗ FAIL: Bulk RHS = {max_error:.3e} > {threshold:.0e}")
        return False


def verify_shear_rhs_equilibrium():
    """Test shear stress RHS = 0 at equilibrium."""
    print("\n" + "=" * 60)
    print("TEST 2: Shear RHS at Equilibrium")
    print("=" * 60)

    fields, relaxation, coeffs = setup_equilibrium_state()

    # Compute all required quantities
    theta = relaxation._compute_expansion_scalar(fields.u_mu)
    sigma_munu = relaxation._compute_shear_tensor(fields.u_mu)
    omega_munu = relaxation._compute_vorticity_tensor(fields.u_mu)
    I_mu = relaxation._compute_chemical_potential_gradient(fields, fields.u_mu)

    # Should be zero at equilibrium (no flow)
    sigma_max = np.max(np.abs(sigma_munu))
    print(f"σ^μν (shear tensor):   max |σ^μν| = {sigma_max:.3e}")

    # Compute shear RHS
    dpi_dt = relaxation._shear_rhs(
        pi_munu=fields.pi_munu,
        Pi=fields.Pi,
        V_mu=fields.V_mu,
        theta=theta,
        sigma_munu=sigma_munu,
        omega_munu=omega_munu,
        nabla_mu_over_T=I_mu,
        temperature=fields.temperature,
    )

    max_error = np.max(np.abs(dpi_dt))
    print(f"Shear RHS:             max |dπ^μν/dt| = {max_error:.3e}")

    threshold = 1e-14
    if max_error < threshold:
        print(f"✓ PASS: Shear RHS < {threshold:.0e} at equilibrium")
        return True
    else:
        print(f"✗ FAIL: Shear RHS = {max_error:.3e} > {threshold:.0e}")
        return False


def verify_diffusion_rhs_equilibrium():
    """Test diffusion current RHS = 0 at equilibrium."""
    print("\n" + "=" * 60)
    print("TEST 3: Diffusion RHS at Equilibrium")
    print("=" * 60)

    fields, relaxation, coeffs = setup_equilibrium_state()

    # Compute required quantities
    theta = relaxation._compute_expansion_scalar(fields.u_mu)
    I_mu = relaxation._compute_chemical_potential_gradient(fields, fields.u_mu)

    I_mu_max = np.max(np.abs(I_mu))
    print(f"I^μ = ∇^μ(μ/T):        max |I^μ| = {I_mu_max:.3e}")

    # Compute diffusion RHS
    dV_dt = relaxation._diffusion_rhs(
        V_mu=fields.V_mu,
        pi_munu=fields.pi_munu,
        theta=theta,
        nabla_mu_over_T=I_mu,
        temperature=fields.temperature,
    )

    max_error = np.max(np.abs(dV_dt))
    print(f"Diffusion RHS:         max |dV^μ/dt| = {max_error:.3e}")

    threshold = 1e-14
    if max_error < threshold:
        print(f"✓ PASS: Diffusion RHS < {threshold:.0e} at equilibrium")
        return True
    else:
        print(f"✗ FAIL: Diffusion RHS = {max_error:.3e} > {threshold:.0e}")
        return False


def main():
    """Run all equilibrium verification tests."""
    print("\n" + "=" * 60)
    print("EQUILIBRIUM RHS VERIFICATION")
    print("=" * 60)
    print("\nPhysics: At equilibrium (uniform fields, no gradients)")
    print("Expected: All RHS = 0 (dissipation vanishes)")
    print("\nThis verifies:")
    print("  1. Bulk pressure relaxation: dΠ/dt = 0")
    print("  2. Shear stress relaxation: dπ^μν/dt = 0")
    print("  3. Diffusion current relaxation: dV^μ/dt = 0")

    results = []
    results.append(verify_bulk_rhs_equilibrium())
    results.append(verify_shear_rhs_equilibrium())
    results.append(verify_diffusion_rhs_equilibrium())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All equilibrium RHS tests PASSED")
        print("\nConclusion: Relaxation equations correctly preserve equilibrium")
        return 0
    else:
        print(f"✗ {total - passed} test(s) FAILED")
        print("\nAction required: Check relaxation equation implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
