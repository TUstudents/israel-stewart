#!/usr/bin/env python3
"""
Diagnose chemical potential computation for diffusion.

Check:
1. Is μ_B/T being computed correctly?
2. Is ∇(μ_B/T) being computed correctly?
3. Does the magnitude match expectations?
"""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def main():
    print("=" * 80)
    print("CHEMICAL POTENTIAL DIAGNOSTIC")
    print("=" * 80)
    print()

    # Setup matching test
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)

    # Strong particle gradient
    X, Y, Z = grid.meshgrid()
    fields.rho[:] = 1.0
    fields.n[:] = 0.5 + 0.2 * np.sin(X)
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0

    # Set temperature for diffusion physics
    fields.update_temperature_from_eos(eos_type="radiation")

    fields.apply_constraints()

    coeffs = TransportCoefficients(
        shear_viscosity=0.05,
        bulk_viscosity=0.05,
        diffusion_coefficient=0.2,
        shear_relaxation_time=0.1,
        bulk_relaxation_time=0.05,
        diffusion_relaxation_time=0.1,
    )

    solver = SpectralISHydrodynamics(grid, fields, coeffs)

    print("Field Profiles")
    print("-" * 80)

    # Check temperature
    print(f"Energy density ρ: {np.min(fields.rho):.6f} to {np.max(fields.rho):.6f}")
    print(f"Pressure p: {np.min(fields.pressure):.6f} to {np.max(fields.pressure):.6f}")
    print(f"Temperature T: {np.min(fields.temperature):.6f} to {np.max(fields.temperature):.6f}")
    print(f"  → Temperature is {'uniform' if np.ptp(fields.temperature) < 1e-10 else 'varying'}")
    print()

    # Check particle density
    print(f"Particle density n: {np.min(fields.n):.6f} to {np.max(fields.n):.6f}")
    grad_n_x, grad_n_y, grad_n_z = solver.spectral.spatial_gradient(fields.n)
    print(f"Max |∇n|: {np.max(np.abs(grad_n_x)):.6e}")
    print()

    # Compute chemical potential
    mu_over_T = fields.compute_chemical_potential_over_temperature(eos_type="radiation")
    print(f"Chemical potential μ/T: {np.min(mu_over_T):.6f} to {np.max(mu_over_T):.6f}")
    print()

    # Analytical expectation for uniform T
    zeta_3 = 1.202056903
    prefactor = zeta_3 / (np.pi**2)
    T_mean = np.mean(fields.temperature)
    n_eq = prefactor * T_mean**3

    mu_expected_min = np.log(np.min(fields.n) / n_eq)
    mu_expected_max = np.log(np.max(fields.n) / n_eq)

    print(f"Expected μ/T (analytical): {mu_expected_min:.6f} to {mu_expected_max:.6f}")
    print(f"Equilibrium n_eq(T): {n_eq:.6e}")
    print()

    # Compute gradient of μ/T
    grad_mu_x, grad_mu_y, grad_mu_z = solver.spectral.spatial_gradient(mu_over_T)
    grad_mu_mag = np.sqrt(grad_mu_x**2 + grad_mu_y**2 + grad_mu_z**2)

    print("∇(μ/T) magnitude:")
    print(f"  Max |∇(μ/T)|: {np.max(grad_mu_mag):.6e}")
    print(
        f"  Components: ∂μ/∂x={np.max(np.abs(grad_mu_x)):.6e}, "
        f"∂μ/∂y={np.max(np.abs(grad_mu_y)):.6e}, "
        f"∂μ/∂z={np.max(np.abs(grad_mu_z)):.6e}"
    )
    print()

    # Analytical expectation: ∇(μ/T) = (1/n) ∇n for uniform T
    # At peak of gradient (X=0): ∇n ≈ 0.2, n ≈ 0.5
    # So ∇(μ/T) ≈ 0.2/0.5 = 0.4

    # Find point with max gradient
    max_grad_idx = np.unravel_index(np.argmax(grad_mu_mag), grad_mu_mag.shape)
    n_at_max = fields.n[max_grad_idx]
    grad_n_at_max = grad_n_x[max_grad_idx]
    grad_mu_at_max = grad_mu_x[max_grad_idx]

    analytical_grad_mu = grad_n_at_max / n_at_max

    print("At point of maximum gradient:")
    print(f"  n = {n_at_max:.6f}")
    print(f"  ∇n = {grad_n_at_max:.6e}")
    print(f"  Computed ∇(μ/T) = {grad_mu_at_max:.6e}")
    print(f"  Expected ∇(μ/T) = ∇n/n = {analytical_grad_mu:.6e}")
    print(
        f"  Ratio: {grad_mu_at_max/analytical_grad_mu:.6f}"
        if analytical_grad_mu != 0
        else "  Ratio: undefined"
    )
    print()

    # Expected diffusion current
    expected_V = -coeffs.diffusion_coefficient * np.max(grad_mu_mag)
    print("Expected equilibrium diffusion current:")
    print(f"  V ~ -D ∇(μ/T) ~ {expected_V:.6e}")
    print()

    # Evolve and check actual V
    print("=" * 80)
    print("After Evolution")
    print("-" * 80)

    dt = 0.002
    for _ in range(50):  # Enough time to approach equilibrium
        solver.time_step(dt)

    V_mag = np.max(np.linalg.norm(solver.fields.V_mu[..., 1:4], axis=-1))
    print(f"Actual max |V^i|: {V_mag:.6e}")
    print(f"Ratio actual/expected: {V_mag/np.abs(expected_V):.6f}")
    print()

    # Check if it's still evolving
    V_old = V_mag
    for _ in range(50):
        solver.time_step(dt)
    V_new = np.max(np.linalg.norm(solver.fields.V_mu[..., 1:4], axis=-1))

    print(f"After 50 more steps: {V_new:.6e}")
    print(f"Change: {V_new - V_old:.6e} ({100*(V_new/V_old - 1):.2f}%)")
    print()

    if np.abs(V_new - V_old) / V_old < 0.01:
        print("✅ Diffusion current has reached equilibrium")
    else:
        print("⚠️  Diffusion current still evolving")

    print()


if __name__ == "__main__":
    main()
