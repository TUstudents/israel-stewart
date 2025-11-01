#!/usr/bin/env python3
"""
Diagnostic script to investigate momentum balance equation residuals.

Tests whether the 30% residual in test_momentum_balance_equation is due to:
1. Numerical truncation (finite difference vs RK2)
2. Product rule inconsistency (evolving ρ and u separately)
3. Actual implementation bug in momentum evolution

The test that fails checks: ∂_t(ρu^j) + ∇·T^{ij} = 0 pointwise
"""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations import ConservationLaws
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def main():
    print("=" * 80)
    print("MOMENTUM BALANCE DIAGNOSTIC")
    print("=" * 80)
    print()

    # Setup: Same as test_momentum_balance_equation
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)

    # Initial conditions with gradient
    X, Y, Z = grid.meshgrid()
    fields.rho[:] = 1.0 + 0.05 * np.sin(X)
    fields.n[:] = 0.5 + 0.02 * np.sin(X)
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0

    # Add momentum
    fields.u_mu[..., 1] = 0.05 * np.sin(X)
    fields.apply_constraints()

    coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        diffusion_coefficient=0.05,
        shear_relaxation_time=0.1,
        bulk_relaxation_time=0.05,
        diffusion_relaxation_time=0.1,
    )

    solver = SpectralISHydrodynamics(grid, fields, coeffs)
    # ConservationLaws needs the spectral solver object, not the hydrodynamics wrapper
    conservation = ConservationLaws(fields, coeffs, solver.spectral)

    # ========================================================================
    # CHECK 1: Balance equation at t=0 (should be exact)
    # ========================================================================
    print("CHECK 1: Momentum balance at t=0 (initial state)")
    print("-" * 80)

    # Get RHS from conservation laws at t=0
    evolution_rhs = conservation.evolution_equations()
    dmom_dt_analytical = evolution_rhs.get("dmom_dt", np.zeros_like(fields.u_mu[..., 1:4]))

    # Compute stress tensor at t=0
    T = conservation.stress_energy_tensor()
    momentum_flux = T[..., 1:4, 1]  # T^{i1} for i=1,2,3 (x-momentum)
    div_flux = grid.divergence(momentum_flux, order=2)

    # Balance: d(ρu^1)/dt = -∇·T^{i1}
    expected_dmom_dt = -div_flux

    # Compare
    residual_t0 = dmom_dt_analytical[..., 0] - expected_dmom_dt  # x-component only
    max_residual_t0 = np.max(np.abs(residual_t0))
    typical_scale_t0 = np.max(np.abs(dmom_dt_analytical[..., 0])) + 1e-15

    print(
        f"Analytical d(ρu^1)/dt range: [{np.min(dmom_dt_analytical[..., 0]):.6e}, "
        f"{np.max(dmom_dt_analytical[..., 0]):.6e}]"
    )
    print(
        f"Expected from -∇·T^{{i1}} range: [{np.min(expected_dmom_dt):.6e}, "
        f"{np.max(expected_dmom_dt):.6e}]"
    )
    print(f"Max residual: {max_residual_t0:.6e}")
    print(f"Relative residual: {max_residual_t0 / typical_scale_t0:.6e}")

    if max_residual_t0 / typical_scale_t0 < 1e-10:
        print("✅ PASS: Balance equation exact at t=0")
    else:
        print(
            f"❌ FAIL: Balance equation has {max_residual_t0 / typical_scale_t0:.2e} error at t=0"
        )
        print("This indicates a bug in conservation law implementation!")

    print()

    # ========================================================================
    # CHECK 2: Evolution with different timesteps
    # ========================================================================
    print("CHECK 2: Convergence with timestep refinement")
    print("-" * 80)

    dt_values = [0.001, 0.0005, 0.00025]
    residuals = []

    for dt in dt_values:
        # Reset to initial state
        fields_test = ISFieldConfiguration(grid)
        fields_test.rho[:] = 1.0 + 0.05 * np.sin(X)
        fields_test.n[:] = 0.5 + 0.02 * np.sin(X)
        fields_test.pressure[:] = fields_test.rho / 3.0
        fields_test.u_mu[..., 0] = 1.0
        fields_test.u_mu[..., 1] = 0.05 * np.sin(X)
        fields_test.apply_constraints()

        solver_test = SpectralISHydrodynamics(grid, fields_test, coeffs)

        # Save initial momentum density
        momentum_0 = (fields_test.rho * fields_test.u_mu[..., 1]).copy()

        # Take timestep
        solver_test.time_step(dt)

        # Compute ∂_t(ρu^1) numerically
        momentum_1 = solver_test.fields.rho * solver_test.fields.u_mu[..., 1]
        dmomentum_dt_numerical = (momentum_1 - momentum_0) / dt

        # Compute expected from stress tensor after evolution
        conservation_evolved = ConservationLaws(solver_test.fields, coeffs, solver_test.spectral)
        T_evolved = conservation_evolved.stress_energy_tensor()
        momentum_flux_evolved = T_evolved[..., 1:4, 1]
        div_flux_evolved = grid.divergence(momentum_flux_evolved, order=2)
        expected_dmomentum_dt = -div_flux_evolved

        # Residual
        residual = dmomentum_dt_numerical - expected_dmomentum_dt
        max_residual = np.max(np.abs(residual))
        typical_scale = np.max(np.abs(dmomentum_dt_numerical)) + 1e-15
        relative_residual = max_residual / typical_scale

        residuals.append(relative_residual)
        print(f"dt = {dt:.6f}: relative residual = {relative_residual:.6e}")

    # Check convergence rate
    if len(residuals) >= 2:
        ratio_1 = residuals[0] / residuals[1]
        ratio_2 = residuals[1] / residuals[2]
        print(f"\nConvergence ratio (should be ~2 for RK2): {ratio_1:.2f}, {ratio_2:.2f}")

        if 1.5 < ratio_1 < 2.5 and 1.5 < ratio_2 < 2.5:
            print("✅ PASS: Residual scales as O(dt^2), consistent with RK2 truncation")
        else:
            print("⚠️  WARNING: Unexpected convergence rate")

    print()

    # ========================================================================
    # CHECK 3: Product rule analysis
    # ========================================================================
    print("CHECK 3: Product rule decomposition")
    print("-" * 80)

    # Reset and evolve with dt=0.001
    fields_test = ISFieldConfiguration(grid)
    fields_test.rho[:] = 1.0 + 0.05 * np.sin(X)
    fields_test.n[:] = 0.5 + 0.02 * np.sin(X)
    fields_test.pressure[:] = fields_test.rho / 3.0
    fields_test.u_mu[..., 0] = 1.0
    fields_test.u_mu[..., 1] = 0.05 * np.sin(X)
    fields_test.apply_constraints()

    solver_test = SpectralISHydrodynamics(grid, fields_test, coeffs)

    # Save initial values
    rho_0 = fields_test.rho.copy()
    u_0 = fields_test.u_mu[..., 1].copy()
    momentum_0 = (rho_0 * u_0).copy()

    dt = 0.001
    solver_test.time_step(dt)

    rho_1 = solver_test.fields.rho.copy()
    u_1 = solver_test.fields.u_mu[..., 1].copy()
    momentum_1 = (rho_1 * u_1).copy()

    # Three ways to compute d(ρu)/dt:
    # Method 1: Direct finite difference of product
    dmomentum_dt_direct = (momentum_1 - momentum_0) / dt

    # Method 2: Product rule with separate derivatives
    drho_dt = (rho_1 - rho_0) / dt
    du_dt = (u_1 - u_0) / dt
    dmomentum_dt_product = u_0 * drho_dt + rho_0 * du_dt

    # Method 3: What conservation laws give
    conservation_final = ConservationLaws(solver_test.fields, coeffs, solver_test.spectral)
    T_final = conservation_final.stress_energy_tensor()
    flux_final = T_final[..., 1:4, 1]
    div_flux_final = grid.divergence(flux_final, order=2)
    dmomentum_dt_conservation = -div_flux_final

    print(
        f"Direct d(ρu)/dt range: [{np.min(dmomentum_dt_direct):.6e}, "
        f"{np.max(dmomentum_dt_direct):.6e}]"
    )
    print(
        f"Product rule d(ρu)/dt range: [{np.min(dmomentum_dt_product):.6e}, "
        f"{np.max(dmomentum_dt_product):.6e}]"
    )
    print(
        f"Conservation -∇·T range: [{np.min(dmomentum_dt_conservation):.6e}, "
        f"{np.max(dmomentum_dt_conservation):.6e}]"
    )

    diff_direct_product = np.max(np.abs(dmomentum_dt_direct - dmomentum_dt_product))
    diff_direct_conservation = np.max(np.abs(dmomentum_dt_direct - dmomentum_dt_conservation))

    print(f"\n|Direct - Product Rule|: {diff_direct_product:.6e}")
    print(f"|Direct - Conservation|: {diff_direct_conservation:.6e}")

    if diff_direct_product < 1e-10:
        print("✅ Product rule consistent (no issue with evolving ρ and u separately)")
    else:
        print(f"⚠️  Product rule discrepancy: {diff_direct_product:.2e}")

    print()

    # ========================================================================
    # CHECK 4: Spatial pattern analysis
    # ========================================================================
    print("CHECK 4: Spatial pattern of residuals")
    print("-" * 80)

    residual_pattern = dmomentum_dt_direct - dmomentum_dt_conservation

    print(f"Residual mean: {np.mean(residual_pattern):.6e}")
    print(f"Residual std: {np.std(residual_pattern):.6e}")
    print(f"Residual max: {np.max(residual_pattern):.6e}")
    print(f"Residual min: {np.min(residual_pattern):.6e}")

    # Check if residual has same spatial pattern as initial perturbation
    correlation = np.corrcoef(residual_pattern.flatten(), np.sin(X).flatten())[0, 1]
    print(f"Correlation with sin(x) perturbation: {correlation:.3f}")

    if abs(correlation) > 0.9:
        print("⚠️  Residual strongly correlated with initial perturbation")
        print("    This suggests systematic error in gradient computation")
    else:
        print("✅ Residual not strongly correlated with initial perturbation")

    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if max_residual_t0 / typical_scale_t0 > 1e-10:
        print("🔴 CRITICAL: Balance equation NOT satisfied at t=0")
        print("   This is a BUG in conservation law implementation")
    elif residuals[0] > 0.1:
        print("🟡 WARNING: Large residual (30%) after time evolution")
        if 1.5 < ratio_1 < 2.5:
            print("   Residual scales properly with dt → likely numerical truncation")
            print("   Consider: reduce dt, reduce |τω|, or relax test tolerance")
        else:
            print("   Residual does NOT scale properly → possible bug in time stepping")
    else:
        print("🟢 OK: Residuals within acceptable range")

    print()


if __name__ == "__main__":
    main()
