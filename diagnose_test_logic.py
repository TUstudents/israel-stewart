#!/usr/bin/env python3
"""
Diagnose the test logic issue: comparing time-averaged derivative
with point-wise divergence at final time.
"""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations import ConservationLaws
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def main():
    print("=" * 80)
    print("TEST LOGIC DIAGNOSTIC")
    print("=" * 80)
    print()

    # Setup
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    X, Y, Z = grid.meshgrid()
    fields.rho[:] = 1.0 + 0.05 * np.sin(X)
    fields.n[:] = 0.5 + 0.02 * np.sin(X)
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0
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
    conservation_init = ConservationLaws(fields, coeffs, solver.spectral)

    # Save initial state
    rho_0 = fields.rho.copy()
    momentum_0 = (fields.rho * fields.u_mu[..., 1]).copy()

    # Get stress tensor and divergence at t=0
    T_0 = conservation_init.stress_energy_tensor()
    energy_flux_0 = T_0[..., 1:4, 0]
    div_energy_0 = solver.spectral.spatial_divergence(energy_flux_0)
    drho_dt_expected_0 = -div_energy_0

    momentum_flux_0 = T_0[..., 1:4, 1]
    div_momentum_0 = solver.spectral.spatial_divergence(momentum_flux_0)
    dmom_dt_expected_0 = -div_momentum_0

    # Take timestep
    dt = 0.001
    solver.time_step(dt)

    # Compute finite difference derivatives
    drho_dt_fd = (solver.fields.rho - rho_0) / dt
    momentum_1 = solver.fields.rho * solver.fields.u_mu[..., 1]
    dmom_dt_fd = (momentum_1 - momentum_0) / dt

    # Get stress tensor and divergence at t=dt
    conservation_final = ConservationLaws(solver.fields, coeffs, solver.spectral)
    T_final = conservation_final.stress_energy_tensor()
    energy_flux_final = T_final[..., 1:4, 0]
    div_energy_final = solver.spectral.spatial_divergence(energy_flux_final)
    drho_dt_expected_final = -div_energy_final

    momentum_flux_final = T_final[..., 1:4, 1]
    div_momentum_final = solver.spectral.spatial_divergence(momentum_flux_final)
    dmom_dt_expected_final = -div_momentum_final

    print("ENERGY CONSERVATION:")
    print("-" * 80)
    print(f"Finite difference ∂_t ρ: [{np.min(drho_dt_fd):.6e}, {np.max(drho_dt_fd):.6e}]")
    print(
        f"-∇·T^{{i0}} at t=0:      [{np.min(drho_dt_expected_0):.6e}, {np.max(drho_dt_expected_0):.6e}]"
    )
    print(
        f"-∇·T^{{i0}} at t=dt:     [{np.min(drho_dt_expected_final):.6e}, {np.max(drho_dt_expected_final):.6e}]"
    )
    print()

    # Compare with initial time
    residual_vs_init = drho_dt_fd - drho_dt_expected_0
    rel_res_init = np.max(np.abs(residual_vs_init)) / (np.max(np.abs(drho_dt_fd)) + 1e-15)

    # Compare with final time (what current test does)
    residual_vs_final = drho_dt_fd - drho_dt_expected_final
    rel_res_final = np.max(np.abs(residual_vs_final)) / (np.max(np.abs(drho_dt_fd)) + 1e-15)

    # Compare with average
    drho_dt_expected_avg = -(div_energy_0 + div_energy_final) / 2
    residual_vs_avg = drho_dt_fd - drho_dt_expected_avg
    rel_res_avg = np.max(np.abs(residual_vs_avg)) / (np.max(np.abs(drho_dt_fd)) + 1e-15)

    print(f"Residual vs t=0:       {rel_res_init:.6e}")
    print(f"Residual vs t=dt:      {rel_res_final:.6e}")
    print(f"Residual vs average:   {rel_res_avg:.6e}")
    print()

    print("MOMENTUM CONSERVATION:")
    print("-" * 80)
    print(f"Finite difference ∂_t(ρu^1): [{np.min(dmom_dt_fd):.6e}, {np.max(dmom_dt_fd):.6e}]")
    print(
        f"-∇·T^{{i1}} at t=0:          [{np.min(dmom_dt_expected_0):.6e}, {np.max(dmom_dt_expected_0):.6e}]"
    )
    print(
        f"-∇·T^{{i1}} at t=dt:         [{np.min(dmom_dt_expected_final):.6e}, {np.max(dmom_dt_expected_final):.6e}]"
    )
    print()

    # Compare with initial time
    residual_mom_init = dmom_dt_fd - dmom_dt_expected_0
    rel_res_mom_init = np.max(np.abs(residual_mom_init)) / (np.max(np.abs(dmom_dt_fd)) + 1e-15)

    # Compare with final time
    residual_mom_final = dmom_dt_fd - dmom_dt_expected_final
    rel_res_mom_final = np.max(np.abs(residual_mom_final)) / (np.max(np.abs(dmom_dt_fd)) + 1e-15)

    # Compare with average
    dmom_dt_expected_avg = -(div_momentum_0 + div_momentum_final) / 2
    residual_mom_avg = dmom_dt_fd - dmom_dt_expected_avg
    rel_res_mom_avg = np.max(np.abs(residual_mom_avg)) / (np.max(np.abs(dmom_dt_fd)) + 1e-15)

    print(f"Residual vs t=0:       {rel_res_mom_init:.6e}")
    print(f"Residual vs t=dt:      {rel_res_mom_final:.6e}")
    print(f"Residual vs average:   {rel_res_mom_avg:.6e}")
    print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    if rel_res_avg < 0.1:
        print("✅ Using AVERAGE of initial and final divergence gives good results")
        print("   This is the correct comparison for finite difference derivative")
    elif rel_res_init < 0.1:
        print("✅ Using INITIAL divergence gives good results")
        print("   This suggests using explicit method for comparison")
    else:
        print("⚠️  Neither comparison gives good results")
        print("   This may be due to being close to regime boundary")

    print()


if __name__ == "__main__":
    main()
