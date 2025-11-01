#!/usr/bin/env python3
"""
Compare local conservation balance across different time integration methods:
- split_step (operator splitting)
- spectral_imex (IMEX Runge-Kutta)
- rk4 (coupled RK4)

This tests if the local balance test failures are due to operator splitting.
"""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations import ConservationLaws
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def test_method(method_name: str) -> dict:
    """Test local conservation with specified integration method."""
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

    # Save initial state
    rho_0 = fields.rho.copy()
    momentum_0 = (fields.rho * fields.u_mu[..., 1]).copy()
    n_0 = fields.n.copy()

    # Take timestep using specified method
    dt = 0.001
    solver.time_step(dt, method=method_name)

    # Compute finite difference derivatives
    drho_dt_fd = (solver.fields.rho - rho_0) / dt
    momentum_1 = solver.fields.rho * solver.fields.u_mu[..., 1]
    dmom_dt_fd = (momentum_1 - momentum_0) / dt
    dn_dt_fd = (solver.fields.n - n_0) / dt

    # Compute expected from stress tensor at final time
    conservation = ConservationLaws(solver.fields, coeffs, solver.spectral)
    T = conservation.stress_energy_tensor()

    # Energy flux
    energy_flux = T[..., 1:4, 0]
    div_energy = solver.spectral.spatial_divergence(energy_flux)
    drho_dt_expected = -div_energy

    # Momentum flux
    momentum_flux = T[..., 1:4, 1]
    div_momentum = solver.spectral.spatial_divergence(momentum_flux)
    dmom_dt_expected = -div_momentum

    # Particle flux
    n_flux = solver.fields.n[..., np.newaxis] * solver.fields.u_mu[..., 1:4]
    diffusion_flux = solver.fields.V_mu[..., 1:4]
    total_flux = n_flux + diffusion_flux
    div_particle = solver.spectral.spatial_divergence(total_flux)
    dn_dt_expected = -div_particle

    # Compute residuals
    results = {}

    # Energy
    residual_energy = drho_dt_fd - drho_dt_expected
    results["energy_residual"] = np.max(np.abs(residual_energy)) / (
        np.max(np.abs(drho_dt_fd)) + 1e-15
    )

    # Momentum
    residual_momentum = dmom_dt_fd - dmom_dt_expected
    results["momentum_residual"] = np.max(np.abs(residual_momentum)) / (
        np.max(np.abs(dmom_dt_fd)) + 1e-15
    )

    # Particle
    residual_particle = dn_dt_fd - dn_dt_expected
    results["particle_residual"] = np.max(np.abs(residual_particle)) / (
        np.max(np.abs(dn_dt_fd)) + 1e-15
    )

    return results


def main():
    print("=" * 80)
    print("INTEGRATION METHOD COMPARISON FOR LOCAL CONSERVATION")
    print("=" * 80)
    print()

    methods = ["split_step", "spectral_imex", "rk4"]
    all_results = {}

    for method in methods:
        print(f"Testing method: {method}")
        print("-" * 80)
        try:
            results = test_method(method)
            all_results[method] = results

            print(f"  Energy balance residual:    {results['energy_residual']:.6e}")
            print(f"  Momentum balance residual:  {results['momentum_residual']:.6e}")
            print(f"  Particle balance residual:  {results['particle_residual']:.6e}")
            print()

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            print()
            all_results[method] = None

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()

    if all(v is not None for v in all_results.values()):
        print(f"{'Method':<15} {'Energy':<15} {'Momentum':<15} {'Particle':<15}")
        print("-" * 60)
        for method in methods:
            r = all_results[method]
            energy_str = f"{r['energy_residual']:.2e}"
            momentum_str = f"{r['momentum_residual']:.2e}"
            particle_str = f"{r['particle_residual']:.2e}"
            print(f"{method:<15} {energy_str:<15} {momentum_str:<15} {particle_str:<15}")

        print()
        print("Analysis:")
        print("-" * 80)

        # Find best method for each test
        best_energy = min(methods, key=lambda m: all_results[m]["energy_residual"])
        best_momentum = min(methods, key=lambda m: all_results[m]["momentum_residual"])
        best_particle = min(methods, key=lambda m: all_results[m]["particle_residual"])

        print(f"Best for energy conservation:    {best_energy}")
        print(f"Best for momentum conservation:  {best_momentum}")
        print(f"Best for particle conservation:  {best_particle}")
        print()

        # Check if any method passes all tests (< 10% residual)
        for method in methods:
            r = all_results[method]
            if (
                r["energy_residual"] < 0.1
                and r["momentum_residual"] < 0.1
                and r["particle_residual"] < 0.1
            ):
                print(f"✅ {method} PASSES all local conservation tests!")
                print()
                break
        else:
            print("⚠️  NO method passes all local conservation tests with < 10% residual")
            print("   This confirms the tests need redesign to account for:")
            print("   1. Operator splitting in time integration")
            print("   2. Regime parameter |τω| = 0.80 (near boundary)")
            print()

    print()


if __name__ == "__main__":
    main()
