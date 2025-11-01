#!/usr/bin/env python3
"""
Diagnose whether the momentum balance discrepancy is due to
spectral divergence vs finite difference divergence mismatch.
"""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations import ConservationLaws
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def main():
    print("=" * 80)
    print("DIVERGENCE METHOD COMPARISON")
    print("=" * 80)
    print()

    # Setup grid
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)

    # Initial conditions
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
    conservation = ConservationLaws(fields, coeffs, solver.spectral)

    # Get stress tensor at t=0
    T = conservation.stress_energy_tensor()

    print("Comparing divergence methods for momentum flux T^{i1}:")
    print("-" * 80)
    print()

    # Extract momentum flux (T^i1 for i=1,2,3)
    momentum_flux = T[..., 1:4, 1]

    # Method 1: Spectral divergence (used by evolution_equations)
    div_spectral = solver.spectral.spatial_divergence(momentum_flux)

    # Method 2: Finite difference (used by test)
    div_finite_diff = grid.divergence(momentum_flux, order=2)

    print(f"Spectral divergence range: [{np.min(div_spectral):.6e}, {np.max(div_spectral):.6e}]")
    print(
        f"Finite diff divergence range: [{np.min(div_finite_diff):.6e}, {np.max(div_finite_diff):.6e}]"
    )
    print()

    # Compute difference
    diff = div_spectral - div_finite_diff
    max_diff = np.max(np.abs(diff))
    typical_scale = np.max(np.abs(div_spectral)) + 1e-15
    relative_diff = max_diff / typical_scale

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Relative difference: {relative_diff:.6e} ({relative_diff * 100:.2f}%)")
    print()

    # Check if this matches the observed discrepancy
    if 0.03 < relative_diff < 0.05:
        print("✅ This matches the 3.6% discrepancy seen in the diagnostic!")
        print("   The issue is: evolution_equations() uses spectral divergence")
        print("   while test uses finite difference divergence.")
    elif relative_diff < 1e-10:
        print("❌ Divergence methods agree - the bug is elsewhere")
    else:
        print(f"⚠️  Divergence methods differ by {relative_diff * 100:.2f}%")

    print()
    print("=" * 80)
    print("TESTING evolution_equations() CONSISTENCY")
    print("=" * 80)
    print()

    # Now check if evolution_equations() is self-consistent
    # Get d(ρu^j)/dt from evolution_equations
    evolution_rhs = conservation.evolution_equations()
    dmom_dt_evolution = evolution_rhs.get("dmom_dt", np.zeros_like(fields.u_mu[..., 1:4]))

    # Compute -∇·T^{ij} using SAME method as evolution_equations (spectral)
    dmom_dt_from_T_spectral = np.zeros_like(dmom_dt_evolution)
    for j in range(1, 4):
        momentum_flux_j = T[..., 1:4, j]
        dmom_dt_from_T_spectral[..., j - 1] = -solver.spectral.spatial_divergence(momentum_flux_j)

    # Compare (should be exact)
    diff_spectral = dmom_dt_evolution[..., 0] - dmom_dt_from_T_spectral[..., 0]  # x-component
    max_diff_spectral = np.max(np.abs(diff_spectral))
    typical_scale_spectral = np.max(np.abs(dmom_dt_evolution[..., 0])) + 1e-15
    relative_diff_spectral = max_diff_spectral / typical_scale_spectral

    print(
        f"evolution_equations() d(ρu^1)/dt range: [{np.min(dmom_dt_evolution[..., 0]):.6e}, "
        f"{np.max(dmom_dt_evolution[..., 0]):.6e}]"
    )
    print(
        f"-∇·T^{{i1}} (spectral) range: [{np.min(dmom_dt_from_T_spectral[..., 0]):.6e}, "
        f"{np.max(dmom_dt_from_T_spectral[..., 0]):.6e}]"
    )
    print(f"Max difference: {max_diff_spectral:.6e}")
    print(f"Relative difference: {relative_diff_spectral:.6e}")
    print()

    if relative_diff_spectral < 1e-10:
        print("✅ evolution_equations() is self-consistent with stress tensor!")
        print("   The test failure is due to using different divergence methods:")
        print("   - evolution_equations(): spectral divergence (FFT)")
        print("   - test: finite difference")
    else:
        print(
            f"❌ CRITICAL BUG: evolution_equations() differs from -∇·T by {relative_diff_spectral:.2e}"
        )
        print("   This indicates an actual bug in the implementation!")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("The test should use the SAME divergence method as evolution_equations().")
    print("For SpectralISHydrodynamics, this means using spectral divergence,")
    print("not finite difference.")
    print()


if __name__ == "__main__":
    main()
