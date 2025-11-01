#!/usr/bin/env python3
"""
Diagnose why diffusion is not reducing particle gradient in test_diffusion_conserves_particles.

Investigation:
1. Is V^μ actually evolving?
2. Is V^μ contributing to particle flux?
3. Is the relaxation equation for V^μ working?
4. What is the magnitude of diffusion vs advection?
"""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.equations import ConservationLaws
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def main():
    print("=" * 80)
    print("DIFFUSION FAILURE DIAGNOSTIC")
    print("=" * 80)
    print()

    # Setup matching test_diffusion_conserves_particles
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)

    # Strong particle gradient to drive diffusion
    X, Y, Z = grid.meshgrid()
    fields.rho[:] = 1.0
    fields.n[:] = 0.5 + 0.2 * np.sin(X)  # Large gradient
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0

    fields.apply_constraints()

    # Large diffusion coefficient
    coeffs = TransportCoefficients(
        shear_viscosity=0.05,
        bulk_viscosity=0.05,
        diffusion_coefficient=0.2,  # Large diffusion
        shear_relaxation_time=0.1,
        bulk_relaxation_time=0.05,
        diffusion_relaxation_time=0.1,
    )

    solver = SpectralISHydrodynamics(grid, fields, coeffs)

    print("Initial State at t=0")
    print("-" * 80)

    # Check initial particle gradient
    n_gradient_0 = np.max(fields.n) - np.min(fields.n)
    print(f"Particle gradient: {n_gradient_0:.6f}")
    print(f"n range: [{np.min(fields.n):.6f}, {np.max(fields.n):.6f}]")
    print()

    # Check initial diffusion current
    V_mag_0 = np.max(np.abs(fields.V_mu))
    print(f"Max |V^μ|: {V_mag_0:.6e}")
    print(
        f"V^μ components: V^0={np.max(np.abs(fields.V_mu[..., 0])):.6e}, "
        f"V^1={np.max(np.abs(fields.V_mu[..., 1])):.6e}, "
        f"V^2={np.max(np.abs(fields.V_mu[..., 2])):.6e}, "
        f"V^3={np.max(np.abs(fields.V_mu[..., 3])):.6e}"
    )
    print()

    # Note: Skip direct relaxation equation call for now - we'll check through solver
    print("(Relaxation equations checked via solver evolution)")
    print()

    # Check particle number gradient (what drives diffusion)
    conservation = ConservationLaws(fields, coeffs, solver.spectral)

    # Compute ∇n which drives diffusion in Landau frame
    grad_n_x, grad_n_y, grad_n_z = solver.spectral.spatial_gradient(fields.n)
    grad_n = np.stack([grad_n_x, grad_n_y, grad_n_z], axis=-1)

    print(f"Max |∇n|: {np.max(np.linalg.norm(grad_n, axis=-1)):.6e}")
    print(
        f"Gradient components: ∂n/∂x={np.max(np.abs(grad_n_x)):.6e}, "
        f"∂n/∂y={np.max(np.abs(grad_n_y)):.6e}, "
        f"∂n/∂z={np.max(np.abs(grad_n_z)):.6e}"
    )
    print()

    # Evolve one step and check changes
    print("=" * 80)
    print("After 1 timestep (dt=0.002)")
    print("-" * 80)

    dt = 0.002
    solver.time_step(dt)

    # Check particle gradient change
    n_gradient_1 = np.max(solver.fields.n) - np.min(solver.fields.n)
    print(f"Particle gradient: {n_gradient_1:.6f} (change: {n_gradient_1 - n_gradient_0:.6e})")
    print(f"n range: [{np.min(solver.fields.n):.6f}, {np.max(solver.fields.n):.6f}]")
    print()

    # Check diffusion current evolution
    V_mag_1 = np.max(np.abs(solver.fields.V_mu))
    print(f"Max |V^μ|: {V_mag_1:.6e} (change: {V_mag_1 - V_mag_0:.6e})")
    print(
        f"V^μ components: V^0={np.max(np.abs(solver.fields.V_mu[..., 0])):.6e}, "
        f"V^1={np.max(np.abs(solver.fields.V_mu[..., 1])):.6e}, "
        f"V^2={np.max(np.abs(solver.fields.V_mu[..., 2])):.6e}, "
        f"V^3={np.max(np.abs(solver.fields.V_mu[..., 3])):.6e}"
    )
    print()

    # Compute actual particle flux
    n_flux = solver.fields.n[..., np.newaxis] * solver.fields.u_mu[..., 1:4]
    diffusion_flux = solver.fields.V_mu[..., 1:4]
    total_flux = n_flux + diffusion_flux

    print("Particle flux magnitudes:")
    print(f"  Advection (nu^i): {np.max(np.linalg.norm(n_flux, axis=-1)):.6e}")
    print(f"  Diffusion (V^i):  {np.max(np.linalg.norm(diffusion_flux, axis=-1)):.6e}")
    print(f"  Total:            {np.max(np.linalg.norm(total_flux, axis=-1)):.6e}")
    print()

    # Check flux divergence
    div_advection = solver.spectral.spatial_divergence(n_flux)
    div_diffusion = solver.spectral.spatial_divergence(diffusion_flux)
    div_total = solver.spectral.spatial_divergence(total_flux)

    print("Flux divergences:")
    print(f"  ∇·(nu^i): {np.max(np.abs(div_advection)):.6e}")
    print(f"  ∇·(V^i):  {np.max(np.abs(div_diffusion)):.6e}")
    print(f"  ∇·Total:  {np.max(np.abs(div_total)):.6e}")
    print()

    # Evolve longer and track
    print("=" * 80)
    print("Long evolution (100 steps, dt=0.002)")
    print("-" * 80)

    # Reset
    fields2 = ISFieldConfiguration(grid)
    fields2.rho[:] = 1.0
    fields2.n[:] = 0.5 + 0.2 * np.sin(X)
    fields2.pressure[:] = fields2.rho / 3.0
    fields2.u_mu[..., 0] = 1.0
    fields2.apply_constraints()

    solver2 = SpectralISHydrodynamics(grid, fields2, coeffs)

    n_gradient_history = []
    V_max_history = []

    for step in range(100):
        n_gradient_history.append(np.max(solver2.fields.n) - np.min(solver2.fields.n))
        V_max_history.append(np.max(np.abs(solver2.fields.V_mu[..., 1:4])))
        solver2.time_step(dt)

    n_gradient_final = np.max(solver2.fields.n) - np.min(solver2.fields.n)
    V_max_final = np.max(np.abs(solver2.fields.V_mu[..., 1:4]))

    print(f"Initial gradient: {n_gradient_history[0]:.6f}")
    print(f"Final gradient:   {n_gradient_final:.6f}")
    print(
        f"Change:           {n_gradient_final - n_gradient_history[0]:.6e} ({100*(n_gradient_final/n_gradient_history[0] - 1):.2f}%)"
    )
    print()

    print(f"Initial max |V^i|: {V_max_history[0]:.6e}")
    print(f"Final max |V^i|:   {V_max_final:.6e}")
    print()

    # Check if gradient is changing at all
    gradient_range = max(n_gradient_history) - min(n_gradient_history)
    print(f"Gradient variation during evolution: {gradient_range:.6e}")
    print(f"  Max gradient: {max(n_gradient_history):.6f}")
    print(f"  Min gradient: {min(n_gradient_history):.6f}")
    print()

    # Diagnostic: Is V^μ orthogonal to u_μ?
    print("=" * 80)
    print("CONSTRAINT CHECKS")
    print("-" * 80)

    u_lower = solver2.fields.u_mu.copy()
    u_lower[..., 0] *= -1  # Lower time index
    dot_product = np.einsum("...i,...i->...", solver2.fields.V_mu, u_lower)
    print(f"V^μ u_μ orthogonality: max|V·u| = {np.max(np.abs(dot_product)):.6e}")
    print("  (should be < 1e-10 for Landau frame)")
    print()

    # KEY DIAGNOSTIC: Check if relaxation equation is actually being called
    print("=" * 80)
    print("SOLVER INTEGRATION METHOD CHECK")
    print("-" * 80)
    print(
        f"Integration method: {solver2._current_method if hasattr(solver2, '_current_method') else 'unknown'}"
    )
    print()

    # Check if diffusion is enabled
    print(f"Diffusion coefficient D: {coeffs.diffusion_coefficient}")
    print(f"Diffusion relaxation time τ_V: {coeffs.diffusion_relaxation_time}")
    print(
        f"Expected equilibrium V ~ -D ∇(n/T) ~ -D ∇n ~ -{coeffs.diffusion_coefficient} × {np.max(np.linalg.norm(grad_n, axis=-1)):.3e} ~ {-coeffs.diffusion_coefficient * np.max(np.linalg.norm(grad_n, axis=-1)):.3e}"
    )
    print()

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if V_max_final < 1e-10:
        print("❌ ISSUE: Diffusion current V^μ is essentially zero!")
        print("   → Relaxation equation may not be running or not producing V^μ")
        print("   → Check if diffusion_coefficient is being used in relaxation equation")
    elif gradient_range < 0.01 * n_gradient_history[0]:
        print("❌ ISSUE: Gradient is not changing despite non-zero V^μ")
        print(
            f"   → Gradient variation: {gradient_range:.6e} vs initial: {n_gradient_history[0]:.6f}"
        )
        print("   → V^μ may not be coupled to particle conservation equation")
    else:
        print("✅ Diffusion appears to be working")

    print()


if __name__ == "__main__":
    main()
