#!/usr/bin/env python3
"""
Example: Save sound wave evolution to HDF5 trajectory.

Demonstrates:
1. Setting up a sound wave initial condition
2. Running time evolution with trajectory saving
3. Reading the trajectory back
4. Analyzing wave propagation
"""

import matplotlib.pyplot as plt
import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
from israel_stewart.utils.io import TrajectoryReader


def main():
    """Run wave evolution and save trajectory."""

    print("=" * 70)
    print("EXAMPLE: Sound Wave Evolution with HDF5 Trajectory Saving")
    print("=" * 70)
    print()

    # ==========================================================================
    # 1. Setup simulation
    # ==========================================================================

    print("Setting up simulation...")

    # Create 3D spatial grid
    # NOTE: For pure 3+1D evolution, we use minimal nt (just for storage)
    nx, ny, nz = 32, 32, 32
    nt = 1  # Minimal time dimension (only current state)

    grid = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(nt, nx, ny, nz),
        boundary_conditions="periodic",
    )

    # Initialize fields
    fields = ISFieldConfiguration(grid)

    # Sound wave parameters
    c_s = np.sqrt(1.0 / 3.0)  # Conformal sound speed
    rho_0 = 1.0
    amplitude = 0.05
    wavelength = 2 * np.pi
    k = 2 * np.pi / wavelength

    # Get spatial coordinates
    x = grid.coordinates["x"]
    y = grid.coordinates["y"]
    z = grid.coordinates["z"]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Initialize with plane wave in x-direction
    # Density perturbation: ρ = ρ_0 + A sin(kx)
    density_perturbation = amplitude * np.sin(k * X)

    # Handle both 3D and 4D storage
    if fields.rho.ndim == 4:
        # 4D storage: initialize last time slice
        fields.rho[-1, :, :, :] = rho_0 + density_perturbation
        fields.pressure[-1, :, :, :] = fields.rho[-1, :, :, :] / 3.0

        # Velocity: u^x = (c_s/ρ_0) A sin(kx)
        u_x = (c_s / rho_0) * density_perturbation
        fields.u_mu[-1, :, :, :, 0] = np.sqrt(1.0 + u_x**2)
        fields.u_mu[-1, :, :, :, 1] = u_x
        fields.u_mu[-1, :, :, :, 2:] = 0.0
    else:
        # 3D storage: initialize directly
        fields.rho[:] = rho_0 + density_perturbation
        fields.pressure[:] = fields.rho / 3.0

        u_x = (c_s / rho_0) * density_perturbation
        fields.u_mu[..., 0] = np.sqrt(1.0 + u_x**2)
        fields.u_mu[..., 1] = u_x
        fields.u_mu[..., 2:] = 0.0

    # Zero viscosity for ideal fluid
    coeffs = TransportCoefficients(shear_viscosity=0.0, bulk_viscosity=0.0)

    # Create solver
    hydro = SpectralISHydrodynamics(grid, fields, coeffs)

    print(f"  Grid: {nx}x{ny}x{nz}")
    print(f"  Wave: λ = {wavelength:.3f}, amplitude = {amplitude:.3f}")
    print(f"  Sound speed: c_s = {c_s:.4f}")
    print()

    # ==========================================================================
    # 2. Run evolution with trajectory saving
    # ==========================================================================

    print("Running evolution...")
    print("  Saving trajectory to: wave_evolution.h5")

    t_final = 5.0
    snapshot_interval = 0.1

    hydro.evolve(
        t_final=t_final,
        save_trajectory={
            "filename": "wave_evolution.h5",
            "interval": snapshot_interval,
            "save_initial": True,
        },
    )

    print()

    # ==========================================================================
    # 3. Read trajectory and analyze
    # ==========================================================================

    print("Reading trajectory...")

    reader = TrajectoryReader("wave_evolution.h5")

    print(f"  Snapshots: {reader.get_n_snapshots()}")
    print(f"  Time range: {reader.get_times()[[0, -1]]}")
    print()

    # Extract density evolution at center point
    center_idx = (nx // 2, ny // 2, nz // 2)
    rho_timeseries = reader.get_field_timeseries("rho", center_idx)
    times = reader.get_times()

    print(f"Time series at center {center_idx}:")
    print(f"  Initial density: {rho_timeseries[0]:.6f}")
    print(f"  Final density: {rho_timeseries[-1]:.6f}")
    print()

    # ==========================================================================
    # 4. Visualize results
    # ==========================================================================

    print("Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Density time series at center
    ax = axes[0, 0]
    ax.plot(times, rho_timeseries, "b-", linewidth=2)
    ax.axhline(rho_0, color="k", linestyle="--", alpha=0.5, label="ρ₀")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density at center")
    ax.set_title("Density Evolution at Center Point")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Initial density profile
    snapshot_0 = reader.get_snapshot(0)
    ax = axes[0, 1]
    ax.plot(x, snapshot_0["rho"][:, ny // 2, nz // 2], "b-", linewidth=2, label="Initial")
    ax.axhline(rho_0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Initial Density Profile (y=π, z=π)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Final density profile
    snapshot_final = reader.get_snapshot(-1)
    ax = axes[1, 0]
    ax.plot(
        x,
        snapshot_final["rho"][:, ny // 2, nz // 2],
        "r-",
        linewidth=2,
        label=f"t={times[-1]:.1f}",
    )
    ax.plot(
        x,
        snapshot_0["rho"][:, ny // 2, nz // 2],
        "b--",
        alpha=0.5,
        linewidth=1,
        label="Initial",
    )
    ax.axhline(rho_0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Final Density Profile (y=π, z=π)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Spacetime diagram (x vs t)
    ax = axes[1, 1]
    # Extract density along x-axis (y=π, z=π) for all times
    rho_xt = np.zeros((len(times), nx))
    for i in range(len(times)):
        snapshot = reader.get_snapshot(i)
        rho_xt[i, :] = snapshot["rho"][:, ny // 2, nz // 2]

    im = ax.pcolormesh(
        x,
        times,
        rho_xt,
        shading="auto",
        cmap="RdBu_r",
        vmin=rho_0 - amplitude,
        vmax=rho_0 + amplitude,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("Time")
    ax.set_title("Spacetime Diagram: ρ(x, t)")
    plt.colorbar(im, ax=ax, label="Density")

    # Add characteristic lines (wave propagation)
    for x0 in [0, np.pi, 2 * np.pi]:
        ax.plot([x0, x0 + c_s * t_final], [0, t_final], "w--", alpha=0.5, linewidth=1)
        ax.plot([x0, x0 - c_s * t_final], [0, t_final], "w--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig("wave_evolution.png", dpi=150, bbox_inches="tight")
    print("  Saved plot: wave_evolution.png")

    reader.close()

    print()
    print("=" * 70)
    print("✅ Example complete!")
    print("=" * 70)
    print()
    print("Output files:")
    print("  - wave_evolution.h5  (HDF5 trajectory)")
    print("  - wave_evolution.png (visualization)")


if __name__ == "__main__":
    main()
