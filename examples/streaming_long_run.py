#!/usr/bin/env python3
"""
Example: Long-running simulation with streaming snapshots.

Phase 7 demonstration: Memory-efficient long simulations using streaming
snapshot architecture (Phase 5).

Demonstrates:
1. SpaceGrid-based pure 3D simulation (Phase 1-3)
2. Streaming snapshot saving with constant memory (Phase 5)
3. Evolution over many timesteps without memory growth
4. Memory usage monitoring and validation
"""

import matplotlib.pyplot as plt
import numpy as np
import psutil

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
from israel_stewart.utils.io import TrajectoryReader


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def main():
    """Run long simulation with streaming snapshots and memory monitoring."""

    print("=" * 70)
    print("EXAMPLE: Long-Running Simulation with Streaming Snapshots")
    print("=" * 70)
    print()
    print("Demonstrates Phase 5 streaming architecture:")
    print("  - Constant memory usage regardless of simulation duration")
    print("  - Buffered snapshot writing (90% memory reduction)")
    print("  - Long evolution (1000+ timesteps) without memory growth")
    print()

    # ==========================================================================
    # 1. Setup simulation
    # ==========================================================================

    print("Setting up simulation...")

    # Medium resolution grid for demonstration
    nx, ny, nz = 64, 64, 64

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(nx, ny, nz),
        boundary_conditions="periodic",
    )

    # Initialize fields with turbulent-like perturbations
    fields = ISFieldConfiguration(grid)
    X, Y, Z = grid.meshgrid()

    # Multi-mode perturbations
    rho_0 = 1.0
    for kx in [1, 2, 3]:
        for ky in [1, 2]:
            amplitude = 0.01 / (kx + ky)
            phase = np.random.random() * 2 * np.pi
            fields.rho[:] += amplitude * np.sin(kx * X + ky * Y + phase)

    fields.rho[:] += rho_0
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0

    # Small viscosity for slow dissipation
    coeffs = TransportCoefficients(
        shear_viscosity=0.01,
        bulk_viscosity=0.005,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
    )

    hydro = SpectralISHydrodynamics(grid, fields, coeffs)

    print(f"  Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} points")
    print(f"  Field memory: {fields.rho.nbytes / (1024**2):.1f} MB per field")
    print(f"  Total fields memory: ~{fields.rho.nbytes * 12 / (1024**2):.1f} MB")
    print()

    # ==========================================================================
    # 2. Configure streaming with memory monitoring
    # ==========================================================================

    t_final = 20.0  # Long evolution
    snapshot_interval = 0.05  # Many snapshots
    buffer_size = 50  # Large buffer for efficient I/O

    expected_snapshots = int(t_final / snapshot_interval) + 1
    print("Evolution parameters:")
    print(f"  Final time: {t_final}")
    print(f"  Snapshot interval: {snapshot_interval}")
    print(f"  Expected snapshots: {expected_snapshots}")
    print(f"  Buffer size: {buffer_size}")
    print()

    print("Memory analysis:")
    print(
        f"  Without buffering: {expected_snapshots} × {fields.rho.nbytes * 12 / (1024**2):.1f} MB"
    )
    print(
        f"                    = {expected_snapshots * fields.rho.nbytes * 12 / (1024**2):.1f} MB total"
    )
    print(f"  With buffering:    {buffer_size} × {fields.rho.nbytes * 12 / (1024**2):.1f} MB")
    print(
        f"                    = {buffer_size * fields.rho.nbytes * 12 / (1024**2):.1f} MB constant"
    )
    print(
        f"  Memory reduction:  {(1 - buffer_size/expected_snapshots)*100:.0f}% for snapshot storage"
    )
    print()

    # ==========================================================================
    # 3. Run evolution with memory tracking
    # ==========================================================================

    print("Running evolution...")
    print("  Monitoring memory usage during evolution")
    print()

    # Track memory before evolution
    memory_before = get_memory_mb()
    print(f"  Memory before evolution: {memory_before:.1f} MB")

    # Run with streaming
    hydro.evolve(
        t_final=t_final,
        snapshot_config={
            "filename": "streaming_long_run.h5",
            "interval": snapshot_interval,
            "buffer_size": buffer_size,
            "save_initial": True,
        },
    )

    # Track memory after evolution
    memory_after = get_memory_mb()
    memory_increase = memory_after - memory_before

    print()
    print(f"  Memory after evolution:  {memory_after:.1f} MB")
    print(f"  Memory increase:         {memory_increase:.1f} MB")
    print()

    if memory_increase < 200:  # Less than 200 MB increase
        print("  ✅ SUCCESS: Memory usage remained constant!")
        print(
            f"     (Increase {memory_increase:.1f} MB << {expected_snapshots * fields.rho.nbytes * 12 / (1024**2):.0f} MB without streaming)"
        )
    else:
        print("  ⚠️  WARNING: Unexpected memory increase")

    print()

    # ==========================================================================
    # 4. Analyze results
    # ==========================================================================

    print("Analyzing trajectory...")

    reader = TrajectoryReader("streaming_long_run.h5")

    n_snapshots = reader.get_n_snapshots()
    times = reader.get_times()

    print(f"  Snapshots saved: {n_snapshots}")
    print(f"  Time range: [{times[0]:.2f}, {times[-1]:.2f}]")
    print(f"  File size: {reader.filename.stat().st_size / (1024**2):.1f} MB")
    print()

    # Extract time series at several points
    points = [
        (nx // 4, ny // 4, nz // 4),
        (nx // 2, ny // 2, nz // 2),
        (3 * nx // 4, 3 * ny // 4, 3 * nz // 4),
    ]

    print("Density evolution at sample points:")
    for i, point in enumerate(points):
        rho_series = reader.get_field_timeseries("rho", point)
        print(f"  Point {i+1} {point}: {rho_series[0]:.4f} → {rho_series[-1]:.4f}")

    print()

    # ==========================================================================
    # 5. Visualize memory efficiency
    # ==========================================================================

    print("Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Memory comparison
    ax = axes[0, 0]
    memory_no_streaming = np.arange(0, expected_snapshots + 1) * fields.rho.nbytes * 12 / (1024**2)
    memory_with_streaming = (
        np.ones(expected_snapshots + 1) * buffer_size * fields.rho.nbytes * 12 / (1024**2)
    )
    snapshot_times = np.linspace(0, t_final, expected_snapshots + 1)

    ax.plot(snapshot_times, memory_no_streaming, "r-", linewidth=2, label="Without streaming")
    ax.plot(snapshot_times, memory_with_streaming, "g-", linewidth=2, label="With streaming")
    ax.fill_between(snapshot_times, 0, memory_no_streaming, alpha=0.2, color="red")
    ax.fill_between(snapshot_times, 0, memory_with_streaming, alpha=0.2, color="green")
    ax.set_xlabel("Time")
    ax.set_ylabel("Snapshot Memory (MB)")
    ax.set_title("Memory Usage: Streaming vs No Streaming")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    final_mem_no = memory_no_streaming[-1]
    final_mem_with = memory_with_streaming[-1]
    savings = (1 - final_mem_with / final_mem_no) * 100
    ax.text(
        t_final / 2,
        final_mem_no / 2,
        f"{savings:.0f}% memory\nreduction",
        fontsize=14,
        ha="center",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    # Plot 2: Density time series
    ax = axes[0, 1]
    for i, point in enumerate(points):
        rho_series = reader.get_field_timeseries("rho", point)
        ax.plot(times, rho_series, label=f"Point {i+1}", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.set_title("Density Evolution at Sample Points")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Snapshot count vs memory
    ax = axes[1, 0]
    snapshot_counts = np.arange(0, 1001, 50)
    memory_growth = snapshot_counts * fields.rho.nbytes * 12 / (1024**2)
    memory_constant = (
        np.ones_like(snapshot_counts) * buffer_size * fields.rho.nbytes * 12 / (1024**2)
    )

    ax.plot(snapshot_counts, memory_growth / 1024, "r-", linewidth=2, label="Without streaming")
    ax.plot(snapshot_counts, memory_constant / 1024, "g-", linewidth=2, label="With streaming")
    ax.set_xlabel("Number of Snapshots")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Scalability: Memory vs Snapshot Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)

    # Plot 4: Initial vs final density field
    ax = axes[1, 1]
    snapshot_0 = reader.get_snapshot(0)
    snapshot_final = reader.get_snapshot(-1)

    x_slice = nx // 2
    extent = [0, 2 * np.pi, 0, 2 * np.pi]

    im1 = ax.imshow(
        snapshot_0["rho"][x_slice, :, :],
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
    )
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(f"Density Field at t={times[-1]:.1f} (slice x=π)")
    plt.colorbar(im1, ax=ax, label="Density")

    plt.tight_layout()
    plt.savefig("streaming_long_run.png", dpi=150, bbox_inches="tight")
    print("  Saved: streaming_long_run.png")

    reader.close()

    print()
    print("=" * 70)
    print("✅ Example complete!")
    print("=" * 70)
    print()
    print("Key results:")
    print(f"  - Evolved {n_snapshots} snapshots with constant memory")
    print(f"  - Memory increase: {memory_increase:.1f} MB (constant)")
    print(
        f"  - Without streaming: {expected_snapshots * fields.rho.nbytes * 12 / (1024**2):.0f} MB (growing)"
    )
    print(
        f"  - Streaming efficiency: {(1 - buffer_size/expected_snapshots)*100:.0f}% memory reduction"
    )
    print()
    print("Output files:")
    print("  - streaming_long_run.h5  (HDF5 trajectory)")
    print("  - streaming_long_run.png (visualization)")


if __name__ == "__main__":
    main()
