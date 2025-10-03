"""Test Phase 5: Streaming Architecture for Memory-Efficient Simulations."""

import tempfile
from pathlib import Path

import h5py
import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
from israel_stewart.utils.streaming import SnapshotStream, StreamingSimulation


def test_snapshot_stream_basic():
    """Test basic SnapshotStream functionality."""
    print("Test 1: Basic SnapshotStream operations...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Create stream
        stream = SnapshotStream(
            filename=filename,
            grid=grid,
            coeffs=None,
            interval=0.1,
            buffer_size=5,
        )

        # Test should_save
        assert stream.should_save(0.0), "Should save at t=0"

        # Save first snapshot to update last_snapshot_time
        stream.save(0.0, fields)

        # Now test interval checking
        assert not stream.should_save(0.05), "Should not save too soon"
        assert stream.should_save(0.1), "Should save after interval"
        assert len(stream.buffer) == 1, f"Buffer should have 1 snapshot, got {len(stream.buffer)}"

        stream.save(0.1, fields)
        assert len(stream.buffer) == 2, f"Buffer should have 2 snapshots, got {len(stream.buffer)}"

        # Test manual flush
        stream.flush()
        assert len(stream.buffer) == 0, "Buffer should be empty after flush"

        # Test automatic flush at buffer_size
        for i in range(5):
            t = 0.2 + i * 0.1
            stream.save(t, fields)

        # Should have auto-flushed
        assert len(stream.buffer) == 0, "Buffer should auto-flush at buffer_size"

        stream.close()

        # Verify file exists
        assert Path(filename).exists(), "HDF5 file should exist"

        print("✓ SnapshotStream basic operations work")
        print("✓ Buffering and auto-flush verified")
        print(f"✓ Total snapshots saved: {stream._total_snapshots}")

    finally:
        Path(filename).unlink(missing_ok=True)

    return True


def test_snapshot_stream_context_manager():
    """Test SnapshotStream as context manager."""
    print("\nTest 2: SnapshotStream context manager...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Use context manager
        with SnapshotStream(filename, grid, interval=0.1, buffer_size=3) as stream:
            stream.save(0.0, fields)
            stream.save(0.1, fields)
            stream.save(0.2, fields)
            # Should auto-close and flush on exit

        # Verify file was written
        assert Path(filename).exists(), "File should exist after context exit"

        with h5py.File(filename, "r") as f:
            assert "snapshots" in f, "Snapshots group should exist"

        print("✓ Context manager handles resource cleanup")

    finally:
        Path(filename).unlink(missing_ok=True)

    return True


def test_spacegrid_to_spacetimegrid_conversion():
    """Test SpaceGrid to SpacetimeGrid conversion for metadata."""
    print("\nTest 3: Grid conversion for HDF5 metadata...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2.0), (0.0, 3.0), (0.0, 4.0)],
        grid_points=(16, 24, 32),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filename = tmp.name

    try:
        stream = SnapshotStream(filename, grid, interval=0.1, buffer_size=2)

        # Verify internal conversion happened
        spacetime_grid = stream._spacegrid_to_spacetimegrid(grid)

        assert spacetime_grid.coordinate_system == "cartesian"
        assert spacetime_grid.spatial_ranges == grid.spatial_ranges
        assert spacetime_grid.grid_points == (1, 16, 24, 32)  # nt=1
        assert spacetime_grid.boundary_conditions == "periodic"

        stream.save(0.0, fields)
        stream.close()

        print("✓ SpaceGrid converted to SpacetimeGrid for metadata")
        print(f"✓ Spatial ranges preserved: {spacetime_grid.spatial_ranges}")
        print(f"✓ Grid points: {spacetime_grid.grid_points}")

    finally:
        Path(filename).unlink(missing_ok=True)

    return True


def test_field_deep_copy():
    """Test that fields are deep copied to avoid modification."""
    print("\nTest 4: Field deep copy during buffering...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filename = tmp.name

    try:
        stream = SnapshotStream(filename, grid, interval=0.1, buffer_size=10)

        # Save initial state
        stream.save(0.0, fields)
        initial_rho = stream.buffer[0][1].rho.copy()

        # Modify original fields
        fields.rho[:] = 2.0

        # Buffered snapshot should be unchanged
        buffered_rho = stream.buffer[0][1].rho
        assert np.allclose(buffered_rho, 1.0), "Buffered snapshot should not be modified"
        assert not np.allclose(buffered_rho, fields.rho), "Should be independent copy"

        stream.close()

        print("✓ Fields are deep copied (original modifications don't affect buffer)")

    finally:
        Path(filename).unlink(missing_ok=True)

    return True


def test_streaming_simulation():
    """Test StreamingSimulation wrapper class."""
    print("\nTest 5: StreamingSimulation wrapper...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filename = tmp.name

    try:
        with StreamingSimulation(
            filename=filename,
            grid=grid,
            fields=fields,
            coeffs=coeffs,
            snapshot_interval=0.1,
            buffer_size=5,
            save_initial=True,
        ) as sim:
            # Verify initial snapshot was saved
            assert sim.stream._total_snapshots == 1, "Initial snapshot should be saved"

            # Manual time stepping (simulated)
            sim.current_time = 0.1
            if sim.stream.should_save(sim.current_time):
                sim.stream.save(sim.current_time, fields)

            sim.current_time = 0.2
            if sim.stream.should_save(sim.current_time):
                sim.stream.save(sim.current_time, fields)

        # Verify snapshots were written
        assert Path(filename).exists()

        print("✓ StreamingSimulation wrapper works correctly")
        print("✓ Initial snapshot saved automatically")

    finally:
        Path(filename).unlink(missing_ok=True)

    return True


def test_spectral_solver_integration():
    """Test integration with SpectralISHydrodynamics.evolve()."""
    print("\nTest 6: Integration with SpectralISHydrodynamics...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    X, Y, Z = grid.meshgrid()

    # Sound wave initial condition
    k = 1.0
    amplitude = 0.01
    fields.rho[:] = 1.0 + amplitude * np.sin(k * X)
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0

    coeffs = TransportCoefficients(
        shear_viscosity=0.01,
        bulk_viscosity=0.005,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
    )

    hydro = SpectralISHydrodynamics(grid, fields, coeffs)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Test new snapshot_config parameter
        snapshot_config = {
            "filename": filename,
            "interval": 0.05,
            "buffer_size": 5,
            "save_initial": True,
        }

        # Evolve with streaming
        t_final = 0.2
        hydro.evolve(t_final=t_final, snapshot_config=snapshot_config)

        # Verify snapshots were written
        assert Path(filename).exists(), "HDF5 file should exist"

        with h5py.File(filename, "r") as f:
            assert "snapshots" in f, "Snapshots group should exist"
            snapshots = f["snapshots"]

            # Should have saved: t=0, 0.05, 0.1, 0.15, 0.2 = 5 snapshots
            # (exact count depends on adaptive stepping, so check minimum)
            num_snapshots = len(snapshots.keys())
            assert num_snapshots >= 3, f"Should have at least 3 snapshots, got {num_snapshots}"

            print("✓ SpectralISHydrodynamics.evolve() with snapshot_config works")
            print(f"✓ Saved {num_snapshots} snapshots to HDF5")

    finally:
        Path(filename).unlink(missing_ok=True)

    return True


def test_backward_compatibility():
    """Test that old save_trajectory parameter still works."""
    print("\nTest 7: Backward compatibility with save_trajectory...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    coeffs = TransportCoefficients(
        shear_viscosity=0.01,
        bulk_viscosity=0.005,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
    )

    hydro = SpectralISHydrodynamics(grid, fields, coeffs)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Test old save_trajectory parameter (deprecated but should still work)
        save_trajectory = {"filename": filename, "interval": 0.05}

        hydro.evolve(t_final=0.1, save_trajectory=save_trajectory)

        # Verify file was created
        assert Path(filename).exists(), "Old save_trajectory should still work"

        print("✓ Backward compatibility maintained with save_trajectory")

    finally:
        Path(filename).unlink(missing_ok=True)

    return True


def test_memory_efficiency():
    """Test that streaming uses constant memory."""
    print("\nTest 8: Memory efficiency with buffering...")

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(32, 32, 32),  # Larger grid
        boundary_conditions="periodic",
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Small buffer to test constant memory
        buffer_size = 3
        stream = SnapshotStream(filename, grid, interval=0.01, buffer_size=buffer_size)

        # Save many snapshots
        num_snapshots = 20
        for i in range(num_snapshots):
            t = i * 0.01
            stream.save(t, fields)

            # Buffer should never exceed buffer_size
            assert len(stream.buffer) <= buffer_size, f"Buffer exceeded limit: {len(stream.buffer)}"

        stream.close()

        # Verify all snapshots were written
        assert stream._total_snapshots == num_snapshots

        print(f"✓ Saved {num_snapshots} snapshots with buffer_size={buffer_size}")
        print(f"✓ Memory usage constant (buffer never exceeded {buffer_size})")

    finally:
        Path(filename).unlink(missing_ok=True)

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5: Streaming Architecture Tests")
    print("=" * 60)

    tests = [
        test_snapshot_stream_basic,
        test_snapshot_stream_context_manager,
        test_spacegrid_to_spacetimegrid_conversion,
        test_field_deep_copy,
        test_streaming_simulation,
        test_spectral_solver_integration,
        test_backward_compatibility,
        test_memory_efficiency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✓ ALL TESTS PASSED - Phase 5 complete!")
        print("\nStreaming architecture successfully implemented:")
        print("- Buffered snapshot writing ✓")
        print("- Automatic flushing ✓")
        print("- Memory-efficient long simulations ✓")
        print("- SpaceGrid compatibility ✓")
        print("- Backward compatibility ✓")
        exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        exit(1)
