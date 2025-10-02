"""Tests for HDF5 trajectory I/O functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.utils.io import TrajectoryReader, TrajectoryWriter


@pytest.fixture
def test_grid():
    """Create a small test grid."""
    return SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(1, 8, 8, 8),
        boundary_conditions="periodic",
    )


@pytest.fixture
def test_fields(test_grid):
    """Create test field configuration."""
    fields = ISFieldConfiguration(test_grid)

    # Initialize with simple pattern
    x = test_grid.coordinates["x"]
    X = np.meshgrid(x, x, x, indexing="ij")[0]

    # Handle both 3D and 4D storage
    if fields.rho.ndim == 4:
        fields.rho[-1, :, :, :] = 1.0 + 0.1 * X
        fields.pressure[-1, :, :, :] = fields.rho[-1, :, :, :] / 3.0
        fields.u_mu[-1, :, :, :, 0] = 1.0
    else:
        fields.rho[:] = 1.0 + 0.1 * X
        fields.pressure[:] = fields.rho / 3.0
        fields.u_mu[..., 0] = 1.0

    return fields


@pytest.fixture
def test_coeffs():
    """Create test transport coefficients."""
    return TransportCoefficients(shear_viscosity=0.1, bulk_viscosity=0.05)


def test_trajectory_writer_creation(test_grid, test_coeffs, tmp_path):
    """Test creating a trajectory writer."""
    filename = tmp_path / "test.h5"

    writer = TrajectoryWriter(filename, test_grid, test_coeffs)
    assert writer.filename == filename
    assert writer._snapshot_count == 0

    writer.close()
    assert filename.exists()


def test_write_single_snapshot(test_grid, test_fields, test_coeffs, tmp_path):
    """Test writing a single snapshot."""
    filename = tmp_path / "test.h5"

    writer = TrajectoryWriter(filename, test_grid, test_coeffs)
    writer.write_snapshot(0.0, test_fields)
    assert writer._snapshot_count == 1

    writer.close()


def test_write_multiple_snapshots(test_grid, test_fields, test_coeffs, tmp_path):
    """Test writing multiple snapshots."""
    filename = tmp_path / "test.h5"

    writer = TrajectoryWriter(filename, test_grid, test_coeffs)

    # Write 5 snapshots with evolving density
    for i in range(5):
        t = i * 0.1
        # Modify fields slightly
        if test_fields.rho.ndim == 4:
            test_fields.rho[-1, :, :, :] += 0.01
        else:
            test_fields.rho += 0.01

        writer.write_snapshot(t, test_fields)

    assert writer._snapshot_count == 5
    writer.close()


def test_read_trajectory(test_grid, test_fields, test_coeffs, tmp_path):
    """Test reading a trajectory."""
    filename = tmp_path / "test.h5"

    # Write trajectory
    writer = TrajectoryWriter(filename, test_grid, test_coeffs)
    times = [0.0, 0.1, 0.2, 0.3]
    for t in times:
        writer.write_snapshot(t, test_fields)
    writer.close()

    # Read trajectory
    reader = TrajectoryReader(filename)
    assert reader.get_n_snapshots() == len(times)

    times_read = reader.get_times()
    assert np.allclose(times_read, times)

    reader.close()


def test_round_trip_data_integrity(test_grid, test_fields, test_coeffs, tmp_path):
    """Test that data is preserved in write-read cycle."""
    filename = tmp_path / "test.h5"

    # Extract current state from fields
    if test_fields.rho.ndim == 4:
        rho_original = test_fields.rho[-1, :, :, :].copy()
        u_mu_original = test_fields.u_mu[-1, :, :, :, :].copy()
    else:
        rho_original = test_fields.rho.copy()
        u_mu_original = test_fields.u_mu.copy()

    # Write
    writer = TrajectoryWriter(filename, test_grid, test_coeffs)
    writer.write_snapshot(0.0, test_fields)
    writer.close()

    # Read
    reader = TrajectoryReader(filename)
    snapshot = reader.get_snapshot(0)

    # Verify data integrity
    assert np.allclose(snapshot["rho"], rho_original, rtol=1e-10)
    assert np.allclose(snapshot["u_mu"], u_mu_original, rtol=1e-10)

    reader.close()


def test_get_snapshot_at_time(test_grid, test_fields, test_coeffs, tmp_path):
    """Test retrieving snapshot at specific time."""
    filename = tmp_path / "test.h5"

    # Write snapshots at specific times
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    writer = TrajectoryWriter(filename, test_grid, test_coeffs)
    for t in times:
        writer.write_snapshot(t, test_fields)
    writer.close()

    # Read snapshot at t=1.0
    reader = TrajectoryReader(filename)
    snapshot = reader.get_snapshot_at_time(1.0)
    assert abs(snapshot["time"] - 1.0) < 1e-10

    # Test tolerance
    snapshot = reader.get_snapshot_at_time(1.01, tolerance=0.02)
    assert abs(snapshot["time"] - 1.0) < 0.02

    # Test failure when out of tolerance
    with pytest.raises(ValueError, match="No snapshot found"):
        reader.get_snapshot_at_time(3.0, tolerance=0.01)  # Way outside range

    reader.close()


def test_field_timeseries(test_grid, test_fields, test_coeffs, tmp_path):
    """Test extracting field time series."""
    filename = tmp_path / "test.h5"

    # Write snapshots with linearly increasing density
    writer = TrajectoryWriter(filename, test_grid, test_coeffs)
    times = np.linspace(0, 1, 10)
    for i, t in enumerate(times):
        # Modify density
        if test_fields.rho.ndim == 4:
            test_fields.rho[-1, :, :, :] = 1.0 + 0.1 * i
        else:
            test_fields.rho[:] = 1.0 + 0.1 * i

        writer.write_snapshot(t, test_fields)
    writer.close()

    # Read time series at a point
    reader = TrajectoryReader(filename)
    point = (4, 4, 4)
    rho_series = reader.get_field_timeseries("rho", point)

    # Verify linear increase
    assert len(rho_series) == len(times)
    expected = 1.0 + 0.1 * np.arange(len(times))
    assert np.allclose(rho_series, expected, rtol=1e-10)

    reader.close()


def test_context_manager(test_grid, test_fields, test_coeffs, tmp_path):
    """Test using writer/reader as context managers."""
    filename = tmp_path / "test.h5"

    # Writer as context manager
    with TrajectoryWriter(filename, test_grid, test_coeffs) as writer:
        writer.write_snapshot(0.0, test_fields)
        writer.write_snapshot(0.1, test_fields)

    # Reader as context manager
    with TrajectoryReader(filename) as reader:
        assert reader.get_n_snapshots() == 2
        snapshot = reader.get_snapshot(0)
        assert "rho" in snapshot


def test_metadata_preservation(test_grid, test_coeffs, tmp_path):
    """Test that metadata is correctly saved and loaded."""
    filename = tmp_path / "test.h5"

    # Write with coefficients
    writer = TrajectoryWriter(filename, test_grid, test_coeffs)
    writer.close()

    # Read and verify metadata
    reader = TrajectoryReader(filename)
    assert reader.metadata["solver_type"] == "SpectralISHydrodynamics"
    assert "creation_date" in reader.metadata

    assert reader.grid_attrs["coordinate_system"] == "cartesian"
    assert reader.grid_attrs["boundary_conditions"] == "periodic"

    assert reader.coeffs_attrs["shear_viscosity"] == 0.1
    assert reader.coeffs_attrs["bulk_viscosity"] == 0.05

    reader.close()


def test_max_snapshots_limit(test_grid, test_fields, test_coeffs, tmp_path):
    """Test that max_snapshots limit is respected."""
    filename = tmp_path / "test.h5"

    max_snaps = 3
    writer = TrajectoryWriter(filename, test_grid, test_coeffs, max_snapshots=max_snaps)

    # Try to write more than max
    for i in range(5):
        if i >= max_snaps:
            with pytest.warns(UserWarning):
                writer.write_snapshot(i * 0.1, test_fields)
        else:
            writer.write_snapshot(i * 0.1, test_fields)

    # Should have exactly max_snapshots
    assert writer._snapshot_count == max_snaps
    writer.close()

    reader = TrajectoryReader(filename)
    assert reader.get_n_snapshots() == max_snaps
    reader.close()


def test_compression(test_grid, test_fields, test_coeffs, tmp_path):
    """Test that compression reduces file size."""
    file_compressed = tmp_path / "compressed.h5"
    file_uncompressed = tmp_path / "uncompressed.h5"

    # Write with compression
    with TrajectoryWriter(file_compressed, test_grid, test_coeffs, compression="gzip") as w:
        for i in range(10):
            w.write_snapshot(i * 0.1, test_fields)

    # Write without compression
    with TrajectoryWriter(file_uncompressed, test_grid, test_coeffs, compression=None) as w:
        for i in range(10):
            w.write_snapshot(i * 0.1, test_fields)

    # Compressed should be smaller (or at least not larger)
    assert file_compressed.stat().st_size <= file_uncompressed.stat().st_size


@pytest.mark.parametrize(
    "field_name", ["rho", "pressure", "temperature", "Pi", "u_mu", "q_mu", "pi_munu"]
)
def test_all_fields_saved(test_grid, test_fields, test_coeffs, field_name, tmp_path):
    """Test that all fields are correctly saved."""
    filename = tmp_path / "test.h5"

    with TrajectoryWriter(filename, test_grid, test_coeffs) as writer:
        writer.write_snapshot(0.0, test_fields)

    with TrajectoryReader(filename) as reader:
        snapshot = reader.get_snapshot(0)
        assert field_name in snapshot
        assert snapshot[field_name] is not None


def test_reader_repr(test_grid, test_fields, test_coeffs, tmp_path):
    """Test reader string representation."""
    filename = tmp_path / "test.h5"

    with TrajectoryWriter(filename, test_grid, test_coeffs) as writer:
        writer.write_snapshot(0.0, test_fields)
        writer.write_snapshot(1.0, test_fields)

    with TrajectoryReader(filename) as reader:
        repr_str = repr(reader)
        assert "test.h5" in repr_str
        assert "snapshots=2" in repr_str
