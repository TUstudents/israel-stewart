"""
HDF5-based I/O utilities for Israel-Stewart hydrodynamics simulations.

This module provides trajectory storage for time evolution simulations,
enabling efficient incremental writing and reading of field snapshots.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from ..core.fields import ISFieldConfiguration, TransportCoefficients
    from ..core.spacetime_grid import SpacetimeGrid


class TrajectoryWriter:
    """
    HDF5 writer for time evolution trajectories.

    Writes simulation snapshots incrementally during evolution with:
    - Efficient chunked storage for large simulations
    - Compression to reduce file size
    - Comprehensive metadata for reproducibility
    - Diagnostic data storage

    Example:
        ```python
        writer = TrajectoryWriter("simulation.h5", grid, coeffs)
        for t in time_steps:
            fields = evolve_one_step(fields, dt)
            writer.write_snapshot(t, fields)
        writer.close()
        ```
    """

    def __init__(
        self,
        filename: str | Path,
        grid: "SpacetimeGrid",
        coeffs: Optional["TransportCoefficients"] = None,
        max_snapshots: Optional[int] = None,
        compression: str = "gzip",
        compression_opts: int = 4,
    ):
        """
        Initialize trajectory writer.

        Args:
            filename: HDF5 file path
            grid: SpacetimeGrid with spatial grid information
            coeffs: Transport coefficients (optional, for metadata)
            max_snapshots: Maximum number of snapshots (None for unlimited)
            compression: HDF5 compression type ('gzip', 'lzf', None)
            compression_opts: Compression level (0-9 for gzip)
        """
        try:
            import h5py
        except ImportError as err:
            raise ImportError("h5py required for trajectory I/O") from err

        self.filename = Path(filename)
        self.grid = grid
        self.coeffs = coeffs
        self.max_snapshots = max_snapshots
        self.compression = compression
        self.compression_opts = compression_opts

        # Open file and create structure
        self.h5file = h5py.File(self.filename, "w")
        self._snapshot_count = 0
        self._setup_file_structure()

    def _setup_file_structure(self) -> None:
        """Create HDF5 file structure with metadata and datasets."""
        import h5py

        # Metadata group
        meta = self.h5file.create_group("metadata")
        meta.attrs["creation_date"] = datetime.now().isoformat()
        meta.attrs["version"] = "1.0"
        meta.attrs["solver_type"] = "SpectralISHydrodynamics"

        # Grid information
        grid_group = meta.create_group("grid")
        grid_group.attrs["coordinate_system"] = self.grid.coordinate_system

        # Spatial grid only (3D)
        if hasattr(self.grid, "grid_points"):
            # Extract spatial dimensions only (exclude time if present)
            grid_points = self.grid.grid_points
            if len(grid_points) == 4:
                # Has time dimension, extract spatial only
                _, nx, ny, nz = grid_points
                spatial_shape = (nx, ny, nz)
            else:
                spatial_shape = grid_points
            grid_group.attrs["grid_points"] = spatial_shape
        else:
            spatial_shape = self.grid.shape

        grid_group.attrs["boundary_conditions"] = getattr(
            self.grid, "boundary_conditions", "periodic"
        )

        # Save coordinate arrays
        coords_group = grid_group.create_group("coordinates")
        if hasattr(self.grid, "coordinates"):
            # Save spatial coordinates only
            for coord_name in ["x", "y", "z"]:
                if coord_name in self.grid.coordinates:
                    # Only add compression parameters if compression is enabled
                    if self.compression:
                        coords_group.create_dataset(
                            coord_name,
                            data=self.grid.coordinates[coord_name],
                            compression=self.compression,
                            compression_opts=self.compression_opts,
                        )
                    else:
                        coords_group.create_dataset(
                            coord_name,
                            data=self.grid.coordinates[coord_name],
                        )

        # Metric information
        if hasattr(self.grid, "metric") and self.grid.metric is not None:
            metric_group = grid_group.create_group("metric")
            metric_group.attrs["metric_type"] = type(self.grid.metric).__name__
            metric_group.attrs["signature"] = getattr(
                self.grid.metric, "signature", "mostly_plus"
            )

        # Transport coefficients
        if self.coeffs is not None:
            coeffs_group = meta.create_group("transport_coefficients")
            coeffs_group.attrs["shear_viscosity"] = float(self.coeffs.shear_viscosity)
            coeffs_group.attrs["bulk_viscosity"] = float(self.coeffs.bulk_viscosity)
            if hasattr(self.coeffs, "shear_relaxation_time") and self.coeffs.shear_relaxation_time is not None:
                coeffs_group.attrs["shear_relaxation_time"] = float(
                    self.coeffs.shear_relaxation_time
                )
            if hasattr(self.coeffs, "bulk_relaxation_time") and self.coeffs.bulk_relaxation_time is not None:
                coeffs_group.attrs["bulk_relaxation_time"] = float(
                    self.coeffs.bulk_relaxation_time
                )

        # Snapshots group - will contain time series datasets
        self.snapshots_group = self.h5file.create_group("snapshots")

        # Get spatial shape from grid
        if hasattr(self.grid, "grid_points"):
            points = self.grid.grid_points
            if len(points) == 4:
                _, nx, ny, nz = points
            else:
                nx, ny, nz = points
        else:
            nx, ny, nz = self.grid.shape[-3:]

        # Helper to add compression kwargs only when needed
        def compression_kwargs():
            """Return compression kwargs if compression is enabled."""
            if self.compression:
                return {
                    "compression": self.compression,
                    "compression_opts": self.compression_opts,
                }
            return {}

        # Create extensible datasets for field snapshots
        # Shape: (n_snapshots, nx, ny, nz) - time series of 3D fields
        maxshape_scalar = (
            None if self.max_snapshots is None else self.max_snapshots,
            nx,
            ny,
            nz,
        )
        maxshape_vector = (
            None if self.max_snapshots is None else self.max_snapshots,
            nx,
            ny,
            nz,
            4,
        )
        maxshape_tensor = (
            None if self.max_snapshots is None else self.max_snapshots,
            nx,
            ny,
            nz,
            4,
            4,
        )

        # Time array
        self.snapshots_group.create_dataset(
            "times",
            shape=(0,),
            maxshape=(None if self.max_snapshots is None else self.max_snapshots,),
            dtype=np.float64,
            chunks=True,
        )

        # Scalar fields
        for field_name in ["rho", "n", "pressure", "temperature", "Pi"]:
            self.snapshots_group.create_dataset(
                field_name,
                shape=(0, nx, ny, nz),
                maxshape=maxshape_scalar,
                dtype=np.float64,
                chunks=(1, nx, ny, nz),  # Chunk along time for sequential writes
                **compression_kwargs(),
            )

        # Vector fields
        for field_name in ["u_mu", "q_mu"]:
            self.snapshots_group.create_dataset(
                field_name,
                shape=(0, nx, ny, nz, 4),
                maxshape=maxshape_vector,
                dtype=np.float64,
                chunks=(1, nx, ny, nz, 4),
                **compression_kwargs(),
            )

        # Tensor field
        self.snapshots_group.create_dataset(
            "pi_munu",
            shape=(0, nx, ny, nz, 4, 4),
            maxshape=maxshape_tensor,
            dtype=np.float64,
            chunks=(1, nx, ny, nz, 4, 4),
            **compression_kwargs(),
        )

        # Diagnostics group (optional)
        self.diagnostics_group = self.h5file.create_group("diagnostics")
        for diag_name in ["total_energy", "total_momentum", "total_entropy"]:
            if diag_name == "total_momentum":
                shape = (0, 3)
                maxshape = (
                    None if self.max_snapshots is None else self.max_snapshots,
                    3,
                )
            else:
                shape = (0,)
                maxshape = (None if self.max_snapshots is None else self.max_snapshots,)

            self.diagnostics_group.create_dataset(
                diag_name,
                shape=shape,
                maxshape=maxshape,
                dtype=np.float64,
                chunks=True,
            )

    def write_snapshot(
        self, t: float, fields: "ISFieldConfiguration", diagnostics: Optional[dict] = None
    ) -> None:
        """
        Write a snapshot at time t.

        Args:
            t: Current simulation time
            fields: ISFieldConfiguration with current field values
            diagnostics: Optional dictionary of diagnostic values
        """
        if self._snapshot_count >= (self.max_snapshots or float("inf")):
            warnings.warn(
                f"Maximum snapshots ({self.max_snapshots}) reached. "
                "Snapshot not written.",
                stacklevel=2,
            )
            return

        # Extend datasets
        new_size = self._snapshot_count + 1

        # Resize all datasets
        self.snapshots_group["times"].resize((new_size,))
        for field_name in ["rho", "n", "pressure", "temperature", "Pi"]:
            if field_name in self.snapshots_group:
                self.snapshots_group[field_name].resize((new_size, *fields.rho.shape[-3:]))

        for field_name in ["u_mu", "q_mu"]:
            if field_name in self.snapshots_group:
                self.snapshots_group[field_name].resize(
                    (new_size, *fields.rho.shape[-3:], 4)
                )

        self.snapshots_group["pi_munu"].resize((new_size, *fields.rho.shape[-3:], 4, 4))

        # Write time
        self.snapshots_group["times"][self._snapshot_count] = t

        # Extract current state (handle both 3D and 4D storage)
        # If fields are 4D (nt, nx, ny, nz), take last time slice
        # If fields are 3D (nx, ny, nz), use directly

        def get_current_slice(field: np.ndarray) -> np.ndarray:
            """Extract current spatial slice from field (handle 3D or 4D)."""
            if field.ndim >= 3 and field.shape[0] == getattr(self.grid, "nt", 1):
                # Has time dimension, take last slice
                return field[-1, ...]
            return field

        # Write scalar fields
        for field_name in ["rho", "n", "pressure", "temperature", "Pi"]:
            if hasattr(fields, field_name):
                field_data = getattr(fields, field_name)
                current_data = get_current_slice(field_data)
                self.snapshots_group[field_name][self._snapshot_count] = current_data

        # Write vector fields
        for field_name in ["u_mu", "q_mu"]:
            if hasattr(fields, field_name):
                field_data = getattr(fields, field_name)
                current_data = get_current_slice(field_data)
                self.snapshots_group[field_name][self._snapshot_count] = current_data

        # Write tensor field
        current_pi = get_current_slice(fields.pi_munu)
        self.snapshots_group["pi_munu"][self._snapshot_count] = current_pi

        # Write diagnostics if provided
        if diagnostics is not None:
            for diag_name, value in diagnostics.items():
                if diag_name in self.diagnostics_group:
                    self.diagnostics_group[diag_name].resize((new_size,))
                    self.diagnostics_group[diag_name][self._snapshot_count] = value

        self._snapshot_count += 1

        # Flush every 10 snapshots for safety
        if self._snapshot_count % 10 == 0:
            self.h5file.flush()

    def close(self) -> None:
        """Close HDF5 file and flush all data."""
        if hasattr(self, "h5file") and self.h5file:
            self.h5file.flush()
            self.h5file.close()
            print(f" Trajectory saved: {self.filename} ({self._snapshot_count} snapshots)")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file on context exit."""
        self.close()

    def __del__(self):
        """Ensure file is closed on deletion."""
        if hasattr(self, "h5file"):
            try:
                self.close()
            except Exception:
                pass


class TrajectoryReader:
    """
    Read HDF5 trajectory files.

    Provides convenient access to time evolution data:
    - Individual snapshots by index or time
    - Time series at spatial points
    - Diagnostic data
    - Metadata

    Example:
        ```python
        reader = TrajectoryReader("simulation.h5")
        times = reader.get_times()
        snapshot_10 = reader.get_snapshot(10)
        rho_evolution = reader.get_field_timeseries("rho", point=(4, 4, 4))
        ```
    """

    def __init__(self, filename: str | Path):
        """
        Open trajectory file for reading.

        Args:
            filename: Path to HDF5 trajectory file
        """
        try:
            import h5py
        except ImportError as err:
            raise ImportError("h5py required for trajectory I/O") from err

        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.filename}")

        self.h5file = h5py.File(self.filename, "r")

        # Load metadata
        self.metadata = dict(self.h5file["metadata"].attrs)
        self.grid_attrs = dict(self.h5file["metadata/grid"].attrs)

        # Check if coefficients exist
        if "transport_coefficients" in self.h5file["metadata"]:
            self.coeffs_attrs = dict(
                self.h5file["metadata/transport_coefficients"].attrs
            )
        else:
            self.coeffs_attrs = {}

        self.snapshots = self.h5file["snapshots"]
        self.diagnostics = self.h5file.get("diagnostics", None)

    def get_times(self) -> np.ndarray:
        """Get array of snapshot times."""
        return self.snapshots["times"][:]

    def get_n_snapshots(self) -> int:
        """Get number of snapshots in trajectory."""
        return len(self.snapshots["times"])

    def get_snapshot(self, index: int) -> dict[str, np.ndarray]:
        """
        Get snapshot at given index.

        Args:
            index: Snapshot index (0 to n_snapshots-1)

        Returns:
            Dictionary with all fields at this snapshot
        """
        if not 0 <= index < self.get_n_snapshots():
            raise IndexError(
                f"Snapshot index {index} out of range [0, {self.get_n_snapshots()})"
            )

        snapshot = {"time": self.snapshots["times"][index]}

        # Load all fields
        for field_name in ["rho", "n", "pressure", "temperature", "Pi"]:
            if field_name in self.snapshots:
                snapshot[field_name] = self.snapshots[field_name][index, ...]

        for field_name in ["u_mu", "q_mu"]:
            if field_name in self.snapshots:
                snapshot[field_name] = self.snapshots[field_name][index, ...]

        if "pi_munu" in self.snapshots:
            snapshot["pi_munu"] = self.snapshots["pi_munu"][index, ...]

        return snapshot

    def get_snapshot_at_time(self, t: float, tolerance: float = 1e-6) -> dict[str, np.ndarray]:
        """
        Get snapshot closest to given time.

        Args:
            t: Target time
            tolerance: Maximum time difference to accept

        Returns:
            Dictionary with snapshot data

        Raises:
            ValueError: If no snapshot within tolerance
        """
        times = self.get_times()
        idx = np.argmin(np.abs(times - t))
        if abs(times[idx] - t) > tolerance:
            raise ValueError(
                f"No snapshot found at t={t} (closest is t={times[idx]}, "
                f"diff={abs(times[idx] - t)})"
            )
        return self.get_snapshot(idx)

    def get_field_timeseries(
        self, field_name: str, spatial_point: Optional[tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Get time series of a field at a spatial point or whole field.

        Args:
            field_name: Name of field ('rho', 'pressure', 'Pi', etc.)
            spatial_point: Spatial indices (ix, iy, iz). If None, returns full field.

        Returns:
            Time series array. Shape depends on field and spatial_point:
            - Scalar field, point given: (n_snapshots,)
            - Scalar field, no point: (n_snapshots, nx, ny, nz)
            - Vector field, point given: (n_snapshots, 4)
            - etc.
        """
        if field_name not in self.snapshots:
            raise KeyError(f"Field '{field_name}' not found in trajectory")

        field_data = self.snapshots[field_name]

        if spatial_point is None:
            return field_data[:]
        else:
            ix, iy, iz = spatial_point
            return field_data[:, ix, iy, iz]

    def get_diagnostics(self) -> dict[str, np.ndarray]:
        """Get all diagnostic time series."""
        if self.diagnostics is None:
            return {}

        diag_data = {}
        for name in self.diagnostics.keys():
            diag_data[name] = self.diagnostics[name][:]
        return diag_data

    def close(self) -> None:
        """Close HDF5 file."""
        if hasattr(self, "h5file") and self.h5file:
            self.h5file.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file on context exit."""
        self.close()

    def __del__(self):
        """Ensure file is closed."""
        if hasattr(self, "h5file"):
            try:
                self.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        n_snaps = self.get_n_snapshots()
        t_range = self.get_times()[[0, -1]] if n_snaps > 0 else []
        return (
            f"TrajectoryReader('{self.filename.name}', "
            f"snapshots={n_snaps}, time_range={t_range})"
        )
