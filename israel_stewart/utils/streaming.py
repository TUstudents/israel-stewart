"""
Streaming snapshot architecture for memory-efficient hydrodynamics simulations.

This module provides buffered snapshot writing with automatic flushing to enable
long-running simulations with constant memory usage.
"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..core.spacegrid import SpaceGrid
from ..core.spacetime_grid import SpacetimeGrid
from ..utils.io import TrajectoryWriter

if TYPE_CHECKING:
    from ..core.fields import ISFieldConfiguration, TransportCoefficients


class SnapshotStream:
    """
    Buffered snapshot streaming with automatic flushing.

    Enables memory-efficient long-running simulations by buffering snapshots
    in memory and periodically flushing to HDF5 files. Automatically handles
    SpaceGrid to SpacetimeGrid conversion for trajectory metadata.

    Example:
        >>> grid = SpaceGrid("cartesian", [(0, 1)] * 3, (64, 64, 64))
        >>> stream = SnapshotStream(
        ...     filename="output.h5", grid=grid, coeffs=coeffs, interval=0.1, buffer_size=20
        ... )
        >>> stream.save(0.0, fields)  # Save initial
        >>> # ... time evolution ...
        >>> stream.save(0.1, fields)  # Buffered
        >>> stream.flush()  # Write to disk
        >>> stream.close()
    """

    def __init__(
        self,
        filename: str,
        grid: SpaceGrid,
        coeffs: Optional["TransportCoefficients"] = None,
        interval: float = 0.1,
        buffer_size: int = 10,
    ):
        """
        Initialize snapshot stream.

        Args:
            filename: Output HDF5 file path
            grid: SpaceGrid defining spatial domain
            coeffs: Transport coefficients for metadata
            interval: Minimum time interval between snapshots (default: 0.1)
            buffer_size: Number of snapshots to buffer before flushing (default: 10)
        """
        self.filename = filename
        self.grid = grid
        self.coeffs = coeffs
        self.interval = interval
        self.buffer_size = buffer_size

        # Create SpacetimeGrid for trajectory metadata
        spacetime_grid = self._spacegrid_to_spacetimegrid(grid)
        self.writer = TrajectoryWriter(filename, spacetime_grid, coeffs)

        # Buffering state
        self.buffer: list[tuple[float, ISFieldConfiguration]] = []
        self.last_snapshot_time = -np.inf
        self._total_snapshots = 0
        self._is_closed = False

    def should_save(self, t: float) -> bool:
        """
        Check if snapshot should be saved at time t.

        Args:
            t: Current simulation time

        Returns:
            True if sufficient time has elapsed since last snapshot
        """
        return (t - self.last_snapshot_time) >= self.interval

    def save(self, t: float, fields: "ISFieldConfiguration") -> None:
        """
        Save snapshot to buffer with automatic flushing.

        Args:
            t: Simulation time
            fields: Field configuration to save (will be deep copied)

        Raises:
            RuntimeError: If stream has been closed
        """
        if self._is_closed:
            raise RuntimeError("Cannot save to closed stream")

        # Deep copy fields to avoid modifications during buffering
        snapshot = self._copy_fields(fields)
        self.buffer.append((t, snapshot))
        self._total_snapshots += 1

        # Automatic flush when buffer is full
        if len(self.buffer) >= self.buffer_size:
            self.flush()

        # Update last snapshot time
        self.last_snapshot_time = t

    def flush(self) -> None:
        """
        Write all buffered snapshots to disk and clear buffer.

        This is automatically called when buffer is full, but can also be
        called manually to ensure data is written.
        """
        if not self.buffer:
            return

        # Write all buffered snapshots
        for t, fields_copy in self.buffer:
            self.writer.write_snapshot(t, fields_copy)

        # Clear buffer
        self.buffer.clear()

    def close(self) -> None:
        """
        Flush remaining snapshots and close the trajectory file.

        Should be called at end of simulation to ensure all data is written.
        """
        if self._is_closed:
            return

        # Flush any remaining snapshots
        self.flush()

        # Close the HDF5 file
        self.writer.close()
        self._is_closed = True

    def _copy_fields(self, fields: "ISFieldConfiguration") -> "ISFieldConfiguration":
        """
        Create deep copy of field configuration.

        Args:
            fields: Original field configuration

        Returns:
            Deep copy of fields
        """
        # Use the built-in copy method with all fields
        return fields.copy()

    def _spacegrid_to_spacetimegrid(self, grid: SpaceGrid) -> SpacetimeGrid:
        """
        Convert SpaceGrid to SpacetimeGrid for trajectory metadata.

        Creates a minimal SpacetimeGrid with nt=1 (single time slice) that
        preserves all spatial grid information for HDF5 metadata.

        Args:
            grid: SpaceGrid to convert

        Returns:
            SpacetimeGrid with spatial information and nt=1
        """
        return SpacetimeGrid(
            coordinate_system=grid.coordinate_system,
            time_range=(0.0, 1.0),  # Dummy time range (actual times in snapshots)
            spatial_ranges=grid.spatial_ranges,
            grid_points=(1, *grid.grid_points),  # nt=1 for metadata only
            metric=grid.metric if hasattr(grid, "metric") else None,
            boundary_conditions=grid.boundary_conditions,
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures stream is closed."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._is_closed else "open"
        return (
            f"SnapshotStream(filename='{self.filename}', "
            f"buffer_size={self.buffer_size}, "
            f"interval={self.interval}, "
            f"buffered={len(self.buffer)}, "
            f"total={self._total_snapshots}, "
            f"status={status})"
        )


class StreamingSimulation:
    """
    High-level wrapper for streaming hydrodynamics simulations.

    Provides a convenient interface for running memory-efficient simulations
    with automatic snapshot management.

    Example:
        >>> with StreamingSimulation(
        ...     filename="output.h5", grid=grid, fields=fields, coeffs=coeffs, snapshot_interval=0.1
        ... ) as sim:
        ...     sim.run(t_final=10.0, dt=0.01)
    """

    def __init__(
        self,
        filename: str,
        grid: SpaceGrid,
        fields: "ISFieldConfiguration",
        coeffs: Optional["TransportCoefficients"] = None,
        snapshot_interval: float = 0.1,
        buffer_size: int = 10,
        save_initial: bool = True,
    ):
        """
        Initialize streaming simulation.

        Args:
            filename: Output HDF5 file
            grid: SpaceGrid defining domain
            fields: Initial field configuration
            coeffs: Transport coefficients
            snapshot_interval: Time between snapshots
            buffer_size: Snapshots to buffer before flushing
            save_initial: Whether to save initial conditions
        """
        self.grid = grid
        self.fields = fields
        self.coeffs = coeffs

        # Create snapshot stream
        self.stream = SnapshotStream(
            filename=filename,
            grid=grid,
            coeffs=coeffs,
            interval=snapshot_interval,
            buffer_size=buffer_size,
        )

        # Save initial conditions if requested
        if save_initial:
            self.stream.save(0.0, fields)

        self.current_time = 0.0

    def step(self, dt: float, solver: Any) -> None:
        """
        Advance simulation by one time step.

        Args:
            dt: Time step
            solver: Solver with time_step(dt) method
        """
        solver.time_step(dt)
        self.current_time += dt

        # Save snapshot if interval has elapsed
        if self.stream.should_save(self.current_time):
            self.stream.save(self.current_time, self.fields)

    def run(self, t_final: float, dt: float, solver: Any) -> None:
        """
        Run simulation to final time with automatic snapshots.

        Args:
            t_final: Final simulation time
            dt: Time step
            solver: Solver with time_step(dt) method
        """
        while self.current_time < t_final:
            dt_step = min(dt, t_final - self.current_time)
            self.step(dt_step, solver)

    def close(self) -> None:
        """Close stream and finalize output."""
        self.stream.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
