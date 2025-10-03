"""
Pytest configuration with hard memory limits to prevent system crashes.

Also provides common fixtures for SpaceGrid-based testing (Phase 6).
"""

import os
import signal
import threading
import time

import numpy as np
import psutil
import pytest

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid


class MemoryMonitor:
    """Monitor memory usage and terminate pytest if limit exceeded."""

    def __init__(self, max_memory_gb=6.0, check_interval=1.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Main monitoring loop - runs in background thread."""
        process = psutil.Process()

        while self.monitoring:
            try:
                # Get memory info for current process and all children
                memory_info = process.memory_info()
                total_memory = memory_info.rss

                # Add memory from child processes
                for child in process.children(recursive=True):
                    try:
                        child_memory = child.memory_info()
                        total_memory += child_memory.rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                memory_gb = total_memory / (1024**3)

                if total_memory > self.max_memory_bytes:
                    print(
                        f"\n!!! MEMORY LIMIT EXCEEDED: {memory_gb:.2f} GB > {self.max_memory_bytes/(1024**3):.1f} GB !!!"
                    )
                    print("!!! TERMINATING PYTEST TO PREVENT SYSTEM CRASH !!!")

                    # Force terminate the entire process group
                    try:
                        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
                    except:
                        os.kill(os.getpid(), signal.SIGKILL)

                elif memory_gb > 4.0:  # Warning at 4GB
                    print(f"\nWARNING: High memory usage: {memory_gb:.2f} GB")

            except Exception as e:
                print(f"Memory monitoring error: {e}")

            time.sleep(self.check_interval)


# Global memory monitor instance
memory_monitor = MemoryMonitor(max_memory_gb=6.0, check_interval=0.5)


@pytest.fixture(scope="session", autouse=True)
def memory_protection():
    """Automatically enable memory protection for all tests."""
    print("\n=== MEMORY PROTECTION ENABLED: 6GB HARD LIMIT ===")
    memory_monitor.start_monitoring()

    yield

    memory_monitor.stop_monitoring()
    print("\n=== MEMORY PROTECTION DISABLED ===")


@pytest.fixture(autouse=True)
def memory_check_per_test():
    """Check memory before each test and provide warnings."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024**2)  # MB

    if initial_memory > 1000:  # Warning if over 1GB before test
        print(f"\nWARNING: Starting test with {initial_memory:.0f} MB memory usage")

    yield

    final_memory = process.memory_info().rss / (1024**2)  # MB
    memory_increase = final_memory - initial_memory

    if memory_increase > 500:  # Warning if test increased memory by 500MB+
        print(f"\nWARNING: Test increased memory by {memory_increase:.0f} MB")


# ============================================================================
# Phase 6: SpaceGrid Test Fixtures
# ============================================================================


@pytest.fixture
def small_grid_3d():
    """Small 3D spatial grid for fast tests (8³)."""
    return SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(8, 8, 8),
        boundary_conditions="periodic",
    )


@pytest.fixture
def medium_grid_3d():
    """Medium 3D spatial grid for standard tests (16³)."""
    return SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 1.0)] * 3,
        grid_points=(16, 16, 16),
        boundary_conditions="periodic",
    )


@pytest.fixture
def periodic_grid_3d():
    """3D grid with periodic boundaries for wave tests (32³)."""
    return SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(32, 32, 32),
        boundary_conditions="periodic",
    )


@pytest.fixture
def test_fields_3d(small_grid_3d):
    """3D field configuration with simple initialization."""
    fields = ISFieldConfiguration(small_grid_3d)

    # Initialize with uniform state
    fields.rho[:] = 1.0
    fields.pressure[:] = 1.0 / 3.0
    fields.u_mu[..., 0] = 1.0  # Rest frame

    return fields


@pytest.fixture
def test_fields_wave(periodic_grid_3d):
    """3D field configuration with sound wave perturbation."""
    fields = ISFieldConfiguration(periodic_grid_3d)

    # Sound wave initial condition
    X, Y, Z = periodic_grid_3d.meshgrid()
    k = 1.0
    amplitude = 0.01

    fields.rho[:] = 1.0 + amplitude * np.sin(k * X)
    fields.pressure[:] = fields.rho / 3.0
    fields.u_mu[..., 0] = 1.0

    return fields


@pytest.fixture
def test_coeffs():
    """Standard transport coefficients for testing."""
    return TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
    )
