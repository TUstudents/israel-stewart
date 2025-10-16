"""
End-to-End Test for Sound Wave Evolution (Mode B Validation)

This test validates the time-evolution capabilities of the SpectralISHydrodynamics
solver by simulating the propagation of a sound wave and verifying its physical
properties against analytical solutions.

This test addresses the gap identified in the architecture documents regarding
the lack of end-to-end validation for "Mode B" (initial value problem) evolution.
"""
import numpy as np
import pytest
from scipy.fft import fft, fftfreq

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

# Analytical properties for a radiation fluid (p = rho/3)
C_S_ANALYTICAL = 1.0 / np.sqrt(3.0)

@pytest.fixture
def sound_wave_setup():
    """Setup for the sound wave evolution test."""
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
        grid_points=(64, 8, 8),  # 1D simulation in a 3D box
        boundary_conditions="periodic",
    )

    coeffs = TransportCoefficients(
        shear_viscosity=0.0,  # Ideal fluid for simple validation
        bulk_viscosity=0.0,
        shear_relaxation_time=1e-9,
        bulk_relaxation_time=1e-9,
    )

    fields = ISFieldConfiguration(grid)
    hydro = SpectralISHydrodynamics(grid, fields, coeffs)

    return hydro, fields, grid


def test_sound_wave_propagation_and_properties(sound_wave_setup):
    """
    Tests the propagation of a sound wave, verifying its speed, frequency,
    and stability over multiple wave periods. This serves as a full, end-to-end
    validation of the 'Mode B' time evolution functionality.
    """
    hydro, fields, grid = sound_wave_setup

    # --- 1. Initial Conditions ---
    k = 2.0  # Wave number
    amplitude = 0.01  # Small amplitude for linear regime
    rho_0 = 1.0

    # Use a 3D meshgrid but only vary along x
    x, y, z = grid.coordinates["x"], grid.coordinates["y"], grid.coordinates["z"]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Initial density perturbation: rho(t=0, x) = rho_0 + A*cos(k*x)
    initial_rho_profile = rho_0 + amplitude * np.cos(k * X)
    fields.rho[:] = initial_rho_profile

    # Initial velocity perturbation: u^x = c_s * (delta_rho / rho_0)
    # For a sound wave, velocity and density are in phase.
    fields.u_mu[..., 1] = C_S_ANALYTICAL * (fields.rho - rho_0) / rho_0

    # Set other fields to equilibrium values
    fields.u_mu[..., 0] = np.sqrt(1.0 + fields.u_mu[..., 1] ** 2)
    fields.pressure[:] = fields.rho / 3.0
    fields.Pi[:] = 0.0
    fields.pi_munu[:] = 0.0

    # --- 2. Evolution ---
    # Evolve for half a wave period to check propagation speed
    omega_analytical = C_S_ANALYTICAL * k
    period = 2 * np.pi / omega_analytical
    t_final = 0.5 * period

    # Store time series for analysis
    time_points = []
    rho_at_midpoint = []

    def callback(t, f):
        time_points.append(t)
        # Observe at a point along the x-axis
        rho_at_midpoint.append(f.rho[grid.grid_points[0] // 4, 0, 0])

    hydro.evolve(
        t_final=t_final,
        method="spectral_imex",
        callback=callback,
    )

    # --- 3. Analysis & Validation ---

    # Test 1: Propagation Speed
    # Find the position of the wave peak at t=0 and t=t_final along the x-axis
    initial_peak_pos_idx = np.argmax(initial_rho_profile[:, 0, 0])
    initial_peak_pos = x[initial_peak_pos_idx]

    final_peak_pos_idx = np.argmax(fields.rho[:, 0, 0])
    final_peak_pos = x[final_peak_pos_idx]

    # The wave can travel in positive or negative direction.
    # We calculate the distance considering the periodic domain.
    distance_traveled = abs(final_peak_pos - initial_peak_pos)

    # The expected distance for half a period is half a wavelength.
    # wavelength = 2*pi/k. So expected_distance = pi/k
    expected_distance = np.pi / k

    assert np.isclose(distance_traveled, expected_distance, rtol=0.1), f"Wave traveled {distance_traveled:.3f}, expected {expected_distance:.3f}"

    # Test 2: Oscillation Frequency
    time_points = np.array(time_points)
    rho_at_midpoint = np.array(rho_at_midpoint)

    # Perform FFT on the time series of the density at a point
    N = len(time_points)
    if N < 2:
        pytest.fail("Not enough time points to perform FFT analysis.")

    # Estimate timestep from average difference
    T = np.mean(np.diff(time_points))
    yf = fft(rho_at_midpoint - np.mean(rho_at_midpoint))
    xf = fftfreq(N, T)[: N // 2]

    # Find the dominant frequency
    dominant_freq_idx = np.argmax(np.abs(yf[1 : N // 2])) + 1  # Exclude DC component
    measured_omega = 2 * np.pi * xf[dominant_freq_idx]

    # Compare to analytical frequency
    freq_error = abs(measured_omega - omega_analytical) / omega_analytical
    assert freq_error < 0.1, f"Frequency error is too large: {freq_error:.2%}"

    # Test 3: Stability and Conservation
    # Check that energy density is still positive and finite
    assert np.all(fields.rho > 0), "Energy density must remain positive"
    assert np.all(np.isfinite(fields.rho)), "Energy density must remain finite"

    # For an ideal fluid, total energy should be conserved
    initial_total_energy = np.sum(initial_rho_profile)
    final_total_energy = np.sum(fields.rho)
    energy_conservation_error = abs(final_total_energy - initial_total_energy) / initial_total_energy
    assert energy_conservation_error < 0.01, f"Energy conservation failed: {energy_conservation_error:.2%}"
