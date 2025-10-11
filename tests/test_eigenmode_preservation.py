"""
Test that the eigenmode structure is preserved during evolution.

This test verifies that when the simulation is initialized with a pure
analytical eigenmode, the ratios of the fluid variables remain constant
over time, proving the correctness of the initialization and the solver.
"""

import numpy as np
import pytest

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients


def test_eigenmode_ratios_are_preserved():
    """Assert that complex eigenmode ratios are stable over time.

    Uses k=1.0 to test well within the Israel-Stewart regime.
    For τ_max=1.0, c_s≈0.577: |τω| ≈ 0.58 < 1, safely within regime limit.
    See Wagner & Gavassino (2024) and docs/IRED_THEORY.md Part IV.
    """
    # Setup a benchmark with parameters within the Israel-Stewart regime
    coeffs = TransportCoefficients(
        shear_viscosity=0.08,
        bulk_viscosity=0.04,
        shear_relaxation_time=1.0,
        bulk_relaxation_time=0.5,
    )
    print(f"Test using Transport Coefficients: {coeffs}")
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
    )
    # Use k=1.0 (well within regime, |τω| ≈ 0.58 < 1)
    k = 1.0

    # This will use the initialization logic we are about to fix
    benchmark.setup_initial_conditions(wave_number=k)

    # --- Get Analytical Target Ratios ---
    wave_vector = np.array([k, 0.0, 0.0])
    modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
    mode = modes[0]
    omega_complex = complex(mode.frequency, -mode.attenuation)
    dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega_complex, wave_vector)
    U, s, Vh = np.linalg.svd(dispersion_matrix)
    eigenvector = Vh[-1, :].conj()
    if abs(eigenvector[0]) > 1e-12:
        eigenvector = eigenvector / eigenvector[0]

    v_x_ratio_complex = eigenvector[1]
    Pi_ratio_complex = eigenvector[2]
    pi_xx_ratio_complex = eigenvector[3]

    # --- Track Ratios During Simulation ---
    time_points = []
    rho_k_list = []
    v_k_list = []
    Pi_k_list = []
    pi_k_list = []
    # For domain_size=2π and k=1.0: k_idx = 1 (since k = 2π*n/L = n)
    k_idx = 1

    def track_fields(t, fields):
        rho_fft = np.fft.fftn(fields.rho - 1.0)
        v_fft = np.fft.fftn(fields.u_mu[..., 1])
        Pi_fft = np.fft.fftn(fields.Pi)
        pi_fft = np.fft.fftn(fields.pi_munu[..., 1, 1])

        time_points.append(t)
        rho_k_list.append(rho_fft[k_idx, 0, 0])
        v_k_list.append(v_fft[k_idx, 0, 0])
        Pi_k_list.append(Pi_fft[k_idx, 0, 0])
        pi_k_list.append(pi_fft[k_idx, 0, 0])

    track_fields(0.0, benchmark.fields)

    # Evolve for a short time
    benchmark.solver.evolve(t_final=1.0, dt=0.01, method="spectral_imex", callback=track_fields)

    # --- Analyze and Assert ---
    rho_k = np.array(rho_k_list)
    v_k = np.array(v_k_list)
    Pi_k = np.array(Pi_k_list)
    pi_k = np.array(pi_k_list)

    # Calculate complex ratios at t=0 and t=final
    v_ratio_t0 = v_k[0] / rho_k[0]
    Pi_ratio_t0 = Pi_k[0] / rho_k[0]
    pi_ratio_t0 = pi_k[0] / rho_k[0]

    v_ratio_tf = v_k[-1] / rho_k[-1]
    Pi_ratio_tf = Pi_k[-1] / rho_k[-1]
    pi_ratio_tf = pi_k[-1] / rho_k[-1]

    # Assert that the initial ratios are close to the analytical target
    # This will fail before the fix
    assert np.allclose(v_ratio_t0, v_x_ratio_complex, rtol=1e-2)
    assert np.allclose(Pi_ratio_t0, Pi_ratio_complex, rtol=1e-2)
    assert np.allclose(pi_ratio_t0, pi_xx_ratio_complex, rtol=1e-2)

    # Assert that the ratios have not drifted significantly over time
    # Note: At k=1.0 we're well within the regime (|τω| ≈ 0.58 < 1), so we expect
    # excellent stability. Use 15% tolerance to account for numerical discretization.
    assert np.allclose(v_ratio_t0, v_ratio_tf, rtol=0.15)
    assert np.allclose(Pi_ratio_t0, Pi_ratio_tf, rtol=0.15)
    assert np.allclose(pi_ratio_t0, pi_ratio_tf, rtol=0.15)
