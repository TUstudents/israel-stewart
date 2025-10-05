#!/usr/bin/env python3
"""Debug eigenmode initialization - print actual eigenvector values."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

transport_coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=transport_coeffs,
)

wave_number = 1.0
wave_vector = np.array([wave_number, 0.0, 0.0])

print("=" * 80)
print("EIGENVECTOR ANALYSIS")
print("=" * 80)
print()

# Get dispersion relation
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)

if modes:
    mode = modes[0]

    print(f"Wave number k = {wave_number}")
    print(f"Frequency ω = {mode.frequency:.6f}")
    print(f"Attenuation γ = {mode.attenuation:.6f}")
    print()

    # Compute eigenmode structure by solving dispersion matrix
    omega_complex = complex(mode.frequency, -mode.attenuation)
    dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega_complex, wave_vector)

    # Find nullspace eigenvector using SVD
    _, s, Vh = np.linalg.svd(dispersion_matrix)
    eigenvector = Vh[-1, :]  # Last row = eigenvector for smallest singular value

    print(f"Smallest singular value: {s[-1]:.6e}")
    print()

    print("Raw eigenvector (complex):")
    print(f"  δρ:     {eigenvector[0]}")
    print(f"  δv_x:   {eigenvector[1]}")
    print(f"  δΠ:     {eigenvector[2]}")
    print(f"  δπ_xx:  {eigenvector[3]}")
    print()

    # Normalize by density
    if abs(eigenvector[0]) > 1e-10:
        eigenvector = eigenvector / eigenvector[0]

    print("Normalized by δρ:")
    print(f"  δρ:     {eigenvector[0]}")
    print(f"  δv_x:   {eigenvector[1]}")
    print(f"  δΠ:     {eigenvector[2]}")
    print(f"  δπ_xx:  {eigenvector[3]}")
    print()

    print("Real parts (cos(kx) component):")
    print(f"  Re(δρ):     {np.real(eigenvector[0]):.6e}")
    print(f"  Re(δv_x):   {np.real(eigenvector[1]):.6e}")
    print(f"  Re(δΠ):     {np.real(eigenvector[2]):.6e}")
    print(f"  Re(δπ_xx):  {np.real(eigenvector[3]):.6e}")
    print()

    print("Imaginary parts (sin(kx) component):")
    print(f"  Im(δρ):     {np.imag(eigenvector[0]):.6e}")
    print(f"  Im(δv_x):   {np.imag(eigenvector[1]):.6e}")
    print(f"  Im(δΠ):     {np.imag(eigenvector[2]):.6e}")
    print(f"  Im(δπ_xx):  {np.imag(eigenvector[3]):.6e}")
    print()

    print("Magnitudes:")
    print(f"  |δρ|:     {abs(eigenvector[0]):.6e}")
    print(f"  |δv_x|:   {abs(eigenvector[1]):.6e}")
    print(f"  |δΠ|:     {abs(eigenvector[2]):.6e}")
    print(f"  |δπ_xx|:  {abs(eigenvector[3]):.6e}")
    print()

    # For amplitude = 0.01, what do we expect?
    amplitude = 0.01

    print(f"With density amplitude = {amplitude}:")
    print(f"  Π amplitude = {amplitude * np.imag(eigenvector[2]):.6e}")
    print(f"  π_xx amplitude = {amplitude * np.imag(eigenvector[3]):.6e}")
    print()

    # What should these be from Navier-Stokes?
    # For sound wave: v_x ~ ε/c_s, Π ~ -ζ*∂v_x/∂x ~ -ζ*k*v_x
    c_s = 1/np.sqrt(3)
    v_x_amplitude = amplitude / c_s  # ε ~ 0.01, v ~ 0.01*sqrt(3)

    # Expansion: θ = ∂v_x/∂x ~ k*v_x for sin wave
    theta_amplitude = wave_number * v_x_amplitude

    # Navier-Stokes: Π ~ -ζ*θ
    Pi_NS = transport_coeffs.bulk_viscosity * theta_amplitude

    # Shear: σ_xx ~ (2/3)*∂v_x/∂x for longitudinal wave
    sigma_xx_amplitude = (2/3) * wave_number * v_x_amplitude
    pi_xx_NS = 2 * transport_coeffs.shear_viscosity * sigma_xx_amplitude

    print("Expected from Navier-Stokes approximation:")
    print(f"  v_x ~ ε/c_s = {v_x_amplitude:.6e}")
    print(f"  θ ~ k*v_x = {theta_amplitude:.6e}")
    print(f"  Π ~ ζ*θ = {Pi_NS:.6e}")
    print(f"  π_xx ~ 2η*σ_xx = {pi_xx_NS:.6e}")
    print()

    # Compare with eigenvector
    v_x_ratio = np.imag(eigenvector[1])
    Pi_ratio = np.imag(eigenvector[2])
    pi_xx_ratio = np.imag(eigenvector[3])

    print("Ratios from eigenvector:")
    print(f"  v_x/ρ = {v_x_ratio:.6e}")
    print(f"  Π/ρ = {Pi_ratio:.6e}")
    print(f"  π_xx/ρ = {pi_xx_ratio:.6e}")
    print()

    print("Actual field values initialized:")
    print(f"  Π = ρ_amplitude * Im(Π/ρ) = {amplitude * Pi_ratio:.6e}")
    print(f"  π_xx = ρ_amplitude * Im(π_xx/ρ) = {amplitude * pi_xx_ratio:.6e}")

print("=" * 80)
