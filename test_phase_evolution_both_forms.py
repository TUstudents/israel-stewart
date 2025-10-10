#!/usr/bin/env -S uv run python
"""
Test phase evolution with both source term formulations.

Formulation A (dispersion matrix): dΠ/dt = -Π/τ - ζθ/τ
Formulation B (our 'fix'):         dΠ/dt = -Π/τ - ζθ

Which one gives correct eigenmode phase preservation?
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark, SoundWaveAnalysis
from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.metrics import MinkowskiMetric

# Test parameters
k = 8.0
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

# Get analytical eigenmode
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic"
)
metric = MinkowskiMetric()
analytical = SoundWaveAnalysis(grid, metric, coeffs)
wave_vector = np.array([k, 0.0, 0.0])
modes = analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]

print("=" * 80)
print("PHASE EVOLUTION TEST: BOTH FORMULATIONS")
print("=" * 80)
print()

print(f"Wave number: k = {k}")
print(f"Transport coefficients: η={coeffs.shear_viscosity}, ζ={coeffs.bulk_viscosity}")
print(f"                        τ_π={coeffs.shear_relaxation_time}, τ_Π={coeffs.bulk_relaxation_time}")
print()

print("Analytical eigenmode from dispersion matrix:")
print(f"  ω = {mode.frequency:.6f}")
print(f"  γ = {mode.attenuation:.6f}")

# Get eigenmode ratios from dispersion matrix eigenvector
omega_complex = complex(mode.frequency, -mode.attenuation)
dispersion_matrix = analytical._build_dispersion_matrix(omega_complex, wave_vector)
U, s, Vh = np.linalg.svd(dispersion_matrix)
eigenvector = Vh[-1, :].conj()

# Normalize to density component
if abs(eigenvector[0]) > 1e-12:
    eigenvector = eigenvector / eigenvector[0]

v_x_ratio = eigenvector[1]
Pi_ratio = eigenvector[2]
pi_xx_ratio = eigenvector[3]

print(f"  δv_x/δρ  = {v_x_ratio.real:.4f} + {v_x_ratio.imag:+.4f}j")
print(f"  δΠ/δρ    = {Pi_ratio.real:.4f} + {Pi_ratio.imag:+.4f}j")
print(f"  δπ_xx/δρ = {pi_xx_ratio.real:.4f} + {pi_xx_ratio.imag:+.4f}j")
print()

# Key question: what is the PHASE (imaginary part) of Π and π?
Pi_phase_analytical = np.angle(Pi_ratio)
pi_phase_analytical = np.angle(pi_xx_ratio)

print(f"Analytical phases:")
print(f"  Phase(δΠ/δρ):    {Pi_phase_analytical:.6f} rad = {Pi_phase_analytical*180/np.pi:.2f}°")
print(f"  Phase(δπ_xx/δρ): {pi_phase_analytical:.6f} rad = {pi_phase_analytical*180/np.pi:.2f}°")
print()

print("=" * 80)
print("TESTING CURRENT CODE (Formulation B: source without /τ)")
print("=" * 80)
print()

# Run benchmark with current code (without /τ in source)
benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)
benchmark.setup_initial_conditions(wave_number=k)

# Get initial phases
k_idx = 8
rho_fft_0 = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
Pi_fft_0 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
pi_fft_0 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]

Pi_ratio_0 = Pi_fft_0 / rho_fft_0
pi_ratio_0 = pi_fft_0 / rho_fft_0

Pi_phase_0 = np.angle(Pi_ratio_0)
pi_phase_0 = np.angle(pi_ratio_0)

print(f"Initial (t=0):")
print(f"  δΠ/δρ    = {Pi_ratio_0.real:.4f} + {Pi_ratio_0.imag:+.4f}j, phase = {Pi_phase_0*180/np.pi:.2f}°")
print(f"  δπ_xx/δρ = {pi_ratio_0.real:.4f} + {pi_ratio_0.imag:+.4f}j, phase = {pi_phase_0*180/np.pi:.2f}°")
print()

# Evolve
dt = 0.025
n_steps = 100  # t=2.5
for i in range(n_steps):
    benchmark.solver.time_step(dt, method="spectral_imex")

# Get final phases
rho_fft_f = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
Pi_fft_f = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
pi_fft_f = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]

Pi_ratio_f = Pi_fft_f / rho_fft_f
pi_ratio_f = pi_fft_f / rho_fft_f

Pi_phase_f = np.angle(Pi_ratio_f)
pi_phase_f = np.angle(pi_ratio_f)

print(f"After evolution (t={n_steps*dt:.1f}):")
print(f"  δΠ/δρ    = {Pi_ratio_f.real:.4f} + {Pi_ratio_f.imag:+.4f}j, phase = {Pi_phase_f*180/np.pi:.2f}°")
print(f"  δπ_xx/δρ = {pi_ratio_f.real:.4f} + {pi_ratio_f.imag:+.4f}j, phase = {pi_phase_f*180/np.pi:.2f}°")
print()

# Phase drift
dPi_phase = (Pi_phase_f - Pi_phase_0) * 180/np.pi
dpi_phase = (pi_phase_f - pi_phase_0) * 180/np.pi

print(f"Phase drift:")
print(f"  ΔPhase(Π):  {dPi_phase:+.2f}° (should be ~0)")
print(f"  ΔPhase(π):  {dpi_phase:+.2f}° (should be ~0)")
print()

# Check if phase flipped
Pi_sign_flip = (Pi_ratio_0.imag * Pi_ratio_f.imag) < 0
pi_sign_flip = (pi_ratio_0.imag * pi_ratio_f.imag) < 0

if Pi_sign_flip:
    print("✗ Π imaginary part FLIPPED SIGN!")
else:
    print("✓ Π imaginary part maintained sign")

if pi_sign_flip:
    print("✗ π imaginary part FLIPPED SIGN!")
else:
    print("✓ π imaginary part maintained sign")

print()

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

print("Current code (formulation B: source = -ζθ without /τ):")
if Pi_sign_flip or pi_sign_flip:
    print("  ✗ Phase evolution is INCORRECT")
    print("  ✗ Eigenmode structure not preserved")
    print("  → Source term formulation inconsistent with dispersion matrix")
else:
    print("  ✓ Phase evolution looks correct")
    print("  ✓ Eigenmode structure preserved")

print()

print("Note: The 'fix' improved frequency from 33% → 1% error,")
print("but if it causes phase flips, the physics is still wrong.")
print()

print("Next: We need to test formulation A (source = -ζθ/τ)")
print("to see if it preserves phases correctly.")
print()

print("=" * 80)
