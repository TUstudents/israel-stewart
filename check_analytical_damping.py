#!/usr/bin/env python3
"""
Check if the analytical damping prediction is actually correct for the given parameters.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import SoundWaveAnalysis
from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid

# Create grid
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic",
)

# Transport coefficients
transport_coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
)

# Create analytical analysis
metric = MinkowskiMetric()
analytical = SoundWaveAnalysis(grid, metric, transport_coeffs)

# Analyze for k=1.0
wave_number = 1.0
wave_vector = np.array([wave_number, 0.0, 0.0])

print("=" * 80)
print("ANALYTICAL DAMPING VERIFICATION")
print("=" * 80)
print()

# Get analytical mode
modes = analytical.analyze_dispersion_relation(wave_vector)

if modes:
    mode = modes[0]
    print(f"Wave number k: {wave_number}")
    print(f"Frequency ω:   {mode.frequency:.6f}")
    print(f"Damping γ:     {mode.attenuation:.6f}")
    print()

    # Compute dimensionless parameters
    omega_tau_pi = mode.frequency * transport_coeffs.shear_relaxation_time
    omega_tau_Pi = mode.frequency * transport_coeffs.bulk_relaxation_time

    print("Dimensionless parameters:")
    print(f"  ω*τ_π = {omega_tau_pi:.4f}")
    print(f"  ω*τ_Π = {omega_tau_Pi:.4f}")
    print()

    if omega_tau_pi < 1 and omega_tau_Pi < 1:
        print("⚠️  REGIME: Slow oscillations (ω*τ << 1)")
        print("   Dissipative fluxes CAN track the oscillations")
        print("   Should see Navier-Stokes-like damping")
    elif omega_tau_pi > 1 or omega_tau_Pi > 1:
        print("⚠️  REGIME: Fast oscillations (ω*τ ~ 1 or > 1)")
        print("   Dissipative fluxes CANNOT track the oscillations")
        print("   Israel-Stewart damping < Navier-Stokes damping")
    print()

    # Navier-Stokes limit damping (τ → 0):
    # γ_NS = (η_s + 4η/3) * k² / (ε₀ + p₀)
    # For our case: ε₀ = 1, p₀ = 1/3, so ε₀ + p₀ = 4/3

    enthalpy = 4.0 / 3.0
    gamma_NS = (transport_coeffs.bulk_viscosity + 4*transport_coeffs.shear_viscosity/3) * wave_number**2 / enthalpy

    print(f"Navier-Stokes damping (τ → 0): γ_NS = {gamma_NS:.6f}")
    print(f"Israel-Stewart damping:         γ_IS = {mode.attenuation:.6f}")
    print(f"Ratio γ_IS/γ_NS = {mode.attenuation/gamma_NS:.4f}")
    print()

    if mode.attenuation < 0.5 * gamma_NS:
        print("❌ Israel-Stewart damping is MUCH smaller than Navier-Stokes!")
        print("   This is expected when ω*τ ~ 1 (relaxation effects)")
    else:
        print("✓ Israel-Stewart damping is comparable to Navier-Stokes")

    print()

    # Expected decay after simulation time
    t_sim = 32.0
    decay_IS = np.exp(-mode.attenuation * t_sim)
    decay_NS = np.exp(-gamma_NS * t_sim)

    print(f"After t = {t_sim:.1f}:")
    print(f"  Israel-Stewart: A(t)/A(0) = {decay_IS:.4f}")
    print(f"  Navier-Stokes:  A(t)/A(0) = {decay_NS:.4f}")
    print()

    # What we actually measured
    measured_decay = 0.98
    print(f"  Measured:       A(t)/A(0) = {measured_decay:.4f}")
    print()

    if abs(decay_IS - measured_decay) / decay_IS < 0.2:
        print("✓ Measured decay matches Israel-Stewart prediction!")
    elif abs(decay_NS - measured_decay) / decay_NS < 0.2:
        print("? Measured decay matches Navier-Stokes prediction!")
    else:
        print("❌ Measured decay does NOT match either prediction!")
        print(f"   Expected {decay_IS:.4f}, got {measured_decay:.4f}")
        print(f"   Discrepancy: {abs(measured_decay - decay_IS)/decay_IS * 100:.1f}%")

else:
    print("❌ Could not find analytical mode!")

print("=" * 80)
