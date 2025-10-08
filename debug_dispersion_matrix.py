#!/usr/bin/env python
"""
Script to find the critical wave number k_c at which the dispersion
relation becomes acausal for a given set of physical parameters.
"""
import numpy as np
import warnings
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import SoundWaveAnalysis
from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid

def find_causality_breakdown(coeffs, k_max=10.0, steps=100):
    """
    Scans wave numbers to find where the model becomes acausal.
    """
    print("="*80)
    print("Causality Breakdown Analysis")
    print(f"Parameters: {coeffs}")
    print("="*80)

    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(32, 32, 16),
        boundary_conditions="periodic",
    )
    metric = MinkowskiMetric()
    analyzer = SoundWaveAnalysis(grid, metric, coeffs)

    k_values = np.linspace(0.1, k_max, steps)
    max_sound_speeds = []
    first_acausal_k = None

    for k in k_values:
        wave_vector = np.array([k, 0.0, 0.0])
        
        acausal_found_for_k = False
        max_speed_for_k = 0.0

        # We need to check all roots for acausality, not just the physical ones
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Get all roots, including potentially unphysical ones
            complex_roots = analyzer._find_dispersion_roots(k)
            for omega_c in complex_roots:
                speed = omega_c.real / k if k > 0 else 0
                if speed > max_speed_for_k:
                    max_speed_for_k = speed
                if speed > 1.0:
                    acausal_found_for_k = True
        
        max_sound_speeds.append(max_speed_for_k)
        if acausal_found_for_k and first_acausal_k is None:
            first_acausal_k = k

    if first_acausal_k is not None:
        print(f"\nRESULT: Acausality detected!")
        print(f"  -> Critical wave number k_c ≈ {first_acausal_k:.3f}")
        print(f"  -> At this k, a mode with speed c_s > 1 appears.")
    else:
        print(f"\nRESULT: No acausality found up to k_max = {k_max:.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, max_sound_speeds, 'b-', label='Max Sound Speed in Spectrum')
    plt.axhline(1.0, color='r', linestyle='--', label='Speed of Light (c=1)')
    if first_acausal_k is not None:
        plt.axvline(first_acausal_k, color='k', linestyle=':', label=f'k_c ≈ {first_acausal_k:.3f}')
    plt.xlabel("Wave Number k")
    plt.ylabel("Max Phase Velocity (c_s = Re(ω)/k)")
    plt.title(f"Causality Analysis for τ_π={coeffs.shear_relaxation_time}, τ_Π={coeffs.bulk_relaxation_time}")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.5)
    plot_path = 'causality_breakdown.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    print("="*80)


if __name__ == "__main__":
    # Use the known-stable parameters from the last successful run
    stable_coeffs = TransportCoefficients(
        shear_viscosity=0.08,
        bulk_viscosity=0.04,
        shear_relaxation_time=1.0,
        bulk_relaxation_time=0.5,
    )
    find_causality_breakdown(stable_coeffs)
