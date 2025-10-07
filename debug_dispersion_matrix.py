#!/usr/bin/env python
"""
Script to debug the analytical dispersion matrix by testing a range of
relaxation time parameters and observing the effect on causality.
"""
import numpy as np
import warnings
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Assume the necessary classes are in the path
from israel_stewart.benchmarks.sound_waves import SoundWaveAnalysis
from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid

def debug_dispersion_parameters(tau_pi_values, tau_Pi_values, k_val=8.0):
    """
    Tests the dispersion relation for various relaxation times.
    """
    print("="*80)
    print("Dispersion Matrix Debugger")
    print(f"Probing for acausality at k = {k_val}")
    print("="*80)

    # Use a dummy grid, it's not important for the analytical calculation
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=(32, 32, 16),
        boundary_conditions="periodic",
    )
    metric = MinkowskiMetric()

    for tau_pi in tau_pi_values:
        for tau_Pi in tau_Pi_values:
            print(f"--- Testing: τ_π = {tau_pi:.3f}, τ_Π = {tau_Pi:.3f} ---")

            try:
                coeffs = TransportCoefficients(
                    shear_viscosity=0.08,
                    bulk_viscosity=0.04,
                    shear_relaxation_time=tau_pi,
                    bulk_relaxation_time=tau_Pi,
                )

                analyzer = SoundWaveAnalysis(grid, metric, coeffs)
                wave_vector = np.array([k_val, 0.0, 0.0])

                # Capture warnings to detect acausality
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    # The analyze_dispersion_relation function prints the raw roots
                    # which is useful for us.
                    modes = analyzer.analyze_dispersion_relation(wave_vector)
                    
                    print(f"  Physical modes found:")
                    for mode in modes:
                        omega_complex = complex(mode.frequency, -mode.attenuation)
                        sound_speed = mode.frequency / k_val if k_val > 0 else 0
                        print(f"    - ω = {omega_complex:.4f}, c_s = {sound_speed:.4f}")

                    if any("Sound speed outside physical range" in str(warn.message) for warn in w):
                        print("  RESULT: !! ACAUSAL MODE DETECTED !!")
                    elif not modes:
                        print("  RESULT: !! NO PHYSICAL MODES FOUND !!")
                    else:
                        print("  RESULT: Physical modes appear causal.")

            except Exception as e:
                print(f"  !! ERROR during analysis: {e}")
            
            print("-" * 50)
            print()


if __name__ == "__main__":
    # Define ranges of relaxation times to test
    # Let's test around the values the user provided
    tau_pi_range = [0.3, 0.5, 1.0, 1.5]
    tau_Pi_range = [0.3, 0.5, 1.0, 1.5]

    debug_dispersion_parameters(tau_pi_range, tau_Pi_range)
