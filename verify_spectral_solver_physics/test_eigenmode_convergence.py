#!/usr/bin/env python
"""
Test eigenmode preservation with different timesteps to verify convergence.

Since RHS now matches analytical perfectly, any remaining drift should
decrease as O(dt^4) for RK4.
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
    lambda_pi_pi=0.0,
    lambda_pi_Pi=0.0,
    xi_1=0.0,
    xi_2=0.0,
)

k = 8.0
t_final = 0.1  # Short time for convergence test

print("="*80)
print("EIGENMODE CONVERGENCE TEST")
print("="*80)
print()
print(f"Testing eigenmode preservation with k={k}, t_final={t_final}")
print()

# Test different timesteps
timesteps = [0.01, 0.005, 0.0025]

results = []

for dt in timesteps:
    # Create fresh benchmark
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2*np.pi,
        grid_points=(32, 32, 16),
        transport_coeffs=coeffs
    )
    benchmark.setup_initial_conditions(wave_number=k)

    # Get initial ratios
    k_idx = 8
    rho_fft_0 = np.fft.fftn(benchmark.fields.rho - 1.0)
    v_fft_0 = np.fft.fftn(benchmark.fields.u_mu[..., 1])

    rho_k_0 = rho_fft_0[k_idx, 0, 0]
    v_k_0 = v_fft_0[k_idx, 0, 0]

    v_ratio_0 = v_k_0 / rho_k_0

    # Evolve
    n_steps = int(t_final / dt)
    for _ in range(n_steps):
        benchmark.solver.time_step(dt, method="rk4")

    # Get final ratios
    rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
    v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])

    rho_k = rho_fft[k_idx, 0, 0]
    v_k = v_fft[k_idx, 0, 0]

    v_ratio = v_k / rho_k

    # Compute error
    error = abs(v_ratio - v_ratio_0) / abs(v_ratio_0)

    results.append({
        'dt': dt,
        'n_steps': n_steps,
        'v_ratio_0': v_ratio_0,
        'v_ratio_final': v_ratio,
        'error': error
    })

    print(f"dt = {dt:.4f}, n_steps = {n_steps:3d}:")
    print(f"  Initial: v/ρ = {v_ratio_0}")
    print(f"  Final:   v/ρ = {v_ratio}")
    print(f"  Error: {error*100:.4f}%")
    print()

print("="*80)
print("CONVERGENCE ANALYSIS")
print("="*80)
print()

# Check convergence rate
if len(results) >= 2:
    for i in range(len(results)-1):
        dt1 = results[i]['dt']
        dt2 = results[i+1]['dt']
        err1 = results[i]['error']
        err2 = results[i+1]['error']

        if err1 > 1e-14 and err2 > 1e-14:
            ratio = dt1 / dt2
            expected_reduction = ratio**4  # RK4 is O(dt^4)
            actual_reduction = err1 / err2

            print(f"dt: {dt1:.4f} → {dt2:.4f}  (factor of {ratio:.1f})")
            print(f"  Error: {err1*100:.4e}% → {err2*100:.4e}%")
            print(f"  Reduction: {actual_reduction:.2f}× (expected {expected_reduction:.2f}× for RK4)")

            if abs(actual_reduction - expected_reduction) / expected_reduction < 0.5:
                print(f"  ✓ Consistent with O(dt^4) convergence")
            else:
                print(f"  ⚠ Does not match O(dt^4)")
            print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()

final_error = results[-1]['error'] * 100

if final_error < 0.1:
    print(f"✓ EXCELLENT: Error = {final_error:.4f}% at dt={results[-1]['dt']}")
    print("  Eigenmode is preserved to high accuracy")
    print("  → Both fixes (linearized momentum + stress tensor sign) work!")
elif final_error < 1.0:
    print(f"✓ GOOD: Error = {final_error:.4f}% at dt={results[-1]['dt']}")
    print("  Eigenmode is well preserved")
    print("  → Fixes are working, small error is numerical truncation")
elif final_error < 5.0:
    print(f"⚠ MODERATE: Error = {final_error:.2f}% at dt={results[-1]['dt']}")
    print("  Some drift remains")
else:
    print(f"✗ LARGE: Error = {final_error:.2f}% at dt={results[-1]['dt']}")
    print("  Significant drift persists")

print()
print("="*80)
