"""Compare IMEX vs RK4 for long-time stability."""
import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

def test_integrator(method_name):
    coeffs = TransportCoefficients(
        shear_viscosity=0.08,
        bulk_viscosity=0.04,
        shear_relaxation_time=1.0,
        bulk_relaxation_time=0.5,
    )
    
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
    )
    k = 8.0
    benchmark.setup_initial_conditions(wave_number=k)
    
    k_idx = 8
    A_0 = abs(np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0])
    
    # Evolve to t=1.0 (shorter to save time)
    dt = 0.01
    n_steps = int(1.0 / dt)
    
    for i in range(n_steps):
        benchmark.solver.time_step(dt, method=method_name)
    
    A_final = abs(np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0])
    ratio = A_final / A_0
    gamma = -np.log(ratio) / 1.0
    
    return A_0, A_final, ratio, gamma

print("=" * 70)
print("COMPARING TIME INTEGRATORS")
print("=" * 70)
print()

print("Testing spectral_imex...")
A0_imex, Af_imex, ratio_imex, gamma_imex = test_integrator("spectral_imex")

print("Testing rk4...")
A0_rk4, Af_rk4, ratio_rk4, gamma_rk4 = test_integrator("rk4")

print()
print(f"{'Method':<15} {'A(0)':<10} {'A(1.0)':<10} {'Ratio':<10} {'γ':<10}")
print("-" * 70)
print(f"{'spectral_imex':<15} {A0_imex:<10.2f} {Af_imex:<10.2f} {ratio_imex:<10.4f} {gamma_imex:<10.6f}")
print(f"{'rk4':<15} {A0_rk4:<10.2f} {Af_rk4:<10.2f} {ratio_rk4:<10.4f} {gamma_rk4:<10.6f}")
print()
print(f"Analytical γ: 0.200454")
print()

if ratio_imex > 1.05:
    print("⚠️  IMEX shows GROWTH (unstable)")
else:
    print("✓ IMEX is stable")

if ratio_rk4 > 1.05:
    print("⚠️  RK4 shows GROWTH (unstable)")
else:
    print("✓ RK4 is stable")
