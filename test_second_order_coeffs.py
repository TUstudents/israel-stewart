"""Test with and without second-order transport coefficients."""
import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

def run_test(with_second_order=False):
    if with_second_order:
        # Include second-order coefficients (typical values)
        coeffs = TransportCoefficients(
            shear_viscosity=0.08,
            bulk_viscosity=0.04,
            shear_relaxation_time=1.0,
            bulk_relaxation_time=0.5,
            lambda_pi_pi=0.1,
            lambda_pi_Pi=0.05,
        )
        label = "WITH second-order"
    else:
        # Zero second-order coefficients (current)
        coeffs = TransportCoefficients(
            shear_viscosity=0.08,
            bulk_viscosity=0.04,
            shear_relaxation_time=1.0,
            bulk_relaxation_time=0.5,
            lambda_pi_pi=0.0,
            lambda_pi_Pi=0.0,
        )
        label = "WITHOUT second-order"
    
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
    )
    k = 8.0
    benchmark.setup_initial_conditions(wave_number=k)
    
    # Track amplitude
    k_idx = 8
    times = []
    amplitudes = []
    
    for i in range(31):
        t = i * 0.01
        rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
        amp = abs(rho_fft[k_idx, 0, 0])
        times.append(t)
        amplitudes.append(amp)
        
        if i < 30:
            benchmark.solver.time_step(0.01, method="spectral_imex")
    
    # Fit damping
    log_amps = np.log(np.array(amplitudes))
    coeffs_fit = np.polyfit(times, log_amps, 1)
    gamma = -coeffs_fit[0]
    
    return gamma, label

print("=" * 70)
print("TESTING SECOND-ORDER COEFFICIENT EFFECT")
print("=" * 70)
print()

gamma_without, label_without = run_test(with_second_order=False)
print(f"{label_without:25s}: γ = {gamma_without:.6f}")

gamma_with, label_with = run_test(with_second_order=True)
print(f"{label_with:25s}: γ = {gamma_with:.6f}")

print()
print(f"Analytical γ: 0.200454")
print()
print(f"Difference: {abs(gamma_with - gamma_without):.6f}")

if abs(gamma_with - gamma_without) > 0.01:
    print("⚠️  Second-order coefficients significantly affect damping!")
else:
    print("✓ Second-order coefficients have minimal effect")
