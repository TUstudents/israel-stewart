"""Test what happens during long evolution (like benchmark)."""
import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

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

# Track over LONG time (like benchmark does)
k_idx = 8
times = []
amplitudes = []

print("Tracking mode amplitude over long evolution...")
print("t\t|ρ_k|\t\tγ(0→t)")
print("-" * 50)

dt = 0.005
for i in range(int(3.45 / dt) + 1):  # Evolve to t=3.45 like benchmark
    t = i * dt
    rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
    amp = abs(rho_fft[k_idx, 0, 0])
    times.append(t)
    amplitudes.append(amp)
    
    # Print every 100 steps
    if i % 100 == 0 and i > 0:
        # Compute damping from t=0 to current t
        log_amp_ratio = np.log(amplitudes[-1] / amplitudes[0])
        gamma_apparent = -log_amp_ratio / t
        print(f"{t:.2f}\t{amp:.2f}\t\t{gamma_apparent:.6f}")
    
    if i < int(3.45 / dt):
        benchmark.solver.time_step(dt, method="spectral_imex")

print()
print("=" * 50)
print(f"Initial amplitude: {amplitudes[0]:.2f}")
print(f"Final amplitude: {amplitudes[-1]:.2f}")
print(f"Ratio: {amplitudes[-1]/amplitudes[0]:.4f}")
print()

# Fit different time ranges
def fit_range(t_start, t_end):
    mask = (np.array(times) >= t_start) & (np.array(times) <= t_end)
    t_fit = np.array(times)[mask]
    a_fit = np.array(amplitudes)[mask]
    log_a = np.log(a_fit)
    coeffs = np.polyfit(t_fit, log_a, 1)
    return -coeffs[0]

gamma_early = fit_range(0, 0.5)
gamma_mid = fit_range(0.5, 1.5)
gamma_late = fit_range(2.0, 3.4)

print(f"Damping γ from different time ranges:")
print(f"  t ∈ [0.0, 0.5]: γ = {gamma_early:.6f}")
print(f"  t ∈ [0.5, 1.5]: γ = {gamma_mid:.6f}")
print(f"  t ∈ [2.0, 3.4]: γ = {gamma_late:.6f}")
print()
print(f"Analytical: γ = 0.200454")
