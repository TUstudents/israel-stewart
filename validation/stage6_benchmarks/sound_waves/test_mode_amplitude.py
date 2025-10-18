"""Track Fourier mode amplitude directly."""
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
k =8.0
benchmark.setup_initial_conditions(wave_number=k)

k_idx = 8
def get_mode_complex(fields):
    rho_fft = np.fft.fftn(fields.rho - 1.0)
    return rho_fft[k_idx, 0, 0]

# Track over short time
times = []
amplitudes = []
phases = []

print("t\t|ρ_k|\t\tphase\t\tln(|ρ_k|)")
print("-" * 60)

for i in range(31):
    t = i * 0.01
    rho_k = get_mode_complex(benchmark.fields)
    amp = abs(rho_k)
    phase = np.angle(rho_k)
    
    times.append(t)
    amplitudes.append(amp)
    phases.append(phase)
    
    if i % 5 == 0:
        print(f"{t:.2f}\t{amp:.4f}\t\t{phase:.4f}\t\t{np.log(amp):.4f}")
    
    if i < 30:
        benchmark.solver.time_step(0.01, method="spectral_imex")

# Fit damping
times_np = np.array(times)
log_amps = np.log(np.array(amplitudes))
coeffs = np.polyfit(times_np, log_amps, 1)
slope = coeffs[0]
gamma_fitted = -slope  # Convention: A = A0·exp(-γt)

print("\n" + "=" * 60)
print(f"Fitted slope of ln(A): {slope:.6f}")
print(f"Damping rate γ = -slope: {gamma_fitted:.6f}")
print(f"Expected analytical γ: 0.200454")
print()
if gamma_fitted > 0:
    print("✓ STABLE (positive damping)")
elif gamma_fitted < 0:
    print("⚠️  UNSTABLE (negative damping, amplitude growing!)")
else:
    print("NEUTRAL")
