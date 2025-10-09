"""Check if four-velocity normalization drifts."""
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

def check_normalization(fields):
    """Check u_μ u^μ = -1"""
    u0 = fields.u_mu[..., 0]
    u_spatial = fields.u_mu[..., 1:4]
    norm_sq = -u0**2 + np.sum(u_spatial**2, axis=-1)
    max_violation = np.max(np.abs(norm_sq + 1.0))
    return max_violation

print("Four-velocity normalization check:")
print("t\tmax|u·u + 1|")
print("-" * 40)

times = [0]
violations = [check_normalization(benchmark.fields)]
print(f"{0:.2f}\t{violations[0]:.3e}")

dt = 0.01
for i in range(100):
    t = (i+1) * dt
    benchmark.solver.time_step(dt, method="spectral_imex")
    violation = check_normalization(benchmark.fields)
    
    if (i+1) % 20 == 0:
        print(f"{t:.2f}\t{violation:.3e}")
    
    times.append(t)
    violations.append(violation)

print()
if violations[-1] > 1e-10:
    print(f"⚠️  Normalization violated by {violations[-1]:.3e}")
else:
    print("✓ Normalization preserved")
