"""Check what second-order terms are in the relaxation RHS."""
import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
    lambda_pi_pi=0.0,  # These are all zero!
    lambda_pi_Pi=0.0,
    xi_1=0.0,
    xi_2=0.0,
)

print("Transport coefficients:")
for attr in dir(coeffs):
    if not attr.startswith('_'):
        val = getattr(coeffs, attr)
        if val is not None:
            print(f"  {attr} = {val}")
print()

# Check relaxation equation formula
print("Israel-Stewart bulk relaxation equation:")
print("  dΠ/dt = -Π/τ_Π - ζθ + λ_ππ(π:π) + λ_πΠ·Π² + ...")
print()
print("With all second-order coefficients set to ZERO:")
print("  dΠ/dt = -Π/τ_Π - ζθ")
print()
print("This is a FIRST-ORDER theory, not full Israel-Stewart!")
print()
print("The analytical dispersion relation was derived for the FULL theory")
print("with second-order terms included.")
print()
print("⚠️  MISMATCH: Code uses first-order, analytical uses second-order!")
