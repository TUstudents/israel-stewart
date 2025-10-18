#!/usr/bin/env python3
"""
Debug dispersion relation for k=0.5 with regime-valid parameters.
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import create_numerical_benchmark_with_ired

# Create benchmark with regime-valid parameters (large σ)
benchmark, ired_model = create_numerical_benchmark_with_ired(
    temperature=0.4,  # 400 MeV
    cross_section=10000.0,  # Very large for regime validity
    truncation="41",
    domain_size=50.0,  # Large domain
    grid_points=(16, 16, 16),
)

# Setup with k=0.5
k = 0.5
wave_vector = np.array([k, 0.0, 0.0])

print("=" * 80)
print("DISPERSION RELATION DEBUG FOR k=0.5")
print("=" * 80)

# Print IReD transport coefficients
print("\n📊 TRANSPORT COEFFICIENTS:")
print(f"Temperature T = {ired_model.temperature:.3f} GeV")
print(f"Cross-section σ = {ired_model.cross_section:.1f} fm²")
print(f"Shear viscosity η = {ired_model.shear_viscosity():.6e} GeV³")
print(f"Bulk viscosity ζ = {ired_model.bulk_viscosity():.6e} GeV³")
print(f"Shear relaxation τ_π = {ired_model.shear_relaxation_time(time_unit='natural'):.6e} GeV⁻¹")
print(f"Bulk relaxation τ_Π = {ired_model.bulk_relaxation_time(time_unit='natural'):.6e} GeV⁻¹")

# Compute regime parameter
c_s = 1.0 / np.sqrt(3.0)
omega_estimate = c_s * k
tau_max = max(
    ired_model.shear_relaxation_time(time_unit="natural"),
    ired_model.bulk_relaxation_time(time_unit="natural")
)
regime_param = abs(tau_max * omega_estimate)

print(f"\n🔬 REGIME VALIDITY:")
print(f"Sound speed c_s = {c_s:.6f}")
print(f"Expected frequency ω ~ c_s × k = {omega_estimate:.6f} GeV")
print(f"Max relaxation time τ_max = {tau_max:.6e} GeV⁻¹")
print(f"|τω| = {regime_param:.6e}")
if regime_param < 1.0:
    print("✓ REGIME VALID (|τω| < 1)")
else:
    print(f"✗ REGIME VIOLATION (|τω| = {regime_param:.2f} > 1)")

# Check background state
epsilon0 = np.mean(benchmark.analytical.background_fields.rho)
p0 = np.mean(benchmark.analytical.background_fields.pressure)
enthalpy = epsilon0 + p0

print(f"\n📐 BACKGROUND STATE:")
print(f"Energy density ε₀ = {epsilon0:.6e} GeV⁴")
print(f"Pressure p₀ = {p0:.6e} GeV⁴")
print(f"Enthalpy ε₀+p₀ = {enthalpy:.6e} GeV⁴")
print(f"Sound speed² c_s² = p₀/ε₀ = {p0/epsilon0:.6f}")

# Estimate damping
eta = ired_model.shear_viscosity()
zeta = ired_model.bulk_viscosity()
gamma_NS = (zeta + 4.0 * eta / 3.0) * k**2 / enthalpy

print(f"\n🌊 EXPECTED DAMPING:")
print(f"Navier-Stokes estimate Γ_NS = (ζ + 4η/3) k²/(ε+p) = {gamma_NS:.6e} GeV")
print(f"Damping/frequency ratio Γ/ω = {gamma_NS/omega_estimate:.6e}")
if gamma_NS / omega_estimate < 0.01:
    print("⚠️  WARNING: Damping extremely small (< 1% of frequency)")
    print("   Nearly ideal fluid - viscous effects negligible")

# Try to find roots
print(f"\n🔎 ROOT FINDING:")
print("-" * 80)

try:
    # Call the dispersion analysis
    modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)

    print(f"Found {len(modes)} modes:")
    for i, mode in enumerate(modes):
        print(f"\n  Mode {i+1}:")
        print(f"    Frequency ω = {mode.frequency:.6e} GeV")
        print(f"    Attenuation Γ = {mode.attenuation:.6e} GeV")
        print(f"    Phase velocity v_φ = {mode.phase_velocity:.6f}")
        print(f"    Group velocity |v_g| = {np.linalg.norm(mode.group_velocity):.6f}")

        # Classify mode
        omega_complex = complex(mode.frequency, -mode.attenuation)
        if mode.frequency > 0.1 * omega_estimate and abs(mode.attenuation) < abs(mode.frequency):
            mode_type = "SOUND"
        elif abs(mode.frequency) < 0.1 * omega_estimate and mode.attenuation < -1e-10:
            mode_type = "VISCOUS"
        else:
            mode_type = "OTHER"
        print(f"    Type: {mode_type}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Manually test determinant function
print(f"\n🧪 MANUAL DETERMINANT TEST:")
print("-" * 80)

# Test at expected sound mode location
omega_test = complex(omega_estimate, -gamma_NS)
print(f"Testing at ω = {omega_test.real:.6e} - i {-omega_test.imag:.6e}")

try:
    det_value = benchmark.analytical._determinant_function(omega_test, k)
    print(f"det(M) = {det_value.real:.6e} + i {det_value.imag:.6e}")
    print(f"|det(M)| = {abs(det_value):.6e}")

    # Try to find root from this guess
    from scipy import optimize

    def det_real_imag(x):
        omega = complex(x[0], x[1])
        det = benchmark.analytical._determinant_function(omega, k)
        return [np.real(det), np.imag(det)]

    result = optimize.root(
        det_real_imag,
        [np.real(omega_test), np.imag(omega_test)],
        method="hybr",
        options={"xtol": 1e-10},
    )

    if result.success:
        omega_root = complex(result.x[0], result.x[1])
        det_check = benchmark.analytical._determinant_function(omega_root, k)
        print(f"\n✓ Root finding SUCCESS:")
        print(f"  ω = {omega_root.real:.6e} - i {-omega_root.imag:.6e}")
        print(f"  |det(M)| at root = {abs(det_check):.6e}")
    else:
        print(f"\n✗ Root finding FAILED: {result.message}")

except Exception as e:
    print(f"❌ ERROR in manual test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
