"""Test the sign of expansion scalar for a simple velocity field."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

transport_coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=transport_coeffs,
)

wave_number = 1.0
benchmark.setup_initial_conditions(wave_number=wave_number)

print("=" * 80)
print("EXPANSION SCALAR SIGN TEST")
print("=" * 80)
print()

# Get fields
u_mu = benchmark.fields.u_mu
X, Y, Z = benchmark.grid.meshgrid()

# For sin(kx) wave:
#   v_x ~ sin(kx)
#   θ = ∇·v = ∂v_x/∂x ~ k*cos(kx)

print("Velocity field:")
print(f"  u^x at x=0:   {u_mu[0, 16, 8, 1]:.6e}")  # sin(0) = 0
print(f"  u^x at x=π/2: {u_mu[8, 16, 8, 1]:.6e}")  # sin(π/2) = max
print(f"  u^x at x=π:   {u_mu[16, 16, 8, 1]:.6e}")  # sin(π) = 0
print()

# Compute expansion using relaxation equations
theta = benchmark.solver.relaxation._compute_expansion_scalar(u_mu)

print("Expansion scalar θ = ∇·v:")
print(f"  θ at x=0:   {theta[0, 16, 8]:.6e}")  # cos(0) = +1
print(f"  θ at x=π/2: {theta[8, 16, 8]:.6e}")  # cos(π/2) = 0
print(f"  θ at x=π:   {theta[16, 16, 8]:.6e}")  # cos(π) = -1
print()

# Expected: θ ~ k*cos(kx) for v_x ~ sin(kx)
# At x=0: θ should be ~ +k*v_x_amplitude
# At x=π: θ should be ~ -k*v_x_amplitude

v_amplitude = np.max(np.abs(u_mu[..., 1]))
expected_theta_amplitude = wave_number * v_amplitude

print(f"Expected θ amplitude: ~{expected_theta_amplitude:.6e}")
print(f"Actual θ amplitude:   {np.max(np.abs(theta)):.6e}")
print()

# Check sign convention
if theta[0, 16, 8] > 0 and u_mu[8, 16, 8, 1] > 0:
    print("✓ Sign convention looks correct:")
    print("  θ > 0 at x=0 where dv_x/dx > 0")
    print("  v_x > 0 at x=π/2 (velocity antinode)")
elif theta[0, 16, 8] < 0:
    print("❌ SIGN ERROR: θ has wrong sign!")
    print("  θ should be > 0 at x=0 where dv_x/dx > 0")

print()
print("Now check bulk pressure source term:")
print()

# From relaxation equations
rhs = benchmark.solver.relaxation.compute_relaxation_rhs(benchmark.fields)

# Extract bulk RHS (first part of vector)
nx, ny, nz = benchmark.grid_points
dPi_dt = rhs[: nx * ny * nz].reshape(nx, ny, nz)

print("Bulk RHS (dΠ/dt from sources only, no linear term):")
print(f"  dΠ/dt at x=0:   {dPi_dt[0, 16, 8]:.6e}")
print(f"  dΠ/dt at x=π/2: {dPi_dt[8, 16, 8]:.6e}")
print(f"  dΠ/dt at x=π:   {dPi_dt[16, 16, 8]:.6e}")
print()

# Expected from leading-order Navier-Stokes: dΠ/dt = -ζ*θ
# But Israel-Stewart includes additional coupling terms:
#   - Linear relaxation: -Π/τ_Π (removed in source term, handled by IMEX)
#   - Nonlinear: -λ_πΠ·π^μν·π_μν / (2·τ_Π)
#   - Higher-order: -ξ_Π terms
expected_dPi_NS = -transport_coeffs.bulk_viscosity * theta[0, 16, 8]
actual_dPi_at_0 = dPi_dt[0, 16, 8]

print(f"Navier-Stokes expectation: -ζ*θ = {expected_dPi_NS:.6e}")
print(f"Actual (with IS couplings):        {actual_dPi_at_0:.6e}")
print()

# Check if sign is consistent
if np.sign(expected_dPi_NS) == np.sign(actual_dPi_at_0):
    ratio = actual_dPi_at_0 / expected_dPi_NS
    print(f"✓ Source term sign is CORRECT (ratio: {ratio:.2f})")
    print("  Difference due to IS coupling terms (λ_πΠ, ξ_Π, etc.)")
else:
    print("✗ Source term sign is WRONG!")
    print("  Expected and actual have opposite signs")

print("=" * 80)
