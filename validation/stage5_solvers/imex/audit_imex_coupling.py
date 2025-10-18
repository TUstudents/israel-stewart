"""Audit IMEX stage coupling in detail."""
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

print("=" * 80)
print("IMEX COUPLING AUDIT")
print("=" * 80)
print()

# Get initial state in Fourier space
k_idx = 8
def get_fourier_state():
    rho_k = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
    v_k = np.fft.fftn(benchmark.fields.u_mu[..., 1])[k_idx, 0, 0]
    Pi_k = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
    pi_k = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]
    return rho_k, v_k, Pi_k, pi_k

print("INITIAL STATE (Fourier mode k=8):")
rho_0, v_0, Pi_0, pi_0 = get_fourier_state()
print(f"  ρ_k = {rho_0}")
print(f"  v_k = {v_0}")
print(f"  Π_k = {Pi_0}")
print(f"  π_k = {pi_0}")
print()

# Analytical eigenmode ratios
r_v = v_0 / rho_0
r_Pi = Pi_0 / rho_0
r_pi = pi_0 / rho_0
print("Eigenmode ratios:")
print(f"  v/ρ  = {r_v}")
print(f"  Π/ρ  = {r_Pi}")
print(f"  π/ρ  = {r_pi}")
print()

# Expected evolution: all fields decay as exp(-iωt)
omega = 5.457140
gamma = 0.200454
omega_c = complex(omega, -gamma)
dt = 0.01

# Analytical expectation after dt
rho_expected = rho_0 * np.exp(-1j * omega_c * dt)
v_expected = v_0 * np.exp(-1j * omega_c * dt)
Pi_expected = Pi_0 * np.exp(-1j * omega_c * dt)
pi_expected = pi_0 * np.exp(-1j * omega_c * dt)

print("EXPECTED AFTER dt=0.01 (analytical eigenmode):")
print(f"  ρ_k = {rho_expected}")
print(f"  v_k = {v_expected}")
print(f"  Π_k = {Pi_expected}")
print(f"  π_k = {pi_expected}")
print()

# Take one IMEX step
benchmark.solver.time_step(dt, method="spectral_imex")

print("ACTUAL AFTER dt=0.01 (IMEX step):")
rho_1, v_1, Pi_1, pi_1 = get_fourier_state()
print(f"  ρ_k = {rho_1}")
print(f"  v_k = {v_1}")
print(f"  Π_k = {Pi_1}")
print(f"  π_k = {pi_1}")
print()

# Check ratios after evolution
r_v_new = v_1 / rho_1
r_Pi_new = Pi_1 / rho_1
r_pi_new = pi_1 / rho_1

print("EIGENMODE RATIO PRESERVATION:")
print(f"  v/ρ  initial: {r_v}")
print(f"  v/ρ  final:   {r_v_new}")
print(f"  Change: {abs(r_v_new - r_v)/abs(r_v)*100:.4f}%")
print()
print(f"  Π/ρ  initial: {r_Pi}")
print(f"  Π/ρ  final:   {r_Pi_new}")
print(f"  Change: {abs(r_Pi_new - r_Pi)/abs(r_Pi)*100:.4f}%")
print()
print(f"  π/ρ  initial: {r_pi}")
print(f"  π/ρ  final:   {r_pi_new}")
print(f"  Change: {abs(r_pi_new - r_pi)/abs(r_pi)*100:.4f}%")
print()

# Check if each field matches analytical
print("FIELD-BY-FIELD ERRORS:")
print(f"  ρ error: {abs(rho_1 - rho_expected)/abs(rho_expected)*100:.4f}%")
print(f"  v error: {abs(v_1 - v_expected)/abs(v_expected)*100:.4f}%")
print(f"  Π error: {abs(Pi_1 - Pi_expected)/abs(Pi_expected)*100:.4f}%")
print(f"  π error: {abs(pi_1 - pi_expected)/abs(pi_expected)*100:.4f}%")
print()

# Now check the COUPLING: does conservation RHS depend correctly on Π, π?
# Compute ∂_x(T^xx) manually
print("=" * 80)
print("CHECKING STRESS TENSOR COUPLING")
print("=" * 80)
print()

# Get stress tensor
T = benchmark.solver.conservation.stress_energy_tensor()
T_xx = T[..., 1, 1]
T_xx_k = np.fft.fftn(T_xx)[k_idx, 0, 0]

# Expected from eigenmode: T^xx = p + Π - π (Convention B)
p_k = rho_0 / 3.0
T_xx_expected = p_k + Pi_0 - pi_0

print(f"T^xx(k) expected: {T_xx_expected}")
print(f"T^xx(k) actual:   {T_xx_k}")
print(f"Error: {abs(T_xx_k - T_xx_expected)/abs(T_xx_expected)*100:.4f}%")
print()

# Check momentum equation: dv/dt should depend on ∇·T
# In Fourier space: d(h·v)/dt = -ik·T^xx
cons_rhs = benchmark.solver.conservation.evolution_equations()
dmom_dt = cons_rhs["dmom_dt"]
dmom_k = np.fft.fftn(dmom_dt[..., 0])[k_idx, 0, 0]

expected_dmom = -1j * k * T_xx_k
print("Momentum equation check:")
print(f"  d(h·v)/dt expected: {expected_dmom}")
print(f"  d(h·v)/dt actual:   {dmom_k}")
print(f"  Ratio: {dmom_k / expected_dmom}")
print()

# Check relaxation equation: dΠ/dt should depend on θ = ∇·v
# Compute θ in Fourier space
velocity_spatial = benchmark.fields.u_mu[..., 1:4]
theta = benchmark.solver.spectral.spatial_divergence(velocity_spatial)
theta_k = np.fft.fftn(theta)[k_idx, 0, 0]

# Expected: θ_k = ik·v_k
expected_theta = 1j * k * v_0
print("Expansion scalar check:")
print(f"  θ_k expected: {expected_theta}")
print(f"  θ_k actual:   {theta_k}")
print(f"  Error: {abs(theta_k - expected_theta)/abs(expected_theta)*100:.4f}%")
print()

# Check relaxation RHS
relax_rhs = benchmark.solver.relaxation.compute_relaxation_rhs(benchmark.fields)
Pi_size = benchmark.fields.Pi.size
dPi_dt = relax_rhs[:Pi_size].reshape(benchmark.fields.Pi.shape)
dPi_k = np.fft.fftn(dPi_dt)[k_idx, 0, 0]

# Expected: dΠ/dt = -Π/τ_Π - ζθ
tau_Pi = 0.5
zeta = 0.04
expected_dPi = -Pi_0/tau_Pi - zeta*theta_k

print("Bulk relaxation check:")
print(f"  dΠ/dt expected: {expected_dPi}")
print(f"  dΠ/dt actual:   {dPi_k}")
print(f"  Error: {abs(dPi_k - expected_dPi)/abs(expected_dPi)*100:.4f}%")
print()

print("=" * 80)
print("CONCLUSION:")
if abs(T_xx_k - T_xx_expected)/abs(T_xx_expected) < 0.01:
    print("✓ Stress tensor coupling is correct")
else:
    print("✗ Stress tensor coupling has error")

if abs(theta_k - expected_theta)/abs(expected_theta) < 0.01:
    print("✓ Expansion scalar coupling is correct")
else:
    print("✗ Expansion scalar coupling has error")

if abs(dmom_k / expected_dmom - 1.0) < 0.01:
    print("✓ Momentum equation uses correct stress tensor")
else:
    print("✗ Momentum equation has coupling error")

if abs(dPi_k / expected_dPi - 1.0) < 0.01:
    print("✓ Relaxation equation uses correct expansion scalar")
else:
    print("✗ Relaxation equation has coupling error")
