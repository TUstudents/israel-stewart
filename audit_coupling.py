"""Audit the coupling between conservation and relaxation in IMEX."""
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

# Get initial state
k_idx = 8
rho_k_0 = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
v_k_0 = np.fft.fftn(benchmark.fields.u_mu[..., 1])[k_idx, 0, 0]
Pi_k_0 = np.fft.fftn(benchmark.fields.Pi)[k_idx, 0, 0]
pi_k_0 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])[k_idx, 0, 0]

print("COUPLING AUDIT")
print("=" * 70)
print()
print("Initial eigenmode (k=8):")
print(f"  ρ_k  = {rho_k_0}")
print(f"  v_k  = {v_k_0}")
print(f"  Π_k  = {Pi_k_0}")
print(f"  π_k  = {pi_k_0}")
print()

# Analytical ratios
r_v = v_k_0 / rho_k_0
r_Pi = Pi_k_0 / rho_k_0  
r_pi = pi_k_0 / rho_k_0

print("Eigenmode ratios:")
print(f"  v/ρ  = {r_v}")
print(f"  Π/ρ  = {r_Pi}")
print(f"  π/ρ  = {r_pi}")
print()

# Compute what the RHS SHOULD be for a pure eigenmode
omega = 5.457140
gamma = 0.200454
omega_complex = complex(omega, -gamma)

# For eigenmode: d(field)/dt = -iω·field
drho_dt_expected = -1j * omega_complex * rho_k_0
dv_dt_expected = -1j * omega_complex * v_k_0
dPi_dt_expected = -1j * omega_complex * Pi_k_0
dpi_dt_expected = -1j * omega_complex * pi_k_0

print("Expected RHS for eigenmode (Fourier space):")
print(f"  dρ/dt  = {drho_dt_expected}")
print(f"  dv/dt  = {dv_dt_expected}")
print(f"  dΠ/dt  = {dPi_dt_expected}")
print(f"  dπ/dt  = {dpi_dt_expected}")
print()

# Now compute actual RHS from the solver
# Get full RHS (conservation + relaxation)
cons_rhs = benchmark.solver.conservation.evolution_equations()
relax_rhs = benchmark.solver.relaxation.compute_relaxation_rhs(benchmark.fields)

# Unpack
drho_dt_cons = cons_rhs["drho_dt"]
dmom_dt_cons = cons_rhs["dmom_dt"]

# Convert momentum to velocity (using linearization)
h_0 = 4.0/3.0
dv_dt_cons = dmom_dt_cons[..., 0] / h_0  # x-component

# Relaxation
Pi_size = benchmark.fields.Pi.size
pi_size = benchmark.fields.pi_munu.size
dPi_dt_relax = relax_rhs[:Pi_size].reshape(benchmark.fields.Pi.shape)
dpi_dt_relax = relax_rhs[Pi_size:Pi_size+pi_size].reshape(benchmark.fields.pi_munu.shape)

# FFT
drho_k = np.fft.fftn(drho_dt_cons)[k_idx, 0, 0]
dv_k = np.fft.fftn(dv_dt_cons)[k_idx, 0, 0]
dPi_k = np.fft.fftn(dPi_dt_relax)[k_idx, 0, 0]
dpi_k = np.fft.fftn(dpi_dt_relax[..., 1, 1])[k_idx, 0, 0]

print("Actual RHS from solver (Fourier space):")
print(f"  dρ/dt  = {drho_k}")
print(f"  dv/dt  = {dv_k}")
print(f"  dΠ/dt  = {dPi_k}")
print(f"  dπ/dt  = {dpi_k}")
print()

print("Comparison (actual / expected):")
print(f"  dρ/dt: {drho_k / drho_dt_expected}")
print(f"  dv/dt: {dv_k / dv_dt_expected}")
print(f"  dΠ/dt: {dPi_k / dPi_dt_expected}")
print(f"  dπ/dt: {dpi_k / dpi_dt_expected}")
print()

# Check if RHS preserves eigenmode ratios
drho_v_ratio = dv_k / drho_k
drho_Pi_ratio = dPi_k / drho_k
drho_pi_ratio = dpi_k / drho_k

print("RHS ratios (should match eigenmode ratios if preserved):")
print(f"  dv/dρ: {drho_v_ratio} (expect {r_v})")
print(f"  dΠ/dρ: {drho_Pi_ratio} (expect {r_Pi})")
print(f"  dπ/dρ: {drho_pi_ratio} (expect {r_pi})")
print()

if np.allclose(drho_v_ratio, r_v, rtol=0.01):
    print("✓ dv/dρ ratio preserved")
else:
    print(f"✗ dv/dρ ratio ERROR: {abs(drho_v_ratio - r_v)/abs(r_v)*100:.1f}%")

if np.allclose(drho_Pi_ratio, r_Pi, rtol=0.01):
    print("✓ dΠ/dρ ratio preserved")
else:
    print(f"✗ dΠ/dρ ratio ERROR: {abs(drho_Pi_ratio - r_Pi)/abs(r_Pi)*100:.1f}%")
    
if np.allclose(drho_pi_ratio, r_pi, rtol=0.01):
    print("✓ dπ/dρ ratio preserved")
else:
    print(f"✗ dπ/dρ ratio ERROR: {abs(drho_pi_ratio - r_pi)/abs(r_pi)*100:.1f}%")
