"""Minimal test of corrected physics - direct field setup."""
import numpy as np
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

# Create grid
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic"
)

# Transport coefficients
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

# Initialize fields
fields = ISFieldConfiguration(grid)
X, Y, Z = grid.meshgrid()

# Set up simple plane wave in x-direction with k=1
k = 1.0
rho_0 = 1.0
amplitude = 0.01

# From dispersion relation eigenmode
omega = 0.599320
gamma = 0.045869

# Eigenmode ratios (from dispersion relation)
v_ratio = 0.4495 - 0.0344j
Pi_ratio = 0.003871 - 0.01721j
pi_ratio = -0.01988 + 0.03777j

# Set initial conditions (real parts only)
fields.rho[:] = rho_0 + amplitude * np.cos(k * X)
fields.pressure[:] = fields.rho / 3.0
fields.u_mu[..., 0] = 1.0  # u^0 = 1 (rest frame)
fields.u_mu[..., 1] = amplitude * v_ratio.real * np.cos(k * X)
fields.Pi[:] = amplitude * Pi_ratio.real * np.cos(k * X)
fields.pi_munu[..., 0, 0] = amplitude * pi_ratio.real * np.cos(k * X)

# Create solver
solver = SpectralISHydrodynamics(
    grid=grid,
    fields=fields,
    coeffs=coeffs
)

print("=" * 80)
print("MINIMAL CORRECTED PHYSICS TEST")
print("=" * 80)
print(f"\nAnalytical predictions:")
print(f"  ω = {omega:.6f}")
print(f"  γ = {gamma:.6f}")
print()

k_idx = 1  # k=1 mode

# Get initial amplitude
rho_k_0 = np.fft.fftn(fields.rho)[k_idx, 0, 0]
amplitude_0 = abs(rho_k_0)

# Evolve
dt = 0.01
n_steps = 300
times = []
amplitudes = []
phases = []

print("Evolving for 300 steps (t = 3.0)...")
for i in range(n_steps):
    t = i * dt

    # Record every 10 steps
    if i % 10 == 0:
        rho_k = np.fft.fftn(fields.rho)[k_idx, 0, 0]
        times.append(t)
        amplitudes.append(abs(rho_k))
        phases.append(np.angle(rho_k))
        if i % 100 == 0:
            print(f"  t = {t:.2f}, |ρ_k| = {abs(rho_k):.6f}")

    solver.time_step(dt, method="spectral_imex")

times = np.array(times)
amplitudes = np.array(amplitudes)
phases = np.unwrap(np.array(phases))

# Measure damping from amplitude
log_amp = np.log(amplitudes)
gamma_fit = -np.polyfit(times, log_amp, 1)[0]

# Measure frequency from phase
omega_fit = -np.polyfit(times, phases, 1)[0]

# Calculate errors
freq_error = abs(omega_fit - omega) / omega * 100
gamma_error = abs(gamma_fit - gamma) / gamma * 100

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\nFrequency:")
print(f"  Analytical:  ω = {omega:.6f}")
print(f"  Measured:    ω = {omega_fit:.6f}")
print(f"  Error:       {freq_error:.2f}%")
print()
print(f"Damping:")
print(f"  Analytical:  γ = {gamma:.6f}")
print(f"  Measured:    γ = {gamma_fit:.6f}")
print(f"  Error:       {gamma_error:.2f}%")
print()

# Check if unstable
if gamma_fit < 0:
    print("⚠️  WARNING: Negative damping detected - numerical instability!")
    print(f"    γ = {gamma_fit:.6f} (should be positive)")
else:
    print("✓ Damping is positive (stable)")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
if freq_error < 1.0:
    print("✓ Frequency error < 1% confirms source term fix is CORRECT")
else:
    print(f"✗ Frequency error {freq_error:.2f}% suggests physics issue")

if gamma_fit < 0:
    print("✗ Negative damping confirms IMEX numerical instability")
    print("  (This is separate from the source term fix)")
elif gamma_error > 20:
    print(f"⚠️  Large damping error ({gamma_error:.1f}%) but still positive")
else:
    print("✓ Damping reasonably accurate")
