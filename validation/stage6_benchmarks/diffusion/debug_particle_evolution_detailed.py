#!/usr/bin/env python3
"""
Comprehensive diagnostic script to trace particle density evolution instability.

This script traces the complete chain: n → μ/T → ∇(μ/T) → V → dn/dt → n
to identify exactly where the exponential growth instability originates.

Expected behavior: Exponential decay with Γ = Dk² ~ 4e-5
Observed behavior: Exponential growth with Γ = -2.2 (wrong sign, 55,000× too large)

This indicates either:
1. Sign error in one of the chain steps
2. Positive feedback loop instead of negative feedback
3. Numerical instability from timestep constraint violation
"""

import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired
from israel_stewart.core.tensor_utils import optimized_einsum
from israel_stewart.equations.ired_simple import HardSphereIReD


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_field_stats(name, field, show_perturbation=False):
    """Print statistics for a field."""
    print(f"\n{name}:")
    print(f"  min   = {field.min():+.6e}")
    print(f"  max   = {field.max():+.6e}")
    print(f"  mean  = {field.mean():+.6e}")
    print(f"  std   = {field.std():+.6e}")
    print(f"  RMS   = {np.sqrt(np.mean(field**2)):+.6e}")

    if show_perturbation and field.std() > 1e-15:
        # For sinusoidal perturbations, amplitude ≈ √2 * RMS of perturbation
        perturbation = field - field.mean()
        amplitude = np.sqrt(np.mean(perturbation**2))
        print(f"  Perturbation amplitude: {amplitude:.6e}")


def check_analytical_vs_numerical(name, analytical, numerical, tolerance=1e-10):
    """Compare analytical and numerical values."""
    error = np.abs(numerical - analytical)
    rel_error = error / (np.abs(analytical) + 1e-15)
    max_error = error.max()
    max_rel_error = rel_error.max()

    status = "✓" if max_error < tolerance else "✗"
    print(f"\n{name}:")
    print(f"  Analytical: {analytical.mean():.6e} (mean)")
    print(f"  Numerical:  {numerical.mean():.6e} (mean)")
    print(f"  Max absolute error: {max_error:.6e} {status}")
    print(f"  Max relative error: {max_rel_error:.2%}")

    return max_error < tolerance


# =============================================================================
# Setup: Create diffusion benchmark
# =============================================================================

print_section("SETUP: DIFFUSION BENCHMARK")

# Use same parameters as test
temperature = 0.4  # 400 MeV
cross_section = 1000.0  # Large σ for regime validity
truncation = "41"
perturbation_amplitude = 0.05
wave_number = 0.5  # Low k for regime validity
grid_points = (32, 32, 32)  # Moderate resolution
domain_size = 4 * np.pi

print(f"Temperature T = {temperature} GeV")
print(f"Cross section σ = {cross_section} fm²")
print(f"Wave number k = {wave_number} fm⁻¹")
print(f"Perturbation amplitude = {perturbation_amplitude}")
print(f"Grid: {grid_points}, domain: {domain_size:.2f} fm")

# Create benchmark
benchmark, ired_model = create_diffusion_benchmark_with_ired(
    temperature=temperature,
    cross_section=cross_section,
    truncation=truncation,
    perturbation_amplitude=perturbation_amplitude,
    wave_number=wave_number,
    grid_points=grid_points,
    domain_size=domain_size,
)

# Get IReD coefficients
D = ired_model.diffusion_coefficient()
tau_V = ired_model.diffusion_relaxation_time()
k = wave_number
Gamma_expected = D * k**2

print("\nIReD Transport Coefficients:")
print(f"  Diffusion coefficient D = {D:.6e} GeV²")
print(f"  Diffusion relaxation time τ_V = {tau_V:.6e} GeV⁻¹")
print(f"  Expected decay rate Γ = Dk² = {Gamma_expected:.6e} GeV")

# Check timestep constraint
dt_suggested = 0.1 * tau_V / (k**2 * D + 1e-15)
print("\nTimestep constraint:")
print(f"  Suggested dt < 0.1 τ_V/(k²D) = {dt_suggested:.6e} GeV⁻¹")

# =============================================================================
# Initial State (t=0)
# =============================================================================

print_section("INITIAL STATE (t=0)")

fields = benchmark.fields
X, Y, Z = benchmark.grid.meshgrid()

# Particle density
print_field_stats("Particle density n", fields.n, show_perturbation=True)

# Background values
n_background = fields.n.mean()
T_background = fields.temperature.mean()
print("\nBackground equilibrium:")
print(f"  n₀ = {n_background:.6e} GeV³")
print(f"  T = {T_background:.6e} GeV")

# Chemical potential μ/T = ln(n/n_eq)
mu_over_T = fields.compute_chemical_potential_over_temperature(eos_type="radiation")
print_field_stats("Chemical potential μ/T", mu_over_T, show_perturbation=True)

# Verify μ/T formula
zeta_3 = 1.202056903
n_eq = (zeta_3 / np.pi**2) * T_background**3
mu_over_T_check = np.log(fields.n / n_eq)
print("\nChemical potential formula check:")
print(f"  n_eq(T) = {n_eq:.6e} GeV³")
print(f"  μ/T from formula: {mu_over_T_check.mean():.6e}")
print(f"  μ/T from fields:  {mu_over_T.mean():.6e}")
print(f"  Match: {np.allclose(mu_over_T, mu_over_T_check, atol=1e-10)}")

# Velocity field (should be rest frame)
print_field_stats("Four-velocity u⁰", fields.u_mu[..., 0])
print_field_stats("Spatial velocity u^x", fields.u_mu[..., 1])

# Diffusion current V^μ
print_field_stats("Diffusion current V⁰", fields.V_mu[..., 0])
print_field_stats("Diffusion current V^x", fields.V_mu[..., 1], show_perturbation=True)
print_field_stats("Diffusion current V^y", fields.V_mu[..., 2])
print_field_stats("Diffusion current V^z", fields.V_mu[..., 3])

# =============================================================================
# Gradient Analysis
# =============================================================================

print_section("GRADIENT ANALYSIS")

# Compute chemical potential gradient using spectral derivatives
solver = benchmark.solver

# Compute ∂_x(μ/T) using the spectral solver
grad_mu_x = solver.spectral.spatial_derivative(mu_over_T, direction=0)
grad_mu_y = solver.spectral.spatial_derivative(mu_over_T, direction=1)
grad_mu_z = solver.spectral.spatial_derivative(mu_over_T, direction=2)

print_field_stats("∂_x(μ/T)", grad_mu_x, show_perturbation=True)
print_field_stats("∂_y(μ/T)", grad_mu_y)
print_field_stats("∂_z(μ/T)", grad_mu_z)

# Analytical gradient for sinusoidal perturbation
# n(x) = n₀(1 + A sin(kx))
# μ/T = ln(n/n_eq) ≈ ln(n₀/n_eq) + A sin(kx) for small A
# ∂_x(μ/T) ≈ A k cos(kx)
A = perturbation_amplitude
grad_mu_x_analytical = A * k * np.cos(k * X)

print("\nGradient comparison (∂_x(μ/T)):")
print(f"  Analytical amplitude: {A * k:.6e}")
print(f"  Numerical amplitude:  {np.sqrt(np.mean(grad_mu_x**2)):.6e}")
check_analytical_vs_numerical("∂_x(μ/T)", grad_mu_x_analytical, grad_mu_x, tolerance=1e-6)

# =============================================================================
# Fick's Law Check
# =============================================================================

print_section("FICK'S LAW VALIDATION")

# Fick's law: V^x = -D ∂_x(μ/T)
V_x_analytical = -D * grad_mu_x_analytical

print(f"Diffusion coefficient D = {D:.6e} GeV²")
print_field_stats("V^x (numerical)", fields.V_mu[..., 1], show_perturbation=True)
print_field_stats("V^x (analytical = -D ∂_x(μ/T))", V_x_analytical, show_perturbation=True)

ficks_law_check = check_analytical_vs_numerical(
    "Fick's law: V^x vs -D∂_x(μ/T)", V_x_analytical, fields.V_mu[..., 1], tolerance=1e-6
)

# Check sign: V should be OPPOSITE to grad(μ/T)
# When ∂_x(μ/T) > 0 (μ increasing), V^x should be < 0 (flow DOWN gradient)
correlation = np.corrcoef(grad_mu_x.flatten(), fields.V_mu[..., 1].flatten())[0, 1]
print("\nSign check:")
print(f"  Correlation(∂_x(μ/T), V^x) = {correlation:.4f}")
print("  Expected: ~ -1.0 (opposite signs)")
print(f"  Status: {'✓ CORRECT' if correlation < -0.99 else '✗ WRONG SIGN'}")

# =============================================================================
# Conservation Law RHS at t=0
# =============================================================================

print_section("CONSERVATION LAW RHS AT t=0")

# Get conservation law RHS
conservation = solver.conservation
conservation_rhs = conservation.evolution_equations()

print_field_stats("dn/dt (conservation)", conservation_rhs["dn_dt"], show_perturbation=True)

# Analytical prediction: dn/dt = -∂_x(n u^x + V^x)
# For rest frame (u^x = 0): dn/dt = -∂_x(V^x)
# For V^x = -D ∂_x(μ/T) with μ/T = A sin(kx):
# dn/dt = -∂_x(-D A k cos(kx)) = D A k² sin(kx)
# But wait, this is positive! We need the NEGATIVE for decay!

# Let's compute it numerically from the analytical V
dn_dt_analytical_from_V = -solver.spectral.spatial_derivative(V_x_analytical, direction=0)

print("\ndn/dt analysis:")
print_field_stats("dn/dt (from conservation code)", conservation_rhs["dn_dt"])
print_field_stats("dn/dt (from -∂_x(V_x_analytical))", dn_dt_analytical_from_V)

check_analytical_vs_numerical(
    "dn/dt", dn_dt_analytical_from_V, conservation_rhs["dn_dt"], tolerance=1e-8
)

# Check particle current
N_i = fields.n[..., np.newaxis] * fields.u_mu[..., 1:4] + fields.V_mu[..., 1:4]
print_field_stats("Particle current N^x (n u^x + V^x)", N_i[..., 0], show_perturbation=True)

# Since u^x ≈ 0, N^x ≈ V^x
print("\nParticle current check (rest frame):")
print("  N^x ≈ V^x (since u^x ≈ 0)")
print(f"  Max |N^x - V^x| = {np.abs(N_i[..., 0] - fields.V_mu[..., 1]).max():.6e}")

# =============================================================================
# Relaxation Law RHS at t=0
# =============================================================================

print_section("RELAXATION LAW RHS AT t=0")

# Get relaxation law RHS
relaxation = solver.relaxation
relaxation_rhs = relaxation.relaxation_equations()

print_field_stats("dV^x/dt (relaxation)", relaxation_rhs["dV_mu"][..., 1], show_perturbation=True)

# Analytical: dV^x/dt = -V^x/τ_V - D ∂_x(μ/T)
linear_term = -fields.V_mu[..., 1] / tau_V
forcing_term = -D * grad_mu_x_analytical
dV_x_dt_analytical = linear_term + forcing_term

print("\nRelaxation equation decomposition:")
print_field_stats("Linear term -V^x/τ_V", linear_term)
print_field_stats("Forcing term -D∂_x(μ/T)", forcing_term)
print_field_stats("Total dV^x/dt (analytical)", dV_x_dt_analytical)

check_analytical_vs_numerical(
    "dV^x/dt", dV_x_dt_analytical, relaxation_rhs["dV_mu"][..., 1], tolerance=1e-6
)

# =============================================================================
# Time Evolution Analysis
# =============================================================================

print_section("TIME EVOLUTION (First 5 timesteps)")

# Store initial state
n_initial = fields.n.copy()
V_x_initial = fields.V_mu[..., 1].copy()

# Evolve with small timesteps
dt = min(0.01, dt_suggested * 0.5)  # Use half the suggested timestep for safety
print(f"\nUsing timestep dt = {dt:.6e} GeV⁻¹")
print(f"(Suggested: dt < {dt_suggested:.6e} GeV⁻¹)")

# Track evolution
times = [0.0]
n_rms = [np.sqrt(np.mean(n_initial**2))]
V_x_rms = [np.sqrt(np.mean(V_x_initial**2))]
n_amplitude = [np.sqrt(np.mean((n_initial - n_initial.mean()) ** 2))]

for step in range(5):
    # Evolve one step
    t_current = times[-1]
    solver.evolve(t_final=t_current + dt, dt=dt, method="rk4")

    times.append(t_current + dt)
    n_rms.append(np.sqrt(np.mean(fields.n**2)))
    V_x_rms.append(np.sqrt(np.mean(fields.V_mu[..., 1] ** 2)))
    n_amplitude.append(np.sqrt(np.mean((fields.n - fields.n.mean()) ** 2)))

    # Compute growth rates
    dn_rms = n_rms[-1] - n_rms[-2]
    dV_rms = V_x_rms[-1] - V_x_rms[-2]

    print(f"\nStep {step+1} (t = {times[-1]:.6e} GeV⁻¹):")
    print(f"  n RMS:  {n_rms[-1]:.6e}  (Δ = {dn_rms:+.6e})")
    print(f"  V^x RMS: {V_x_rms[-1]:.6e}  (Δ = {dV_rms:+.6e})")
    print(f"  n amplitude: {n_amplitude[-1]:.6e}")

    # Estimate instantaneous growth rate
    if step > 0:
        log_ratio = np.log(n_amplitude[-1] / n_amplitude[-2] + 1e-15)
        Gamma_instant = -log_ratio / dt  # Negative because we want decay rate
        print(f"  Instantaneous Γ: {Gamma_instant:+.6e} GeV (expected: {Gamma_expected:+.6e})")

# =============================================================================
# Summary and Diagnosis
# =============================================================================

print_section("DIAGNOSIS SUMMARY")

# Compute measured growth rate from amplitude evolution
times_array = np.array(times)
amplitudes_array = np.array(n_amplitude)

# Fit exponential
log_amplitudes = np.log(amplitudes_array + 1e-15)
coeffs = np.polyfit(times_array, log_amplitudes, deg=1)
Gamma_measured = -coeffs[0]  # Negative slope = decay rate

print("\nMeasured decay rate (from n amplitude):")
print(f"  Γ_measured = {Gamma_measured:+.6e} GeV")
print(f"  Γ_expected = {Gamma_expected:+.6e} GeV")
print(f"  Ratio: {Gamma_measured / Gamma_expected:+.2f}")
print(f"  Error: {abs(Gamma_measured - Gamma_expected) / Gamma_expected * 100:.1f}%")

if Gamma_measured < 0:
    print("\n  ⚠️  EXPONENTIAL GROWTH DETECTED (Γ < 0)")
    print("      This indicates a POSITIVE FEEDBACK LOOP or SIGN ERROR")
elif abs(Gamma_measured - Gamma_expected) / Gamma_expected > 0.5:
    print("\n  ⚠️  WRONG MAGNITUDE (>50% error)")
    print("      Check numerical accuracy or coupling terms")
else:
    print("\n  ✓  Decay rate matches expected value")

print("\n" + "=" * 80)
print("KEY FINDINGS".center(80))
print("=" * 80)

findings = []

# Check each component
if not ficks_law_check:
    findings.append("❌ Fick's law VIOLATED at t=0 - Check V initialization or D coefficient")

if correlation > -0.9:
    findings.append("❌ V and ∇(μ/T) have WRONG SIGN correlation - Check sign in relaxation.py")

if Gamma_measured < 0:
    findings.append("❌ EXPONENTIAL GROWTH - Positive feedback loop detected")
    findings.append("   Likely causes:")
    findings.append("   1. Wrong sign in particle current N^i = n u^i + V^i (conservation.py:239)")
    findings.append("   2. Wrong sign in dn/dt = -∂_i N^i (conservation.py:243)")
    findings.append("   3. Wrong sign in dV/dt forcing term (relaxation.py:442)")

if abs(Gamma_measured) > 10 * abs(Gamma_expected):
    findings.append(f"❌ Decay rate {abs(Gamma_measured / Gamma_expected):.0f}× too large")
    findings.append("   Possible causes:")
    findings.append("   1. Timestep constraint violated")
    findings.append("   2. Wrong coefficient (D, τ_V) being used")
    findings.append("   3. Numerical instability in IMEX scheme")

if len(findings) == 0:
    findings.append("✓ All diagnostics pass - System is working correctly")

for finding in findings:
    print(finding)

print("\n" + "=" * 80)
