"""Diagnose sign errors in RHS terms at t=0."""

import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired

# Create benchmark
benchmark, ired_model = create_diffusion_benchmark_with_ired(
    temperature=0.4,
    cross_section=1000.0,
    truncation="41",
    perturbation_amplitude=0.05,
    wave_number=0.5,
    grid_points=(16, 16, 16),
    domain_size=4 * np.pi,
)

fields = benchmark.fields
solver = benchmark.solver
grid = fields.grid

print("=" * 80)
print("INITIAL STATE AT t=0")
print("=" * 80)

# Get spatial coordinates
X = grid.meshgrid()[0]  # x-coordinate
x_1d = X[:, 0, 0]  # Extract 1D x-array

# Print particle density
print("\nParticle density n:")
print(f"  mean = {fields.n.mean():.6e}")
print(f"  min = {fields.n.min():.6e}")
print(f"  max = {fields.n.max():.6e}")

# Print temperature
print("\nTemperature T:")
print(f"  mean = {fields.temperature.mean():.6e}")
print(f"  min = {fields.temperature.min():.6e}")
print(f"  max = {fields.temperature.max():.6e}")

# Print chemical potential
mu_over_T = fields.compute_chemical_potential_over_temperature()
print("\nChemical potential μ/T:")
print(f"  mean = {mu_over_T.mean():.6e}")
print(f"  min = {mu_over_T.min():.6e}")
print(f"  max = {mu_over_T.max():.6e}")

# Print diffusion current V^x
V_x = fields.V_mu[..., 1]
print("\nDiffusion current V^x:")
print(f"  mean = {V_x.mean():.6e}")
print(f"  min = {V_x.min():.6e}")
print(f"  max = {V_x.max():.6e}")

print("\n" + "=" * 80)
print("RHS TERMS AT t=0")
print("=" * 80)

# Get conservation RHS
conservation_rhs = solver.conservation.evolution_equations()

dn_dt = conservation_rhs["dn_dt"]
print("\nParticle evolution dn/dt:")
print(f"  mean = {dn_dt.mean():.6e}")
print(f"  min = {dn_dt.min():.6e}")
print(f"  max = {dn_dt.max():.6e}")

# Get relaxation RHS (returns flat array: [dPi, dpi_munu, dV_mu])
relaxation_rhs_flat = solver.relaxation.compute_relaxation_rhs(fields)

# Unpack the flat array to get dV_mu
grid_size = int(np.prod(grid.shape))
offset = grid_size + 16 * grid_size  # Skip dPi (grid_size) and dpi_munu (16*grid_size)
dV_mu_flat = relaxation_rhs_flat[offset : offset + 4 * grid_size]
dV_mu = dV_mu_flat.reshape(grid.shape + (4,))
dV_x_dt = dV_mu[..., 1]
print("\nDiffusion current evolution dV^x/dt:")
print(f"  mean = {dV_x_dt.mean():.6e}")
print(f"  min = {dV_x_dt.min():.6e}")
print(f"  max = {dV_x_dt.max():.6e}")

print("\n" + "=" * 80)
print("SIGN CORRELATION CHECKS")
print("=" * 80)

# Check sign correlation: dn/dt should anti-correlate with (n - n₀) for decay
# For exponential decay: n = n₀(1 + A sin(kx) exp(-Γt))
# => dn/dt = -Γ(n - n₀), so correlation should be NEGATIVE
n_perturbation = fields.n - fields.n.mean()
corr_n_dndt = np.corrcoef(n_perturbation.flat, dn_dt.flat)[0, 1]
print(f"\nCorrelation(n - n₀, dn/dt): {corr_n_dndt:.4f}")
print("  Expected: NEGATIVE (restoring force for decay)")

# Check: dV/dt should anti-correlate with V
# Because dV/dt = -V/τ - D∇(μ/T), both terms oppose V
corr_V_dVdt = np.corrcoef(V_x.flat, dV_x_dt.flat)[0, 1]
print(f"\nCorrelation(V^x, dV^x/dt): {corr_V_dVdt:.4f}")
print("  Expected: NEGATIVE (restoring force opposes displacement)")

# Compute individual terms in dV/dt
tau_V = ired_model.diffusion_relaxation_time()
D = ired_model.diffusion_coefficient()

# Linear damping term: -V/τ
V_damping = -fields.V_mu / tau_V
V_x_damping = V_damping[..., 1]

print("\nLinear damping term -V^x/τ_V:")
print(f"  mean = {V_x_damping.mean():.6e}")
print(f"  Correlation(V^x, -V^x/τ_V): {np.corrcoef(V_x.flat, V_x_damping.flat)[0, 1]:.4f}")
print("  Expected: -1.0 (perfect anti-correlation)")

# Forcing term: -D∇(μ/T)
# This should drive V toward -D∇(μ/T) (Fick's law equilibrium)
grad_mu_x, grad_mu_y, grad_mu_z = solver.spectral.spatial_gradient(mu_over_T)
forcing = -D * grad_mu_x

print("\nForcing term -D ∂_x(μ/T):")
print(f"  mean = {forcing.mean():.6e}")

# At equilibrium, V^x = -D ∂_x(μ/T), so forcing should equal V^x
print("\nFick's law check (forcing should equal V^x at equilibrium):")
print(f"  Correlation(V^x, -D∂_x(μ/T)): {np.corrcoef(V_x.flat, forcing.flat)[0, 1]:.4f}")
print("  Expected: +1.0 (Fick's law satisfied)")

# Check if dV/dt = damping + forcing
dV_reconstructed = V_x_damping + forcing
print("\nReconstruction check dV^x/dt = -V^x/τ + (-D∂_x(μ/T)):")
print(
    f"  Correlation(dV^x/dt, reconstructed): {np.corrcoef(dV_x_dt.flat, dV_reconstructed.flat)[0, 1]:.4f}"
)
print("  Expected: +1.0 (exact match)")

print("\n" + "=" * 80)
print("PHYSICAL EXPECTATIONS")
print("=" * 80)

print("\nPhysics check:")
print("1. At t=0, V^x satisfies Fick's law: V^x = -D∂_x(μ/T)")
print("2. Damping -V/τ opposes V, trying to restore V→0")
print("3. Forcing -D∂_x(μ/T) tries to maintain Fick's law")
print("4. Net: dV/dt = -V/τ - D∂_x(μ/T)")
print("   Since V = -D∂_x(μ/T) initially, we have:")
print("   dV/dt = -V/τ - V = -V(1 + 1/τ)")
print("   This should decay V exponentially!")

# Check actual values
ratio = V_x / forcing
print("\nRatio V^x / (-D∂_x(μ/T)):")
print(f"  mean = {np.abs(ratio[np.abs(forcing) > 1e-10]).mean():.6f}")
print("  Expected: 1.0 (Fick's law satisfied)")

# So dV/dt should be:
expected_dVdt = -V_x / tau_V - V_x
print("\nExpected dV^x/dt = -V^x/τ_V - V^x = -V^x(1/τ_V + 1):")
print(f"  mean = {expected_dVdt.mean():.6e}")
print(f"  actual mean = {dV_x_dt.mean():.6e}")

# Check sign
if np.sign(expected_dVdt.mean()) == np.sign(dV_x_dt.mean()):
    print("  ✓ Sign matches!")
else:
    print("  ✗ SIGN ERROR! Wrong sign in dV/dt")

# Check particle evolution
# dn/dt = -∂_x(N^x) = -∂_x(n u^x + V^x)
# In rest frame u^x = 0, so dn/dt = -∂_x(V^x)
div_V = solver.spectral.spatial_divergence(fields.V_mu[..., 1:4])
expected_dndt = -div_V

print("\nExpected dn/dt = -∂_i(V^i):")
print(f"  mean = {expected_dndt.mean():.6e}")
print(f"  actual mean = {dn_dt.mean():.6e}")

if np.sign(expected_dndt.mean()) == np.sign(dn_dt.mean()):
    print("  ✓ Sign matches!")
else:
    print("  ✗ SIGN ERROR! Wrong sign in dn/dt")

print("\n" + "=" * 80)
print("GROWTH/DECAY ANALYSIS")
print("=" * 80)

# If dV/dt and V have same sign → growth (WRONG!)
# If dV/dt and V have opposite sign → decay (CORRECT!)
if corr_V_dVdt > 0:
    print("\n⚠ WARNING: Positive correlation between V and dV/dt → EXPONENTIAL GROWTH!")
    print("This means V pushes itself to grow, not decay.")
elif corr_V_dVdt < 0:
    print("\n✓ Negative correlation between V and dV/dt → exponential decay (correct)")
else:
    print("\n? Zero correlation - unclear dynamics")

# Similarly for n
if corr_n_dndt > 0:
    print("\n⚠ WARNING: Positive correlation between (n-n₀) and dn/dt → EXPONENTIAL GROWTH!")
elif corr_n_dndt < 0:
    print("\n✓ Negative correlation between (n-n₀) and dn/dt → exponential decay (correct)")
