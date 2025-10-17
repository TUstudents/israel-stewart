#!/usr/bin/env python3
"""Trace V evolution term-by-term to find the bug."""

import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired
from israel_stewart.core.tensor_utils import optimized_einsum

# Create benchmark
benchmark, ired = create_diffusion_benchmark_with_ired(
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

print("=" * 80)
print("TRACE V EVOLUTION TERM-BY-TERM AT t=0")
print("=" * 80)

# Get coefficients
D = ired.diffusion_coefficient()
tau_V = ired.diffusion_relaxation_time()

print("\nTransport coefficients:")
print(f"  D = {D:.6e} GeV²")
print(f"  τ_V = {tau_V:.6e} GeV⁻¹")
print(f"  1/τ_V = {1/tau_V:.6e} GeV")

# Get V at t=0
V_mu = fields.V_mu
V_x = V_mu[..., 1]

print("\nInitial V^x:")
print(f"  min = {V_x.min():.6e}")
print(f"  max = {V_x.max():.6e}")
print(f"  RMS = {np.sqrt(np.mean(V_x**2)):.6e}")

# Compute chemical potential gradient
mu_over_T = fields.compute_chemical_potential_over_temperature()
grad_mu_x, _, _ = solver.spectral.spatial_gradient(mu_over_T)

print("\nChemical potential gradient ∂_x(μ/T):")
print(f"  min = {grad_mu_x.min():.6e}")
print(f"  max = {grad_mu_x.max():.6e}")
print(f"  RMS = {np.sqrt(np.mean(grad_mu_x**2)):.6e}")

# Compute individual RHS terms
print(f"\n{'='*80}")
print("RHS TERMS FOR dV^x/dt")
print("=" * 80)

# Term 1: -V/τ_V (linear damping)
term1 = -V_mu / tau_V
term1_x = term1[..., 1]

print("\nTerm 1: -V^x/τ_V (linear damping)")
print(f"  Formula: -V^x / {tau_V:.6e}")
print(f"  min = {term1_x.min():.6e}")
print(f"  max = {term1_x.max():.6e}")
print(f"  RMS = {np.sqrt(np.mean(term1_x**2)):.6e}")
print(f"  Correlation with V^x: {np.corrcoef(V_x.flat, term1_x.flat)[0, 1]:.6f}")
print("  Expected correlation: -1.0 (anti-correlated)")

# Term 2: -D ∇(μ/T) (Fick's law driving term)
# Need to compute the full 4-vector gradient

# Compute perpendicular projector
u_mu = fields.u_mu
g_inv = solver.conservation.metric.inverse
if isinstance(g_inv, np.ndarray) and g_inv.ndim == 2:
    g_inv_broadcast = np.broadcast_to(g_inv, fields.u_mu.shape[:-1] + (4, 4))
else:
    g_inv_broadcast = g_inv

delta = g_inv_broadcast + optimized_einsum("...i,...j->...ij", u_mu, u_mu)

# Gradient as 4-vector (only spatial components non-zero in rest frame)
grad_mu_4vec = np.zeros(fields.u_mu.shape)
grad_mu_4vec[..., 1] = grad_mu_x  # x-component

# Project: ∇^μ(μ/T) = Δ^μν ∇_ν(μ/T)
nabla_mu_projected = optimized_einsum("...ij,...j->...i", delta, grad_mu_4vec)

term2 = -D * nabla_mu_projected
term2_x = term2[..., 1]

print("\nTerm 2: -D ∇^x(μ/T) (Fick's law forcing)")
print(f"  Formula: -{D:.6e} × ∂_x(μ/T)")
print(f"  min = {term2_x.min():.6e}")
print(f"  max = {term2_x.max():.6e}")
print(f"  RMS = {np.sqrt(np.mean(term2_x**2)):.6e}")
print(f"  Correlation with V^x: {np.corrcoef(V_x.flat, term2_x.flat)[0, 1]:.6f}")
print("  Expected correlation: +1.0 (Fick's law satisfied)")

# Total RHS from implementation
relaxation_rhs_flat = solver.relaxation.compute_relaxation_rhs(fields)
grid_size = int(np.prod(solver.grid.shape))
offset = grid_size + 16 * grid_size
dV_mu_flat = relaxation_rhs_flat[offset : offset + 4 * grid_size]
dV_mu = dV_mu_flat.reshape(solver.grid.shape + (4,))
dV_x_dt_impl = dV_mu[..., 1]

print("\nImplementation RHS: dV^x/dt")
print(f"  min = {dV_x_dt_impl.min():.6e}")
print(f"  max = {dV_x_dt_impl.max():.6e}")
print(f"  RMS = {np.sqrt(np.mean(dV_x_dt_impl**2)):.6e}")

# Reconstructed RHS
dV_x_dt_reconstructed = term1_x + term2_x

print("\nReconstructed RHS: term1 + term2")
print(f"  min = {dV_x_dt_reconstructed.min():.6e}")
print(f"  max = {dV_x_dt_reconstructed.max():.6e}")
print(f"  RMS = {np.sqrt(np.mean(dV_x_dt_reconstructed**2)):.6e}")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print("=" * 80)

print(
    f"\nCorrelation(implementation, reconstructed): {np.corrcoef(dV_x_dt_impl.flat, dV_x_dt_reconstructed.flat)[0, 1]:.6f}"
)
print("  Expected: 1.0 (perfect match)")

diff = dV_x_dt_impl - dV_x_dt_reconstructed
print("\nDifference (impl - reconstructed):")
print(f"  max |diff| = {np.max(np.abs(diff)):.6e}")
print(f"  RMS diff = {np.sqrt(np.mean(diff**2)):.6e}")
print(f"  Relative RMS error = {np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(dV_x_dt_impl**2)):.6e}")

if np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(dV_x_dt_impl**2)) < 0.01:
    print("  ✓ Implementation matches reconstruction (< 1% error)")
else:
    print("  ✗ Significant discrepancy!")

# Analyze magnitude of terms
print(f"\n{'='*80}")
print("MAGNITUDE ANALYSIS")
print("=" * 80)

print("\nRMS magnitudes:")
print(f"  |V^x| = {np.sqrt(np.mean(V_x**2)):.6e}")
print(f"  |-V^x/τ_V| = {np.sqrt(np.mean(term1_x**2)):.6e}")
print(f"  |-D∇(μ/T)| = {np.sqrt(np.mean(term2_x**2)):.6e}")
print(f"  |dV^x/dt| = {np.sqrt(np.mean(dV_x_dt_impl**2)):.6e}")

print("\nRatios:")
print(f"  |-V/τ| / |V| = {np.sqrt(np.mean(term1_x**2)) / np.sqrt(np.mean(V_x**2)):.6e}")
print(f"  Expected: 1/τ_V = {1/tau_V:.6e} GeV")

print("\nInstantaneous decay rate:")
inst_rate = np.sqrt(np.mean(dV_x_dt_impl**2)) / np.sqrt(np.mean(V_x**2))
print(f"  |dV/dt| / |V| = {inst_rate:.6e} GeV")
print("  Eigenmode Γ_slow = 2.733e-3 GeV")
print(f"  Fast mode 1/τ_V = {1/tau_V:.6e} GeV")
print(f"  Ratio to slow mode: {inst_rate / 2.733e-3:.1f}×")
