#!/usr/bin/env python3
"""Debug script to trace particle density evolution."""

import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired
from israel_stewart.equations.ired_simple import HardSphereIReD

# Create a simple diffusion benchmark
benchmark, ired_model = create_diffusion_benchmark_with_ired(
    temperature=0.4,
    cross_section=1000.0,
    truncation="41",
    perturbation_amplitude=0.05,
    wave_number=0.5,
    grid_points=(16, 16, 16),  # Small grid for debugging
    domain_size=4 * np.pi,
)

# Check initial state
print("=" * 80)
print("INITIAL STATE (t=0)")
print("=" * 80)
fields = benchmark.fields

print("\nParticle density n:")
print(f"  min = {fields.n.min():.6e}")
print(f"  max = {fields.n.max():.6e}")
print(f"  mean = {fields.n.mean():.6e}")
print(f"  std = {fields.n.std():.6e}")

print("\nDiffusion current V^μ:")
for i, label in enumerate(["t", "x", "y", "z"]):
    print(
        f"  V^{label}: min={fields.V_mu[..., i].min():.6e}, max={fields.V_mu[..., i].max():.6e}, mean={fields.V_mu[..., i].mean():.6e}"
    )

print("\nEnergy density ρ:")
print(f"  min = {fields.rho.min():.6e}")
print(f"  max = {fields.rho.max():.6e}")

# Check conservation equations at t=0
print("\n" + "=" * 80)
print("CONSERVATION EQUATIONS AT t=0")
print("=" * 80)

conservation = benchmark.solver.conservation
evolution_rhs = conservation.evolution_equations()

print("\ndn/dt:")
print(f"  min = {evolution_rhs['dn_dt'].min():.6e}")
print(f"  max = {evolution_rhs['dn_dt'].max():.6e}")
print(f"  mean = {evolution_rhs['dn_dt'].mean():.6e}")
print(f"  std = {evolution_rhs['dn_dt'].std():.6e}")

print("\ndρ/dt:")
print(f"  min = {evolution_rhs['drho_dt'].min():.6e}")
print(f"  max = {evolution_rhs['drho_dt'].max():.6e}")

# Check relaxation equations
print("\n" + "=" * 80)
print("RELAXATION EQUATIONS AT t=0")
print("=" * 80)

relaxation = benchmark.solver.relaxation
relaxation_rhs = relaxation.relaxation_equations()

print("\ndV^μ/dt:")
for i, label in enumerate(["t", "x", "y", "z"]):
    dV = relaxation_rhs["dV_mu"][..., i]
    print(f"  dV^{label}/dt: min={dV.min():.6e}, max={dV.max():.6e}, mean={dV.mean():.6e}")

# Evolve a single timestep and check changes
print("\n" + "=" * 80)
print("AFTER ONE TIMESTEP (dt=0.01)")
print("=" * 80)

n_initial = fields.n.copy()
V_initial = fields.V_mu.copy()

benchmark.solver.evolve(t_final=0.01, dt=0.01, method="rk4")

print("\nChange in particle density:")
dn = fields.n - n_initial
print(f"  min Δn = {dn.min():.6e}")
print(f"  max Δn = {dn.max():.6e}")
print(f"  mean Δn = {dn.mean():.6e}")

print("\nChange in diffusion current V^x:")
dV_x = fields.V_mu[..., 1] - V_initial[..., 1]
print(f"  min ΔV^x = {dV_x.min():.6e}")
print(f"  max ΔV^x = {dV_x.max():.6e}")
print(f"  mean ΔV^x = {dV_x.mean():.6e}")

# Check if V^x amplitude is growing or decaying
V_x_rms_initial = np.sqrt(np.mean(V_initial[..., 1] ** 2))
V_x_rms_final = np.sqrt(np.mean(fields.V_mu[..., 1] ** 2))

print("\nV^x RMS amplitude:")
print(f"  Initial: {V_x_rms_initial:.6e}")
print(f"  After dt: {V_x_rms_final:.6e}")
print(f"  Change: {V_x_rms_final - V_x_rms_initial:.6e}")
print(f"  Growth rate: {(V_x_rms_final - V_x_rms_initial) / (V_x_rms_initial * 0.01):.6e}")

# Expected decay
D = ired_model.diffusion_coefficient()
k = 0.5
Gamma_expected = D * k**2
print(f"\nExpected decay rate Γ = Dk² = {Gamma_expected:.6e}")
print(f"Expected RMS change in 1 timestep = -{Gamma_expected * 0.01 * V_x_rms_initial:.6e}")
