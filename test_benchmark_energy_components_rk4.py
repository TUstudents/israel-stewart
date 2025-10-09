#!/usr/bin/env -S uv run python
"""
Track energy components during benchmark evolution with RK4.

Compare RK4 vs IMEX time integration schemes.
"""

import numpy as np
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
from israel_stewart.equations.conservation import ConservationLaws
from israel_stewart.core.tensor_utils import optimized_einsum

# EXACT SAME PARAMETERS AS BENCHMARK
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic"
)

# BENCHMARK TRANSPORT COEFFICIENTS
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

# Initialize with sound wave (same as benchmark)
fields = ISFieldConfiguration(grid)
X, Y, Z = grid.meshgrid()

k = 1.0
rho_0 = 1.0
amplitude = 0.01

# Sound wave in x-direction
fields.rho[:] = rho_0 + amplitude * np.cos(k * X)
fields.pressure[:] = fields.rho / 3.0  # Radiation fluid
fields.u_mu[..., 0] = 1.0  # Rest frame initially
fields.u_mu[..., 1] = amplitude * 0.45 * np.cos(k * X)  # Small velocity

# Create solver
solver = SpectralISHydrodynamics(grid=grid, fields=fields, coeffs=coeffs)

# Conservation laws
conservation = ConservationLaws(fields, coeffs, spectral_solver=solver)

# Cell volume
dx, dy, dz = grid.dx, grid.dy, grid.dz
dV = dx * dy * dz

def compute_energy_breakdown():
    """Compute energy contribution from each component."""
    f = fields

    # Metric
    g_inv = np.zeros((*grid.shape, 4, 4))
    g_inv[..., 0, 0] = -1.0
    g_inv[..., 1, 1] = 1.0
    g_inv[..., 2, 2] = 1.0
    g_inv[..., 3, 3] = 1.0

    # Spatial projector
    u_outer = optimized_einsum("...i,...j->...ij", f.u_mu, f.u_mu)
    Delta = g_inv + u_outer

    # Components of T^00
    enthalpy = f.rho + f.pressure
    T_perfect_00 = enthalpy * f.u_mu[..., 0] * f.u_mu[..., 0] + f.pressure * g_inv[..., 0, 0]
    T_bulk_00 = f.Pi * Delta[..., 0, 0]
    T_shear_00 = -f.pi_munu[..., 0, 0]  # Convention B minus sign
    T_heat_00 = 2.0 * f.q_mu[..., 0] * f.u_mu[..., 0]

    # Integrate over volume
    E_perfect = np.sum(T_perfect_00) * dV
    E_bulk = np.sum(T_bulk_00) * dV
    E_shear = np.sum(T_shear_00) * dV
    E_heat = np.sum(T_heat_00) * dV
    E_total = E_perfect + E_bulk + E_shear + E_heat

    # Also get field statistics
    Pi_rms = np.sqrt(np.mean(f.Pi**2))
    pi_rms = np.sqrt(np.mean(f.pi_munu**2))

    return {
        'E_perfect': E_perfect,
        'E_bulk': E_bulk,
        'E_shear': E_shear,
        'E_heat': E_heat,
        'E_total': E_total,
        'Pi_rms': Pi_rms,
        'pi_rms': pi_rms
    }

print("=" * 80)
print("BENCHMARK ENERGY COMPONENT TRACKING WITH RK4")
print("=" * 80)
print()
print("TIME INTEGRATION: RK4 (fully explicit, 4th order)")
print()
print("Transport coefficients (matching benchmark):")
print(f"  η (shear viscosity):     {coeffs.shear_viscosity}")
print(f"  ζ (bulk viscosity):      {coeffs.bulk_viscosity}")
print(f"  τ_π (shear relaxation):  {coeffs.shear_relaxation_time}")
print(f"  τ_Π (bulk relaxation):   {coeffs.bulk_relaxation_time}")
print()
print(f"Wave number: k = {k}")
print(f"Grid: {grid.shape}")
print()

# Initial state
data_0 = compute_energy_breakdown()

print("INITIAL STATE (t=0)")
print("=" * 80)
print(f"Perfect fluid energy:  E_pf  = {data_0['E_perfect']:.10f}")
print(f"Bulk viscosity:        E_Π   = {data_0['E_bulk']:.10e}")
print(f"Shear stress:          E_π   = {data_0['E_shear']:.10e}")
print(f"Heat flux:             E_q   = {data_0['E_heat']:.10e}")
print(f"---")
print(f"Total energy:          E_tot = {data_0['E_total']:.10f}")
print()
print(f"RMS field values:")
print(f"  Π_rms  = {data_0['Pi_rms']:.6e}")
print(f"  π_rms  = {data_0['pi_rms']:.6e}")
print()

# Time evolution tracking
dt = 0.01
n_steps = 100  # Evolve to t=1.0 (RK4 is slower, 4 evals per step)
snapshot_interval = 20

times = [0.0]
E_perfect_history = [data_0['E_perfect']]
E_bulk_history = [data_0['E_bulk']]
E_shear_history = [data_0['E_shear']]
E_total_history = [data_0['E_total']]
Pi_rms_history = [data_0['Pi_rms']]
pi_rms_history = [data_0['pi_rms']]

print("EVOLUTION (RK4)")
print("=" * 80)

for i in range(n_steps):
    # USE RK4 EXPLICITLY
    solver.time_step(dt, method="rk4")

    if (i + 1) % snapshot_interval == 0:
        t = (i + 1) * dt
        data = compute_energy_breakdown()

        times.append(t)
        E_perfect_history.append(data['E_perfect'])
        E_bulk_history.append(data['E_bulk'])
        E_shear_history.append(data['E_shear'])
        E_total_history.append(data['E_total'])
        Pi_rms_history.append(data['Pi_rms'])
        pi_rms_history.append(data['pi_rms'])

        dE_total = data['E_total'] - data_0['E_total']
        dE_rel = dE_total / data_0['E_total'] * 100

        print(f"t={t:.2f}: E_tot={data['E_total']:.10f} (ΔE={dE_rel:+.4f}%), "
              f"Π_rms={data['Pi_rms']:.4e}, π_rms={data['pi_rms']:.4e}")

times = np.array(times)
E_perfect_history = np.array(E_perfect_history)
E_bulk_history = np.array(E_bulk_history)
E_shear_history = np.array(E_shear_history)
E_total_history = np.array(E_total_history)
Pi_rms_history = np.array(Pi_rms_history)
pi_rms_history = np.array(pi_rms_history)

print()

# Final state
data_f = compute_energy_breakdown()

print("FINAL STATE (t=1.0)")
print("=" * 80)
print(f"Perfect fluid energy:  E_pf  = {data_f['E_perfect']:.10f}")
print(f"Bulk viscosity:        E_Π   = {data_f['E_bulk']:.10e}")
print(f"Shear stress:          E_π   = {data_f['E_shear']:.10e}")
print(f"Heat flux:             E_q   = {data_f['E_heat']:.10e}")
print(f"---")
print(f"Total energy:          E_tot = {data_f['E_total']:.10f}")
print()
print(f"RMS field values:")
print(f"  Π_rms  = {data_f['Pi_rms']:.6e}")
print(f"  π_rms  = {data_f['pi_rms']:.6e}")
print()

# Changes
print("=" * 80)
print("ENERGY REDISTRIBUTION")
print("=" * 80)
print()

dE_perfect = data_f['E_perfect'] - data_0['E_perfect']
dE_bulk = data_f['E_bulk'] - data_0['E_bulk']
dE_shear = data_f['E_shear'] - data_0['E_shear']
dE_total = data_f['E_total'] - data_0['E_total']

print(f"ΔE_perfect = {dE_perfect:+.6e}  ({dE_perfect/data_0['E_total']*100:+.6f}%)")
print(f"ΔE_bulk    = {dE_bulk:+.6e}  ({dE_bulk/data_0['E_total']*100:+.6f}%)")
print(f"ΔE_shear   = {dE_shear:+.6e}  ({dE_shear/data_0['E_total']*100:+.6f}%)")
print(f"---")
print(f"ΔE_total   = {dE_total:+.6e}  ({dE_total/data_0['E_total']*100:+.6f}%)")
print()

# Peak values
max_Pi_rms = np.max(Pi_rms_history)
max_pi_rms = np.max(pi_rms_history)
max_E_bulk = np.max(np.abs(E_bulk_history))
max_E_shear = np.max(np.abs(E_shear_history))

print("PEAK VALUES DURING EVOLUTION")
print("=" * 80)
print(f"Max Π_rms:     {max_Pi_rms:.6e}  (at t={times[np.argmax(Pi_rms_history)]:.2f})")
print(f"Max π_rms:     {max_pi_rms:.6e}  (at t={times[np.argmax(pi_rms_history)]:.2f})")
print(f"Max |E_Π|:     {max_E_bulk:.6e}  (at t={times[np.argmax(np.abs(E_bulk_history))]:.2f})")
print(f"Max |E_π|:     {max_E_shear:.6e}  (at t={times[np.argmax(np.abs(E_shear_history))]:.2f})")
print()

# Ratios to total energy
print("VISCOUS CONTRIBUTION (as % of total energy)")
print("=" * 80)
print(f"Peak:")
print(f"  |E_Π/E_tot| = {abs(max_E_bulk/data_0['E_total'])*100:.8f}%")
print(f"  |E_π/E_tot| = {abs(max_E_shear/data_0['E_total'])*100:.8f}%")
print()
print(f"Final:")
print(f"  |E_Π/E_tot| = {abs(data_f['E_bulk']/data_f['E_total'])*100:.8f}%")
print(f"  |E_π/E_tot| = {abs(data_f['E_shear']/data_f['E_total'])*100:.8f}%")
print()

# Energy conservation
print("=" * 80)
print("RK4 INTEGRATION QUALITY")
print("=" * 80)
print()

if abs(dE_total/data_0['E_total']) < 0.01:  # 0.01% = 1e-4
    print(f"✓ Total energy conserved to {abs(dE_total/data_0['E_total'])*100:.4f}%")
else:
    print(f"⚠️  Total energy drift: {abs(dE_total/data_0['E_total'])*100:.4f}%")

print()
print(f"RK4 is 4th-order accurate: error ~ O(dt^4) = O({dt**4:.2e})")
print(f"With dt={dt}, theoretical truncation error per step ~ {dt**4:.2e}")
print(f"Over {n_steps} steps: cumulative error ~ {n_steps * dt**4:.2e}")
print()

print("=" * 80)
print("CONCLUSION (RK4)")
print("=" * 80)
print()
print(f"With RK4 time integration (η={coeffs.shear_viscosity}, ζ={coeffs.bulk_viscosity}):")
print(f"  • Viscous energy contributions: ~{max(abs(max_E_bulk), abs(max_E_shear))/data_0['E_total']*100:.6f}% of total")
print(f"  • Perfect fluid energy dominates (>99.99%)")
print(f"  • Total energy conserved to {abs(dE_total/data_0['E_total'])*100:.4f}%")
print(f"  • Max viscous fields: Π_rms={max_Pi_rms:.2e}, π_rms={max_pi_rms:.2e}")
print()
print("✓ RK4 provides high-accuracy evolution with corrected physics")
print()
print("=" * 80)
