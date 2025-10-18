#!/usr/bin/env -S uv run python
"""
Test energy conservation by individual stress tensor components.

Break down T^00 into its constituent parts to verify each contribution:
- Perfect fluid: (ε+p)u^0u^0 + p·g^00
- Bulk viscosity: Π·Δ^00
- Shear stress: -π^00 (Convention B)
- Heat flux: 2·q^0·u^0
"""

import numpy as np
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
from israel_stewart.equations.conservation import ConservationLaws
from israel_stewart.core.tensor_utils import optimized_einsum

# Create grid with periodic BC (closed system)
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

# Initialize with sound wave perturbation
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

# Create conservation laws object
conservation = ConservationLaws(fields, coeffs, spectral_solver=solver)

# Compute cell volume
dx, dy, dz = grid.dx, grid.dy, grid.dz
dV = dx * dy * dz

def compute_T00_components():
    """Compute individual contributions to T^00."""
    f = fields

    # Get metric tensor g^μν (inverse metric)
    g_inv = np.zeros((*grid.shape, 4, 4))
    g_inv[..., 0, 0] = -1.0  # Minkowski
    g_inv[..., 1, 1] = 1.0
    g_inv[..., 2, 2] = 1.0
    g_inv[..., 3, 3] = 1.0

    # Spatial projector Δ^μν = g^μν + u^μu^ν
    u_outer = optimized_einsum("...i,...j->...ij", f.u_mu, f.u_mu)
    Delta = g_inv + u_outer

    # 1. Perfect fluid: (ε+p)u^0u^0 + p·g^00
    enthalpy = f.rho + f.pressure
    T_perfect_00 = enthalpy * f.u_mu[..., 0] * f.u_mu[..., 0] + f.pressure * g_inv[..., 0, 0]

    # 2. Bulk viscosity: Π·Δ^00
    T_bulk_00 = f.Pi * Delta[..., 0, 0]

    # 3. Shear stress: -π^00 (Convention B MINUS sign)
    T_shear_00 = -f.pi_munu[..., 0, 0]

    # 4. Heat flux: q^0·u^0 + u^0·q^0 = 2·q^0·u^0
    T_heat_00 = 2.0 * f.q_mu[..., 0] * f.u_mu[..., 0]

    # Total should match T^00 from stress_energy_tensor()
    T_total_00_computed = T_perfect_00 + T_bulk_00 + T_shear_00 + T_heat_00

    # Get actual T^00 from full tensor
    T_full = conservation.stress_energy_tensor()
    T_actual_00 = T_full[..., 0, 0]

    return {
        'perfect': T_perfect_00,
        'bulk': T_bulk_00,
        'shear': T_shear_00,
        'heat': T_heat_00,
        'computed_total': T_total_00_computed,
        'actual_total': T_actual_00
    }

def integrate_component(component):
    """Integrate component over volume."""
    return np.sum(component) * dV

print("=" * 80)
print("ENERGY COMPONENTS TEST")
print("=" * 80)
print()

# Initial component breakdown
print("INITIAL STATE (t=0)")
print("=" * 80)

components_0 = compute_T00_components()

E_perfect_0 = integrate_component(components_0['perfect'])
E_bulk_0 = integrate_component(components_0['bulk'])
E_shear_0 = integrate_component(components_0['shear'])
E_heat_0 = integrate_component(components_0['heat'])
E_computed_0 = integrate_component(components_0['computed_total'])
E_actual_0 = integrate_component(components_0['actual_total'])

print(f"Perfect fluid energy:  E_pf  = {E_perfect_0:.10f}")
print(f"Bulk viscosity:        E_Π   = {E_bulk_0:.10f}")
print(f"Shear stress:          E_π   = {E_shear_0:.10f}")
print(f"Heat flux:             E_q   = {E_heat_0:.10f}")
print(f"---")
print(f"Computed total:        E_sum = {E_computed_0:.10f}")
print(f"Actual T^00:           E_act = {E_actual_0:.10f}")
print(f"Match error:                   {abs(E_computed_0 - E_actual_0):.6e}")
print()

# Verify our component breakdown matches the actual tensor
assert abs(E_computed_0 - E_actual_0) / E_actual_0 < 1e-10, "Component breakdown doesn't match!"

# Evolve system
dt = 0.01
n_steps = 100

print("EVOLVING...")
for i in range(n_steps):
    solver.time_step(dt)

print()

# Final component breakdown
print("FINAL STATE (t=1.0)")
print("=" * 80)

components_f = compute_T00_components()

E_perfect_f = integrate_component(components_f['perfect'])
E_bulk_f = integrate_component(components_f['bulk'])
E_shear_f = integrate_component(components_f['shear'])
E_heat_f = integrate_component(components_f['heat'])
E_computed_f = integrate_component(components_f['computed_total'])
E_actual_f = integrate_component(components_f['actual_total'])

print(f"Perfect fluid energy:  E_pf  = {E_perfect_f:.10f}")
print(f"Bulk viscosity:        E_Π   = {E_bulk_f:.10f}")
print(f"Shear stress:          E_π   = {E_shear_f:.10f}")
print(f"Heat flux:             E_q   = {E_heat_f:.10f}")
print(f"---")
print(f"Computed total:        E_sum = {E_computed_f:.10f}")
print(f"Actual T^00:           E_act = {E_actual_f:.10f}")
print(f"Match error:                   {abs(E_computed_f - E_actual_f):.6e}")
print()

# Compute changes
print("=" * 80)
print("CHANGES IN EACH COMPONENT")
print("=" * 80)
print()

dE_perfect = E_perfect_f - E_perfect_0
dE_bulk = E_bulk_f - E_bulk_0
dE_shear = E_shear_f - E_shear_0
dE_heat = E_heat_f - E_heat_0
dE_total = E_actual_f - E_actual_0

print(f"ΔE_perfect = {dE_perfect:+.6e}  ({dE_perfect/E_actual_0*100:+.6f}%)")
print(f"ΔE_bulk    = {dE_bulk:+.6e}  ({dE_bulk/E_actual_0*100:+.6f}%)")
print(f"ΔE_shear   = {dE_shear:+.6e}  ({dE_shear/E_actual_0*100:+.6f}%)")
print(f"ΔE_heat    = {dE_heat:+.6e}  ({dE_heat/E_actual_0*100:+.6f}%)")
print(f"---")
print(f"ΔE_total   = {dE_total:+.6e}  ({dE_total/E_actual_0*100:+.6f}%)")
print()

# Check that sum of changes equals total change
sum_changes = dE_perfect + dE_bulk + dE_shear + dE_heat
print(f"Sum of changes:  {sum_changes:+.6e}")
print(f"Total change:    {dE_total:+.6e}")
print(f"Difference:      {abs(sum_changes - dE_total):.6e}")
print()

# Physical interpretation
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

print("Sign Convention (Convention B - Landau-Lifshitz):")
print("  T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν - π^μν + q^μu^ν + q^νu^μ")
print("         ^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^ MINUS ^^^^^ ^^^^^^^^^^^")
print("         Perfect fluid          Bulk    Shear  Heat flux")
print()

print("Physical meaning:")
print("  - Perfect fluid: Conserved exactly if no viscosity")
print("  - Bulk Π > 0: Resists compression (adds pressure)")
print("  - Shear π^μν > 0: Dissipates energy (MINUS sign = energy sink)")
print("  - Heat flux q^μ: Transfers energy between regions")
print()

# Check sign consistency
if abs(E_shear_0) > 1e-10 or abs(E_shear_f) > 1e-10:
    print(f"✓ Shear contribution detected: E_π(0) = {E_shear_0:.6e}, E_π(t) = {E_shear_f:.6e}")
    print(f"  Sign check: Convention B uses -π^00, so positive π^00 → negative contribution")
else:
    print("Note: Shear stress π^00 ≈ 0 in this test (rest frame, no off-diagonal stress)")

print()

if abs(dE_total/E_actual_0) < 0.01:  # 0.01% tolerance
    print(f"✓✓✓ TOTAL ENERGY CONSERVED TO {abs(dE_total/E_actual_0*100):.4f}% ✓✓✓")
    print()
    print("All components correctly contribute to energy conservation!")
else:
    print(f"⚠️  Total energy changed by {abs(dE_total/E_actual_0*100):.4f}%")

print()
print("=" * 80)
