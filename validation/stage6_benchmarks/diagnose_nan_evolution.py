#!/usr/bin/env python3
"""Diagnose where NaN first appears during diffusion evolution."""

import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired


def check_fields(fields, label=""):
    """Check all fields for NaN/Inf and report."""
    issues = []

    if np.any(~np.isfinite(fields.rho)):
        issues.append(f"rho has {np.sum(~np.isfinite(fields.rho))} non-finite values")
    if np.any(~np.isfinite(fields.n)):
        issues.append(f"n has {np.sum(~np.isfinite(fields.n))} non-finite values")
    if np.any(~np.isfinite(fields.pressure)):
        issues.append(f"pressure has {np.sum(~np.isfinite(fields.pressure))} non-finite values")
    if np.any(~np.isfinite(fields.temperature)):
        issues.append(
            f"temperature has {np.sum(~np.isfinite(fields.temperature))} non-finite values"
        )
    if np.any(~np.isfinite(fields.u_mu)):
        issues.append(f"u_mu has {np.sum(~np.isfinite(fields.u_mu))} non-finite values")
    if np.any(~np.isfinite(fields.Pi)):
        issues.append(f"Pi has {np.sum(~np.isfinite(fields.Pi))} non-finite values")
    if np.any(~np.isfinite(fields.pi_munu)):
        issues.append(f"pi_munu has {np.sum(~np.isfinite(fields.pi_munu))} non-finite values")
    if np.any(~np.isfinite(fields.V_mu)):
        issues.append(f"V_mu has {np.sum(~np.isfinite(fields.V_mu))} non-finite values")

    if issues:
        print(f"\n⚠ {label}: Non-finite values detected:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"✓ {label}: All fields finite")
        return True


def print_field_stats(fields, label=""):
    """Print statistics for all fields."""
    print(f"\n{label} Field Statistics:")
    print(
        f"  rho: min={np.nanmin(fields.rho):.6e}, max={np.nanmax(fields.rho):.6e}, mean={np.nanmean(fields.rho):.6e}"
    )
    print(
        f"  n: min={np.nanmin(fields.n):.6e}, max={np.nanmax(fields.n):.6e}, mean={np.nanmean(fields.n):.6e}"
    )
    print(
        f"  pressure: min={np.nanmin(fields.pressure):.6e}, max={np.nanmax(fields.pressure):.6e}, mean={np.nanmean(fields.pressure):.6e}"
    )
    print(
        f"  temperature: min={np.nanmin(fields.temperature):.4f}, max={np.nanmax(fields.temperature):.4f}, mean={np.nanmean(fields.temperature):.4f}"
    )
    print(f"  u_mu: min={np.nanmin(fields.u_mu):.6e}, max={np.nanmax(fields.u_mu):.6e}")
    print(f"  Pi: min={np.nanmin(fields.Pi):.6e}, max={np.nanmax(fields.Pi):.6e}")
    print(f"  pi_munu: min={np.nanmin(fields.pi_munu):.6e}, max={np.nanmax(fields.pi_munu):.6e}")
    print(f"  V_mu: min={np.nanmin(fields.V_mu):.6e}, max={np.nanmax(fields.V_mu):.6e}")


# Test IReD regime (should be the most stable)
print("=" * 80)
print("DIAGNOSING NaN EVOLUTION IN IRED REGIME")
print("=" * 80)

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

print("\n" + "=" * 80)
print("INITIAL STATE (t=0)")
print("=" * 80)
check_fields(fields, "Initial")
print_field_stats(fields, "Initial")

# Try evolving one small step
print("\n" + "=" * 80)
print("ATTEMPTING SINGLE TIMESTEP")
print("=" * 80)

dt = 0.1  # Very small timestep
print(f"Using dt = {dt} GeV⁻¹")

try:
    # Manually take one RK4 step to see what happens
    print("\nComputing RHS at t=0...")

    # Get RHS
    rhs = solver._compute_full_coupled_rhs(fields)

    print("RHS computed. Checking for non-finite values...")
    if np.any(~np.isfinite(rhs["drho_dt"])):
        print(f"  ⚠ drho_dt has {np.sum(~np.isfinite(rhs['drho_dt']))} non-finite values")
    else:
        print(
            f"  ✓ drho_dt finite (min={np.min(rhs['drho_dt']):.6e}, max={np.max(rhs['drho_dt']):.6e})"
        )

    if np.any(~np.isfinite(rhs["dn_dt"])):
        print(f"  ⚠ dn_dt has {np.sum(~np.isfinite(rhs['dn_dt']))} non-finite values")
    else:
        print(f"  ✓ dn_dt finite (min={np.min(rhs['dn_dt']):.6e}, max={np.max(rhs['dn_dt']):.6e})")

    if np.any(~np.isfinite(rhs["du_dt"])):
        print(f"  ⚠ du_dt has {np.sum(~np.isfinite(rhs['du_dt']))} non-finite values")
    else:
        print(f"  ✓ du_dt finite (min={np.min(rhs['du_dt']):.6e}, max={np.max(rhs['du_dt']):.6e})")

    if np.any(~np.isfinite(rhs["dPi_dt"])):
        print(f"  ⚠ dPi_dt has {np.sum(~np.isfinite(rhs['dPi_dt']))} non-finite values")
    else:
        print(
            f"  ✓ dPi_dt finite (min={np.min(rhs['dPi_dt']):.6e}, max={np.max(rhs['dPi_dt']):.6e})"
        )

    if np.any(~np.isfinite(rhs["dpi_munu_dt"])):
        print(f"  ⚠ dpi_munu_dt has {np.sum(~np.isfinite(rhs['dpi_munu_dt']))} non-finite values")
    else:
        print(
            f"  ✓ dpi_munu_dt finite (min={np.min(rhs['dpi_munu_dt']):.6e}, max={np.max(rhs['dpi_munu_dt']):.6e})"
        )

    if np.any(~np.isfinite(rhs["dV_mu_dt"])):
        print(f"  ⚠ dV_mu_dt has {np.sum(~np.isfinite(rhs['dV_mu_dt']))} non-finite values")
    else:
        print(
            f"  ✓ dV_mu_dt finite (min={np.min(rhs['dV_mu_dt']):.6e}, max={np.max(rhs['dV_mu_dt']):.6e})"
        )

    # Try full evolution
    print(f"\n{'='*80}")
    print("ATTEMPTING EVOLUTION WITH solver.evolve()")
    print(f"{'='*80}")

    n_initial = fields.n.copy()
    V_mu_initial = fields.V_mu.copy()

    solver.evolve(t_final=dt, dt=dt, method="rk4")

    print(f"\nAfter one timestep (dt={dt}):")
    check_fields(fields, "After 1 step")
    print_field_stats(fields, "After 1 step")

    # Check changes
    dn = fields.n - n_initial
    dV = fields.V_mu - V_mu_initial

    print("\nChanges in fields:")
    print(f"  Δn: min={np.nanmin(dn):.6e}, max={np.nanmax(dn):.6e}, mean={np.nanmean(dn):.6e}")
    print(f"  ΔV_mu: min={np.nanmin(dV):.6e}, max={np.nanmax(dV):.6e}")

except Exception as e:
    print(f"\n✗ Exception during evolution: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
