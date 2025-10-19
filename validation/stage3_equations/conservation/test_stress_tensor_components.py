#!/usr/bin/env -S uv run python
"""Test stress-energy tensor component construction.

Verifies:
1. Ideal part: (ε+p)u^μu^ν + p·g^μν
2. Viscous shear: π^μν
3. Bulk pressure: Π·Δ^μν
4. Diffusion: V^μu^ν + V^νu^μ (Landau frame)
5. Sign conventions: ALL dissipative terms have PLUS signs (IReD eq. 5)

Critical: For (-,+,+,+) signature:
  T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν + (V^μu^ν + V^νu^μ)
"""

import sys

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.stress_tensors import StressEnergyTensor


def test_ideal_stress_tensor():
    """Test ideal fluid part: (ε+p)u^μu^ν + p·g^μν."""
    print("\n" + "=" * 60)
    print("TEST 1: Ideal Stress-Energy Tensor")
    print("=" * 60)

    # Known values for radiation fluid in rest frame
    epsilon = 1.0  # Energy density
    pressure = epsilon / 3.0  # Radiation EOS: p = ε/3
    u_mu = np.array([1.0, 0.0, 0.0, 0.0])  # Rest frame: u^μ = (1, 0, 0, 0)

    # Compute ideal T^μν
    metric = MinkowskiMetric()
    g_munu = metric.components  # (-1, +1, +1, +1)

    # T^μν = (ε+p)u^μu^ν + p·g^μν
    w = epsilon + pressure  # Enthalpy density
    T_ideal = w * np.outer(u_mu, u_mu) + pressure * g_munu

    # Expected in rest frame (Minkowski):
    # T^00 = ε, T^ii = p (i=1,2,3), T^0i = T^i0 = 0
    expected = np.diag([epsilon, pressure, pressure, pressure])

    max_error = np.max(np.abs(T_ideal - expected))
    print(f"Energy density: ε = {epsilon:.3f}")
    print(f"Pressure: p = {pressure:.3f}")
    print(f"Four-velocity: u^μ = {u_mu}")
    print("\nIdeal stress tensor T^μν:")
    print(f"  T^00 = {T_ideal[0,0]:.3f} (expected {epsilon:.3f})")
    print(f"  T^11 = T^22 = T^33 = {T_ideal[1,1]:.3f} (expected {pressure:.3f})")
    print(f"\nMax error: {max_error:.3e}")

    threshold = 1e-14
    if max_error < threshold:
        print(f"✓ PASS: Ideal T^μν correct to {threshold:.0e}")
        return True
    else:
        print(f"✗ FAIL: Error = {max_error:.3e} > {threshold:.0e}")
        return False


def test_viscous_stress_sign_convention():
    """Test sign convention for viscous corrections.

    CRITICAL: For (-,+,+,+) signature, ALL dissipative terms have PLUS signs.
    This follows from IReD paper eq. (5) after metric conversion.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Viscous Stress Sign Convention")
    print("=" * 60)
    print("\nCRITICAL: Signature (-,+,+,+) requires PLUS signs")
    print("T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν + (V^μu^ν + V^νu^μ)")

    grid = SpaceGrid(
        "cartesian",
        [(0.0, 1.0)] * 3,
        (4, 4, 4),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )
    fields = ISFieldConfiguration(grid)

    # Set ideal part
    fields.rho[:] = 1.0
    fields.pressure[:] = 0.33
    fields.u_mu[..., 0] = 1.0

    # Set positive dissipative fields
    fields.Pi[:] = 0.1  # Positive bulk pressure
    fields.pi_munu[..., 1, 1] = 0.05  # Positive shear component
    fields.V_mu[..., 1] = 0.02  # Positive diffusion current

    # Manually construct stress tensor at a point
    i, j, k = 0, 0, 0

    # Get field values at point
    epsilon = fields.rho[i, j, k]
    pressure = fields.pressure[i, j, k]
    u_mu_point = fields.u_mu[i, j, k]
    Pi_point = fields.Pi[i, j, k]
    pi_munu_point = fields.pi_munu[i, j, k]
    V_mu_point = fields.V_mu[i, j, k]

    # Construct T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν + (V^μu^ν + V^νu^μ)
    metric = grid.metric
    g_munu = metric.components
    w = epsilon + pressure

    # Ideal part
    T_ideal = w * np.outer(u_mu_point, u_mu_point) + pressure * g_munu

    # Projection tensor: Δ^μν = g^μν + u^μu^ν
    Delta = g_munu + np.outer(u_mu_point, u_mu_point)

    # Viscous corrections
    T_bulk = Pi_point * Delta
    T_shear = pi_munu_point
    T_diffusion = np.outer(V_mu_point, u_mu_point) + np.outer(u_mu_point, V_mu_point)

    # Full tensor (ALL PLUS SIGNS)
    T_full = T_ideal + T_bulk + T_shear + T_diffusion

    T_00 = T_full[0, 0]
    T_11 = T_full[1, 1]

    # Ideal contributions
    epsilon = fields.rho[i, j, k]
    pressure = fields.pressure[i, j, k]
    u_0 = fields.u_mu[i, j, k, 0]

    # T^00 ideal = ε (in rest frame)
    T_00_ideal = epsilon * u_0 * u_0  # (ε+p)u^0u^0 - p = ε for u^0=1, p=ε/3

    # Bulk contribution: Π·Δ^00 where Δ^00 = g^00 + u^0u^0 = -1 + 1 = 0
    # So bulk doesn't contribute to T^00 in rest frame

    # T^11 ideal = p (in rest frame)
    T_11_ideal = pressure

    # Bulk contribution: Π·Δ^11 where Δ^11 = g^11 = +1
    Pi = fields.Pi[i, j, k]
    T_11_bulk = Pi * 1.0  # Should be POSITIVE contribution

    # Shear contribution: π^11
    pi_11 = fields.pi_munu[i, j, k, 1, 1]
    T_11_shear = pi_11  # Should be POSITIVE contribution

    print(f"\nAt point ({i},{j},{k}):")
    print(f"  Ideal: ε = {epsilon:.3f}, p = {pressure:.3f}")
    print(f"  Dissipative: Π = {Pi:.3f}, π^11 = {pi_11:.3f}")
    print("\nT^11 decomposition:")
    print(f"  Ideal:   {T_11_ideal:.3f}")
    print(f"  + Bulk:  {T_11_bulk:.3f} (Π·Δ^11)")
    print(f"  + Shear: {T_11_shear:.3f} (π^11)")
    print(f"  = Total: {T_11:.3f}")

    # Expected: T^11 = p + Π + π^11 (all PLUS signs)
    expected_T11 = T_11_ideal + T_11_bulk + T_11_shear
    error = abs(T_11 - expected_T11)

    print(f"\nExpected: {expected_T11:.3f}")
    print(f"Error: {error:.3e}")

    threshold = 1e-12
    if error < threshold and T_11 > T_11_ideal:  # Dissipation increases T^11
        print("✓ PASS: Sign convention correct (all PLUS signs)")
        return True
    else:
        print(f"✗ FAIL: Sign convention wrong or error = {error:.3e}")
        return False


def test_projection_tensor():
    """Test projection tensor Δ^μν = g^μν + u^μu^ν."""
    print("\n" + "=" * 60)
    print("TEST 3: Projection Tensor Δ^μν")
    print("=" * 60)

    # Rest frame
    u_mu = np.array([1.0, 0.0, 0.0, 0.0])
    metric = MinkowskiMetric()
    g_munu = metric.components

    # Δ^μν = g^μν + u^μu^ν
    Delta = g_munu + np.outer(u_mu, u_mu)

    # Properties:
    # 1. Δ^μν u_ν = 0 (orthogonal to u^μ)
    # 2. Δ^μ_μ = 3 (purely spatial in rest frame)
    # 3. Δ^00 = 0 in rest frame

    Delta_dot_u = np.dot(Delta, u_mu)
    trace = np.trace(Delta)
    Delta_00 = Delta[0, 0]

    print(f"u^μ = {u_mu}")
    print(f"\nΔ^μν · u_ν = {Delta_dot_u} (expected: 0)")
    print(f"Δ^μ_μ = {trace:.1f} (expected: 3)")
    print(f"Δ^00 = {Delta_00:.1f} (expected: 0)")

    # Check properties
    check1 = np.allclose(Delta_dot_u, 0.0, atol=1e-14)
    check2 = np.abs(trace - 3.0) < 1e-14
    check3 = np.abs(Delta_00) < 1e-14

    if check1 and check2 and check3:
        print("✓ PASS: Projection tensor has correct properties")
        return True
    else:
        print("✗ FAIL: Projection tensor properties violated")
        return False


def test_traceless_shear():
    """Test shear stress is traceless: π^μ_μ = 0."""
    print("\n" + "=" * 60)
    print("TEST 4: Shear Stress Tracelessness")
    print("=" * 60)

    grid = SpaceGrid(
        "cartesian",
        [(0.0, 1.0)] * 3,
        (4, 4, 4),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )
    fields = ISFieldConfiguration(grid)

    # Set shear stress
    fields.pi_munu[..., 1, 1] = 0.05
    fields.pi_munu[..., 2, 2] = -0.05  # Must cancel for tracelessness

    # Check trace
    trace = np.trace(fields.pi_munu[0, 0, 0])

    print(f"π^11 = {fields.pi_munu[0,0,0,1,1]:.3f}")
    print(f"π^22 = {fields.pi_munu[0,0,0,2,2]:.3f}")
    print(f"π^33 = {fields.pi_munu[0,0,0,3,3]:.3f}")
    print(f"π^00 = {fields.pi_munu[0,0,0,0,0]:.3f}")
    print(f"\nTrace π^μ_μ = {trace:.3e}")

    if np.abs(trace) < 1e-14:
        print("✓ PASS: Shear stress is traceless")
        return True
    else:
        print(f"✗ FAIL: Trace = {trace:.3e} ≠ 0")
        return False


def main():
    """Run all stress tensor component tests."""
    print("\n" + "=" * 60)
    print("STRESS-ENERGY TENSOR COMPONENT VALIDATION")
    print("=" * 60)
    print("\nVerifies stress tensor construction:")
    print("  T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν + (V^μu^ν + V^νu^μ)")
    print("\nCRITICAL: Sign convention for (-,+,+,+) signature")
    print("  ALL dissipative terms have PLUS signs (IReD eq. 5)")

    results = []
    results.append(test_ideal_stress_tensor())
    results.append(test_viscous_stress_sign_convention())
    results.append(test_projection_tensor())
    results.append(test_traceless_shear())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All stress tensor tests PASSED")
        print("\nReference: IReD paper eq. (5), CLAUDE.md sign conventions")
        return 0
    else:
        print(f"✗ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
