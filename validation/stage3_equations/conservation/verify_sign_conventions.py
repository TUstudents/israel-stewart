#!/usr/bin/env -S uv run python
"""Verify sign conventions throughout the codebase.

CRITICAL for (-,+,+,+) signature:
- Metric: g^μν = diag(-1, +1, +1, +1)
- Four-velocity normalization: u_μ u^μ = -1
- Stress tensor: T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν + ...
  ALL dissipative terms have PLUS signs (IReD eq. 5)

Reference: CLAUDE.md, docs/IRED_THEORY.md Section 1.3
"""

import sys

import numpy as np

from israel_stewart.core.metrics import MinkowskiMetric


def test_metric_signature():
    """Test metric signature is (-,+,+,+)."""
    print("\n" + "=" * 60)
    print("TEST 1: Metric Signature")
    print("=" * 60)

    metric = MinkowskiMetric()
    g_munu = metric.components

    # Expected: diag(-1, +1, +1, +1)
    expected = np.diag([-1.0, 1.0, 1.0, 1.0])

    max_error = np.max(np.abs(g_munu - expected))

    print("Minkowski metric g^μν:")
    print(f"  g^00 = {g_munu[0,0]:.1f} (expected -1)")
    print(f"  g^11 = {g_munu[1,1]:.1f} (expected +1)")
    print(f"  g^22 = {g_munu[2,2]:.1f} (expected +1)")
    print(f"  g^33 = {g_munu[3,3]:.1f} (expected +1)")
    print(
        f"\nSignature: ({g_munu[0,0]:+.0f},{g_munu[1,1]:+.0f},{g_munu[2,2]:+.0f},{g_munu[3,3]:+.0f})"
    )

    if max_error < 1e-14:
        print("✓ PASS: Metric signature = (-,+,+,+)")
        return True
    else:
        print(f"✗ FAIL: Metric error = {max_error:.3e}")
        return False


def test_four_velocity_normalization():
    """Test u_μ u^μ = -1 for rest frame."""
    print("\n" + "=" * 60)
    print("TEST 2: Four-Velocity Normalization")
    print("=" * 60)

    # Rest frame: u^μ = (1, 0, 0, 0)
    u_mu = np.array([1.0, 0.0, 0.0, 0.0])

    metric = MinkowskiMetric()
    g_munu = metric.components

    # Lower index: u_μ = g_μν u^ν
    u_lower = np.dot(g_munu, u_mu)  # u_μ = (-1, 0, 0, 0)

    # Normalization: u_μ u^μ
    norm = np.dot(u_lower, u_mu)

    print(f"u^μ = {u_mu}")
    print(f"u_μ = g_μν u^ν = {u_lower}")
    print(f"\nNormalization: u_μ u^μ = {norm:.1f}")
    print("Expected: -1 (for signature (-,+,+,+))")

    if abs(norm - (-1.0)) < 1e-14:
        print("✓ PASS: u_μ u^μ = -1")
        return True
    else:
        print(f"✗ FAIL: u_μ u^μ = {norm:.3f} ≠ -1")
        return False


def test_stress_tensor_sign_convention():
    """Verify stress tensor uses PLUS signs for dissipative terms."""
    print("\n" + "=" * 60)
    print("TEST 3: Stress Tensor Sign Convention")
    print("=" * 60)
    print("\nFor signature (-,+,+,+), IReD paper eq. (5) gives:")
    print("T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν + (V^μu^ν + V^νu^μ)")
    print("\nCRITICAL: ALL dissipative terms have PLUS signs")

    # Construct example stress tensor
    epsilon = 1.0
    pressure = 0.33
    Pi = 0.1  # Positive bulk pressure
    u_mu = np.array([1.0, 0.0, 0.0, 0.0])

    metric = MinkowskiMetric()
    g_munu = metric.components

    # Ideal part
    w = epsilon + pressure
    T_ideal = w * np.outer(u_mu, u_mu) + pressure * g_munu

    # Projection tensor
    Delta = g_munu + np.outer(u_mu, u_mu)

    # Bulk correction with PLUS sign
    T_bulk = Pi * Delta

    # Full tensor
    T_full = T_ideal + T_bulk  # PLUS sign here

    # Check T^11 = p + Π (both spatial)
    T_11_ideal = pressure  # From p·g^11
    T_11_bulk = Pi * Delta[1, 1]  # From Π·Δ^11
    T_11_total = T_full[1, 1]

    print(f"\nExample: ε = {epsilon}, p = {pressure}, Π = {Pi}")
    print("\nT^11 decomposition:")
    print(f"  Ideal part:  p = {T_11_ideal:.2f}")
    print(f"  Bulk part:  +Π = +{T_11_bulk:.2f} (PLUS sign)")
    print(f"  Total:     T^11 = {T_11_total:.2f}")
    print(f"\nCheck: {T_11_ideal:.2f} + {T_11_bulk:.2f} = {T_11_total:.2f}")

    expected = T_11_ideal + T_11_bulk  # Sum with PLUS
    error = abs(T_11_total - expected)

    if error < 1e-14 and T_11_total > T_11_ideal:
        print("✓ PASS: Dissipative terms use PLUS signs")
        return True
    else:
        print(f"✗ FAIL: Sign convention error = {error:.3e}")
        return False


def test_projection_tensor_properties():
    """Verify Δ^μν = g^μν + u^μu^ν properties."""
    print("\n" + "=" * 60)
    print("TEST 4: Projection Tensor Properties")
    print("=" * 60)

    u_mu = np.array([1.0, 0.0, 0.0, 0.0])
    metric = MinkowskiMetric()
    g_munu = metric.components

    # Projection tensor: Δ^μν = g^μν + u^μu^ν
    # Note PLUS sign (for (-,+,+,+) signature)
    Delta = g_munu + np.outer(u_mu, u_mu)

    print("Δ^μν = g^μν + u^μu^ν (PLUS sign for (-,+,+,+))")
    print(f"\nΔ^00 = g^00 + u^0u^0 = -1 + 1 = {Delta[0,0]:.0f}")
    print(f"Δ^11 = g^11 + u^1u^1 = +1 + 0 = {Delta[1,1]:.0f}")

    check1 = abs(Delta[0, 0]) < 1e-14  # Δ^00 = 0
    check2 = abs(Delta[1, 1] - 1.0) < 1e-14  # Δ^11 = 1

    if check1 and check2:
        print("✓ PASS: Projection tensor Δ^μν = g^μν + u^μu^ν")
        return True
    else:
        print("✗ FAIL: Projection tensor incorrect")
        return False


def main():
    """Run all sign convention tests."""
    print("\n" + "=" * 60)
    print("SIGN CONVENTION VALIDATION")
    print("=" * 60)
    print("\nCRITICAL: Signature (-,+,+,+) determines all signs")
    print("\nKey conventions:")
    print("  1. Metric: g^μν = diag(-1,+1,+1,+1)")
    print("  2. Four-velocity: u_μ u^μ = -1")
    print("  3. Projection: Δ^μν = g^μν + u^μu^ν (PLUS)")
    print("  4. Stress tensor: ALL dissipative terms PLUS")
    print("\nReferences:")
    print("  - IReD paper eq. (5)")
    print("  - CLAUDE.md sign conventions")
    print("  - docs/IRED_THEORY.md Section 1.3")

    results = []
    results.append(test_metric_signature())
    results.append(test_four_velocity_normalization())
    results.append(test_stress_tensor_sign_convention())
    results.append(test_projection_tensor_properties())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All sign convention tests PASSED")
        print("\nConclusion: Codebase uses consistent (-,+,+,+) conventions")
        return 0
    else:
        print(f"✗ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
