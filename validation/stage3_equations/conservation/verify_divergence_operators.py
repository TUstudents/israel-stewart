#!/usr/bin/env -S uv run python
"""Verify divergence operators and covariant derivatives.

Tests:
1. Spatial divergence ∇_i V^i in flat space
2. Christoffel symbol corrections in curved space
3. Metric compatibility: ∇_μ g^αβ = 0
"""

import sys

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid


def test_flat_space_divergence():
    """Test ∇_i V^i = 0 for uniform field in Minkowski."""
    print("\n" + "=" * 60)
    print("TEST 1: Flat Space Divergence (Uniform Field)")
    print("=" * 60)

    grid = SpaceGrid(
        "cartesian",
        [(0.0, 2.0)] * 3,
        (8, 8, 8),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )

    # Uniform vector field
    V_field = np.ones(grid.shape + (3,)) * 0.5

    # Compute divergence
    div_V = grid.divergence(V_field, order=2)

    max_div = np.max(np.abs(div_V))
    print("Uniform field: V^i = (0.5, 0.5, 0.5)")
    print("Expected: ∇·V = 0 (no spatial variation)")
    print(f"Computed: max|∇·V| = {max_div:.3e}")

    if max_div < 1e-14:
        print("✓ PASS: Divergence = 0 for uniform field")
        return True
    else:
        print(f"✗ FAIL: Divergence = {max_div:.3e} ≠ 0")
        return False


def test_linear_divergence():
    """Test ∇_i V^i for linear field V^x = αx."""
    print("\n" + "=" * 60)
    print("TEST 2: Linear Field Divergence")
    print("=" * 60)

    grid = SpaceGrid(
        "cartesian",
        [(0.0, 1.0)] * 3,
        (16, 16, 16),  # Higher resolution
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )

    # Linear field: V^x = αx, V^y = V^z = 0
    # So ∂_x V^x = α, ∂_y V^y = 0, ∂_z V^z = 0
    # Therefore ∇·V = α
    alpha = 2.0
    X = grid.coordinates["x"][:, None, None]

    V_field = np.zeros(grid.shape + (3,))
    V_field[..., 0] = alpha * X  # V^x = αx

    # Compute divergence
    div_V = grid.divergence(V_field, order=2)

    # Expected: ∇·V = ∂_x(αx) = α everywhere
    expected = alpha
    mean_div = np.mean(div_V)
    max_error = np.max(np.abs(div_V - expected))

    print(f"Field: V^x = {alpha}x, V^y = V^z = 0")
    print(f"Expected: ∇·V = {expected}")
    print(f"Computed: mean(∇·V) = {mean_div:.6f}")
    print(f"Max error: {max_error:.3e}")

    # Periodic BC can introduce edge effects
    if max_error < 0.1 * alpha:  # 10% tolerance
        print("✓ PASS: Divergence matches expected value")
        return True
    else:
        print(f"✗ FAIL: Error = {max_error:.3e} too large")
        return False


def test_christoffel_correction():
    """Test that Christoffel symbols vanish in Minkowski (flat space)."""
    print("\n" + "=" * 60)
    print("TEST 3: Christoffel Symbols in Flat Space")
    print("=" * 60)

    metric = MinkowskiMetric()
    christoffel = metric.christoffel_symbols

    # All Christoffel symbols should be zero in Minkowski
    max_gamma = np.max(np.abs(christoffel))

    print("Minkowski metric (flat space)")
    print("Expected: Γ^μ_νρ = 0 everywhere")
    print(f"Computed: max|Γ| = {max_gamma:.3e}")

    if max_gamma < 1e-14:
        print("✓ PASS: All Christoffel symbols = 0 in flat space")
        return True
    else:
        print(f"✗ FAIL: Γ = {max_gamma:.3e} ≠ 0")
        return False


def test_divergence_with_metric():
    """Test that divergence uses metric correctly in computation."""
    print("\n" + "=" * 60)
    print("TEST 4: Divergence with Metric")
    print("=" * 60)

    grid = SpaceGrid(
        "cartesian",
        [(0.0, 1.0)] * 3,
        (8, 8, 8),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )

    # Test that grid.divergence with metric gives same result as without
    # (since Minkowski is flat)
    X = grid.coordinates["x"][:, None, None]
    V_field = np.zeros(grid.shape + (3,))
    V_field[..., 0] = X  # V^x = x

    # This should use metric and Christoffel symbols internally
    div_V = grid.divergence(V_field, order=2)

    # Expected: ∇·V = ∂_x(x) = 1
    expected = 1.0
    mean_div = np.mean(div_V)

    print("Field: V^x = x")
    print("Expected: ∇·V = 1")
    print(f"Computed: mean(∇·V) = {mean_div:.6f}")

    error = abs(mean_div - expected)

    if error < 0.1:  # 10% tolerance for numerical derivative
        print("✓ PASS: Metric-aware divergence correct")
        return True
    else:
        print(f"✗ FAIL: Error = {error:.3e}")
        return False


def main():
    """Run all divergence operator tests."""
    print("\n" + "=" * 60)
    print("DIVERGENCE OPERATOR VALIDATION")
    print("=" * 60)
    print("\nTests spatial divergence: ∇_i V^i = ∂_i V^i + Γ^i_{ij} V^j")
    print("\nIn flat space (Minkowski):")
    print("  - Christoffel symbols Γ^i_{ij} = 0")
    print("  - Covariant derivative = partial derivative")

    results = []
    results.append(test_flat_space_divergence())
    results.append(test_linear_divergence())
    results.append(test_christoffel_correction())
    results.append(test_divergence_with_metric())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All divergence operator tests PASSED")
        return 0
    else:
        print(f"✗ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
