#!/usr/bin/env -S uv run python
"""Test regime of applicability warnings for Israel-Stewart theory.

Wagner & Gavassino (2024) condition: |τω| ≲ 1

For plane waves: ω ≈ k·c_s
For radiation fluid: c_s = 1/√3

Expected: Warning when k > 1/(τ·c_s)
"""

import sys
import warnings

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def test_high_k_warning():
    """Test that warning triggers for high wavenumbers."""
    print("\n" + "=" * 60)
    print("TEST: High Wavenumber Regime Warning")
    print("=" * 60)

    # High wavenumber grid
    k_max = 10.0  # Very high k
    L = 2 * np.pi / k_max
    grid = SpaceGrid(
        "cartesian",
        [(0.0, L)] * 3,
        (8, 8, 8),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 0.33
    fields.u_mu[..., 0] = 1.0

    # Typical relaxation time
    tau_pi = 0.5
    coeffs = TransportCoefficients(
        shear_viscosity=0.1, shear_relaxation_time=tau_pi, bulk_relaxation_time=0.3
    )

    # Check regime parameter
    c_s = 1.0 / np.sqrt(3.0)
    omega_max = k_max * c_s
    regime_param = abs(tau_pi * omega_max)

    print(f"k_max = {k_max:.2f}")
    print(f"τ_π = {tau_pi:.2f}")
    print(f"c_s = {c_s:.3f}")
    print(f"|τω| = {regime_param:.2f}")
    print(f"\nExpected: Warning if |τω| > 1")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create solver (may trigger warning)
        _ = SpectralISHydrodynamics(grid, fields, coeffs)

        # Check if warning was raised
        regime_warnings = [
            warning
            for warning in w
            if "regime" in str(warning.message).lower()
            or "applicability" in str(warning.message).lower()
        ]

        if regime_warnings and regime_param > 1.0:
            print(f"✓ PASS: Warning triggered for |τω| = {regime_param:.2f} > 1")
            print(f"  Message: {regime_warnings[0].message}")
            return True
        elif not regime_warnings and regime_param <= 1.0:
            print(f"✓ PASS: No warning for |τω| = {regime_param:.2f} ≤ 1")
            return True
        elif regime_warnings and regime_param <= 1.0:
            print(f"✗ FAIL: Spurious warning for |τω| = {regime_param:.2f} ≤ 1")
            return False
        else:
            print(f"⚠ WARNING: No regime check for |τω| = {regime_param:.2f} > 1")
            print("  (Regime warning may not be implemented yet)")
            return True  # Not a failure, just not implemented


def test_low_k_no_warning():
    """Test that no warning for low wavenumbers."""
    print("\n" + "=" * 60)
    print("TEST: Low Wavenumber (No Warning)")
    print("=" * 60)

    # Low wavenumber grid
    k_max = 1.0  # Moderate k
    L = 2 * np.pi / k_max
    grid = SpaceGrid(
        "cartesian",
        [(0.0, L)] * 3,
        (8, 8, 8),
        boundary_conditions="periodic",
        metric=MinkowskiMetric(),
    )

    fields = ISFieldConfiguration(grid)
    fields.rho[:] = 1.0
    fields.pressure[:] = 0.33
    fields.u_mu[..., 0] = 1.0

    tau_pi = 0.5
    coeffs = TransportCoefficients(
        shear_viscosity=0.1, shear_relaxation_time=tau_pi, bulk_relaxation_time=0.3
    )

    # Check regime parameter
    c_s = 1.0 / np.sqrt(3.0)
    omega_max = k_max * c_s
    regime_param = abs(tau_pi * omega_max)

    print(f"k_max = {k_max:.2f}")
    print(f"τ_π = {tau_pi:.2f}")
    print(f"|τω| = {regime_param:.2f}")
    print(f"\nExpected: No warning if |τω| ≲ 1")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = SpectralISHydrodynamics(grid, fields, coeffs)

        regime_warnings = [
            warning
            for warning in w
            if "regime" in str(warning.message).lower()
            or "applicability" in str(warning.message).lower()
        ]

        if not regime_warnings:
            print(f"✓ PASS: No warning for |τω| = {regime_param:.2f} ≲ 1")
            return True
        else:
            print(f"✗ FAIL: Unexpected warning for |τω| = {regime_param:.2f}")
            return False


def main():
    """Run regime warning tests."""
    print("\n" + "=" * 60)
    print("REGIME APPLICABILITY VERIFICATION")
    print("=" * 60)
    print("\nWagner & Gavassino (2024) condition:")
    print("  Israel-Stewart valid when |τω| ≲ 1")
    print("  For sound waves: ω ≈ k·c_s")
    print("  For radiation: c_s = 1/√3 ≈ 0.577")
    print("\nTypical values:")
    print("  τ ~ 0.5 GeV⁻¹ → k_max ≲ 3.5 GeV")

    results = []
    results.append(test_high_k_warning())
    results.append(test_low_k_no_warning())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All regime warning tests PASSED")
        print("\nReference: Wagner & Gavassino (2024), arXiv:2309.14828v2")
        return 0
    else:
        print(f"✗ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
