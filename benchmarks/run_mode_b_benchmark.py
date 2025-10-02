#!/usr/bin/env python3
"""
Mode B Benchmark: Time Evolution Validation

Tests the time evolution capabilities of SpectralISHydrodynamics using
the evolve() method. Validates actual hydrodynamic dynamics.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


@dataclass
class ModeBBenchmarkResult:
    """Results from a Mode B benchmark test (time evolution)."""

    test_name: str
    passed: bool
    error_metric: float
    tolerance: float
    analytical_value: float
    numerical_value: float
    computation_time: float = 0.0
    grid_resolution: tuple[int, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert numpy types to Python native types
        if isinstance(result.get("passed"), np.bool_):
            result["passed"] = bool(result["passed"])
        return result


class ModeBBenchmark:
    """
    Mode B benchmark suite for time evolution validation.

    Tests actual hydrodynamic dynamics using evolve() method.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize Mode B benchmark.

        Args:
            config: Configuration dictionary with benchmark parameters
        """
        self.config = config or self._default_config()
        self.results: dict[str, ModeBBenchmarkResult] = {}

        # Output directory for results
        self.output_dir = Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

        print("=" * 70)
        print("MODE B BENCHMARK: Time Evolution Validation")
        print("=" * 70)
        print(f"Configuration: {self.config}")
        print()

    def _default_config(self) -> dict:
        """Create default benchmark configuration."""
        return {
            "wave_propagation": {
                "grid_points": (5, 8, 8, 8),  # Minimal grid for testing
                "t_final": 0.5,  # Short evolution time for testing
                "amplitude": 0.01,
                "sigma": 0.5,  # Wave packet width
                "rho_0": 1.0,
            },
        }

    def run_all_tests(self) -> dict[str, ModeBBenchmarkResult]:
        """
        Run complete Mode B benchmark suite.

        Returns:
            Dictionary of test results
        """
        print("Starting Mode B benchmark suite...")
        print()

        # Test 1: Wave propagation
        print("[1/1] Wave Propagation (Time Evolution)")
        print("-" * 70)
        self.results["wave_propagation"] = self.test_wave_propagation()
        self._print_result(self.results["wave_propagation"])
        print()

        # Generate reports
        self._save_results()
        self._generate_markdown_report()

        return self.results

    def test_wave_propagation(self) -> ModeBBenchmarkResult:
        """
        Test 1: Sound wave propagation using evolve() method.

        Tests actual time evolution with Gaussian wave packet.
        Measures wave speed, amplitude preservation, and dispersion.
        """
        start_time = time.time()
        config = self.config["wave_propagation"]

        # Create grid with pre-allocated time slices
        # Note: SpectralISHydrodynamics requires full 4D spacetime grid
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, config["t_final"]),
            spatial_ranges=[(0.0, 2 * np.pi)] * 3,
            grid_points=config["grid_points"],
            boundary_conditions="periodic",
        )

        fields = ISFieldConfiguration(grid)

        # Physical parameters
        c_s = np.sqrt(1.0 / 3.0)
        rho_0 = config["rho_0"]
        amplitude = config["amplitude"]
        sigma = config["sigma"]

        # Center of wave packet
        x0, y0, z0 = np.pi, np.pi, np.pi

        # Create spatial meshgrid (only for t=0)
        x = grid.coordinates["x"]
        y = grid.coordinates["y"]
        z = grid.coordinates["z"]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Gaussian wave packet at t=0
        r_squared = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
        gaussian = np.exp(-r_squared / (2 * sigma**2))

        # Initialize density at t=0 (only first time slice)
        fields.rho[0, :, :, :] = rho_0 + amplitude * gaussian

        # Initialize pressure (conformal EOS)
        fields.pressure[0, :, :, :] = fields.rho[0, :, :, :] / 3.0

        # Initialize velocity (x-direction propagation)
        u_x_initial = (c_s / rho_0) * amplitude * gaussian
        fields.u_mu[0, :, :, :, 1] = u_x_initial
        fields.u_mu[0, :, :, :, 2] = 0.0
        fields.u_mu[0, :, :, :, 3] = 0.0
        fields.u_mu[0, :, :, :, 0] = np.sqrt(1.0 + u_x_initial**2)

        # Zero viscosity for ideal fluid
        coeffs = TransportCoefficients(shear_viscosity=0.0, bulk_viscosity=0.0)

        # Create solver
        hydro = SpectralISHydrodynamics(grid, fields, coeffs)

        # Storage for tracking wave packet
        times = []
        x_positions = []
        amplitudes = []
        widths = []

        def track_wave_packet(t: float, step: int, fields_current: ISFieldConfiguration) -> None:
            """Callback to track wave packet properties at each timestep."""
            # Get latest time slice (for evolve, this is dynamic)
            # We need to access the current field state
            rho_current = fields_current.rho[0, :, :, :]  # Latest time slice

            # Find peak position along x-axis (integrate over y,z)
            rho_x_profile = np.mean(rho_current, axis=(1, 2))  # Average over y,z
            peak_idx = np.argmax(rho_x_profile)
            x_peak = x[peak_idx]

            # Measure amplitude
            amp_current = np.max(rho_current) - rho_0

            # Measure width (FWHM approximation)
            half_max = rho_0 + amp_current / 2
            above_half = rho_x_profile > half_max
            if np.any(above_half):
                width_estimate = np.sum(above_half) * (x[1] - x[0])
            else:
                width_estimate = sigma  # Fallback

            # Record
            times.append(t)
            x_positions.append(x_peak)
            amplitudes.append(amp_current)
            widths.append(width_estimate)

        # Initial measurement
        track_wave_packet(0.0, 0, fields)

        # Evolve forward in time
        try:
            hydro.evolve(t_final=config["t_final"], output_callback=track_wave_packet)
        except Exception as e:
            # If evolve fails, return failure result
            computation_time = time.time() - start_time
            return ModeBBenchmarkResult(
                test_name="Wave Propagation (Mode B)",
                passed=False,
                error_metric=1.0,
                tolerance=0.01,
                analytical_value=c_s,
                numerical_value=0.0,
                computation_time=computation_time,
                grid_resolution=config["grid_points"],
                details={
                    "error": str(e),
                    "note": "evolve() method failed - Mode B not supported or implementation issue",
                },
            )

        # Analysis: Wave speed
        times = np.array(times)
        x_positions = np.array(x_positions)
        amplitudes = np.array(amplitudes)
        widths = np.array(widths)

        # Fit linear displacement: x(t) = x0 + v*t
        if len(times) > 2:
            # Handle periodic boundary conditions
            # Unwrap positions to handle wrapping at 2π
            x_unwrapped = np.unwrap(x_positions, period=2 * np.pi)
            fit_coeffs = np.polyfit(times, x_unwrapped, 1)
            v_measured = fit_coeffs[0]

            # Wave speed error
            wave_speed_error = abs(v_measured - c_s) / c_s

            # Amplitude preservation
            amplitude_decay = abs(amplitudes[-1] - amplitudes[0]) / amplitudes[0]

            # Dispersion (width change)
            width_change = abs(widths[-1] - widths[0]) / widths[0]

            # Success criteria
            wave_speed_ok = wave_speed_error < 0.01  # 1%
            amplitude_ok = amplitude_decay < 0.02  # 2%
            dispersion_ok = width_change < 0.05  # 5%

            passed = wave_speed_ok and amplitude_ok and dispersion_ok
            error_metric = max(wave_speed_error, amplitude_decay, width_change)

        else:
            # Not enough data points
            v_measured = 0.0
            wave_speed_error = 1.0
            amplitude_decay = 1.0
            width_change = 1.0
            passed = False
            error_metric = 1.0

        computation_time = time.time() - start_time

        return ModeBBenchmarkResult(
            test_name="Wave Propagation (Mode B)",
            passed=passed,
            error_metric=error_metric,
            tolerance=0.01,
            analytical_value=c_s,
            numerical_value=v_measured,
            computation_time=computation_time,
            grid_resolution=config["grid_points"],
            details={
                "sound_speed_analytical": float(c_s),
                "sound_speed_measured": float(v_measured),
                "wave_speed_error": float(wave_speed_error),
                "amplitude_initial": float(amplitudes[0]) if len(amplitudes) > 0 else 0.0,
                "amplitude_final": float(amplitudes[-1]) if len(amplitudes) > 0 else 0.0,
                "amplitude_decay": float(amplitude_decay),
                "width_initial": float(widths[0]) if len(widths) > 0 else 0.0,
                "width_final": float(widths[-1]) if len(widths) > 0 else 0.0,
                "width_change": float(width_change),
                "num_timesteps": len(times),
                "times_recorded": [float(t) for t in times],
                "positions_recorded": [float(x) for x in x_positions],
            },
        )

    def _print_result(self, result: ModeBBenchmarkResult) -> None:
        """Print formatted test result."""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"Status: {status}")
        print(f"Error: {result.error_metric:.6f} (tolerance: {result.tolerance:.6f})")
        print(f"Time: {result.computation_time:.2f}s")
        if "wave_speed_error" in result.details:
            print(f"Wave speed error: {result.details['wave_speed_error']:.4f}")
            print(f"Amplitude decay: {result.details['amplitude_decay']:.4f}")
            print(f"Width change: {result.details['width_change']:.4f}")

    def _save_results(self) -> None:
        """Save results to JSON file."""
        results_dict = {name: result.to_dict() for name, result in self.results.items()}

        output_file = self.output_dir / "mode_b_results.json"
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n📊 Results saved to: {output_file}")

    def _generate_markdown_report(self) -> None:
        """Generate markdown report."""
        report_lines = [
            "# Mode B Benchmark Report: Time Evolution",
            "",
            "## Summary",
            "",
            "| Test | Status | Error | Tolerance | Time |",
            "|------|--------|-------|-----------|------|",
        ]

        for _name, result in self.results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report_lines.append(
                f"| {result.test_name} | {status} | {result.error_metric:.6f} | "
                f"{result.tolerance:.6f} | {result.computation_time:.2f}s |"
            )

        report_lines.extend(
            [
                "",
                "## Detailed Results",
                "",
            ]
        )

        for _name, result in self.results.items():
            report_lines.extend(
                [
                    f"### {result.test_name}",
                    "",
                    f"- **Status**: {'✅ PASS' if result.passed else '❌ FAIL'}",
                    f"- **Error**: {result.error_metric:.6f}",
                    f"- **Tolerance**: {result.tolerance:.6f}",
                    f"- **Grid**: {result.grid_resolution}",
                    f"- **Time**: {result.computation_time:.2f}s",
                    "",
                    "**Details**:",
                    "```json",
                    json.dumps(result.details, indent=2),
                    "```",
                    "",
                ]
            )

        report_file = self.output_dir / "MODE_B_REPORT.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        print(f"📄 Report saved to: {report_file}")


def main() -> int:
    """Run Mode B benchmark suite."""
    benchmark = ModeBBenchmark()
    results = benchmark.run_all_tests()

    # Summary
    print()
    print("=" * 70)
    print("MODE B BENCHMARK COMPLETE")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r.passed)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests PASSED!")
    else:
        print(f"❌ {total - passed} test(s) FAILED")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
