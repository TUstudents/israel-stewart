#!/usr/bin/env python3
"""
Equilibration Dynamics Validation Test for Spectral Israel-Stewart Solver

This test implements Phase 1.3 of the spectral validation plan, providing
comprehensive validation of equilibration dynamics in relativistic viscous fluids.

Tests:
1. Exponential relaxation to equilibrium
2. Correct relaxation timescales (τ_π, τ_Π)
3. Entropy production and second law validation
4. Approach to Navier-Stokes limit
5. Thermodynamic consistency throughout evolution
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.equilibration import (
    EntropyProductionAnalysis,
    EquilibrationAnalysis,
    EquilibrationBenchmark,
    RelaxationTimeAnalysis,
    create_equilibration_benchmark,
    run_relaxation_analysis,
    validate_entropy_production,
)
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


class TestEquilibrationValidation:
    """Comprehensive equilibration dynamics validation for spectral solver."""

    @pytest.fixture
    def minimal_setup(self):
        """Create minimal setup for equilibration tests."""
        return self._create_minimal_setup()

    @pytest.fixture
    def comprehensive_setup(self):
        """Create comprehensive setup for detailed validation."""
        return self._create_comprehensive_setup()

    def _create_minimal_setup(self):
        """Create minimal setup for equilibration tests."""
        # Small grid for fast testing
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 5.0),
            spatial_ranges=[(0.0, 2 * np.pi)] * 3,
            grid_points=(25, 16, 16, 16),
            boundary_conditions="periodic",
        )

        # Transport coefficients for Israel-Stewart
        transport_coeffs = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
            # Second-order coefficients
            lambda_pi_pi=0.1,
            xi_1=0.2,
        )

        metric = MinkowskiMetric()
        fields = ISFieldConfiguration(grid)

        return grid, transport_coeffs, metric, fields

    def _create_comprehensive_setup(self):
        """Create comprehensive setup for detailed validation."""
        # Larger grid for accuracy testing
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 8.0),
            spatial_ranges=[(0.0, 2 * np.pi)] * 3,
            grid_points=(40, 32, 32, 32),
            boundary_conditions="periodic",
        )

        # Realistic transport coefficients
        transport_coeffs = TransportCoefficients(
            shear_viscosity=0.08,
            bulk_viscosity=0.04,
            shear_relaxation_time=0.6,
            bulk_relaxation_time=0.4,
            # Full second-order coefficients
            lambda_pi_pi=0.15,
            lambda_pi_Pi=0.1,
            xi_1=0.25,
            xi_2=0.1,
        )

        metric = MinkowskiMetric()
        fields = ISFieldConfiguration(grid)

        return grid, transport_coeffs, metric, fields

    def test_exponential_relaxation(self, comprehensive_setup):
        """Test exponential relaxation to equilibrium."""
        grid, transport_coeffs, metric, fields = comprehensive_setup

        # Create equilibration analysis
        analysis = EquilibrationAnalysis(grid, metric, transport_coeffs)

        # Create initial non-equilibrium state
        initial_fields = self._create_perturbed_state(grid, perturbation=0.15)

        # Run relaxation analysis
        properties = analysis.analyze_relaxation_to_equilibrium(
            initial_fields, final_time=6.0, timestep=0.02, method="implicit"
        )

        # Test exponential decay of dissipative fluxes
        bulk_evolution = properties.bulk_pressure_evolution
        shear_evolution = properties.shear_stress_evolution

        # Check that both fluxes decay significantly
        initial_bulk = bulk_evolution[0]
        final_bulk = bulk_evolution[-1]
        bulk_decay_fraction = (initial_bulk - final_bulk) / initial_bulk

        initial_shear = shear_evolution[0]
        final_shear = shear_evolution[-1]
        shear_decay_fraction = (initial_shear - final_shear) / initial_shear

        # Validate exponential decay
        assert (
            bulk_decay_fraction > 0.9
        ), f"Bulk pressure decay insufficient: {bulk_decay_fraction:.3f}"
        assert (
            shear_decay_fraction > 0.9
        ), f"Shear stress decay insufficient: {shear_decay_fraction:.3f}"

        # Test that final state approaches equilibrium
        assert (
            final_bulk < 0.01 * initial_bulk
        ), f"Bulk pressure not sufficiently suppressed: {final_bulk/initial_bulk:.3f}"
        assert (
            final_shear < 0.01 * initial_shear
        ), f"Shear stress not sufficiently suppressed: {final_shear/initial_shear:.3f}"

        print(
            f"✓ Exponential relaxation validated: bulk {bulk_decay_fraction:.3f}, shear {shear_decay_fraction:.3f}"
        )

    def test_relaxation_timescales(self, minimal_setup):
        """Test that measured relaxation times match theoretical values."""
        grid, transport_coeffs, metric, fields = minimal_setup

        # Create equilibration benchmark
        benchmark = EquilibrationBenchmark(grid, metric, transport_coeffs)

        # Create initial state with significant perturbations
        initial_fields = self._create_perturbed_state(grid, perturbation=0.2)

        # Run equilibration analysis
        properties = benchmark.analysis.analyze_relaxation_to_equilibrium(
            initial_fields, final_time=4.0, timestep=0.01
        )

        # Compare with theoretical relaxation times
        comparison = benchmark.relaxation_analysis.compare_with_theory(properties)

        # Check agreement within 20% tolerance
        assert comparison[
            "overall_agreement"
        ], f"Relaxation times do not match theory: {comparison}"

        shear_error = comparison["shear_relaxation"]["relative_error"]
        bulk_error = comparison["bulk_relaxation"]["relative_error"]

        assert shear_error < 0.2, f"Shear relaxation time error too large: {shear_error:.3f}"
        assert bulk_error < 0.2, f"Bulk relaxation time error too large: {bulk_error:.3f}"

        print(
            f"✓ Relaxation timescales validated: τ_π error {shear_error:.3f}, τ_Π error {bulk_error:.3f}"
        )

    def test_entropy_production(self, comprehensive_setup):
        """Test entropy production and second law of thermodynamics."""
        grid, transport_coeffs, metric, fields = comprehensive_setup

        # Create equilibration analysis
        analysis = EquilibrationAnalysis(grid, metric, transport_coeffs)
        entropy_analysis = EntropyProductionAnalysis(analysis)

        # Create initial non-equilibrium state
        initial_fields = self._create_perturbed_state(grid, perturbation=0.12)

        # Run relaxation analysis
        properties = analysis.analyze_relaxation_to_equilibrium(
            initial_fields, final_time=5.0, timestep=0.02
        )

        # Validate second law of thermodynamics
        second_law_validation = entropy_analysis.validate_second_law(properties)

        assert second_law_validation["valid"], f"Second law violated: {second_law_validation}"
        assert second_law_validation["entropy_increases"], "Total entropy does not increase"
        assert second_law_validation[
            "production_positive"
        ], "Average entropy production rate is negative"

        # Check violation statistics
        violation_fraction = second_law_validation["violation_fraction"]
        assert violation_fraction < 0.05, f"Too many entropy violations: {violation_fraction:.3f}"

        # Test entropy production sources
        entropy_sources = entropy_analysis.compute_entropy_production_sources(initial_fields)
        assert (
            entropy_sources["total"] >= 0
        ), f"Negative total entropy production: {entropy_sources['total']}"

        print(
            f"✓ Entropy production validated: violations {violation_fraction:.3f}, production {entropy_sources['total']:.6f}"
        )

    def test_thermodynamic_consistency(self, minimal_setup):
        """Test thermodynamic consistency throughout evolution."""
        grid, transport_coeffs, metric, fields = minimal_setup

        # Create equilibration analysis
        analysis = EquilibrationAnalysis(grid, metric, transport_coeffs)

        # Create initial equilibrium state
        initial_fields = self._create_equilibrium_state(grid, temperature=0.2)

        # Add small perturbations
        initial_fields.Pi.data[:] += 0.05 * np.mean(initial_fields.pressure.data)
        initial_fields.pi_munu.data[..., 1, 1] += 0.03 * np.mean(initial_fields.pressure.data)

        # Run relaxation analysis
        properties = analysis.analyze_relaxation_to_equilibrium(
            initial_fields, final_time=3.0, timestep=0.02
        )

        # Check temperature evolution is reasonable
        temp_evolution = properties.temperature_evolution
        assert np.all(temp_evolution > 0), "Temperature becomes negative"
        assert np.all(np.isfinite(temp_evolution)), "Temperature becomes non-finite"

        # Temperature should remain roughly constant (energy conservation)
        temp_variation = (np.max(temp_evolution) - np.min(temp_evolution)) / np.mean(temp_evolution)
        assert temp_variation < 0.1, f"Temperature varies too much: {temp_variation:.3f}"

        # Entropy should increase monotonically (with small tolerance for numerical errors)
        entropy_evolution = properties.entropy_evolution
        entropy_decreases = np.sum(np.diff(entropy_evolution) < -1e-12)
        assert (
            entropy_decreases < len(entropy_evolution) * 0.05
        ), f"Too many entropy decreases: {entropy_decreases}"

        print(f"✓ Thermodynamic consistency validated: temp variation {temp_variation:.3f}")

    def test_navier_stokes_limit(self, minimal_setup):
        """Test approach to Navier-Stokes limit with small relaxation times."""
        grid, transport_coeffs, metric, fields = minimal_setup

        # Create transport coefficients with very small relaxation times
        ns_coeffs = TransportCoefficients(
            shear_viscosity=transport_coeffs.shear_viscosity,
            bulk_viscosity=transport_coeffs.bulk_viscosity,
            shear_relaxation_time=0.01,  # Very small
            bulk_relaxation_time=0.01,  # Very small
            lambda_pi_pi=0.0,  # Turn off second-order terms
            xi_1=0.0,
        )

        # Create analyses for both IS and NS-like systems
        is_analysis = EquilibrationAnalysis(grid, metric, transport_coeffs)
        ns_analysis = EquilibrationAnalysis(grid, metric, ns_coeffs)

        # Create identical initial states
        initial_fields_is = self._create_perturbed_state(grid, perturbation=0.1)
        initial_fields_ns = self._create_perturbed_state(grid, perturbation=0.1)

        # Run both analyses
        is_properties = is_analysis.analyze_relaxation_to_equilibrium(
            initial_fields_is, final_time=2.0, timestep=0.01
        )
        ns_properties = ns_analysis.analyze_relaxation_to_equilibrium(
            initial_fields_ns, final_time=2.0, timestep=0.01
        )

        # In the Navier-Stokes limit, relaxation should be much faster
        ns_bulk_final = ns_properties.bulk_pressure_evolution[-1]
        ns_shear_final = ns_properties.shear_stress_evolution[-1]

        is_bulk_final = is_properties.bulk_pressure_evolution[-1]
        is_shear_final = is_properties.shear_stress_evolution[-1]

        # NS system should relax faster (smaller final values)
        assert (
            ns_bulk_final < 0.5 * is_bulk_final
        ), f"NS bulk relaxation not faster: {ns_bulk_final/is_bulk_final:.3f}"
        assert (
            ns_shear_final < 0.5 * is_shear_final
        ), f"NS shear relaxation not faster: {ns_shear_final/is_shear_final:.3f}"

        print(f"✓ Navier-Stokes limit validated: bulk speedup {is_bulk_final/ns_bulk_final:.2f}x")

    def test_spectral_solver_integration(self, minimal_setup):
        """Test integration with spectral solver for equilibration."""
        grid, transport_coeffs, metric, fields = minimal_setup

        # Initialize spectral solver
        spectral_solver = SpectralISHydrodynamics(grid, fields, transport_coeffs)

        # Create non-equilibrium initial condition
        initial_fields = self._create_perturbed_state(grid, perturbation=0.08)

        # Copy to solver fields
        fields.rho.data[:] = initial_fields.rho.data[:]
        fields.pressure.data[:] = initial_fields.pressure.data[:]
        fields.u_mu.data[:] = initial_fields.four_velocity.data[:]
        fields.Pi.data[:] = initial_fields.Pi.data[:]
        fields.pi_munu.data[:] = initial_fields.pi_munu.data[:]
        fields.q_mu.data[:] = initial_fields.q_mu.data[:]

        # Record initial state
        initial_bulk = np.mean(np.abs(fields.Pi.data))
        initial_shear = np.sqrt(np.mean(np.sum(fields.pi_munu.data**2, axis=(-2, -1))))

        # Evolve with spectral solver
        dt = 0.02
        n_steps = 50  # Total time = 1.0

        for _step in range(n_steps):
            spectral_solver.time_step(dt, method="spectral_imex")

        # Check final state
        final_bulk = np.mean(np.abs(fields.Pi.data))
        final_shear = np.sqrt(np.mean(np.sum(fields.pi_munu.data**2, axis=(-2, -1))))

        # Should see significant decay
        bulk_decay = (initial_bulk - final_bulk) / initial_bulk
        shear_decay = (initial_shear - final_shear) / initial_shear

        assert bulk_decay > 0.5, f"Insufficient bulk decay with spectral solver: {bulk_decay:.3f}"
        assert (
            shear_decay > 0.5
        ), f"Insufficient shear decay with spectral solver: {shear_decay:.3f}"

        # Fields should remain physical
        assert np.all(fields.rho.data > 0), "Energy density became negative"
        assert np.all(fields.pressure.data > 0), "Pressure became negative"
        assert np.all(np.isfinite(fields.rho.data)), "Non-finite values in energy density"

        print(
            f"✓ Spectral solver integration validated: bulk decay {bulk_decay:.3f}, shear decay {shear_decay:.3f}"
        )

    def test_benchmark_performance(self, minimal_setup):
        """Test performance of equilibration benchmark suite."""
        grid, transport_coeffs, metric, fields = minimal_setup

        start_time = time.time()

        # Create benchmark suite
        benchmark = EquilibrationBenchmark(grid, metric, transport_coeffs)

        # Run comprehensive tests
        results = benchmark.run_comprehensive_tests(initial_perturbation=0.1)

        elapsed_time = time.time() - start_time

        # Check that benchmark completes successfully
        assert results["overall_pass"], f"Benchmark tests failed: {results}"

        # Should complete in reasonable time
        assert elapsed_time < 45.0, f"Benchmark too slow: {elapsed_time:.1f}s"

        # Verify individual test results
        assert results["equilibration_test"]["pass"], "Basic equilibration test failed"
        assert results["relaxation_times_test"]["pass"], "Relaxation times test failed"
        assert results["second_law_test"]["pass"], "Second law test failed"

        print(f"✓ Benchmark performance validated: {elapsed_time:.2f}s, all tests passed")

    def _create_perturbed_state(self, grid, perturbation=0.1):
        """Create initial non-equilibrium state with specified perturbation."""
        fields = ISFieldConfiguration(grid)

        # Background equilibrium state (T = 0.2 GeV)
        T = 0.2
        g_eff = 37.5
        rho = (np.pi**2 / 90.0) * g_eff * T**4
        p = rho / 3.0

        fields.rho.fill(rho)
        fields.pressure.fill(p)
        fields.four_velocity.fill_zero()
        fields.four_velocity.data[..., 0] = 1.0

        # Add non-equilibrium perturbations
        fields.Pi.fill(perturbation * p)
        fields.pi_munu.fill_zero()
        fields.pi_munu.data[..., 1, 1] = perturbation * p * 0.5
        fields.pi_munu.data[..., 2, 2] = -perturbation * p * 0.5
        fields.q_mu.fill_zero()

        return fields

    def _create_equilibrium_state(self, grid, temperature=0.2):
        """Create equilibrium state at specified temperature."""
        fields = ISFieldConfiguration(grid)

        # Thermodynamics for ideal gas
        T = temperature
        g_eff = 37.5
        rho = (np.pi**2 / 90.0) * g_eff * T**4
        p = rho / 3.0

        fields.rho.fill(rho)
        fields.pressure.fill(p)
        fields.four_velocity.fill_zero()
        fields.four_velocity.data[..., 0] = 1.0

        # Zero dissipative fluxes (equilibrium)
        fields.Pi.fill_zero()
        fields.pi_munu.fill_zero()
        fields.q_mu.fill_zero()

        return fields


def run_comprehensive_validation():
    """Run comprehensive equilibration validation suite."""
    print("=" * 70)
    print("EQUILIBRATION VALIDATION - PHASE 1.3")
    print("=" * 70)

    # Initialize test class
    validator = TestEquilibrationValidation()

    # Create setups (call methods directly for standalone execution)
    minimal = validator._create_minimal_setup()
    comprehensive = validator._create_comprehensive_setup()

    print("\n1. Testing exponential relaxation to equilibrium...")
    validator.test_exponential_relaxation(comprehensive)
    print("✓ Exponential relaxation dynamics confirmed")

    print("\n2. Testing relaxation timescales...")
    validator.test_relaxation_timescales(minimal)
    print("✓ Relaxation timescales match theoretical predictions")

    print("\n3. Testing entropy production and second law...")
    validator.test_entropy_production(comprehensive)
    print("✓ Second law of thermodynamics satisfied")

    print("\n4. Testing thermodynamic consistency...")
    validator.test_thermodynamic_consistency(minimal)
    print("✓ Thermodynamic consistency maintained")

    print("\n5. Testing Navier-Stokes limit...")
    validator.test_navier_stokes_limit(minimal)
    print("✓ Proper approach to Navier-Stokes limit")

    print("\n6. Testing spectral solver integration...")
    validator.test_spectral_solver_integration(minimal)
    print("✓ Spectral solver equilibration validated")

    print("\n7. Testing benchmark performance...")
    validator.test_benchmark_performance(minimal)
    print("✓ Performance benchmarks met")

    print("\n" + "=" * 70)
    print("EQUILIBRATION VALIDATION COMPLETE - ALL TESTS PASSED")
    print("✅ Phase 1.3 Implementation Successful")
    print("=" * 70)


if __name__ == "__main__":
    print("Starting Equilibration Validation (Phase 1.3)...")

    try:
        run_comprehensive_validation()
        print("\n🎉 Equilibration validation completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Equilibration validation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
