"""
Tests for transport coefficient calculation module.

Validates kinetic theory and QCD-inspired models for temperature and
density-dependent transport coefficients in Israel-Stewart hydrodynamics.
"""

import numpy as np
import pytest

from israel_stewart.core.constants import KBOLTZ, MPROTON
from israel_stewart.equations.coefficients import (
    KineticTheoryModel,
    QCDInspiredModel,
    TransportCoefficientCalculator,
)


class TestKineticTheoryModel:
    """Test kinetic theory transport coefficient model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = KineticTheoryModel(
            particle_mass=MPROTON,
            cross_section=1e-24,  # cm²
            degrees_of_freedom=3.0
        )
        self.T = 0.150  # GeV (150 MeV)
        self.rho = 0.16  # GeV/fm³ (nuclear density)

    def test_positive_coefficients(self):
        """Test that all coefficients are positive for physical inputs."""
        eta = self.model.shear_viscosity(self.T, self.rho)
        zeta = self.model.bulk_viscosity(self.T, self.rho)
        kappa = self.model.thermal_conductivity(self.T, self.rho)
        tau_pi = self.model.shear_relaxation_time(self.T, self.rho)
        tau_Pi = self.model.bulk_relaxation_time(self.T, self.rho)
        tau_q = self.model.heat_relaxation_time(self.T, self.rho)

        assert eta > 0, f"Shear viscosity should be positive, got {eta}"
        assert zeta >= 0, f"Bulk viscosity should be non-negative, got {zeta}"
        assert kappa > 0, f"Thermal conductivity should be positive, got {kappa}"
        assert tau_pi > 0, f"Shear relaxation time should be positive, got {tau_pi}"
        assert tau_Pi >= 0, f"Bulk relaxation time should be non-negative, got {tau_Pi}"
        assert tau_q > 0, f"Heat relaxation time should be positive, got {tau_q}"

    def test_temperature_scaling(self):
        """Test correct temperature scaling of coefficients."""
        T1, T2 = 0.100, 0.200  # GeV

        eta1 = self.model.shear_viscosity(T1, self.rho)
        eta2 = self.model.shear_viscosity(T2, self.rho)

        # Shear viscosity should increase with temperature (∝ √T for kinetic theory)
        assert eta2 > eta1, "Shear viscosity should increase with temperature"

        # Check approximate scaling
        expected_ratio = np.sqrt(T2 / T1)
        actual_ratio = eta2 / eta1
        assert 0.5 * expected_ratio < actual_ratio < 2.0 * expected_ratio, \
            f"Temperature scaling seems wrong: expected ≈{expected_ratio}, got {actual_ratio}"

    def test_density_scaling(self):
        """Test correct density scaling of coefficients."""
        rho1, rho2 = 0.08, 0.32  # GeV/fm³

        eta1 = self.model.shear_viscosity(self.T, rho1)
        eta2 = self.model.shear_viscosity(self.T, rho2)

        # Higher density → more collisions → lower viscosity
        assert eta2 < eta1, "Shear viscosity should decrease with density"

    def test_zero_inputs(self):
        """Test behavior with zero temperature or density."""
        # Zero temperature
        eta_zero_T = self.model.shear_viscosity(0.0, self.rho)
        assert eta_zero_T == 0.0, "Zero temperature should give zero viscosity"

        # Zero density
        eta_zero_rho = self.model.shear_viscosity(self.T, 0.0)
        assert eta_zero_rho == 0.0, "Zero density should give zero viscosity"

    def test_monatomic_bulk_viscosity(self):
        """Test that monatomic gas has very small bulk viscosity."""
        # For exactly 3 DOF (monatomic), bulk viscosity should be zero
        model_mono = KineticTheoryModel(degrees_of_freedom=3.0)
        zeta = model_mono.bulk_viscosity(self.T, self.rho)
        assert zeta == 0.0, f"Monatomic gas should have zero bulk viscosity, got {zeta}"

    def test_wiedemann_franz_relation(self):
        """Test approximate Wiedemann-Franz relation for thermal conductivity."""
        eta = self.model.shear_viscosity(self.T, self.rho)
        kappa = self.model.thermal_conductivity(self.T, self.rho)

        # For kinetic theory: κ = (15/4) * (k_B/m) * η
        expected_kappa = (15.0 / 4.0) * (KBOLTZ / self.model.particle_mass) * eta

        assert abs(kappa - expected_kappa) / expected_kappa < 1e-10, \
            f"Thermal conductivity doesn't match kinetic theory: expected {expected_kappa}, got {kappa}"

    def test_relaxation_time_consistency(self):
        """Test that relaxation times are consistent with first-order coefficients."""
        eta = self.model.shear_viscosity(self.T, self.rho)
        tau_pi = self.model.shear_relaxation_time(self.T, self.rho)

        # τ_π should be roughly η/P where P is pressure
        number_density = self.rho / self.model.particle_mass
        pressure = number_density * KBOLTZ * self.T

        expected_tau_pi = eta / (1.5 * pressure)  # β ≈ 1.5 in our model

        # Allow factor of 2 difference due to approximations
        ratio = tau_pi / expected_tau_pi
        assert 0.5 < ratio < 2.0, f"Relaxation time scaling seems wrong: expected ≈{expected_tau_pi}, got {tau_pi}"


class TestQCDInspiredModel:
    """Test QCD-inspired transport coefficient model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = QCDInspiredModel(
            critical_temperature=0.170,  # GeV
            eta_over_s_minimum=0.08,
            zeta_over_s_peak=0.10,
            crossover_width=0.020
        )
        self.T_low = 0.120   # Below Tc
        self.T_c = 0.170     # Critical temperature
        self.T_high = 0.300  # Above Tc
        self.rho = 0.16      # GeV/fm³

    def test_positive_coefficients(self):
        """Test that all coefficients are positive."""
        for T in [self.T_low, self.T_c, self.T_high]:
            eta = self.model.shear_viscosity(T, self.rho)
            zeta = self.model.bulk_viscosity(T, self.rho)
            kappa = self.model.thermal_conductivity(T, self.rho)
            tau_pi = self.model.shear_relaxation_time(T, self.rho)
            tau_Pi = self.model.bulk_relaxation_time(T, self.rho)
            tau_q = self.model.heat_relaxation_time(T, self.rho)

            assert eta > 0, f"Shear viscosity should be positive at T={T}, got {eta}"
            assert zeta >= 0, f"Bulk viscosity should be non-negative at T={T}, got {zeta}"
            assert kappa > 0, f"Thermal conductivity should be positive at T={T}, got {kappa}"
            assert tau_pi > 0, f"Shear relaxation time should be positive at T={T}, got {tau_pi}"
            assert tau_Pi >= 0, f"Bulk relaxation time should be non-negative at T={T}, got {tau_Pi}"
            assert tau_q > 0, f"Heat relaxation time should be positive at T={T}, got {tau_q}"

    def test_kss_bound(self):
        """Test that η/s respects the KSS bound."""
        # Compute η/s at critical temperature
        eta = self.model.shear_viscosity(self.T_c, self.rho)

        # Entropy density
        g_eff = 37.5
        s = (np.pi**2 / 90) * g_eff * self.T_c**3

        eta_over_s = eta / s

        # Should be close to the minimum value
        assert abs(eta_over_s - self.model.eta_s_min) < 0.01, \
            f"η/s at Tc should be near minimum: expected {self.model.eta_s_min}, got {eta_over_s}"

    def test_bulk_viscosity_peak(self):
        """Test that bulk viscosity peaks near critical temperature."""
        zeta_low = self.model.bulk_viscosity(self.T_low, self.rho)
        zeta_c = self.model.bulk_viscosity(self.T_c, self.rho)
        zeta_high = self.model.bulk_viscosity(self.T_high, self.rho)

        # Bulk viscosity should peak near Tc
        assert zeta_c > zeta_low, "Bulk viscosity should be higher at Tc than below"
        assert zeta_c > zeta_high, "Bulk viscosity should be higher at Tc than above"

    def test_conformal_limit(self):
        """Test approach to conformal limit at high temperature."""
        T_very_high = 1.0  # GeV (much higher than Tc)

        zeta_high = self.model.bulk_viscosity(T_very_high, self.rho)
        zeta_c = self.model.bulk_viscosity(self.T_c, self.rho)

        # Bulk viscosity should be much smaller at high T (conformal limit)
        assert zeta_high < 0.1 * zeta_c, \
            f"Bulk viscosity should be suppressed at high T: ζ(T={T_very_high})={zeta_high}, ζ(Tc)={zeta_c}"

    def test_temperature_scaling_shear(self):
        """Test temperature scaling of shear viscosity."""
        eta_low = self.model.shear_viscosity(self.T_low, self.rho)
        eta_high = self.model.shear_viscosity(self.T_high, self.rho)

        # η should increase with temperature (more entropy)
        assert eta_high > eta_low, "Shear viscosity should increase with temperature"

    def test_relaxation_time_scales(self):
        """Test that relaxation times have reasonable scales."""
        tau_pi = self.model.shear_relaxation_time(self.T_c, self.rho)

        # Relaxation time should be of order 1/T
        expected_scale = 1.0 / self.T_c  # ≈ 6 fm/c

        # Should be within an order of magnitude
        assert 0.1 * expected_scale < tau_pi < 10 * expected_scale, \
            f"Relaxation time scale seems wrong: expected ~{expected_scale}, got {tau_pi}"


class TestTransportCoefficientCalculator:
    """Test main transport coefficient calculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kinetic_calc = TransportCoefficientCalculator(
            model=KineticTheoryModel(), enable_second_order=True
        )
        self.qcd_calc = TransportCoefficientCalculator(
            model=QCDInspiredModel(), enable_second_order=True
        )
        self.T = 0.150  # GeV
        self.rho = 0.16  # GeV/fm³

    def test_complete_coefficient_computation(self):
        """Test that all coefficients are computed and returned."""
        coeffs = self.kinetic_calc.compute_coefficients(self.T, self.rho)

        # Check that all attributes exist and are reasonable
        assert hasattr(coeffs, 'shear_viscosity')
        assert hasattr(coeffs, 'bulk_viscosity')
        assert hasattr(coeffs, 'thermal_conductivity')
        assert hasattr(coeffs, 'shear_relaxation_time')
        assert hasattr(coeffs, 'bulk_relaxation_time')
        assert hasattr(coeffs, 'heat_relaxation_time')

        # Second-order coefficients
        assert hasattr(coeffs, 'lambda_pi_pi')
        assert hasattr(coeffs, 'lambda_pi_Pi')
        assert hasattr(coeffs, 'xi_1')
        assert hasattr(coeffs, 'xi_2')

        # Check values are reasonable
        assert coeffs.shear_viscosity > 0
        assert coeffs.thermal_conductivity > 0
        assert coeffs.shear_relaxation_time > 0

    def test_second_order_coefficients(self):
        """Test computation of second-order coefficients."""
        coeffs = self.kinetic_calc.compute_coefficients(self.T, self.rho)

        # Second-order coefficients should be computed
        assert coeffs.lambda_pi_pi > 0, "λ_ππ should be positive"
        assert abs(coeffs.lambda_pi_Pi) >= 0, "λ_πΠ should be defined"

        # Check scaling with first-order coefficients
        eta = coeffs.shear_viscosity
        tau_pi = coeffs.shear_relaxation_time

        # λ_ππ should scale with τ_π
        expected_scale = tau_pi
        assert 0.1 * expected_scale < coeffs.lambda_pi_pi < 10 * expected_scale, \
            f"λ_ππ scaling seems wrong: expected ~{expected_scale}, got {coeffs.lambda_pi_pi}"

    def test_caching(self):
        """Test coefficient caching functionality."""
        # First computation
        coeffs1 = self.kinetic_calc.compute_coefficients(self.T, self.rho)
        cache_info1 = self.kinetic_calc.get_cache_info()

        # Second computation with same parameters
        coeffs2 = self.kinetic_calc.compute_coefficients(self.T, self.rho)
        cache_info2 = self.kinetic_calc.get_cache_info()

        # Should be cached (same object or same values)
        assert coeffs1.shear_viscosity == coeffs2.shear_viscosity
        assert cache_info1['cache_size'] <= cache_info2['cache_size']

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Compute some coefficients
        self.kinetic_calc.compute_coefficients(self.T, self.rho)
        assert self.kinetic_calc.get_cache_info()['cache_size'] > 0

        # Clear cache
        self.kinetic_calc.clear_cache()
        assert self.kinetic_calc.get_cache_info()['cache_size'] == 0

    def test_model_comparison(self):
        """Test that different models give different results."""
        kinetic_coeffs = self.kinetic_calc.compute_coefficients(self.T, self.rho)
        qcd_coeffs = self.qcd_calc.compute_coefficients(self.T, self.rho)

        # Should be different (different physics)
        assert kinetic_coeffs.shear_viscosity != qcd_coeffs.shear_viscosity
        assert kinetic_coeffs.bulk_viscosity != qcd_coeffs.bulk_viscosity

    def test_disable_second_order(self):
        """Test disabling second-order coefficient computation."""
        calc_no_second = TransportCoefficientCalculator(
            model=KineticTheoryModel(), enable_second_order=False
        )

        coeffs = calc_no_second.compute_coefficients(self.T, self.rho)

        # Second-order coefficients should be zero or minimal
        assert coeffs.lambda_pi_pi == 0.0
        assert coeffs.lambda_pi_Pi == 0.0
        assert coeffs.xi_1 == 0.0


class TestPhysicalConsistency:
    """Test physical consistency and thermodynamic constraints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calc = TransportCoefficientCalculator(
            model=KineticTheoryModel(), enable_second_order=True
        )

    def test_thermodynamic_constraint(self):
        """Test thermodynamic constraint: ζ + (2/3)η ≥ 0."""
        T_values = np.linspace(0.050, 0.500, 10)  # GeV
        rho = 0.16  # GeV/fm³

        for T in T_values:
            if T > 0:  # Skip zero temperature
                coeffs = self.calc.compute_coefficients(T, rho, validate=False)
                eta = coeffs.shear_viscosity
                zeta = coeffs.bulk_viscosity

                constraint = zeta + (2.0/3.0) * eta
                assert constraint >= -1e-10, \
                    f"Thermodynamic constraint violated at T={T}: ζ + (2/3)η = {constraint}"

    def test_causality_constraint(self):
        """Test that relaxation times are positive (causality)."""
        T_values = np.linspace(0.050, 0.500, 10)
        rho = 0.16

        for T in T_values:
            if T > 0:
                coeffs = self.calc.compute_coefficients(T, rho, validate=False)

                assert coeffs.shear_relaxation_time > 0, \
                    f"Shear relaxation time non-positive at T={T}"
                assert coeffs.bulk_relaxation_time >= 0, \
                    f"Bulk relaxation time negative at T={T}"
                assert coeffs.heat_relaxation_time > 0, \
                    f"Heat relaxation time non-positive at T={T}"

    def test_dimensional_analysis(self):
        """Test dimensional consistency of coefficients."""
        T = 0.150  # GeV
        rho = 0.16  # GeV/fm³

        coeffs = self.calc.compute_coefficients(T, rho)

        # In natural units (ℏ = c = 1):
        # [η] = [energy] * [time] / [volume] = GeV * fm/c / fm³ = GeV/fm²
        # [τ] = [time] = fm/c
        # [λ] = [time] = fm/c

        # Check that values have reasonable magnitudes for natural units
        assert 1e-6 < coeffs.shear_viscosity < 1e2, \
            f"Shear viscosity magnitude seems wrong: {coeffs.shear_viscosity} GeV/fm²"

        assert 1e-3 < coeffs.shear_relaxation_time < 1e2, \
            f"Relaxation time magnitude seems wrong: {coeffs.shear_relaxation_time} fm/c"

    @pytest.mark.parametrize("T,rho", [
        (0.100, 0.08),   # Low T, low ρ
        (0.170, 0.16),   # Critical point
        (0.300, 0.32),   # High T, high ρ
        (0.050, 0.64),   # Low T, high ρ
        (0.400, 0.04),   # High T, low ρ
    ])
    def test_parameter_space_coverage(self, T, rho):
        """Test coefficient computation across parameter space."""
        coeffs = self.calc.compute_coefficients(T, rho)

        # All coefficients should be physically reasonable
        assert coeffs.shear_viscosity > 0
        assert coeffs.bulk_viscosity >= 0
        assert coeffs.thermal_conductivity > 0
        assert coeffs.shear_relaxation_time > 0
        assert coeffs.bulk_relaxation_time >= 0
        assert coeffs.heat_relaxation_time > 0


if __name__ == "__main__":
    pytest.main([__file__])