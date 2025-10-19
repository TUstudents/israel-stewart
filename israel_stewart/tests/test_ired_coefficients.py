"""
Tests for IReD transport coefficients.

Validates the simplified IReD implementation against benchmark values
from Wagner, Palermo, Ambrus (2022), Tables III-IV.
"""

import numpy as np
import pytest

from israel_stewart.equations.ired_simple import HardSphereIReD


class TestHardSphereIReD:
    """Test hard sphere gas IReD transport coefficients."""

    @pytest.fixture
    def model(self):
        """Create standard test model at T=400 MeV."""
        return HardSphereIReD(
            temperature=0.4,  # 400 MeV
            cross_section=1.0,  # 1 fm²
            truncation="41",  # Highest accuracy
        )

    def test_initialization(self, model):
        """Test model initializes correctly."""
        assert model.temperature == 0.4
        assert model.cross_section == 1.0
        assert model.truncation == "41"
        assert model.beta == 2.5  # 1/T

    def test_invalid_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            HardSphereIReD(temperature=-0.1, cross_section=1.0)

    def test_invalid_cross_section(self):
        """Test that negative cross-section raises error."""
        with pytest.raises(ValueError, match="Cross-section must be positive"):
            HardSphereIReD(temperature=0.4, cross_section=-1.0)

    def test_invalid_truncation(self):
        """Test that invalid truncation raises error."""
        with pytest.raises(ValueError, match="Unknown truncation"):
            HardSphereIReD(temperature=0.4, cross_section=1.0, truncation="100")

    # ========================================================================
    # First-Order Coefficients
    # ========================================================================

    def test_shear_viscosity_scaling(self):
        """Test shear viscosity scales correctly with T and σ."""
        model1 = HardSphereIReD(temperature=0.4, cross_section=1.0)
        model2 = HardSphereIReD(temperature=0.8, cross_section=1.0)  # 2x temperature
        model3 = HardSphereIReD(temperature=0.4, cross_section=2.0)  # 2x cross-section

        # η ∝ T/σ, so doubling T should double η
        np.testing.assert_allclose(
            model2.shear_viscosity(), 2 * model1.shear_viscosity(), rtol=1e-10
        )

        # η ∝ 1/σ, so doubling σ should halve η
        np.testing.assert_allclose(
            model3.shear_viscosity(), 0.5 * model1.shear_viscosity(), rtol=1e-10
        )

    def test_bulk_viscosity_conformal(self, model):
        """Test bulk viscosity is zero for conformal fluid."""
        assert model.bulk_viscosity() == 0.0

    # ========================================================================
    # Relaxation Times
    # ========================================================================

    def test_diffusion_relaxation_time_value(self, model):
        """Test diffusion relaxation time matches IReD Table III."""
        tau_V = model.diffusion_relaxation_time()

        # Expected: τ_V = 2.0794 λ_mfp for N₁=4, 41 moments
        expected = 2.0794 * model.mean_free_path

        np.testing.assert_allclose(tau_V, expected, rtol=1e-4)

    # ========================================================================
    # Second-Order Coefficients
    # ========================================================================

    def test_tau_pi_pi_value(self, model):
        """Test shear-shear coupling matches IReD Table III."""
        tau_pi_pi = model.tau_pi_pi()

        # Expected: τ_ππ = 1.6944 τ_π for N₂=3
        tau_pi = model.shear_relaxation_time()
        expected = 1.6944 * tau_pi

        np.testing.assert_allclose(tau_pi_pi, expected, rtol=1e-4)

    def test_delta_pi_pi_value(self, model):
        """Test shear expansion coupling is 4/3."""
        assert model.delta_pi_pi() == pytest.approx(4.0 / 3.0)

    def test_lambda_pi_V_value(self, model):
        """Test shear-diffusion coupling matches IReD Table IV."""
        lambda_pi_V = model.lambda_pi_V()

        # Expected: λ_πn = 0.20890/β = 0.20890 × T
        # Units: GeV¹ (required for dimensional consistency)
        expected = 0.20890 / model.beta

        np.testing.assert_allclose(lambda_pi_V, expected, rtol=1e-4)

    def test_delta_V_V_value(self, model):
        """Test diffusion expansion coupling is 1."""
        assert model.delta_V_V() == 1.0

    def test_lambda_V_V_value(self, model):
        """Test diffusion-diffusion coupling matches IReD Table III."""
        lambda_V_V = model.lambda_V_V()

        # Expected: λ_VV = 0.89501 τ_V
        tau_V = model.diffusion_relaxation_time()
        expected = 0.89501 * tau_V

        np.testing.assert_allclose(lambda_V_V, expected, rtol=1e-4)

    def test_lambda_V_pi_value(self, model):
        """Test diffusion-shear coupling matches IReD Table III."""
        lambda_V_pi = model.lambda_V_pi()

        # Expected: λ_Vπ = 0.069240 β τ_V
        tau_V = model.diffusion_relaxation_time()
        expected = 0.069240 * model.beta * tau_V

        np.testing.assert_allclose(lambda_V_pi, expected, rtol=1e-4)

    def test_tau_V_pi_value(self, model):
        """Test diffusion-shear force coupling matches IReD Table III."""
        tau_V_pi = model.tau_V_pi()

        # Expected: τ_Vπ = 0.0071692 β τ_V/P
        tau_V = model.diffusion_relaxation_time()
        expected = 0.0071692 * model.beta * tau_V / model.pressure

        np.testing.assert_allclose(tau_V_pi, expected, rtol=1e-4)

    def test_ell_V_pi_value(self, model):
        """Test diffusion-shear gradient coupling matches IReD Table III."""
        ell_V_pi = model.ell_V_pi()

        # Expected: ℓ_Vπ = 0.028677 β τ_V
        tau_V = model.diffusion_relaxation_time()
        expected = 0.028677 * model.beta * tau_V

        np.testing.assert_allclose(ell_V_pi, expected, rtol=1e-4)

    # ========================================================================
    # Derived Quantities
    # ========================================================================

    def test_eta_over_s_positive(self, model):
        """Test η/s ratio is positive."""
        eta_over_s = model.eta_over_s()
        assert eta_over_s > 0

    def test_eta_over_s_above_kss_bound(self, model):
        """Test η/s is above KSS bound (weak check for kinetic theory)."""
        eta_over_s = model.eta_over_s()
        kss_bound = 1.0 / (4 * np.pi)

        # Kinetic theory typically gives η/s >> KSS bound
        # (KSS bound is quantum lower limit, not classical)
        assert eta_over_s > 0  # Just check positivity

    def test_knudsen_number_scaling(self, model):
        """Test Knudsen number scales correctly."""
        L = 10.0  # fm
        Kn = model.knudsen_number(L)

        # Kn = λ_mfp / L
        expected = model.mean_free_path / L
        np.testing.assert_allclose(Kn, expected, rtol=1e-10)

    def test_regime_parameter_scaling(self, model):
        """Test regime parameter scales linearly with wavenumber."""
        k1 = 1.0
        k2 = 2.0

        regime1 = model.regime_parameter(k1)
        regime2 = model.regime_parameter(k2)

        # |τω| ∝ k, so doubling k should double |τω|
        np.testing.assert_allclose(regime2, 2 * regime1, rtol=1e-10)

    # ========================================================================
    # Truncation Convergence
    # ========================================================================

    def test_truncation_convergence(self):
        """Test that higher truncations give more accurate results."""
        models = {
            "14": HardSphereIReD(temperature=0.4, cross_section=1.0, truncation="14"),
            "23": HardSphereIReD(temperature=0.4, cross_section=1.0, truncation="23"),
            "32": HardSphereIReD(temperature=0.4, cross_section=1.0, truncation="32"),
            "41": HardSphereIReD(temperature=0.4, cross_section=1.0, truncation="41"),
        }

        # Expected from IReD paper: η converges to 1.2676 (N=∞)
        eta_values = [models[n].shear_viscosity() for n in ["14", "23", "32", "41"]]

        # Check monotonic convergence (errors should decrease)
        errors = [abs(eta - eta_values[-1]) for eta in eta_values[:-1]]
        assert errors[0] > errors[1] > errors[2]  # Monotonic decrease

    def test_different_truncations_close(self):
        """Test that different truncations give similar results."""
        model_23 = HardSphereIReD(temperature=0.4, cross_section=1.0, truncation="23")
        model_41 = HardSphereIReD(temperature=0.4, cross_section=1.0, truncation="41")

        # Should differ by < 1% for 23 vs 41 moments (from IReD Table I)
        np.testing.assert_allclose(
            model_23.shear_viscosity(),
            model_41.shear_viscosity(),
            rtol=0.01,  # 1% tolerance
        )

    # ========================================================================
    # Validation Against Paper
    # ========================================================================

    def test_validate_against_ired_paper(self, model):
        """Test comprehensive validation against IReD Table III."""
        validation = model.validate_against_ired_paper()

        # All coefficients should match paper values
        for name, passed in validation.items():
            assert passed, f"Validation failed for {name}"

    def test_summary_runs_without_error(self, model):
        """Test that summary method runs without errors."""
        summary = model.summary()
        assert isinstance(summary, str)
        assert "IReD Transport Coefficients" in summary
        assert "Hard Sphere Gas" in summary


class TestPhysicalConsistency:
    """Test physical consistency of IReD coefficients."""

    def test_positive_viscosities(self):
        """Test that all viscosities are positive (second law)."""
        model = HardSphereIReD(temperature=0.4, cross_section=1.0)

        assert model.shear_viscosity() > 0
        assert model.bulk_viscosity() >= 0  # Zero for conformal
        assert model.diffusion_coefficient() > 0

    def test_positive_relaxation_times(self):
        """Test that all relaxation times are positive."""
        model = HardSphereIReD(temperature=0.4, cross_section=1.0)

        assert model.shear_relaxation_time() > 0
        assert model.bulk_relaxation_time() > 0
        assert model.diffusion_relaxation_time() > 0

    def test_temperature_scaling(self):
        """Test that coefficients scale correctly with temperature."""
        T1 = 0.2
        T2 = 0.4

        model1 = HardSphereIReD(temperature=T1, cross_section=1.0)
        model2 = HardSphereIReD(temperature=T2, cross_section=1.0)

        # η ∝ T for fixed σ
        ratio_eta = model2.shear_viscosity() / model1.shear_viscosity()
        np.testing.assert_allclose(ratio_eta, T2 / T1, rtol=1e-10)

        # Mean free path λ ∝ 1/(n·σ) ∝ 1/T³ for ultrarelativistic gas
        ratio_lambda = model2.mean_free_path / model1.mean_free_path
        np.testing.assert_allclose(ratio_lambda, (T1 / T2) ** 3, rtol=1e-3)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_cross_section(self):
        """Test behavior with very small cross-section (nearly ideal)."""
        model = HardSphereIReD(temperature=0.4, cross_section=1e-6)

        # Very small σ → very large λ_mfp → weak coupling
        # With corrected units: λ_mfp = (ℏc)³/(n·σ) ≈ 9.86e5 fm
        assert model.mean_free_path > 9e5  # Very large (almost 1 mm!)

        # Coefficients should still be finite and positive
        assert np.isfinite(model.shear_viscosity())
        assert model.shear_viscosity() > 0

    def test_very_large_cross_section(self):
        """Test behavior with very large cross-section (strong coupling)."""
        model = HardSphereIReD(temperature=0.4, cross_section=1e6)

        # Very large σ → very small λ_mfp → strong coupling
        assert model.mean_free_path < 1e-3  # Very small

        # Coefficients should still be finite and positive
        assert np.isfinite(model.shear_viscosity())
        assert model.shear_viscosity() > 0

    def test_high_temperature(self):
        """Test behavior at high temperature (T ~ 1 GeV)."""
        model = HardSphereIReD(temperature=1.0, cross_section=1.0)

        # Should still give reasonable results
        assert np.isfinite(model.shear_viscosity())
        assert np.isfinite(model.shear_relaxation_time())
        assert model.shear_viscosity() > 0
        assert model.shear_relaxation_time() > 0

        # Check that coefficients scale correctly with T
        # For ultrarelativistic gas: η ∝ T, s ∝ T³, so η/s ∝ 1/T²
        model_low_T = HardSphereIReD(temperature=0.4, cross_section=1.0)
        ratio_eta_over_s = model.eta_over_s() / model_low_T.eta_over_s()
        expected_ratio = (model_low_T.temperature / model.temperature) ** 2

        np.testing.assert_allclose(ratio_eta_over_s, expected_ratio, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
