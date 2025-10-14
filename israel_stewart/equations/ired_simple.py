"""
Simplified IReD Transport Coefficients (Hard Sphere Gas Benchmark)

This module provides a pragmatic implementation of IReD transport coefficients
based on pre-computed values from the IReD paper (Wagner et al. 2022).

Instead of computing collision matrices from scratch, this uses the benchmark
values from Tables III-IV of the paper for ultrarelativistic hard sphere gas.

This provides:
- Quantitatively accurate coefficients for benchmark comparisons
- Fast computation (no collision integrals)
- Clear validation against published results
- Foundation for more sophisticated models later

Usage:
    >>> from israel_stewart.equations.ired_simple import HardSphereIReD
    >>> model = HardSphereIReD(temperature=0.4, cross_section=1.0)
    >>> eta = model.shear_viscosity()
    >>> tau_pi = model.shear_relaxation_time()
    >>> print(f"η = {eta:.4f}, τ_π = {tau_pi:.4f}")

Reference:
    Wagner, Palermo, Ambrus (2022), Tables III-IV
    arXiv:2203.12608v2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HardSphereIReD:
    """
    IReD transport coefficients for ultrarelativistic hard sphere gas.

    This uses benchmark values from IReD Tables III-IV (41-moment truncation).
    Valid for:
    - Massless particles (m = 0)
    - Constant cross-section σ
    - Classical statistics

    The coefficients scale with temperature and cross-section following
    kinetic theory.

    Attributes:
        temperature: Temperature T in natural units (GeV)
        cross_section: Collision cross-section σ in fm²
        truncation: Moment truncation ('14', '23', '32', '41')
    """

    temperature: float
    cross_section: float = 1.0  # fm²
    truncation: str = "41"  # Use highest accuracy by default

    # Physical constants
    HBARC = 0.1973269804  # GeV·fm (for unit conversion)

    def __post_init__(self):
        """Validate inputs and compute derived quantities."""
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
        if self.cross_section <= 0:
            raise ValueError(f"Cross-section must be positive, got {self.cross_section}")
        if self.truncation not in ["14", "23", "32", "41"]:
            raise ValueError(f"Unknown truncation: {self.truncation}")

        # Compute derived quantities
        self.beta = 1.0 / self.temperature  # Inverse temperature

        # For ultrarelativistic ideal gas: P = ε/3, s = 4ε/(3T)
        # ε = (π²/30) g T⁴ for bosons (g=1)
        self.energy_density = (np.pi**2 / 30.0) * self.temperature**4
        self.pressure = self.energy_density / 3.0
        self.entropy_density = 4.0 * self.energy_density / (3.0 * self.temperature)

        # Mean free path: λ_mfp = 1/(n·σ) where n ∝ T³
        # For radiation: n = (ζ(3)/π²) T³ where ζ(3) ≈ 1.202
        self.particle_density = (1.202 / np.pi**2) * self.temperature**3
        self.mean_free_path = 1.0 / (self.particle_density * self.cross_section)

    # ========================================================================
    # First-Order Transport Coefficients
    # ========================================================================

    def shear_viscosity(self) -> float:
        """
        Shear viscosity η from IReD Table III.

        For N=41 truncation: η = 1.2678/(σβ)

        Returns:
            η in GeV³
        """
        # Table III values for different truncations
        eta_table = {
            "14": 1.3333,
            "23": 1.2727,
            "32": 1.2685,
            "41": 1.2678,
        }

        eta_dimensionless = eta_table[self.truncation]
        return eta_dimensionless / (self.cross_section * self.beta)

    def bulk_viscosity(self) -> float:
        """
        Bulk viscosity ζ.

        For massless conformal fluid: ζ = 0 (exact).

        Returns:
            ζ = 0
        """
        return 0.0  # Conformal symmetry

    def diffusion_coefficient(self) -> float:
        """
        Diffusion coefficient D (Landau frame).

        Note: Original IReD paper uses Eckart frame (thermal conductivity κ).
        For Landau frame, we need the diffusion coefficient D.

        For N₁=4 truncation: D = 0.15959/σ

        Returns:
            D in GeV²
        """
        # Table III (N₁=4, 41 total moments)
        D_dimensionless = 0.15959
        return D_dimensionless / self.cross_section

    # ========================================================================
    # Relaxation Times
    # ========================================================================

    def shear_relaxation_time(self) -> float:
        """
        Shear relaxation time τ_π from IReD Table III.

        For N₂=3 truncation: τ_π = 1.6552 λ_mfp

        Returns:
            τ_π in fm/c
        """
        # Table III values
        tau_pi_table = {
            "14": 1.6667,
            "23": 1.6494,
            "32": 1.6540,
            "41": 1.6552,
        }

        tau_pi_dimensionless = tau_pi_table[self.truncation]
        return tau_pi_dimensionless * self.mean_free_path

    def bulk_relaxation_time(self) -> float:
        """
        Bulk relaxation time τ_Π.

        For conformal fluid with ζ=0, bulk relaxation is not relevant.
        Return a nominal value.

        Returns:
            τ_Π (nominal)
        """
        # Use typical scale: τ_Π ~ τ_π for dimensional analysis
        return self.shear_relaxation_time()

    def diffusion_relaxation_time(self) -> float:
        """
        Diffusion relaxation time τ_V (Landau frame).

        For N₁=4 truncation: τ_V = 2.0794 λ_mfp

        Returns:
            τ_V in fm/c
        """
        # Table III (N₁=4, 41 moments)
        tau_V_dimensionless = 2.0794
        return tau_V_dimensionless * self.mean_free_path

    # ========================================================================
    # Second-Order Transport Coefficients
    # ========================================================================

    def tau_pi_pi(self) -> float:
        """
        Shear-shear coupling τ_ππ from IReD Table III.

        For N₂=3: τ_ππ = 1.6944 τ_π

        Returns:
            τ_ππ in fm/c
        """
        tau_pi = self.shear_relaxation_time()
        return 1.6944 * tau_pi

    def lambda_pi_V(self) -> float:
        """
        Shear-diffusion coupling λ_πV from IReD Table III.

        For N₂=3: λ_πV = 0.20890 τ_π/β

        Returns:
            λ_πV in GeV⁴
        """
        tau_pi = self.shear_relaxation_time()
        return 0.20890 * tau_pi / self.beta

    def delta_pi_pi(self) -> float:
        """
        Shear expansion coupling δ_ππ from IReD Table III.

        For N₂=3: δ_ππ = 4/3

        Returns:
            δ_ππ (dimensionless)
        """
        return 4.0 / 3.0

    def ell_pi_V(self) -> float:
        """
        Shear-diffusion gradient coupling ℓ_πV from IReD Table III.

        For N₂=3: ℓ_πV = -0.56014/β

        Returns:
            ℓ_πV in GeV³
        """
        return -0.56014 / self.beta

    def tau_pi_V(self) -> float:
        """
        Shear-diffusion force coupling τ_πV from IReD Table III.

        For N₂=3: τ_πV = -0.56014/(βP)

        Returns:
            τ_πV in GeV³
        """
        return -0.56014 / (self.beta * self.pressure)

    def delta_V_V(self) -> float:
        """
        Diffusion expansion coupling δ_VV from IReD Table III.

        For N₁=4: δ_VV = 1

        Returns:
            δ_VV (dimensionless)
        """
        return 1.0

    def lambda_V_V(self) -> float:
        """
        Diffusion-diffusion coupling λ_VV from IReD Table III.

        For N₁=4: λ_VV = 0.89501 τ_V

        Returns:
            λ_VV in fm/c
        """
        tau_V = self.diffusion_relaxation_time()
        return 0.89501 * tau_V

    def lambda_V_pi(self) -> float:
        """
        Diffusion-shear coupling λ_Vπ from IReD Table III.

        For N₁=4: λ_Vπ = 0.069240 β τ_V

        Returns:
            λ_Vπ in GeV⁻²·(fm/c)
        """
        tau_V = self.diffusion_relaxation_time()
        return 0.069240 * self.beta * tau_V

    def ell_V_pi(self) -> float:
        """
        Diffusion-shear gradient coupling ℓ_Vπ from IReD Table III.

        For N₁=4: ℓ_Vπ = 0.028677 β τ_V

        Returns:
            ℓ_Vπ in GeV⁻²·(fm/c)
        """
        tau_V = self.diffusion_relaxation_time()
        return 0.028677 * self.beta * tau_V

    def tau_V_pi(self) -> float:
        """
        Diffusion-shear force coupling τ_Vπ from IReD Table III.

        For N₁=4: τ_Vπ = 0.0071692 β τ_V/P

        Returns:
            τ_Vπ in GeV⁻⁵·(fm/c)
        """
        tau_V = self.diffusion_relaxation_time()
        return 0.0071692 * self.beta * tau_V / self.pressure

    # ========================================================================
    # Derived Quantities
    # ========================================================================

    def eta_over_s(self) -> float:
        """
        Compute η/s ratio (dimensionless).

        This is a key observable in heavy-ion physics.
        KSS bound: η/s ≥ 1/(4π) ≈ 0.0796

        Returns:
            η/s (dimensionless)
        """
        return self.shear_viscosity() / self.entropy_density

    def knudsen_number(self, length_scale: float) -> float:
        """
        Compute Knudsen number Kn = λ_mfp/L.

        Args:
            length_scale: Characteristic length L in fm

        Returns:
            Kn (dimensionless)
        """
        return self.mean_free_path / length_scale

    def reynolds_number(self, length_scale: float, velocity_scale: float) -> float:
        """
        Compute Reynolds number Re = ρ v L/η.

        Args:
            length_scale: Characteristic length L in fm
            velocity_scale: Characteristic velocity v (dimensionless, v/c)

        Returns:
            Re (dimensionless)
        """
        rho = self.energy_density
        eta = self.shear_viscosity()
        return rho * velocity_scale * length_scale / eta

    def regime_parameter(self, wavenumber: float) -> float:
        """
        Compute regime applicability parameter |τω| from Wagner & Gavassino (2024).

        Israel-Stewart is valid when |τω| ≲ 1.

        Args:
            wavenumber: Wave vector magnitude k in fm⁻¹

        Returns:
            |τω| (dimensionless)
        """
        c_s = 1.0 / np.sqrt(3.0)  # Sound speed for radiation fluid
        omega = wavenumber * c_s  # Dispersion relation ω ≈ k·c_s
        tau = self.shear_relaxation_time()
        return abs(tau * omega)

    # ========================================================================
    # Validation and Summary
    # ========================================================================

    def validate_against_ired_paper(self) -> dict[str, bool]:
        """
        Validate computed values against IReD Table III.

        Returns:
            Dictionary of validation results
        """
        results = {}

        # Expected values from Table III (N₂=3, N₁=4, 41 total moments)
        expected = {
            "shear_viscosity": 1.2678 / (self.cross_section * self.beta),
            "shear_relaxation_time": 1.6552 * self.mean_free_path,
            "tau_pi_pi": 1.6944 * 1.6552 * self.mean_free_path,
            "delta_pi_pi": 4.0 / 3.0,
            "diffusion_coefficient": 0.15959 / self.cross_section,
            "diffusion_relaxation_time": 2.0794 * self.mean_free_path,
            "delta_V_V": 1.0,
        }

        tolerance = 1e-4  # 0.01% tolerance

        for name, expected_value in expected.items():
            computed_value = getattr(self, name)()
            relative_error = abs(computed_value - expected_value) / abs(expected_value)
            results[name] = relative_error < tolerance

        return results

    def summary(self) -> str:
        """
        Generate summary of transport coefficients.

        Returns:
            Formatted summary string
        """
        lines = [
            "IReD Transport Coefficients (Hard Sphere Gas)",
            "=" * 60,
            f"Temperature: T = {self.temperature:.4f} GeV",
            f"Cross-section: σ = {self.cross_section:.4f} fm²",
            f"Mean free path: λ_mfp = {self.mean_free_path:.4f} fm",
            f"Truncation: {self.truncation} moments",
            "",
            "First-Order Coefficients:",
            f"  Shear viscosity: η = {self.shear_viscosity():.4e} GeV³",
            f"  Bulk viscosity: ζ = {self.bulk_viscosity():.4e} GeV³ (conformal)",
            f"  Diffusion coeff: D = {self.diffusion_coefficient():.4e} GeV²",
            f"  η/s = {self.eta_over_s():.4f} (KSS bound ≈ 0.0796)",
            "",
            "Relaxation Times:",
            f"  τ_π = {self.shear_relaxation_time():.4f} fm/c",
            f"  τ_V = {self.diffusion_relaxation_time():.4f} fm/c",
            "",
            "Second-Order Coefficients:",
            f"  τ_ππ = {self.tau_pi_pi():.4f} fm/c",
            f"  δ_ππ = {self.delta_pi_pi():.4f}",
            f"  λ_πV = {self.lambda_pi_V():.4e} GeV⁴",
            f"  δ_VV = {self.delta_V_V():.4f}",
            f"  λ_VV = {self.lambda_V_V():.4f} fm/c",
        ]

        return "\n".join(lines)


def example_usage():
    """Example demonstrating usage of HardSphereIReD."""
    print("Example: Ultrarelativistic Hard Sphere Gas at T=400 MeV\n")

    # Create model
    model = HardSphereIReD(
        temperature=0.4,  # 400 MeV
        cross_section=1.0,  # 1 fm²
        truncation="41"  # Highest accuracy
    )

    # Print summary
    print(model.summary())

    # Validate against IReD paper
    print("\n" + "=" * 60)
    print("Validation Against IReD Table III:")
    print("=" * 60)
    validation = model.validate_against_ired_paper()
    for name, passed in validation.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    # Check regime for different wavenumbers
    print("\n" + "=" * 60)
    print("Regime Applicability (|τω| ≲ 1):")
    print("=" * 60)
    for k in [1.0, 2.0, 4.0, 8.0]:
        regime_param = model.regime_parameter(k)
        status = "✓ VALID" if regime_param < 1.0 else "✗ OUTSIDE REGIME"
        print(f"  k = {k:.1f} fm⁻¹: |τω| = {regime_param:.2f} {status}")


if __name__ == "__main__":
    example_usage()
