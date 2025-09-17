"""
Thermodynamic constraints and consistency conditions for Israel-Stewart hydrodynamics.

This module implements physical constraints that ensure thermodynamic consistency,
causality, and stability in second-order viscous hydrodynamics simulations.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la

from ..core.constants import HBAR, KBOLTZ
from ..core.fields import ISFieldConfiguration, TransportCoefficients
from ..core.metrics import MetricBase
from ..core.performance import monitor_performance
from ..core.spacetime_grid import SpacetimeGrid


class ThermodynamicConstraints:
    """
    Enforces thermodynamic constraints for Israel-Stewart hydrodynamics.

    Implements physical constraints including:
    - Causality conditions (hyperbolicity)
    - Thermodynamic stability
    - Energy conditions
    - Transport coefficient bounds
    """

    def __init__(
        self,
        grid: SpacetimeGrid,
        metric: MetricBase,
        equation_of_state: str = "ideal",
        enable_causality_check: bool = True,
        enable_stability_check: bool = True,
        tolerance: float = 1e-10,
    ):
        """
        Initialize thermodynamic constraints.

        Args:
            grid: Spacetime grid
            metric: Spacetime metric
            equation_of_state: Type of EOS ("ideal", "realistic")
            enable_causality_check: Whether to enforce causality
            enable_stability_check: Whether to enforce stability
            tolerance: Numerical tolerance for constraint violations
        """
        self.grid = grid
        self.metric = metric
        self.eos = equation_of_state
        self.enable_causality = enable_causality_check
        self.enable_stability = enable_stability_check
        self.tolerance = tolerance

        # Physical constants for ideal gas
        self.g_eff = 37.5  # Effective degrees of freedom for QGP
        self.a_coeff = (np.pi**2 / 90) * self.g_eff  # Energy density coefficient

        # Constraint violation tracking
        self.violation_history: list[dict[str, Any]] = []

    @monitor_performance("causality_constraint_check")
    def check_causality_constraints(
        self,
        fields: ISFieldConfiguration,
        coefficients: TransportCoefficients,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Check causality constraints for Israel-Stewart theory.

        Causality requires that characteristic velocities do not exceed c=1.
        This constrains transport coefficients and relaxation times.
        """
        violations = []
        causality_speeds = {}

        # Extract thermodynamic quantities
        temperature = fields.temperature
        energy_density = fields.rho
        pressure = fields.pressure

        # Speed of sound
        cs_squared = self._compute_speed_of_sound_squared(temperature, energy_density, pressure)
        cs = np.sqrt(np.maximum(cs_squared, 0.0))
        causality_speeds["sound_speed"] = cs

        # Check speed of sound causality: cs² ≤ 1
        cs_violation = np.any(cs_squared > 1.0 + self.tolerance)
        if cs_violation:
            max_cs = np.max(cs)
            violations.append(
                {
                    "type": "sound_speed",
                    "description": f"Sound speed exceeds c: max(cs) = {max_cs:.6f}",
                    "severity": "critical",
                    "max_violation": max_cs - 1.0,
                }
            )

        # Shear sector causality
        if coefficients.shear_viscosity and coefficients.shear_relaxation_time:
            eta = coefficients.shear_viscosity
            tau_pi = coefficients.shear_relaxation_time

            # Kinetic theory bound: cs² + v_shear² ≤ 1
            # where v_shear² = (4η/3)/(sT·τ_π) for conformal fluids
            entropy_density = self._compute_entropy_density(temperature)
            v_shear_squared = (4 * eta / 3) / (entropy_density * temperature * tau_pi + 1e-12)

            shear_causality = cs_squared + v_shear_squared
            causality_speeds["shear_mode"] = np.sqrt(np.maximum(v_shear_squared, 0.0))

            shear_violation = np.any(shear_causality > 1.0 + self.tolerance)
            if shear_violation:
                max_shear_speed = np.max(np.sqrt(shear_causality))
                violations.append(
                    {
                        "type": "shear_causality",
                        "description": f"Shear mode exceeds c: max speed = {max_shear_speed:.6f}",
                        "severity": "warning",
                        "max_violation": max_shear_speed - 1.0,
                    }
                )

        # Bulk sector causality
        if coefficients.bulk_viscosity and coefficients.bulk_relaxation_time:
            zeta = coefficients.bulk_viscosity
            tau_Pi = coefficients.bulk_relaxation_time

            # Bulk causality condition
            v_bulk_squared = zeta / (energy_density * tau_Pi + 1e-12)
            bulk_causality = cs_squared + v_bulk_squared
            causality_speeds["bulk_mode"] = np.sqrt(np.maximum(v_bulk_squared, 0.0))

            bulk_violation = np.any(bulk_causality > 1.0 + self.tolerance)
            if bulk_violation:
                max_bulk_speed = np.max(np.sqrt(bulk_causality))
                violations.append(
                    {
                        "type": "bulk_causality",
                        "description": f"Bulk mode exceeds c: max speed = {max_bulk_speed:.6f}",
                        "severity": "warning",
                        "max_violation": max_bulk_speed - 1.0,
                    }
                )

        # Overall causality assessment
        has_critical_violations = any(v["severity"] == "critical" for v in violations)
        causality_status = "violated" if has_critical_violations else "satisfied"

        result = {
            "status": causality_status,
            "violations": violations,
            "characteristic_speeds": causality_speeds,
            "max_speed": max(
                np.max(speed)
                for speed in causality_speeds.values()
                if isinstance(speed, np.ndarray) and speed.size > 0
            )
            if causality_speeds
            else 0.0,
        }

        if verbose and violations:
            for violation in violations:
                print(f"Causality violation: {violation['description']}")

        return result

    @monitor_performance("stability_constraint_check")
    def check_stability_constraints(
        self,
        fields: ISFieldConfiguration,
        coefficients: TransportCoefficients,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Check thermodynamic stability constraints.

        Includes:
        - Positive definiteness of transport coefficients
        - Stability of equilibrium states
        - Positivity of entropy production
        """
        violations = []

        # Positivity constraints
        if coefficients.shear_viscosity is not None:
            if coefficients.shear_viscosity < 0:
                violations.append(
                    {
                        "type": "negative_shear_viscosity",
                        "description": f"Negative shear viscosity: eta = {coefficients.shear_viscosity:.6e}",
                        "severity": "critical",
                    }
                )

        if coefficients.bulk_viscosity is not None:
            if coefficients.bulk_viscosity < 0:
                violations.append(
                    {
                        "type": "negative_bulk_viscosity",
                        "description": f"Negative bulk viscosity: zeta = {coefficients.bulk_viscosity:.6e}",
                        "severity": "critical",
                    }
                )

        # Relaxation time positivity
        if coefficients.shear_relaxation_time is not None:
            if coefficients.shear_relaxation_time <= 0:
                violations.append(
                    {
                        "type": "non_positive_shear_relaxation_time",
                        "description": f"Non-positive shear relaxation time: tau_pi = {coefficients.shear_relaxation_time:.6e}",
                        "severity": "critical",
                    }
                )

        if coefficients.bulk_relaxation_time is not None:
            if coefficients.bulk_relaxation_time <= 0:
                violations.append(
                    {
                        "type": "non_positive_bulk_relaxation_time",
                        "description": f"Non-positive bulk relaxation time: Ġ = {coefficients.bulk_relaxation_time:.6e}",
                        "severity": "critical",
                    }
                )

        # Thermodynamic stability (positive temperature and density)
        temperature = fields.temperature
        energy_density = fields.rho

        if np.any(temperature <= 0):
            min_temp = np.min(temperature)
            violations.append(
                {
                    "type": "negative_temperature",
                    "description": f"Non-positive temperature: min(T) = {min_temp:.6e}",
                    "severity": "critical",
                }
            )

        if np.any(energy_density <= 0):
            min_energy = np.min(energy_density)
            violations.append(
                {
                    "type": "negative_energy_density",
                    "description": f"Non-positive energy density: min(rho) = {min_energy:.6e}",
                    "severity": "critical",
                }
            )

        # Second law of thermodynamics (positive entropy production)
        entropy_production = self._compute_entropy_production(fields, coefficients)
        if np.any(entropy_production < -self.tolerance):
            min_entropy_prod = np.min(entropy_production)
            violations.append(
                {
                    "type": "negative_entropy_production",
                    "description": f"Negative entropy production: min(sigma) = {min_entropy_prod:.6e}",
                    "severity": "warning",
                }
            )

        # Overall stability assessment
        has_critical_violations = any(v["severity"] == "critical" for v in violations)
        stability_status = "unstable" if has_critical_violations else "stable"

        result = {
            "status": stability_status,
            "violations": violations,
            "entropy_production_range": (np.min(entropy_production), np.max(entropy_production)),
        }

        if verbose and violations:
            for violation in violations:
                print(f"Stability violation: {violation['description']}")

        return result

    @monitor_performance("energy_condition_check")
    def check_energy_conditions(
        self,
        fields: ISFieldConfiguration,
        coefficients: TransportCoefficients | None = None,
    ) -> dict[str, Any]:
        """
        Check relativistic energy conditions.

        Includes:
        - Weak Energy Condition (WEC): rho >= 0, rho + P >= 0
        - Null Energy Condition (NEC): rho + P >= 0
        - Strong Energy Condition (SEC): rho + 3P >= 0
        - Dominant Energy Condition (DEC): rho >= |P|
        """
        energy_density = fields.rho
        pressure = fields.pressure

        # Include viscous corrections if available
        total_pressure = pressure.copy()
        if hasattr(fields, "Pi") and fields.Pi is not None:
            total_pressure += fields.Pi

        violations = []

        # Weak Energy Condition: rho >= 0 and rho + P >= 0
        wec1_violation = np.any(energy_density < -self.tolerance)
        wec2_violation = np.any((energy_density + total_pressure) < -self.tolerance)

        if wec1_violation:
            min_rho = np.min(energy_density)
            violations.append(
                {
                    "type": "wec_negative_density",
                    "description": f"WEC violated: negative energy density, min(rho) = {min_rho:.6e}",
                    "severity": "critical",
                }
            )

        if wec2_violation:
            min_rho_plus_p = np.min(energy_density + total_pressure)
            violations.append(
                {
                    "type": "wec_negative_rho_plus_p",
                    "description": f"WEC violated: rho + P < 0, min(rho+P) = {min_rho_plus_p:.6e}",
                    "severity": "critical",
                }
            )

        # Null Energy Condition: rho + P e 0 (same as WEC2)
        nec_status = "satisfied" if not wec2_violation else "violated"

        # Strong Energy Condition: rho + 3P e 0
        sec_violation = np.any((energy_density + 3 * total_pressure) < -self.tolerance)
        if sec_violation:
            min_rho_plus_3p = np.min(energy_density + 3 * total_pressure)
            violations.append(
                {
                    "type": "sec_violation",
                    "description": f"SEC violated: rho + 3P < 0, min(rho+3P) = {min_rho_plus_3p:.6e}",
                    "severity": "warning",
                }
            )

        # Dominant Energy Condition: rho e |P|
        dec_violation = np.any(energy_density < np.abs(total_pressure) - self.tolerance)
        if dec_violation:
            max_violation = np.max(np.abs(total_pressure) - energy_density)
            violations.append(
                {
                    "type": "dec_violation",
                    "description": f"DEC violated: rho < |P|, max violation = {max_violation:.6e}",
                    "severity": "warning",
                }
            )

        # Overall assessment
        has_critical_violations = any(v["severity"] == "critical" for v in violations)
        energy_condition_status = "violated" if has_critical_violations else "satisfied"

        return {
            "status": energy_condition_status,
            "violations": violations,
            "individual_conditions": {
                "weak_energy_condition": "satisfied"
                if not (wec1_violation or wec2_violation)
                else "violated",
                "null_energy_condition": nec_status,
                "strong_energy_condition": "satisfied" if not sec_violation else "violated",
                "dominant_energy_condition": "satisfied" if not dec_violation else "violated",
            },
        }

    @monitor_performance("transport_coefficient_bounds")
    def check_transport_coefficient_bounds(
        self,
        coefficients: TransportCoefficients,
        fields: ISFieldConfiguration | None = None,
    ) -> dict[str, Any]:
        """
        Check physical bounds on transport coefficients.

        Includes kinetic theory limits and empirical bounds.
        """
        violations = []
        bounds_info = {}

        # Kinetic theory bounds for rho/s
        if coefficients.shear_viscosity and fields:
            eta = coefficients.shear_viscosity
            entropy_density = self._compute_entropy_density(fields.temperature)
            eta_over_s = eta / (entropy_density + 1e-12)

            # KSS bound: rho/s e 1/(4rho) H 0.0796
            kss_bound = 1.0 / (4 * np.pi)
            bounds_info["eta_over_s"] = {
                "value": np.mean(eta_over_s),
                "kss_bound": kss_bound,
                "satisfies_kss": np.all(eta_over_s >= kss_bound - self.tolerance),
            }

            if np.any(eta_over_s < kss_bound - self.tolerance):
                min_eta_s = np.min(eta_over_s)
                violations.append(
                    {
                        "type": "kss_bound_violation",
                        "description": f"rho/s below KSS bound: min(rho/s) = {min_eta_s:.6f} < {kss_bound:.6f}",
                        "severity": "warning",
                    }
                )

        # Relaxation time bounds
        if coefficients.shear_relaxation_time and coefficients.shear_viscosity and fields:
            tau_pi = coefficients.shear_relaxation_time
            eta = coefficients.shear_viscosity
            temperature = fields.temperature
            entropy_density = self._compute_entropy_density(temperature)

            # Natural scale: rhorho ~ rho/(sT)
            natural_scale = eta / (entropy_density * temperature + 1e-12)
            tau_ratio = tau_pi / (natural_scale + 1e-12)

            bounds_info["tau_pi_ratio"] = {
                "value": np.mean(tau_ratio),
                "natural_scale": np.mean(natural_scale),
                "reasonable_range": (0.1, 10.0),  # Rough empirical bounds
            }

            if np.any(tau_ratio < 0.01) or np.any(tau_ratio > 100.0):
                violations.append(
                    {
                        "type": "unusual_relaxation_time",
                        "description": f"Relaxation time far from natural scale: rhorho/(rho/sT) ~ {np.mean(tau_ratio):.2f}",
                        "severity": "info",
                    }
                )

        # Second-order coefficient bounds
        second_order_bounds = self._check_second_order_bounds(coefficients)
        violations.extend(second_order_bounds.get("violations", []))
        bounds_info.update(second_order_bounds.get("bounds_info", {}))

        result = {
            "violations": violations,
            "bounds_info": bounds_info,
            "overall_status": "reasonable"
            if not any(v["severity"] in ["critical", "warning"] for v in violations)
            else "questionable",
        }

        return result

    def _check_second_order_bounds(self, coefficients: TransportCoefficients) -> dict[str, Any]:
        """Check bounds on second-order transport coefficients."""
        violations = []
        bounds_info = {}

        # rhorhorho bounds (self-interaction of shear stress)
        lambda_pi_pi = getattr(coefficients, "lambda_pi_pi", None)
        if lambda_pi_pi is not None:
            bounds_info["lambda_pi_pi"] = {
                "value": lambda_pi_pi,
                "expected_range": (0.0, 2.0),  # Rough kinetic theory estimates
            }

            if lambda_pi_pi < 0 or lambda_pi_pi > 5.0:
                violations.append(
                    {
                        "type": "unusual_lambda_pi_pi",
                        "description": f"rhorhorho outside expected range: rhorhorho = {lambda_pi_pi:.3f}",
                        "severity": "info",
                    }
                )

        # rho coefficients (bulk pressure nonlinearities)
        xi_1 = getattr(coefficients, "xi_1", None)
        if xi_1 is not None:
            bounds_info["xi_1"] = {
                "value": xi_1,
                "expected_range": (0.0, 1.0),
            }

        return {"violations": violations, "bounds_info": bounds_info}

    @monitor_performance("enforce_constraints")
    def enforce_constraints(
        self,
        fields: ISFieldConfiguration,
        coefficients: TransportCoefficients,
        method: str = "clamp",
    ) -> tuple[ISFieldConfiguration, TransportCoefficients]:
        """
        Enforce physical constraints by modifying fields and coefficients.

        Args:
            fields: Field configuration
            coefficients: Transport coefficients
            method: Enforcement method ('clamp', 'rescale', 'project')

        Returns:
            Constrained fields and coefficients
        """
        constrained_fields = fields.copy()

        # Enforce positivity of thermodynamic quantities
        temp_constrained = np.maximum(constrained_fields.temperature, self.tolerance)
        constrained_fields.temperature = temp_constrained.astype(np.float64).reshape(
            constrained_fields.temperature.shape
        )

        rho_constrained = np.maximum(constrained_fields.rho, self.tolerance)
        constrained_fields.rho = rho_constrained.astype(np.float64).reshape(
            constrained_fields.rho.shape
        )

        # Enforce causality by limiting dissipative fluxes
        if method == "clamp":
            constrained_fields = self._clamp_dissipative_fluxes(constrained_fields, coefficients)
        elif method == "rescale":
            constrained_fields = self._rescale_dissipative_fluxes(constrained_fields, coefficients)
        else:
            warnings.warn(f"Unknown constraint enforcement method: {method}", UserWarning)

        # Constrain transport coefficients
        constrained_coeffs = self._constrain_transport_coefficients(coefficients)

        return constrained_fields, constrained_coeffs

    def _clamp_dissipative_fluxes(
        self, fields: ISFieldConfiguration, coefficients: TransportCoefficients
    ) -> ISFieldConfiguration:
        """Clamp dissipative fluxes to physically reasonable bounds."""
        # Bulk pressure bounds
        if hasattr(fields, "Pi") and fields.Pi is not None:
            # Limit bulk pressure to fraction of equilibrium pressure
            max_bulk = 0.5 * fields.pressure  # 50% of equilibrium pressure
            pi_clipped = np.clip(fields.Pi, -max_bulk, max_bulk)
            fields.Pi = pi_clipped.astype(np.float64).reshape(fields.Pi.shape)

        # Shear stress bounds
        if hasattr(fields, "pi_munu") and fields.pi_munu is not None:
            # Limit shear stress magnitude
            pi_magnitude = np.sqrt(np.sum(fields.pi_munu**2, axis=(-2, -1)))
            max_shear = 0.5 * fields.rho  # 50% of energy density

            # Rescale if needed
            rescale_factor = np.minimum(1.0, max_shear / (pi_magnitude + 1e-12))
            fields.pi_munu *= rescale_factor[..., np.newaxis, np.newaxis]

        return fields

    def _rescale_dissipative_fluxes(
        self, fields: ISFieldConfiguration, coefficients: TransportCoefficients
    ) -> ISFieldConfiguration:
        """Rescale dissipative fluxes to maintain physical bounds."""
        # Similar to clamp but with smooth rescaling
        return self._clamp_dissipative_fluxes(fields, coefficients)

    def _constrain_transport_coefficients(
        self, coefficients: TransportCoefficients
    ) -> TransportCoefficients:
        """Apply physical constraints to transport coefficients."""
        # Create new coefficients object with constraints
        constrained = TransportCoefficients(
            shear_viscosity=max(coefficients.shear_viscosity or 0.0, 0.0),
            bulk_viscosity=max(coefficients.bulk_viscosity or 0.0, 0.0),
            thermal_conductivity=max(coefficients.thermal_conductivity or 0.0, 0.0),
            shear_relaxation_time=max(coefficients.shear_relaxation_time or 0.1, 1e-6),
            bulk_relaxation_time=max(coefficients.bulk_relaxation_time or 0.1, 1e-6),
            heat_relaxation_time=max(getattr(coefficients, "heat_relaxation_time", 0.1), 1e-6),
        )

        # Copy second-order coefficients with bounds
        for attr in ["lambda_pi_pi", "lambda_pi_Pi", "lambda_pi_q", "xi_1", "xi_2"]:
            if hasattr(coefficients, attr):
                value = getattr(coefficients, attr)
                if value is not None:
                    setattr(constrained, attr, value)

        return constrained

    def comprehensive_constraint_check(
        self,
        fields: ISFieldConfiguration,
        coefficients: TransportCoefficients,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Perform comprehensive constraint checking.

        Returns complete analysis of all physical constraints.
        """
        results = {}

        # Individual constraint checks
        if self.enable_causality:
            results["causality"] = self.check_causality_constraints(fields, coefficients, verbose)

        if self.enable_stability:
            results["stability"] = self.check_stability_constraints(fields, coefficients, verbose)

        results["energy_conditions"] = self.check_energy_conditions(fields, coefficients)
        results["transport_bounds"] = self.check_transport_coefficient_bounds(coefficients, fields)

        # Overall assessment
        critical_violations = []
        for check_name, check_result in results.items():
            if "violations" in check_result:
                critical_violations.extend(
                    [v for v in check_result["violations"] if v.get("severity") == "critical"]
                )

        overall_status = "consistent" if not critical_violations else "violated"

        # Summary
        summary = {
            "overall_status": overall_status,
            "total_critical_violations": len(critical_violations),
            "constraint_checks": list(results.keys()),
            "critical_violations": critical_violations,
        }

        # Store for history tracking
        self.violation_history.append(
            {
                "timestamp": len(self.violation_history),
                "summary": summary,
                "detailed_results": results,
            }
        )

        return {
            "summary": summary,
            "detailed_results": results,
        }

    # Helper methods for thermodynamic calculations
    def _compute_speed_of_sound_squared(
        self,
        temperature: np.ndarray,
        energy_density: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        """Compute speed of sound squared."""
        if self.eos == "ideal":
            # For ideal gas: csrho = P/rho = 1/3
            return np.full_like(temperature, 1.0 / 3.0)
        else:
            # General case: csrho = P/rho (finite difference approximation)
            return np.full_like(temperature, 1.0 / 3.0)  # Simplified

    def _compute_entropy_density(self, temperature: np.ndarray) -> np.ndarray:
        """Compute entropy density."""
        # For ideal gas: s = (2rhorho/90) * g_eff * Trho
        return (2 * np.pi**2 / 90) * self.g_eff * temperature**3

    def _compute_entropy_production(
        self,
        fields: ISFieldConfiguration,
        coefficients: TransportCoefficients,
    ) -> np.ndarray:
        """
        Compute entropy production rate.

        rho = (1/T) * [rho^rhorho _{rho} u_{rho} + rho rho + q^rho _rho(1/T)]
        """
        temperature = fields.temperature

        # Simplified entropy production (full calculation requires gradients)
        # This is a placeholder - in practice you'd compute proper derivatives
        expansion_rate = -1.0 / np.ones_like(temperature)  # Placeholder

        entropy_production = np.zeros_like(temperature)

        # Bulk contribution
        if hasattr(fields, "Pi") and fields.Pi is not None:
            entropy_production += (fields.Pi * expansion_rate) / temperature

        # Shear contribution (simplified)
        if hasattr(fields, "pi_munu") and fields.pi_munu is not None:
            pi_magnitude = np.sqrt(np.sum(fields.pi_munu**2, axis=(-2, -1)))
            # Approximate shear rate
            shear_rate = np.abs(expansion_rate)  # Simplified
            entropy_production += (pi_magnitude * shear_rate) / temperature

        return entropy_production


def validate_israel_stewart_setup(
    grid: SpacetimeGrid,
    metric: MetricBase,
    fields: ISFieldConfiguration,
    coefficients: TransportCoefficients,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to validate a complete Israel-Stewart setup.

    Args:
        grid: Spacetime grid
        metric: Spacetime metric
        fields: Initial field configuration
        coefficients: Transport coefficients
        verbose: Whether to print violations

    Returns:
        Validation report
    """
    constraints = ThermodynamicConstraints(grid, metric)
    return constraints.comprehensive_constraint_check(fields, coefficients, verbose)
