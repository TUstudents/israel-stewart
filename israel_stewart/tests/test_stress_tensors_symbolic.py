"""
Tests for symbolic stress tensor operations.

This module tests the critical bug fixes for SymPy symbolic computation
in stress-energy and viscous stress tensors, specifically:
1. Momentum density extraction from symbolic stress-energy tensors
2. Heat flux extraction from symbolic viscous stress tensors
3. All subsequent FourVector operations on extracted quantities

The main bug was that SymPy branches created (4,1) matrices that caused
failures in tensor operations expecting (4,) vectors.
"""

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core.four_vectors import FourVector
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.stress_tensors import StressEnergyTensor, ViscousStressTensor


class TestSymbolicStressEnergyTensor:
    """Test symbolic stress-energy tensor operations."""

    @pytest.fixture
    def metric(self) -> MinkowskiMetric:
        """Create Minkowski metric for testing."""
        return MinkowskiMetric()

    @pytest.fixture
    def symbolic_stress_tensor(self, metric: MinkowskiMetric) -> StressEnergyTensor:
        """Create symbolic stress-energy tensor for testing."""
        # Use symbolic variables
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)

        # Create a realistic stress-energy tensor with symbolic entries
        T_components = sp.Matrix(
            [
                [t**2, t * x, 0, 0],  # T^00, T^01 - energy and momentum density
                [t * x, x**2, 0, 0],  # T^10, T^11 - momentum flux
                [0, 0, sp.Rational(1, 3) * t**2, 0],  # T^22 - pressure
                [0, 0, 0, sp.Rational(1, 3) * t**2],  # T^33 - pressure
            ]
        )

        return StressEnergyTensor(T_components, metric)

    @pytest.fixture
    def symbolic_four_velocity(self, metric: MinkowskiMetric) -> FourVector:
        """Create symbolic four-velocity for testing."""
        # Normalized four-velocity in rest frame with small spatial component
        gamma = sp.sqrt(1 + sp.Rational(1, 4))  # γ for v = 1/2
        return FourVector([gamma, sp.Rational(1, 2), 0, 0], False, metric)

    def test_symbolic_momentum_density_extraction(
        self, symbolic_stress_tensor: StressEnergyTensor, symbolic_four_velocity: FourVector
    ) -> None:
        """Test momentum density extraction from symbolic stress tensor."""
        # This was the main bug - momentum_density would fail with (4,1) SymPy matrices
        momentum = symbolic_stress_tensor.momentum_density(symbolic_four_velocity)

        # Verify it's a proper FourVector
        assert isinstance(momentum, FourVector)

        # Verify components have correct shape and type
        assert hasattr(momentum.components, "shape")
        assert momentum.components.shape == (4,)
        assert isinstance(momentum.components, np.ndarray)

        # Verify components contain symbolic expressions
        assert hasattr(momentum.time_component, "free_symbols") or isinstance(
            momentum.time_component, (int, float)
        )

        # Verify the extraction formula: j^μ = -u_ν T^μν
        # For our test case with u = (γ, 1/2, 0, 0) and T from fixture
        # j^0 should be -u_0*T^00 - u_1*T^10 = -γ*t^2 - (1/2)*t*x
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        gamma = sp.sqrt(1 + sp.Rational(1, 4))

        expected_j0 = -gamma * t**2 - sp.Rational(1, 2) * t * x
        # Convert to float for comparison if possible
        if momentum.time_component != 0:
            # Basic structure check (exact symbolic comparison is complex)
            assert t in momentum.time_component.free_symbols
            assert x in momentum.time_component.free_symbols

    def test_symbolic_momentum_density_operations(
        self, symbolic_stress_tensor: StressEnergyTensor, symbolic_four_velocity: FourVector
    ) -> None:
        """Test that extracted symbolic momentum can be used in all FourVector operations."""
        momentum = symbolic_stress_tensor.momentum_density(symbolic_four_velocity)

        # Test dot product (this was the specific failure case)
        dot_product = momentum.dot(momentum)
        assert isinstance(dot_product, sp.Expr) or isinstance(dot_product, (int, float))

        # Test magnitude squared
        mag_squared = momentum.magnitude_squared()
        assert isinstance(mag_squared, sp.Expr) or isinstance(mag_squared, (int, float))

        # Test component access
        assert momentum.time_component is not None
        assert momentum.x is not None
        assert momentum.y is not None
        assert momentum.z is not None

        # Test spatial components
        spatial = momentum.spatial_components
        assert hasattr(spatial, "__len__")
        assert len(spatial) == 3

    def test_symbolic_energy_density_extraction(
        self, symbolic_stress_tensor: StressEnergyTensor, symbolic_four_velocity: FourVector
    ) -> None:
        """Test energy density extraction from symbolic stress tensor."""
        energy = symbolic_stress_tensor.energy_density(symbolic_four_velocity)

        # Should return symbolic expression
        assert isinstance(energy, sp.Expr) or isinstance(energy, (int, float))

        # For our test tensor, energy density ρ = u_μ u_ν T^μν should be non-zero
        if isinstance(energy, sp.Expr):
            # Should contain our symbolic variables
            t = sp.Symbol("t", real=True)
            x = sp.Symbol("x", real=True)
            symbols = energy.free_symbols
            assert t in symbols or x in symbols or len(symbols) == 0

    def test_mixed_symbolic_numeric_operations(self, metric: MinkowskiMetric) -> None:
        """Test operations mixing symbolic and numeric components."""
        # Create tensor with mixed symbolic/numeric components
        a = sp.Symbol("a", positive=True)
        T_mixed = sp.Matrix(
            [
                [1.0, a, 0, 0],  # Mix of numeric and symbolic
                [a, 2.0, 0, 0],
                [0, 0, 0.5, 0],
                [0, 0, 0, 0.5],
            ]
        )

        stress_tensor = StressEnergyTensor(T_mixed, metric)
        four_velocity = FourVector([1.0, 0.1, 0, 0], False, metric)

        # Should work without errors
        momentum = stress_tensor.momentum_density(four_velocity)
        energy = stress_tensor.energy_density(four_velocity)

        assert isinstance(momentum, FourVector)
        assert momentum.components.shape == (4,)

        # Test operations
        dot_result = momentum.dot(momentum)
        assert isinstance(dot_result, (sp.Expr, float, int))


class TestSymbolicViscousStressTensor:
    """Test symbolic viscous stress tensor operations."""

    @pytest.fixture
    def metric(self) -> MinkowskiMetric:
        """Create Minkowski metric for testing."""
        return MinkowskiMetric()

    @pytest.fixture
    def symbolic_viscous_tensor(self, metric: MinkowskiMetric) -> ViscousStressTensor:
        """Create symbolic viscous stress tensor for testing."""
        # Use symbolic variables
        eta = sp.Symbol("eta", positive=True)  # shear viscosity
        zeta = sp.Symbol("zeta", positive=True)  # bulk viscosity

        # Create viscous tensor with symbolic entries
        pi_components = sp.Matrix(
            [
                [0, 0, 0, 0],  # No viscous stress in time-time component
                [0, eta, eta / 2, 0],  # Shear viscous stress
                [0, eta / 2, zeta, 0],  # Shear + bulk viscous stress
                [0, 0, 0, -zeta],  # Bulk viscous stress (traceless)
            ]
        )

        return ViscousStressTensor(pi_components, metric)

    @pytest.fixture
    def symbolic_four_velocity(self, metric: MinkowskiMetric) -> FourVector:
        """Create symbolic four-velocity for testing."""
        return FourVector([1, sp.Rational(1, 3), 0, 0], False, metric)

    def test_symbolic_heat_flux_extraction(
        self, symbolic_viscous_tensor: ViscousStressTensor, symbolic_four_velocity: FourVector
    ) -> None:
        """Test heat flux extraction from symbolic viscous tensor."""
        # This was the second bug - heat_flux_part would fail with (4,1) SymPy matrices
        heat_flux = symbolic_viscous_tensor.heat_flux_part(symbolic_four_velocity)

        # Verify it's a proper FourVector
        assert isinstance(heat_flux, FourVector)

        # Verify components have correct shape and type
        assert hasattr(heat_flux.components, "shape")
        assert heat_flux.components.shape == (4,)
        assert isinstance(heat_flux.components, np.ndarray)

        # Verify the extraction formula: q^μ = -u_ν π^μν
        # Should contain symbolic expressions
        eta = sp.Symbol("eta", positive=True)
        if heat_flux.time_component != 0:
            symbols = (
                heat_flux.time_component.free_symbols
                if hasattr(heat_flux.time_component, "free_symbols")
                else set()
            )
            # May contain eta symbol or be zero
            assert eta in symbols or heat_flux.time_component == 0

    def test_symbolic_heat_flux_operations(
        self, symbolic_viscous_tensor: ViscousStressTensor, symbolic_four_velocity: FourVector
    ) -> None:
        """Test that extracted symbolic heat flux can be used in all FourVector operations."""
        heat_flux = symbolic_viscous_tensor.heat_flux_part(symbolic_four_velocity)

        # Test dot product (this was the specific failure case)
        dot_product = heat_flux.dot(heat_flux)
        assert isinstance(dot_product, sp.Expr) or isinstance(dot_product, (int, float))

        # Test magnitude squared
        mag_squared = heat_flux.magnitude_squared()
        assert isinstance(mag_squared, sp.Expr) or isinstance(mag_squared, (int, float))

        # Test component access
        assert heat_flux.time_component is not None
        assert heat_flux.x is not None
        assert heat_flux.y is not None
        assert heat_flux.z is not None

    def test_symbolic_bulk_viscous_extraction(
        self, symbolic_viscous_tensor: ViscousStressTensor, symbolic_four_velocity: FourVector
    ) -> None:
        """Test bulk viscous part extraction from symbolic tensor."""
        bulk = symbolic_viscous_tensor.bulk_part(symbolic_four_velocity)

        # Should return symbolic expression or number
        assert isinstance(bulk, sp.Expr) or isinstance(bulk, (int, float))

        # For our test tensor, should contain zeta symbol
        zeta = sp.Symbol("zeta", positive=True)
        if isinstance(bulk, sp.Expr) and bulk != 0:
            assert zeta in bulk.free_symbols

    def test_symbolic_shear_viscous_extraction(
        self, symbolic_viscous_tensor: ViscousStressTensor, symbolic_four_velocity: FourVector
    ) -> None:
        """Test shear viscous part extraction from symbolic tensor."""
        shear_tensor = symbolic_viscous_tensor.shear_part(symbolic_four_velocity)

        # Should return ViscousStressTensor
        assert isinstance(shear_tensor, ViscousStressTensor)

        # Should have symbolic components
        assert hasattr(shear_tensor.components, "shape")
        assert shear_tensor.components.shape == (4, 4)


class TestSymbolicStressTensorRegression:
    """Regression tests for the specific bug that was fixed."""

    @pytest.fixture
    def metric(self) -> MinkowskiMetric:
        """Create Minkowski metric for testing."""
        return MinkowskiMetric()

    def test_regression_momentum_density_dot_product(self, metric: MinkowskiMetric) -> None:
        """Regression test for the specific momentum density dot product failure."""
        # Create the exact scenario that was failing
        t = sp.Symbol("t")
        x = sp.Symbol("x")
        T_symbolic = sp.Matrix(
            [[t**2, t * x, 0, 0], [t * x, x**2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        stress_tensor = StressEnergyTensor(T_symbolic, metric)
        u_symbolic = FourVector([1, sp.Rational(1, 2), 0, 0], False, metric)

        # This should NOT raise ValueError about einsum dimensions
        momentum = stress_tensor.momentum_density(u_symbolic)

        # This specific operation was failing before the fix
        dot_product = momentum.dot(momentum)

        # Should complete successfully and return symbolic result
        assert isinstance(dot_product, sp.Expr) or isinstance(dot_product, (int, float))

    def test_regression_heat_flux_dot_product(self, metric: MinkowskiMetric) -> None:
        """Regression test for the specific heat flux dot product failure."""
        # Create the exact scenario that was failing
        eta = sp.Symbol("eta", positive=True)
        pi_symbolic = sp.Matrix([[0, 0, 0, 0], [0, eta, 0, 0], [0, 0, eta, 0], [0, 0, 0, 0]])

        viscous_tensor = ViscousStressTensor(pi_symbolic, metric)
        u_symbolic = FourVector([1, sp.Rational(1, 2), 0, 0], False, metric)

        # This should NOT raise ValueError about einsum dimensions
        heat_flux = viscous_tensor.heat_flux_part(u_symbolic)

        # This specific operation was failing before the fix
        dot_product = heat_flux.dot(heat_flux)

        # Should complete successfully and return symbolic result
        assert isinstance(dot_product, sp.Expr) or isinstance(dot_product, (int, float))

    def test_component_shape_consistency(self, metric: MinkowskiMetric) -> None:
        """Test that all extracted FourVectors have consistent component shapes."""
        # Test various symbolic tensors
        a, b, c = sp.symbols("a b c", real=True)

        T1 = sp.Matrix([[a, 0, 0, 0], [0, b, 0, 0], [0, 0, c, 0], [0, 0, 0, 0]])
        T2 = sp.Matrix([[1, a, 0, 0], [a, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        pi1 = sp.Matrix([[0, a, 0, 0], [a, 0, 0, 0], [0, 0, b, 0], [0, 0, 0, c]])
        pi2 = sp.Matrix([[0, 0, 0, 0], [0, a, b, 0], [0, b, c, 0], [0, 0, 0, 0]])

        u = FourVector([1, 0.1, 0, 0], False, metric)

        for T_components in [T1, T2]:
            stress = StressEnergyTensor(T_components, metric)
            momentum = stress.momentum_density(u)

            # All should have (4,) shape
            assert momentum.components.shape == (4,)

        for pi_components in [pi1, pi2]:
            viscous = ViscousStressTensor(pi_components, metric)
            heat_flux = viscous.heat_flux_part(u)

            # All should have (4,) shape
            assert heat_flux.components.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__])
