#!/usr/bin/env python3
"""
Test script for memory optimization improvements in the Israel-Stewart spectral solver.

This script compares the original methods with memory-optimized versions
to demonstrate reduced memory allocation and improved performance.
"""

import numpy as np
import time
import tracemalloc

from israel_stewart.core.memory_optimization import (
    memory_usage_report, get_array_pool, get_fft_manager,
    memory_optimized_context
)
from israel_stewart.core.performance import (
    detailed_performance_report, reset_performance_stats
)


def simulate_spectral_operations():
    """Simulate spectral operations to test memory optimization."""

    # Import here to avoid import issues during testing
    from israel_stewart.core import SpacetimeGrid, MinkowskiMetric, TransportCoefficients
    from israel_stewart.core.fields import ISFieldConfiguration
    from israel_stewart.solvers.spectral import SpectralISolver

    print("Setting up spectral solver with memory optimization...")

    # Create test configuration
    grid = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi)],
        grid_points=(8, 32, 32, 16)  # Moderate size for testing
    )

    fields = ISFieldConfiguration(grid)
    transport_coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05
    )

    # Initialize solver (this will set up memory optimization)
    solver = SpectralISolver(grid, fields, transport_coeffs)

    return solver, fields


def test_fft_memory_optimization():
    """Test FFT memory optimization."""
    print("\n🔄 Testing FFT Memory Optimization")
    print("-" * 40)

    solver, fields = simulate_spectral_operations()

    # Test with different array sizes
    test_shapes = [
        (32, 32, 16),
        (64, 64, 32),
    ]

    for shape in test_shapes:
        print(f"\nTesting FFT with shape {shape}:")

        # Create test data
        test_field = np.random.rand(*shape)

        # Test original FFT
        tracemalloc.start()
        start_time = time.perf_counter()

        for _ in range(5):  # Multiple calls to test caching
            result1 = solver.adaptive_fft(test_field)

        original_time = time.perf_counter() - start_time
        original_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Test memory-optimized FFT
        tracemalloc.start()
        start_time = time.perf_counter()

        for _ in range(5):  # Multiple calls to test workspace reuse
            result2 = solver.memory_optimized_fft(test_field)

        optimized_time = time.perf_counter() - start_time
        optimized_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Compare results
        print(f"  Original FFT:   {original_time:.4f}s, {original_memory/(1024**2):.1f}MB peak")
        print(f"  Optimized FFT:  {optimized_time:.4f}s, {optimized_memory/(1024**2):.1f}MB peak")
        print(f"  Speedup:        {original_time/optimized_time:.2f}x")
        print(f"  Memory reduction: {(original_memory-optimized_memory)/original_memory*100:.1f}%")

        # Verify accuracy
        error = np.max(np.abs(result1 - result2))
        print(f"  Max difference: {error:.2e}")


def test_divergence_memory_optimization():
    """Test divergence computation memory optimization."""
    print("\n∇ Testing Divergence Memory Optimization")
    print("-" * 45)

    solver, fields = simulate_spectral_operations()

    # Test with vector field
    test_shape = (32, 32, 16, 3)  # Vector field
    test_vector_field = np.random.rand(*test_shape)

    print(f"Testing divergence with shape {test_shape[:-1]} (3-vector):")

    # Test original divergence
    tracemalloc.start()
    start_time = time.perf_counter()

    for _ in range(3):  # Multiple calls
        result1 = solver.spatial_divergence(test_vector_field)

    original_time = time.perf_counter() - start_time
    original_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Test memory-optimized divergence
    tracemalloc.start()
    start_time = time.perf_counter()

    for _ in range(3):  # Multiple calls to test array pool
        result2 = solver.memory_optimized_divergence(test_vector_field)

    optimized_time = time.perf_counter() - start_time
    optimized_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Compare results
    print(f"  Original divergence:   {original_time:.4f}s, {original_memory/(1024**2):.1f}MB peak")
    print(f"  Optimized divergence:  {optimized_time:.4f}s, {optimized_memory/(1024**2):.1f}MB peak")
    print(f"  Speedup:               {original_time/optimized_time:.2f}x")
    print(f"  Memory reduction:      {(original_memory-optimized_memory)/original_memory*100:.1f}%")

    # Verify accuracy
    error = np.max(np.abs(result1 - result2))
    print(f"  Max difference:        {error:.2e}")


def test_array_pool_efficiency():
    """Test array pool efficiency."""
    print("\n📦 Testing Array Pool Efficiency")
    print("-" * 35)

    array_pool = get_array_pool()
    array_pool.clear_pool()  # Start fresh

    # Test array reuse
    test_shapes = [
        (64, 64, 32),
        (32, 32, 16),
        (64, 64, 32),  # Repeat to test reuse
        (128, 128, 64),
        (32, 32, 16),  # Repeat to test reuse
    ]

    arrays = []
    print("Requesting arrays:")

    for i, shape in enumerate(test_shapes):
        array = array_pool.get_array(shape, np.float64)
        arrays.append(array)
        print(f"  Array {i+1}: shape {shape} - {'New' if shape not in [s.shape for s in arrays[:-1]] else 'Reused'}")

    # Return arrays to pool
    print("\nReturning arrays to pool...")
    for array in arrays:
        array_pool.return_array(array)

    # Test reuse efficiency
    print("\nRequesting same arrays again:")
    for i, shape in enumerate(test_shapes):
        array = array_pool.get_array(shape, np.float64)
        array_pool.return_array(array)

    # Get efficiency stats
    stats = array_pool.get_efficiency_stats()
    print(f"\nArray Pool Efficiency:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  New allocations: {stats['allocations']}")
    print(f"  Array reuses: {stats['reuses']}")
    print(f"  Reuse rate: {stats['reuse_rate']:.1%}")
    print(f"  Arrays in pool: {stats['total_pooled_arrays']}")


def test_fft_workspace_efficiency():
    """Test FFT workspace manager efficiency."""
    print("\n⚡ Testing FFT Workspace Manager")
    print("-" * 35)

    fft_manager = get_fft_manager()
    fft_manager.clear_cache()  # Start fresh

    # Test workspace reuse
    test_shapes = [
        (64, 64, 32),
        (32, 32, 16),
        (64, 64, 32),  # Repeat to test reuse
        (128, 128, 64),
        (32, 32, 16),  # Repeat to test reuse
    ]

    print("Requesting FFT workspaces:")
    for i, shape in enumerate(test_shapes):
        workspace = fft_manager.get_workspace(shape, np.complex128)
        print(f"  Workspace {i+1}: shape {shape}")

    # Get efficiency stats
    stats = fft_manager.get_efficiency_stats()
    print(f"\nFFT Workspace Efficiency:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Workspaces cached: {stats['workspaces_cached']}")
    print(f"  Memory usage: {stats['memory_usage_mb']:.1f}MB")


def run_memory_optimization_test():
    """Run comprehensive memory optimization test."""
    print("🧠 MEMORY OPTIMIZATION TEST SUITE")
    print("=" * 50)

    # Reset performance stats
    reset_performance_stats()

    try:
        # Test individual components
        test_fft_memory_optimization()
        test_divergence_memory_optimization()
        test_array_pool_efficiency()
        test_fft_workspace_efficiency()

        # Generate comprehensive report
        print("\n📊 COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 45)

        detailed_report = detailed_performance_report()
        memory_report = memory_usage_report()

        # Summary statistics
        summary = detailed_report["summary"]
        print(f"\nOperation Summary:")
        print(f"  Total operations: {summary['total_operations']}")
        print(f"  Total time: {summary['total_time']:.3f}s")
        print(f"  Memory optimized operations: {len([op for op in summary['slowest_operations'] if 'memory_optimized' in op[0]])}")

        # Memory optimization statistics
        print(f"\nMemory Optimization Stats:")
        array_stats = memory_report['array_pool']
        print(f"  Array pool reuse rate: {array_stats['reuse_rate']:.1%}")
        print(f"  Arrays in pool: {array_stats['total_pooled_arrays']}")

        fft_stats = memory_report['fft_workspace']
        print(f"  FFT workspace hit rate: {fft_stats['hit_rate']:.1%}")
        print(f"  FFT memory usage: {fft_stats['memory_usage_mb']:.1f}MB")

        # Top optimization targets
        print(f"\nTop Optimization Targets:")
        targets = detailed_report["optimization_targets"][:3]
        for i, target in enumerate(targets, 1):
            print(f"  {i}. {target['operation']} ({target['type']})")
            print(f"     {target['recommendation']}")

        print("\n✅ Memory optimization test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_memory_optimization_test()