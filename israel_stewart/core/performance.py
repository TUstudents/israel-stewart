"""
Performance monitoring utilities for tensor operations.

This module provides performance tracking and optimization utilities
for relativistic hydrodynamics tensor computations.
"""

import functools
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np


class PerformanceMonitor:
    """
    Monitor performance of tensor operations and provide optimization hints.

    Tracks computation time, memory usage, and operation counts to help
    identify bottlenecks in relativistic hydrodynamics calculations.
    """

    def __init__(self):
        """Initialize performance monitoring."""
        self.operation_counts: dict[str, int] = {}
        self.timing_data: dict[str, list] = {}
        self.memory_usage: dict[str, list] = {}
        self.warnings_issued: dict[str, bool] = {}

    def time_operation(self, operation_name: str):
        """
        Decorator to time tensor operations.

        Args:
            operation_name: Name of operation to track

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()

                # Record timing
                elapsed = end_time - start_time
                if operation_name not in self.timing_data:
                    self.timing_data[operation_name] = []
                self.timing_data[operation_name].append(elapsed)

                # Update operation count
                self.operation_counts[operation_name] = self.operation_counts.get(operation_name, 0) + 1

                # Issue performance warnings if needed
                self._check_performance_warnings(operation_name, elapsed)

                return result
            return wrapper
        return decorator

    def _check_performance_warnings(self, operation_name: str, elapsed_time: float) -> None:
        """Check if performance warnings should be issued."""
        # Warn about slow operations (>1 second)
        if elapsed_time > 1.0 and not self.warnings_issued.get(f"{operation_name}_slow", False):
            warnings.warn(f"Operation {operation_name} took {elapsed_time:.2f}s - consider optimization", stacklevel=2)
            self.warnings_issued[f"{operation_name}_slow"] = True

        # Warn about very frequent operations (>1000 calls)
        call_count = self.operation_counts.get(operation_name, 0)
        if call_count > 1000 and not self.warnings_issued.get(f"{operation_name}_frequent", False):
            warnings.warn(f"Operation {operation_name} called {call_count} times - consider caching", stacklevel=2)
            self.warnings_issued[f"{operation_name}_frequent"] = True

    def get_performance_report(self) -> dict[str, Any]:
        """
        Generate performance report.

        Returns:
            Dictionary with performance statistics
        """
        report = {
            'operation_counts': self.operation_counts.copy(),
            'timing_statistics': {},
            'recommendations': []
        }

        # Compute timing statistics
        for op_name, times in self.timing_data.items():
            if times:
                times_array = np.array(times)
                report['timing_statistics'][op_name] = {
                    'mean_time': float(np.mean(times_array)),
                    'std_time': float(np.std(times_array)),
                    'min_time': float(np.min(times_array)),
                    'max_time': float(np.max(times_array)),
                    'total_time': float(np.sum(times_array)),
                    'call_count': len(times)
                }

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> list:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check for frequently called operations
        for op_name, count in self.operation_counts.items():
            if count > 100:
                recommendations.append(f"Consider caching results for frequently called operation: {op_name} ({count} calls)")

        # Check for slow operations
        for op_name, times in self.timing_data.items():
            if times:
                mean_time = np.mean(times)
                if mean_time > 0.1:
                    recommendations.append(f"Slow operation detected: {op_name} (avg {mean_time:.3f}s)")

        return recommendations

    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        self.operation_counts.clear()
        self.timing_data.clear()
        self.memory_usage.clear()
        self.warnings_issued.clear()


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


def monitor_performance(operation_name: str):
    """
    Decorator for monitoring performance of tensor operations.

    Args:
        operation_name: Name to use for tracking this operation

    Returns:
        Decorator function
    """
    return _global_monitor.time_operation(operation_name)


def performance_report() -> dict[str, Any]:
    """
    Get current performance report.

    Returns:
        Performance statistics and recommendations
    """
    return _global_monitor.get_performance_report()


def reset_performance_stats() -> None:
    """Reset all performance monitoring statistics."""
    _global_monitor.reset_stats()


# Optimization utilities
def suggest_einsum_optimization(einsum_string: str, *tensor_shapes) -> str | None:
    """
    Suggest optimized Einstein summation path.

    Args:
        einsum_string: Einstein summation string
        tensor_shapes: Shapes of input tensors

    Returns:
        Optimization suggestion or None
    """
    try:
        # Try to use opt_einsum if available
        import opt_einsum

        # Create dummy arrays with given shapes
        dummy_arrays = [np.zeros(shape) for shape in tensor_shapes]

        # Get optimized path
        path_info = opt_einsum.contract_path(einsum_string, *dummy_arrays, optimize='optimal')

        if len(path_info) > 1 and hasattr(path_info[1], 'largest_intermediate'):
            return f"Use opt_einsum for {einsum_string} - reduces memory by factor of {path_info[1].largest_intermediate:.1e}"

    except (ImportError, Exception):
        pass

    return None


def check_tensor_cache_efficiency(cache_hits: int, cache_misses: int, cache_name: str = "tensor_cache") -> None:
    """
    Check cache efficiency and issue warnings if needed.

    Args:
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        cache_name: Name of cache for warning messages
    """
    total_requests = cache_hits + cache_misses
    if total_requests > 100:  # Only check efficiency after reasonable sample size
        hit_rate = cache_hits / total_requests
        if hit_rate < 0.5:  # Less than 50% hit rate
            warnings.warn(f"Low cache efficiency for {cache_name}: {hit_rate:.1%} hit rate "
                         f"({cache_hits} hits, {cache_misses} misses)", stacklevel=2)
