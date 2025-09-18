"""
Performance monitoring utilities for tensor operations.

This module provides performance tracking and optimization utilities
for relativistic hydrodynamics tensor computations.
"""

import functools
import time
import tracemalloc
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

    def __init__(self) -> None:
        """Initialize performance monitoring."""
        self.operation_counts: dict[str, int] = {}
        self.timing_data: dict[str, list[float]] = {}
        self.memory_usage: dict[str, list[float]] = {}
        self.warnings_issued: dict[str, bool] = {}

    def time_operation(self, operation_name: str) -> Callable[[Callable], Callable]:
        """
        Decorator to time tensor operations.

        Args:
            operation_name: Name of operation to track

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Start timing and memory tracking
                start_time = time.perf_counter()

                # Start memory tracking if tracemalloc is not already started
                memory_started_here = False
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                    memory_started_here = True

                start_memory = tracemalloc.get_traced_memory()[0]  # current memory

                # Execute function
                result = func(*args, **kwargs)

                # Stop timing and memory tracking
                end_time = time.perf_counter()
                end_memory = tracemalloc.get_traced_memory()[0]  # current memory

                # Clean up memory tracing if we started it
                if memory_started_here:
                    tracemalloc.stop()

                # Record timing
                elapsed = end_time - start_time
                if operation_name not in self.timing_data:
                    self.timing_data[operation_name] = []
                self.timing_data[operation_name].append(elapsed)

                # Record memory usage delta
                memory_delta = end_memory - start_memory
                if operation_name not in self.memory_usage:
                    self.memory_usage[operation_name] = []
                self.memory_usage[operation_name].append(memory_delta)

                # Update operation count
                self.operation_counts[operation_name] = (
                    self.operation_counts.get(operation_name, 0) + 1
                )

                # Issue performance warnings if needed
                self._check_performance_warnings(operation_name, elapsed, memory_delta)

                return result

            return wrapper

        return decorator

    def _check_performance_warnings(
        self, operation_name: str, elapsed_time: float, memory_delta: float = 0.0
    ) -> None:
        """Check if performance warnings should be issued."""
        # Warn about slow operations (>1 second)
        if elapsed_time > 1.0 and not self.warnings_issued.get(f"{operation_name}_slow", False):
            warnings.warn(
                f"Operation {operation_name} took {elapsed_time:.2f}s - consider optimization",
                stacklevel=2,
            )
            self.warnings_issued[f"{operation_name}_slow"] = True

        # Warn about high memory usage operations (>100MB)
        memory_mb = memory_delta / (1024 * 1024)
        if memory_mb > 100 and not self.warnings_issued.get(f"{operation_name}_memory", False):
            warnings.warn(
                f"Operation {operation_name} used {memory_mb:.1f}MB - consider memory optimization",
                stacklevel=2,
            )
            self.warnings_issued[f"{operation_name}_memory"] = True

        # Warn about very frequent operations (>1000 calls)
        call_count = self.operation_counts.get(operation_name, 0)
        if call_count > 1000 and not self.warnings_issued.get(f"{operation_name}_frequent", False):
            warnings.warn(
                f"Operation {operation_name} called {call_count} times - consider caching",
                stacklevel=2,
            )
            self.warnings_issued[f"{operation_name}_frequent"] = True

    def get_performance_report(self) -> dict[str, Any]:
        """
        Generate performance report.

        Returns:
            Dictionary with performance statistics
        """
        report: dict[str, Any] = {
            "operation_counts": self.operation_counts.copy(),
            "timing_statistics": {},
            "memory_statistics": {},
            "recommendations": [],
        }

        # Compute timing statistics
        for op_name, times in self.timing_data.items():
            if times:
                times_array = np.array(times)
                report["timing_statistics"][op_name] = {
                    "mean_time": float(np.mean(times_array)),
                    "std_time": float(np.std(times_array)),
                    "min_time": float(np.min(times_array)),
                    "max_time": float(np.max(times_array)),
                    "total_time": float(np.sum(times_array)),
                    "call_count": len(times),
                }

        # Compute memory statistics
        for op_name, memory_deltas in self.memory_usage.items():
            if memory_deltas:
                memory_array = np.array(memory_deltas)
                # Convert bytes to MB for readability
                memory_mb = memory_array / (1024 * 1024)
                report["memory_statistics"][op_name] = {
                    "mean_memory_mb": float(np.mean(memory_mb)),
                    "std_memory_mb": float(np.std(memory_mb)),
                    "min_memory_mb": float(np.min(memory_mb)),
                    "max_memory_mb": float(np.max(memory_mb)),
                    "total_memory_mb": float(np.sum(memory_mb)),
                    "call_count": len(memory_deltas),
                }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> list:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check for frequently called operations
        for op_name, count in self.operation_counts.items():
            if count > 100:
                recommendations.append(
                    f"Consider caching results for frequently called operation: {op_name} ({count} calls)"
                )

        # Check for slow operations
        for op_name, times in self.timing_data.items():
            if times:
                mean_time = np.mean(times)
                if mean_time > 0.1:
                    recommendations.append(
                        f"Slow operation detected: {op_name} (avg {mean_time:.3f}s)"
                    )

        # Check for memory-intensive operations
        for op_name, memory_deltas in self.memory_usage.items():
            if memory_deltas:
                mean_memory_mb = np.mean(memory_deltas) / (1024 * 1024)
                if mean_memory_mb > 50:  # More than 50MB average
                    recommendations.append(
                        f"Memory-intensive operation: {op_name} (avg {mean_memory_mb:.1f}MB per call)"
                    )

                # Check for operations with high memory variance (potential leaks)
                std_memory_mb = np.std(memory_deltas) / (1024 * 1024)
                if std_memory_mb > mean_memory_mb * 2:  # High variance
                    recommendations.append(
                        f"Unstable memory usage in {op_name} - check for memory leaks or inefficient allocation"
                    )

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


def monitor_performance(operation_name: str) -> Callable[[Callable], Callable]:
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
def suggest_einsum_optimization(einsum_string: str, *tensor_shapes: tuple[int, ...]) -> str | None:
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
        path_info = opt_einsum.contract_path(einsum_string, *dummy_arrays, optimize="optimal")

        if len(path_info) > 1 and hasattr(path_info[1], "largest_intermediate"):
            return f"Use opt_einsum for {einsum_string} - reduces memory by factor of {path_info[1].largest_intermediate:.1e}"

    except ImportError:
        # opt_einsum not available - this is expected in some environments
        pass
    except (ValueError, TypeError) as e:
        # Invalid einsum string or tensor shapes
        warnings.warn(f"Invalid einsum optimization request: {e}", stacklevel=2)
    except AttributeError:
        # path_info doesn't have expected attributes - opt_einsum version mismatch
        pass

    return None


def check_tensor_cache_efficiency(
    cache_hits: int, cache_misses: int, cache_name: str = "tensor_cache"
) -> None:
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
            warnings.warn(
                f"Low cache efficiency for {cache_name}: {hit_rate:.1%} hit rate "
                f"({cache_hits} hits, {cache_misses} misses)",
                stacklevel=2,
            )
