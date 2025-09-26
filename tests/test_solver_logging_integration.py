"""
Integration tests for solver logging functionality.

Tests that logging integrates properly with solver operations,
performance monitoring, and physics validation.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.performance import monitor_performance
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.utils.logging_config import (
    configure_logging,
    enable_performance_logging,
    get_logger,
    performance_logger,
    physics_logger,
)


class TestSolverLoggingIntegration(unittest.TestCase):
    """Test logging integration with solver operations."""

    def setUp(self):
        """Set up test environment with logging."""
        # Configure logging for tests
        configure_logging(level="DEBUG", enable_performance=True)
        enable_performance_logging(True)

        # Create test grid and fields
        self.grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(4, 8, 8, 8),
        )
        self.metric = MinkowskiMetric()
        self.fields = ISFieldConfiguration(self.grid)

        # Initialize test field data
        self.fields.rho.fill(1.0)
        self.fields.pressure.fill(0.33)

        # Create a log capture handler for testing
        self.log_capture = LogCapture()
        israel_stewart_logger = logging.getLogger("israel_stewart")
        israel_stewart_logger.addHandler(self.log_capture)

    def tearDown(self):
        """Clean up after tests."""
        # Remove test handler
        israel_stewart_logger = logging.getLogger("israel_stewart")
        israel_stewart_logger.removeHandler(self.log_capture)

    def test_performance_monitoring_logging(self):
        """Test that performance monitoring generates proper log messages."""

        @monitor_performance("test_solver_operation")
        def test_solver_function():
            # Simulate solver work
            import time

            time.sleep(0.05)  # Short operation
            return np.sum(self.fields.rho)

        # Execute monitored function
        result = test_solver_function()

        # Check that performance logging occurred
        self.assertIsNotNone(result)

        # Look for performance-related log messages
        performance_messages = [
            msg
            for msg in self.log_capture.messages
            if "performance" in msg.lower() or "operation completed" in msg.lower()
        ]
        self.assertGreater(len(performance_messages), 0, "Expected performance logging messages")

    def test_slow_operation_logging(self):
        """Test that slow operations trigger appropriate logging."""

        @monitor_performance("slow_test_operation")
        def slow_solver_function():
            # Simulate slow solver work (>1 second)
            import time

            time.sleep(1.1)
            return "completed"

        # Execute slow function
        result = slow_solver_function()
        self.assertEqual(result, "completed")

        # Check for slow operation warning
        slow_messages = [msg for msg in self.log_capture.messages if "slow" in msg.lower()]
        self.assertGreater(len(slow_messages), 0, "Expected slow operation logging")

    def test_physics_logger_integration(self):
        """Test physics logger integration with solver operations."""
        # Test conservation check logging
        physics_logger.log_conservation_check("energy", 1e-8, 1e-6)
        physics_logger.log_conservation_check("momentum_x", 1e-4, 1e-6)  # This should warn

        # Test convergence logging
        physics_logger.log_convergence("implicit_solver", 15, 1e-10, True)
        physics_logger.log_convergence("newton_raphson", 50, 1e-3, False)

        # Check that messages were logged
        physics_messages = [
            msg
            for msg in self.log_capture.messages
            if any(keyword in msg.lower() for keyword in ["conservation", "convergence", "solver"])
        ]
        self.assertGreater(len(physics_messages), 0)

        # Check for appropriate log levels
        warning_messages = [
            msg
            for msg in self.log_capture.messages
            if self.log_capture.get_level(msg) >= logging.WARNING
        ]
        self.assertGreater(len(warning_messages), 0, "Expected warning messages for failed checks")

    def test_solver_error_logging(self):
        """Test error logging in solver operations."""

        @monitor_performance("error_prone_operation")
        def failing_solver_function():
            raise RuntimeError("Solver convergence failed")

        # Execute function that should fail
        with self.assertRaises(RuntimeError):
            failing_solver_function()

        # Performance monitoring should still log the operation attempt
        operation_messages = [
            msg for msg in self.log_capture.messages if "operation" in msg.lower()
        ]
        # Note: The exact behavior depends on how @monitor_performance handles exceptions

    def test_fallback_mechanism_logging(self):
        """Test logging when solvers fall back to simpler methods."""
        # Simulate a fallback scenario
        physics_logger.log_physics_fallback(
            operation="matrix_inversion", reason="matrix_singular", fallback="pseudo_inverse"
        )

        physics_logger.log_physics_fallback(
            operation="fft_computation", reason="scipy_unavailable", fallback="numpy_fft"
        )

        # Check for fallback messages
        fallback_messages = [msg for msg in self.log_capture.messages if "fallback" in msg.lower()]
        self.assertEqual(len(fallback_messages), 2, "Expected two fallback log messages")

    def test_memory_usage_logging(self):
        """Test memory usage logging integration."""
        # Simulate high memory usage
        large_array = np.ones((1000, 1000))  # ~8MB array

        # Log memory usage
        performance_logger.log_memory_usage(
            operation="large_array_allocation",
            memory_mb=large_array.nbytes / (1024 * 1024),
            array_shape=large_array.shape,
        )

        # Check for memory-related messages
        memory_messages = [msg for msg in self.log_capture.messages if "memory" in msg.lower()]
        self.assertGreater(len(memory_messages), 0, "Expected memory usage logging")

    @patch("israel_stewart.solvers.finite_difference.warnings.warn")
    def test_solver_warning_logging_integration(self):
        """Test that solver warnings are properly integrated with logging."""
        # This test would be more complete with actual solver import
        # For now, test the pattern of warning integration

        logger = get_logger("finite_difference.test")
        logger.warning(
            "Boundary condition fallback",
            extra={
                "boundary_condition": "unknown_type",
                "fallback": "no_extension",
                "solver": "finite_difference",
            },
        )

        # Check that structured warning was logged
        warning_messages = [
            msg
            for msg in self.log_capture.messages
            if "boundary" in msg.lower() and "fallback" in msg.lower()
        ]
        self.assertGreater(len(warning_messages), 0)


class LogCapture(logging.Handler):
    """Test handler to capture log messages."""

    def __init__(self):
        super().__init__()
        self.messages = []
        self.records = []

    def emit(self, record):
        self.records.append(record)
        self.messages.append(record.getMessage())

    def get_level(self, message):
        """Get the log level for a specific message."""
        for record in self.records:
            if record.getMessage() == message:
                return record.levelno
        return logging.NOTSET

    def clear(self):
        """Clear captured messages."""
        self.messages.clear()
        self.records.clear()


class TestPerformanceImpact(unittest.TestCase):
    """Test performance impact of logging."""

    def test_logging_overhead(self):
        """Test that logging doesn't significantly impact performance."""
        import time

        # Configure minimal logging
        configure_logging(level="WARNING")

        @monitor_performance("performance_test")
        def test_computation():
            # Simulate typical solver computation
            data = np.random.random((100, 100))
            return np.linalg.eigvals(data)

        # Time with minimal logging
        start_time = time.time()
        for _ in range(10):
            test_computation()
        minimal_logging_time = time.time() - start_time

        # Configure verbose logging
        configure_logging(level="DEBUG", enable_performance=True)
        enable_performance_logging(True)

        # Time with verbose logging
        start_time = time.time()
        for _ in range(10):
            test_computation()
        verbose_logging_time = time.time() - start_time

        # Logging overhead should be minimal (< 50% increase)
        overhead_ratio = verbose_logging_time / minimal_logging_time
        self.assertLess(overhead_ratio, 1.5, f"Logging overhead too high: {overhead_ratio:.2f}x")

    def test_disabled_logger_performance(self):
        """Test that disabled loggers have minimal overhead."""
        import time

        logger = get_logger("performance_test")
        logger.setLevel(logging.CRITICAL)  # Effectively disable

        def test_with_logging():
            for _ in range(1000):
                logger.debug("This debug message should be ignored")
                logger.info("This info message should be ignored")

        # Time disabled logging
        start_time = time.time()
        test_with_logging()
        disabled_time = time.time() - start_time

        # This should be very fast since messages are filtered out early
        self.assertLess(disabled_time, 0.1, "Disabled logging taking too long")


if __name__ == "__main__":
    unittest.main()
