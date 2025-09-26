"""
Unit tests for logging configuration and runtime controls.

Tests the logging infrastructure, environment variable handling,
and runtime logging control functionality.
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from israel_stewart.utils.logging_config import (
    configure_logging,
    enable_memory_debugging,
    enable_performance_logging,
    get_logger,
    get_logging_status,
    performance_logger,
    physics_logger,
    set_log_level,
    setup_from_environment,
)


class TestLoggingConfig(unittest.TestCase):
    """Test logging configuration functionality."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing handlers
        israel_stewart_logger = logging.getLogger("israel_stewart")
        israel_stewart_logger.handlers.clear()

        # Reset logger states
        for name in list(logging.getLogger().manager.loggerDict.keys()):
            if name.startswith("israel_stewart."):
                logger = logging.getLogger(name)
                logger.handlers.clear()
                logger.disabled = False
                logger.setLevel(logging.NOTSET)

    def tearDown(self):
        """Clean up after tests."""
        self.setUp()  # Reuse setup cleanup

    def test_get_logger(self):
        """Test logger creation with proper naming."""
        logger = get_logger("test_module")
        self.assertEqual(logger.name, "israel_stewart.test_module")

    def test_basic_logging_configuration(self):
        """Test basic logging configuration."""
        configure_logging(level="DEBUG", format_type="console")

        logger = get_logger("test")
        self.assertEqual(logger.level, logging.NOTSET)  # Uses parent level

        # Check that root israel_stewart logger is configured
        root_logger = logging.getLogger("israel_stewart")
        self.assertTrue(len(root_logger.handlers) > 0)

    def test_file_logging_configuration(self):
        """Test file logging configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            configure_logging(level="INFO", log_file=log_file)

            # Check that file handler was added
            israel_stewart_logger = logging.getLogger("israel_stewart")
            file_handlers = [
                h
                for h in israel_stewart_logger.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            self.assertTrue(len(file_handlers) > 0)

    def test_performance_logging_configuration(self):
        """Test performance logging configuration."""
        configure_logging(enable_performance=True)

        perf_logger = logging.getLogger("israel_stewart.performance")
        self.assertTrue(len(perf_logger.handlers) > 0)
        self.assertFalse(perf_logger.propagate)

    def test_memory_tracking_configuration(self):
        """Test memory tracking configuration."""
        configure_logging(enable_memory_tracking=True)

        mem_logger = logging.getLogger("israel_stewart.memory")
        self.assertTrue(len(mem_logger.handlers) > 0)
        self.assertFalse(mem_logger.propagate)

    def test_runtime_log_level_adjustment(self):
        """Test runtime log level adjustment."""
        configure_logging(level="INFO")

        # Test level adjustment
        set_log_level("DEBUG")
        israel_stewart_logger = logging.getLogger("israel_stewart")
        self.assertEqual(israel_stewart_logger.level, logging.DEBUG)

        # Test with invalid level (should raise AttributeError)
        with self.assertRaises(AttributeError):
            set_log_level("INVALID_LEVEL")

    def test_performance_logging_toggle(self):
        """Test runtime performance logging toggle."""
        configure_logging()

        # Enable performance logging
        enable_performance_logging(True)
        perf_logger = logging.getLogger("israel_stewart.performance")
        self.assertFalse(perf_logger.disabled)
        self.assertTrue(len(perf_logger.handlers) > 0)

        # Disable performance logging
        enable_performance_logging(False)
        self.assertTrue(perf_logger.disabled)

    def test_memory_debugging_toggle(self):
        """Test runtime memory debugging toggle."""
        configure_logging()

        # Enable memory debugging
        enable_memory_debugging(True)
        mem_logger = logging.getLogger("israel_stewart.memory")
        self.assertFalse(mem_logger.disabled)
        self.assertTrue(len(mem_logger.handlers) > 0)

        # Disable memory debugging
        enable_memory_debugging(False)
        self.assertTrue(mem_logger.disabled)

    def test_logging_status(self):
        """Test logging status reporting."""
        configure_logging(level="INFO", enable_performance=True)

        status = get_logging_status()

        self.assertEqual(status["main_level"], "INFO")
        self.assertTrue(status["performance_enabled"])
        self.assertFalse(status["memory_debugging_enabled"])
        self.assertGreater(status["handlers_count"], 0)
        self.assertIsInstance(status["active_loggers"], list)

    @patch.dict(
        os.environ,
        {
            "ISRAEL_STEWART_LOG_LEVEL": "DEBUG",
            "ISRAEL_STEWART_LOG_FORMAT": "console",
            "ISRAEL_STEWART_LOG_PERFORMANCE": "true",
            "ISRAEL_STEWART_LOG_MEMORY": "true",
        },
    )
    def test_environment_variable_configuration(self):
        """Test configuration from environment variables."""
        setup_from_environment()

        israel_stewart_logger = logging.getLogger("israel_stewart")
        perf_logger = logging.getLogger("israel_stewart.performance")
        mem_logger = logging.getLogger("israel_stewart.memory")

        # Check that configuration was applied
        self.assertTrue(len(israel_stewart_logger.handlers) > 0)
        self.assertTrue(len(perf_logger.handlers) > 0)
        self.assertTrue(len(mem_logger.handlers) > 0)

    @patch.dict(os.environ, {"ISRAEL_STEWART_LOG_FILE": "/tmp/test_israel_stewart.log"})
    def test_environment_file_logging(self):
        """Test file logging from environment variables."""
        setup_from_environment()

        israel_stewart_logger = logging.getLogger("israel_stewart")
        file_handlers = [
            h
            for h in israel_stewart_logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        self.assertTrue(len(file_handlers) > 0)


class TestLoggerClasses(unittest.TestCase):
    """Test logger utility classes."""

    def setUp(self):
        """Set up test environment."""
        configure_logging(level="DEBUG")

    def test_performance_logger(self):
        """Test PerformanceLogger functionality."""
        # Test operation logging
        performance_logger.log_operation("test_op", 1.5, memory_mb=50.0)

        # Test memory logging
        performance_logger.log_memory_usage("test_mem", 100.0, peak_usage=120.0)

    def test_physics_logger(self):
        """Test PhysicsLogger functionality."""
        # Test conservation check logging
        physics_logger.log_conservation_check("energy", 1e-6, 1e-5)  # Passed
        physics_logger.log_conservation_check("momentum", 1e-4, 1e-5)  # Failed

        # Test convergence logging
        physics_logger.log_convergence("implicit_solver", 10, 1e-8, True)  # Converged
        physics_logger.log_convergence("explicit_solver", 100, 1e-3, False)  # Failed

        # Test physics fallback logging
        physics_logger.log_physics_fallback("scipy_solver", "import_error", "numpy_fallback")

        # Test error recovery logging
        physics_logger.log_error_recovery(
            "matrix_inversion", "singular_matrix", "pseudo_inverse", True
        )


class TestLoggingIntegration(unittest.TestCase):
    """Test logging integration with other components."""

    def test_logger_isolation(self):
        """Test that different logger instances don't interfere."""
        configure_logging(level="INFO")

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        # Both should be under israel_stewart namespace
        self.assertTrue(logger1.name.startswith("israel_stewart."))
        self.assertTrue(logger2.name.startswith("israel_stewart."))
        self.assertNotEqual(logger1.name, logger2.name)

    def test_third_party_logger_levels(self):
        """Test that third-party loggers are properly configured."""
        configure_logging()

        matplotlib_logger = logging.getLogger("matplotlib")
        scipy_logger = logging.getLogger("scipy")

        # These should be set to WARNING to reduce noise
        self.assertGreaterEqual(matplotlib_logger.level, logging.WARNING)
        self.assertGreaterEqual(scipy_logger.level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
