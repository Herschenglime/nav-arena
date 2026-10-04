# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ArenaLogger, custom log levels, and CLI logging configuration."""

from __future__ import annotations

import argparse
import io
import logging
import sys
import unittest.mock as mock

import pytest

from nav_arena.utils.logger import (
    SUCCESS,
    ArenaLogFormatter,
    ArenaLogger,
    CarboniteBridgeHandler,
    add_logger_args,
    configure_logging,
    get_logger,
)


def test_custom_success_level():
    """Verify SUCCESS log level registration and integer value."""
    assert SUCCESS == 25
    assert logging.getLevelName(25) == "SUCCESS"
    assert logging.getLevelName("SUCCESS") == 25


def test_arena_logger_instance():
    """Verify get_logger returns an instance of ArenaLogger."""
    logger = get_logger("test.instance")
    assert isinstance(logger, ArenaLogger)
    assert hasattr(logger, "success")
    assert hasattr(logger, "section")
    assert hasattr(logger, "check")


def test_success_dispatch(caplog: pytest.LogCaptureFixture):
    """Verify logger.success dispatches at SUCCESS level (25)."""
    logger = get_logger("test.success_dispatch")
    logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        logger.success("Operation completed successfully!")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == SUCCESS
    assert record.levelname == "SUCCESS"
    assert record.getMessage() == "Operation completed successfully!"


def test_section_formatting(caplog: pytest.LogCaptureFixture):
    """Verify logger.section formats section headers at INFO level."""
    logger = get_logger("test.section")
    logger.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO):
        logger.section("INITIALIZING ROS 2 BRIDGE")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    assert record.getMessage() == "=== INITIALIZING ROS 2 BRIDGE ==="


def test_check_helper_pass_and_fail(caplog: pytest.LogCaptureFixture):
    """Verify logger.check dispatches [PASS] at SUCCESS and [FAIL] at ERROR."""
    logger = get_logger("test.check")
    logger.setLevel(logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        # 1. Passed check without detail
        res1 = logger.check("TF Tree Continuity", True)
        assert res1 is True

        # 2. Passed check with detail
        res2 = logger.check("Clock Jitter", True, "jitter=1.2ms")
        assert res2 is True

        # 3. Failed check without detail
        res3 = logger.check("LiDAR Ray Hits", False)
        assert res3 is False

        # 4. Failed check with detail
        res4 = logger.check("Goal Reached", False, "timeout after 120s")
        assert res4 is False

    assert len(caplog.records) == 4

    rec1 = caplog.records[0]
    assert rec1.levelno == SUCCESS
    assert rec1.getMessage() == "[PASS] TF Tree Continuity"

    rec2 = caplog.records[1]
    assert rec2.levelno == SUCCESS
    assert rec2.getMessage() == "[PASS] Clock Jitter - jitter=1.2ms"

    rec3 = caplog.records[2]
    assert rec3.levelno == logging.ERROR
    assert rec3.getMessage() == "[FAIL] LiDAR Ray Hits"

    rec4 = caplog.records[3]
    assert rec4.levelno == logging.ERROR
    assert rec4.getMessage() == "[FAIL] Goal Reached - timeout after 120s"


def test_log_level_filtering():
    """Verify log records below configured threshold are filtered out."""
    stream = io.StringIO()
    configure_logging(level="INFO", use_color=False, stream=stream)

    logger = get_logger("test.filtering")
    logger.debug("Debug message that should be hidden")
    logger.info("Info message that should appear")
    logger.success("Success message that should appear")

    output = stream.getvalue()
    assert "Debug message that should be hidden" not in output
    assert "[INFO] Info message that should appear" in output
    assert "[SUCCESS] Success message that should appear" in output

    # Now filter at WARNING level
    stream.truncate(0)
    stream.seek(0)
    configure_logging(level="WARNING", use_color=False, stream=stream)

    logger.info("Info message now hidden")
    logger.success("Success message now hidden")
    logger.warning("Warning message that should appear")
    logger.error("Error message that should appear")

    output2 = stream.getvalue()
    assert "Info message now hidden" not in output2
    assert "Success message now hidden" not in output2
    assert "[WARNING] Warning message that should appear" in output2
    assert "[ERROR] Error message that should appear" in output2


def test_cli_add_logger_args():
    """Verify add_logger_args binds --log-level and parses valid values."""
    parser = argparse.ArgumentParser()
    add_logger_args(parser)

    # Default
    args = parser.parse_args([])
    assert args.log_level == "INFO"

    # Lowercase input should parse to uppercase
    args_debug = parser.parse_args(["--log-level", "debug"])
    assert args_debug.log_level == "DEBUG"

    # SUCCESS level
    args_success = parser.parse_args(["--log-level", "SUCCESS"])
    assert args_success.log_level == "SUCCESS"

    # Invalid level should raise SystemExit
    with pytest.raises(SystemExit):
        parser.parse_args(["--log-level", "INVALID_LEVEL"])


def test_ansi_color_formatting():
    """Verify ArenaLogFormatter applies ANSI colors when enabled and none when disabled."""
    record_info = logging.LogRecord("test", logging.INFO, "test.py", 10, "Info test", (), None)
    record_success = logging.LogRecord("test", SUCCESS, "test.py", 11, "Success test", (), None)
    record_error = logging.LogRecord("test", logging.ERROR, "test.py", 12, "Error test", (), None)

    # 1. With color enabled
    formatter_color = ArenaLogFormatter(use_color=True)
    out_info_color = formatter_color.format(record_info)
    out_succ_color = formatter_color.format(record_success)
    out_err_color = formatter_color.format(record_error)

    assert "\033[34m[INFO]\033[0m" in out_info_color
    assert "\033[1;32m[SUCCESS]\033[0m" in out_succ_color
    assert "\033[1;31m[ERROR]\033[0m" in out_err_color

    # 2. With color disabled
    formatter_plain = ArenaLogFormatter(use_color=False)
    out_info_plain = formatter_plain.format(record_info)
    out_succ_plain = formatter_plain.format(record_success)
    out_err_plain = formatter_plain.format(record_error)

    assert "\033[" not in out_info_plain
    assert out_info_plain == "[INFO] Info test"
    assert "\033[" not in out_succ_plain
    assert out_succ_plain == "[SUCCESS] Success test"
    assert "\033[" not in out_err_plain
    assert out_err_plain == "[ERROR] Error test"


def test_carbonite_bridge_dispatch_and_recursion_guard():
    """Verify CarboniteBridgeHandler routes warning/error to carb and guards recursion."""
    mock_carb = mock.MagicMock()
    handler = CarboniteBridgeHandler(level=logging.WARNING)
    handler.setFormatter(logging.Formatter("%(message)s"))

    record_warn = logging.LogRecord("test", logging.WARNING, "test.py", 10, "Sim warning", (), None)
    record_err = logging.LogRecord("test", logging.ERROR, "test.py", 11, "Sim error", (), None)
    record_info = logging.LogRecord("test", logging.INFO, "test.py", 12, "Sim info", (), None)

    with mock.patch.dict(sys.modules, {"carb": mock_carb}):
        handler.emit(record_info)
        assert mock_carb.log_warn.call_count == 0
        assert mock_carb.log_error.call_count == 0

        handler.emit(record_warn)
        assert mock_carb.log_warn.call_count == 1
        mock_carb.log_warn.assert_called_with("Sim warning")

        handler.emit(record_err)
        assert mock_carb.log_error.call_count == 1
        mock_carb.log_error.assert_called_with("Sim error")

        # Test recursion guard: if carb triggers another log inside log_warn
        def reentrant_call(msg):
            handler.emit(record_warn)

        mock_carb.log_warn.side_effect = reentrant_call
        # This must terminate without RecursionError and without reentrant emission
        handler.emit(record_warn)
        assert mock_carb.log_warn.call_count == 2


def test_formatter_preserves_message_tokens():
    """Verify ArenaLogFormatter does not corrupt bracketed strings inside user message."""
    formatter_color = ArenaLogFormatter(use_color=True)
    record = logging.LogRecord("test", logging.INFO, "test.py", 10, "Found [INFO] tag inside payload", (), None)
    out = formatter_color.format(record)
    assert "\033[34m[INFO]\033[0m" in out
    assert "Found [INFO] tag inside payload" in out

    # Test custom format without brackets in fmt
    formatter_custom = ArenaLogFormatter(fmt="%(levelname)s: %(message)s", use_color=True)
    out_custom = formatter_custom.format(record)
    assert "\033[34mINFO\033[0m:" in out_custom
    assert "Found [INFO] tag inside payload" in out_custom


def test_logger_monkey_patched_methods():
    """Verify standard logging.Logger has success, section, and check methods."""
    std_logger = logging.getLogger("test.standard_logger")
    assert hasattr(std_logger, "success")
    assert hasattr(std_logger, "section")
    assert hasattr(std_logger, "check")


def test_get_logger_default_name():
    """Verify get_logger() defaults to 'nav_arena' and preserves root logger class."""
    logger = get_logger()
    assert isinstance(logger, ArenaLogger)
    assert logger.name == "nav_arena"


def test_carbonite_bridge_with_carb_log_handler_active():
    """Verify CarboniteBridgeHandler skips when _CarbLogHandler is present on root."""
    class _CarbLogHandler(logging.StreamHandler):
        pass

    mock_carb = mock.MagicMock()
    handler = CarboniteBridgeHandler(level=logging.WARNING)
    record_warn = logging.LogRecord("test", logging.WARNING, "test.py", 10, "Sim warning", (), None)

    carb_handler = _CarbLogHandler()
    root = logging.getLogger()
    root.addHandler(carb_handler)
    try:
        with mock.patch.dict(sys.modules, {"carb": mock_carb}):
            handler.emit(record_warn)
            assert mock_carb.log_warn.call_count == 0
    finally:
        root.removeHandler(carb_handler)


def test_configure_logging_does_not_hijack_subclass_handlers():
    """Verify configure_logging creates a real StreamHandler rather than hijacking subclasses."""
    class _CarbLogHandler(logging.StreamHandler):
        pass

    carb_handler = _CarbLogHandler()
    root = logging.getLogger()
    root.addHandler(carb_handler)
    try:
        out_stream = io.StringIO()
        configure_logging(level="INFO", stream=out_stream, use_color=False)
        # Ensure a genuine StreamHandler was created and bound
        real_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
        assert len(real_handlers) >= 1
        assert carb_handler.stream != out_stream
    finally:
        root.removeHandler(carb_handler)
