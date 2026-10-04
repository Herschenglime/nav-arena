# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logging and utility primitives for nav_arena."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from typing import Any

SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")


def _logger_success(self: logging.Logger, msg: object, *args: object, **kwargs: object) -> None:
    """Log 'msg % args' with severity 'SUCCESS' (level 25)."""
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, msg, args, **kwargs)


def _logger_section(self: logging.Logger, title: str, *args: object, **kwargs: object) -> None:
    """Format and log a section header at INFO level."""
    self.info(f"=== {title} ===", *args, **kwargs)


def _logger_check(
    self: logging.Logger,
    name: str,
    passed: bool,
    detail: str | None = None,
    *args: object,
    **kwargs: object,
) -> bool:
    """Log a verification check result.

    Args:
        name: Description of the check.
        passed: Whether the verification check passed or failed.
        detail: Optional supplementary detail or metrics.

    Returns:
        bool: The passed status for convenient chaining or assert.
    """
    if passed:
        msg = f"[PASS] {name} - {detail}" if detail else f"[PASS] {name}"
        if hasattr(self, "success"):
            self.success(msg, *args, **kwargs)
        else:
            self.log(SUCCESS, msg, *args, **kwargs)
        return True
    else:
        msg = f"[FAIL] {name} - {detail}" if detail else f"[FAIL] {name}"
        self.error(msg, *args, **kwargs)
        return False


setattr(logging.Logger, "success", _logger_success)
setattr(logging.Logger, "section", _logger_section)
setattr(logging.Logger, "check", _logger_check)


class ArenaLogFormatter(logging.Formatter):
    """Custom formatter with ANSI color support for ArenaLogger."""

    COLOR_CYAN = "\033[36m"
    COLOR_BLUE = "\033[34m"
    COLOR_BOLD_GREEN = "\033[1;32m"
    COLOR_YELLOW = "\033[33m"
    COLOR_BOLD_RED = "\033[1;31m"
    COLOR_BOLD_MAGENTA = "\033[1;35m"
    COLOR_RESET = "\033[0m"

    LEVEL_COLORS = {
        logging.DEBUG: COLOR_CYAN,
        logging.INFO: COLOR_BLUE,
        SUCCESS: COLOR_BOLD_GREEN,
        logging.WARNING: COLOR_YELLOW,
        logging.ERROR: COLOR_BOLD_RED,
        logging.CRITICAL: COLOR_BOLD_MAGENTA,
    }

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_color: bool = True,
    ) -> None:
        if fmt is None:
            fmt = "[%(levelname)s] %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color
        if "[%(levelname)s]" in fmt:
            self._colored_fmt = fmt.replace("[%(levelname)s]", "%(bracketed_level)s")
        else:
            self._colored_fmt = fmt

    def formatMessage(self, record: logging.LogRecord) -> str:
        if self.use_color and record.levelno in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelno]
            if self._colored_fmt != self._fmt:
                record.bracketed_level = f"{color}[{record.levelname}]{self.COLOR_RESET}"
                return self._colored_fmt % record.__dict__
            else:
                orig_levelname = record.levelname
                try:
                    record.levelname = f"{color}{orig_levelname}{self.COLOR_RESET}"
                    return self._fmt % record.__dict__
                finally:
                    record.levelname = orig_levelname
        return super().formatMessage(record)


class ArenaLogger(logging.Logger):
    """Specialized logger for nav_arena verification, simulations, and tools."""

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)

    def success(self, msg: object, *args: object, **kwargs: object) -> None:
        """Log 'msg % args' with severity 'SUCCESS' (level 25)."""
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, **kwargs)

    def section(self, title: str, *args: object, **kwargs: object) -> None:
        """Format and log a section header at INFO level."""
        self.info(f"=== {title} ===", *args, **kwargs)

    def check(
        self,
        name: str,
        passed: bool,
        detail: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> bool:
        """Log a verification check result.

        Args:
            name: Description of the check.
            passed: Whether the verification check passed or failed.
            detail: Optional supplementary detail or metrics.

        Returns:
            bool: The passed status for convenient chaining or assert.
        """
        if passed:
            msg = f"[PASS] {name} - {detail}" if detail else f"[PASS] {name}"
            self.success(msg, *args, **kwargs)
            return True
        else:
            msg = f"[FAIL] {name} - {detail}" if detail else f"[FAIL] {name}"
            self.error(msg, *args, **kwargs)
            return False


class CarboniteBridgeHandler(logging.Handler):
    """Bridge routing WARNING, ERROR, and CRITICAL logs to carb.log_warn / carb.log_error safely without recursion."""

    _thread_local = threading.local()

    def __init__(self, level: int = logging.WARNING) -> None:
        super().__init__(level=level)

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._thread_local, "active", False):
            return
        root = logging.getLogger()
        if any(h.__class__.__name__ == "_CarbLogHandler" for h in root.handlers):
            return
        if "carb" not in sys.modules:
            return
        carb = sys.modules.get("carb")
        if carb is None:
            return
        # If Kit is actively booting (omni.kit.app imported but not running),
        # calling carb.log_* triggers C++ plugin circular dependency.
        if "omni.kit.app" in sys.modules and hasattr(carb, "get_framework"):
            import unittest.mock as mock

            if not isinstance(carb, (mock.MagicMock, mock.Mock)):
                try:
                    import omni.kit.app

                    app = omni.kit.app.get_app()
                    if app is not None and app.is_running():
                        return
                except Exception:
                    return
        try:
            self._thread_local.active = True
            msg = self.format(record)
            if record.levelno >= logging.ERROR and hasattr(carb, "log_error"):
                carb.log_error(msg)
            elif record.levelno >= logging.WARNING and hasattr(carb, "log_warn"):
                carb.log_warn(msg)
        except Exception:
            pass
        finally:
            self._thread_local.active = False


_stream_handler: logging.StreamHandler | None = None
_carb_handler: CarboniteBridgeHandler | None = None


def _parse_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    name = str(level).strip().upper()
    if name == "SUCCESS":
        return SUCCESS
    if hasattr(logging, name):
        val = getattr(logging, name)
        if isinstance(val, int):
            return val
    raise ValueError(f"Unknown log level: '{level}'. Valid levels: DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL")


def configure_logging(
    level: str | int = logging.INFO,
    fmt: str | None = None,
    use_color: bool | None = None,
    stream: Any = sys.stdout,
) -> None:
    """Configure the root logger and attached stream handler.

    Args:
        level: Logging level (e.g. 'INFO', 'DEBUG', 'SUCCESS', logging.INFO).
        fmt: Log record format string. Defaults to '[%(levelname)s] %(message)s'.
        use_color: Whether to format with ANSI colors. If None, auto-detected from stream TTY and NO_COLOR.
        stream: Target stream for console output. Defaults to sys.stdout.
    """
    global _stream_handler, _carb_handler

    level_no = _parse_level(level)

    if use_color is None:
        is_tty = hasattr(stream, "isatty") and stream.isatty()
        no_color = "NO_COLOR" in os.environ
        use_color = is_tty and not no_color

    formatter = ArenaLogFormatter(fmt=fmt, use_color=use_color)
    root = logging.getLogger()
    root.setLevel(level_no)

    if _stream_handler is None or _stream_handler not in root.handlers:
        found = None
        for h in root.handlers:
            if type(h) is logging.StreamHandler:
                found = h
                break
        if found is not None:
            _stream_handler = found
        else:
            _stream_handler = logging.StreamHandler(stream)
            root.addHandler(_stream_handler)

    if stream is not None:
        _stream_handler.setStream(stream)

    _stream_handler.setLevel(level_no)
    _stream_handler.setFormatter(formatter)

    # Attach Carbonite bridge handler if not already present
    if _carb_handler is None or _carb_handler not in root.handlers:
        found_carb = any(isinstance(h, CarboniteBridgeHandler) for h in root.handlers)
        if not found_carb:
            _carb_handler = CarboniteBridgeHandler()
            _carb_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            root.addHandler(_carb_handler)


def get_logger(name: str | None = None) -> ArenaLogger:
    """Retrieve an ArenaLogger instance by name."""
    logger_name = name if name is not None else "nav_arena"
    logging.setLoggerClass(ArenaLogger)
    logger = logging.getLogger(logger_name)
    if not isinstance(logger, ArenaLogger):
        logger.__class__ = ArenaLogger

    root = logging.getLogger()
    if not root.handlers:
        configure_logging()

    return logger


def add_logger_args(parser: argparse.ArgumentParser) -> None:
    """Add standard --log-level argument to an ArgumentParser."""
    parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: INFO)",
    )
