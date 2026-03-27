# Copyright 2025 HOUMO AI
#
# File: logging_format.py
# Description:
#   Logging Format Utilities
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
Logging Format Utilities - Custom formatters for glog-style and colored output.
"""
import sys
import time
import logging
from typing import Dict

# FATAL is alias of CRITICAL in Python logging
logging.FATAL = logging.CRITICAL


def format_message(record: logging.LogRecord) -> str:
    """
    Format the log record message with argument substitution.

    Args:
        record: LogRecord containing the log information

    Returns:
        Formatted message string with arguments substituted
    """
    try:
        return record.msg if not record.args else record.msg % record.args
    except (TypeError, ValueError):
        return str(record.msg)


class BaseFormatter(logging.Formatter):
    """Base formatter with common glog-style formatting utilities."""

    LEVEL_MAP: Dict[int, str] = {}

    def _get_level(self, record: logging.LogRecord) -> str:
        """Get level string for the record."""
        return self.LEVEL_MAP.get(record.levelno, "?")

    def _get_timestamp(self, record: logging.LogRecord) -> tuple:
        """Get formatted date/time components."""
        date = time.localtime(record.created)
        usec = int((record.created - int(record.created)) * 1e6)
        return date, usec

    def _format_glog(self, record: logging.LogRecord) -> str:
        """
        Format record in glog style: I20250326 14:30:45.123456 12345 file.py:42] message
        """
        level = self._get_level(record)
        date, usec = self._get_timestamp(record)

        return "%c%d%02d%02d %02d:%02d:%02d.%06d %s %s:%d] %s" % (
            level,
            date.tm_year,
            date.tm_mon,
            date.tm_mday,
            date.tm_hour,
            date.tm_min,
            date.tm_sec,
            usec,
            record.process or "?????",
            record.filename,
            record.lineno,
            format_message(record),
        )


class LoggingFormatter(BaseFormatter):
    """
    Glog-style formatter with level, timestamp, process ID, file, and line.

    Output format: I20250326 14:30:45.123456 12345 file.py:42] message
    """

    LEVEL_MAP = {
        logging.FATAL: "F",
        logging.ERROR: "E",
        logging.WARN: "W",
        logging.INFO: "I",
        logging.DEBUG: "D",
    }

    def format(self, record: logging.LogRecord) -> str:
        message = self._format_glog(record)
        record.getMessage = lambda: message
        return super().format(record)


class LoggingFormatterWithColor(BaseFormatter):
    """
    Colored glog-style formatter for terminal output.

    Output format: I20250326 14:30:45.123456 12345 file.py:42] message
    (with ANSI colors based on log level)
    """

    LEVEL_MAP = {
        logging.FATAL: "F",
        logging.ERROR: "E",
        logging.WARN: "W",
        logging.INFO: "I",
        logging.DEBUG: "D",
    }

    # ANSI color codes
    RESET = "\x1b[0m"
    COLORS = {
        logging.DEBUG: "\x1b[98;20m",  # grey
        logging.INFO: "\x1b[92;20m",  # green
        logging.WARN: "\x1b[93;20m",  # yellow
        logging.ERROR: "\x1b[91;1m",  # bold red
        logging.FATAL: "\x1b[97;41m",  # white on red background
    }

    def format(self, record: logging.LogRecord) -> str:
        message = self._format_glog(record)

        # Apply color
        color = self.COLORS.get(record.levelno, "")
        if color:
            message = f"{color}{message}{self.RESET}"

        record.getMessage = lambda: message
        return super().format(record)


class FatalExitHandler(logging.Handler):
    """
    Handler that exits the process with error code on FATAL level logs.

    Usage:
        logger = logging.getLogger()
        logger.addHandler(FatalExitHandler())
        logger.fatal("Critical error occurred")  # Will exit with code 1
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Exit process if record level is FATAL/CRITICAL."""
        if record.levelno >= logging.FATAL:
            sys.exit(1)
