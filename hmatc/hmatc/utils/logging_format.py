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
import time
import logging


def format_message(record):
    """
    Format the log record message by handling potential argument substitution.

    Args:
        record: A logging.LogRecord object containing the log information

    Returns:
        str: Formatted message string with arguments properly substituted
    """
    try:
        if len(record.args) == 0:
            record_message = "%s" % record.msg
        else:
            record_message = "%s" % (record.msg % record.args)
    except TypeError:
        record_message = record.msg
    return record_message


class LoggingFormatter(logging.Formatter):
    """
    Custom logging formatter that formats log records in a specific style similar to glog.
    The format includes level, date, time with microseconds, process ID, filename, line number, and message.
    """

    LEVEL_MAP = {
        logging.FATAL: "F",  # FATAL is alias of CRITICAL
        logging.ERROR: "E",
        logging.WARN: "W",
        logging.INFO: "I",
        logging.DEBUG: "D",
    }

    def __init__(self):
        """
        Initialize the LoggingFormatter instance.
        """
        logging.Formatter.__init__(self)

    def format(self, record):
        """
        Format the specified record as text according to the custom format.

        Args:
            record: A logging.LogRecord object to format

        Returns:
            str: Formatted log record string
        """
        try:
            level = LoggingFormatter.LEVEL_MAP[record.levelno]
        except KeyError:
            level = "?"
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        record_message = "%c%d%02d%02d %02d:%02d:%02d.%06d %s %s:%d] %s" % (
            level,
            date.tm_year,
            date.tm_mon,
            date.tm_mday,
            date.tm_hour,
            date.tm_min,
            date.tm_sec,
            date_usec,
            record.process if record.process is not None else "?????",
            record.filename,
            record.lineno,
            format_message(record),
        )
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)


class LoggingFormatterWithColor(logging.Formatter):
    """
    Custom logging formatter that formats log records with color coding based on log level.
    Colors help distinguish different log levels visually in terminal output.
    """

    LEVEL_MAP = {
        logging.FATAL: "FATAL",  # FATAL is alias of CRITICAL
        logging.ERROR: "ERROR",
        logging.WARN: "WARN",
        logging.INFO: "INFO",
        logging.DEBUG: "DEBUG",
    }

    green = "\x1b[92;20m"
    grey = "\x1b[98;20m"
    yellow = "\x1b[93;20m"
    red = "\x1b[91;20m"
    bold_red = "\x1b[91;1m"
    reset = "\x1b[0m"

    COLOR_FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: bold_red,  # red,
        logging.CRITICAL: bold_red,
    }

    def __init__(self):
        """
        Initialize the LoggingFormatterWithColor instance.
        """
        logging.Formatter.__init__(self)

    def format(self, record):
        """
        Format the specified record as text with color coding based on log level.

        Args:
            record: A logging.LogRecord object to format

        Returns:
            str: Formatted log record string with color codes
        """
        try:
            level = LoggingFormatterWithColor.LEVEL_MAP[record.levelno]
        except KeyError:
            level = "?"
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        record_message = "%02d:%02d:%02d.%06d %s:%d [%s] %s" % (
            date.tm_hour,
            date.tm_min,
            date.tm_sec,
            date_usec,
            record.filename,
            record.lineno,
            level,
            format_message(record),
        )
        record_message = (
            LoggingFormatterWithColor.COLOR_FORMATS[record.levelno]
            + record_message
            + "\x1b[0m"
        )
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)
