#!/usr/bin/env python3

import logging
import time


def format_message(record):
    try:
        if len(record.args) == 0:
            record_message = '%s' % record.msg
        else:
            record_message = '%s' % (record.msg % record.args)
    except TypeError:
        record_message = record.msg
    return record_message


class GLogFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.FATAL: 'F',  # FATAL is alias of CRITICAL
        logging.ERROR: 'E',
        logging.WARN: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    def __init__(self):
        logging.Formatter.__init__(self)

    def format(self, record):
        try:
            level = GLogFormatter.LEVEL_MAP[record.levelno]
        except KeyError:
            level = '?'
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        record_message = '%c%d%02d%02d %02d:%02d:%02d.%06d %s %s:%d] %s' % (
            level, date.tm_year, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
            date.tm_sec, date_usec,
            record.process if record.process is not None else '?????',
            record.filename,
            record.lineno,
            format_message(record)
        )
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)


class GLogFormatterWithColor(logging.Formatter):
    LEVEL_MAP = {
        logging.FATAL: 'FATAL',  # FATAL is alias of CRITICAL
        logging.ERROR: 'ERROR',
        logging.WARN: 'WARN',
        logging.INFO: 'INFO',
        logging.DEBUG: 'DEBUG'
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
        logging.CRITICAL: bold_red
    }

    def __init__(self):
        logging.Formatter.__init__(self)

    def format(self, record):
        try:
            level = GLogFormatterWithColor.LEVEL_MAP[record.levelno]
        except KeyError:
            level = '?'
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        record_message = '%02d:%02d:%02d.%06d %s:%d [%s] %s' % (
            date.tm_hour, date.tm_min,
            date.tm_sec, date_usec,
            record.filename,
            record.lineno,
            level,
            format_message(record)
        )
        record_message = GLogFormatterWithColor.COLOR_FORMATS[record.levelno] + record_message + "\x1b[0m"
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)
