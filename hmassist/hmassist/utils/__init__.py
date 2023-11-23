#!/usr/bin/env python3

import logging
from .glog_format import GLogFormatterWithColor

console_handler = logging.StreamHandler()
console_handler.setFormatter(GLogFormatterWithColor())

# set base logger
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.WARNING)

logger = logging.getLogger("hmassist")
logger.setLevel(logging.INFO)
