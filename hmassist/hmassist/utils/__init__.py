#!/usr/bin/env python3

import sys
import logging
from .glog_format import GLogFormatterWithColor

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(GLogFormatterWithColor())

# set base logger
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("hmassist")
logger.setLevel(logging.INFO)
