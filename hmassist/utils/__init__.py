#!/usr/bin/env python3

import logging
from utils.glog_format import GLogFormatterWithColor


logger = logging.getLogger()

console_handler = logging.StreamHandler()
console_handler.setFormatter(GLogFormatterWithColor())

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
