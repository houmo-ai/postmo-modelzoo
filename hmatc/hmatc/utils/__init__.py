import sys
import logging
from .logging_format import LoggingFormatterWithColor
from .utils import read_yaml_to_dict, read_json_to_dict


console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(LoggingFormatterWithColor())

# root_logger = logging.getLogger()
# root_logger.addHandler(console_handler)
# root_logger.setLevel(logging.INFO)

logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
