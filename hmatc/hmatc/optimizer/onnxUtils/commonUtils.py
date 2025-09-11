"""

Author: Nan Xu
Maintainer: Nan xu
Date: 2025/07/14
Company: Houmo

"""
import numpy as np  # type: ignore
from pathlib import Path
import math


def get_dict_default(input_dict, key, default):
    if key not in input_dict.keys():
        return default
    return input_dict[key]


def get_precision_from_ini_default(ini, default: int = 8):
    if ini is not None:
        return ini.precision
    else:
        return default


def get_asym_from_ini_default(ini, default=False):
    if ini is not None:
        return not (ini.ft_quantize_sync or ini.wt_quantize_sync)
    else:
        return default


def get_out_format_from_ini_default(ini, default: str = "NCHW"):
    if ini is not None:
        return ini.output_format
    else:
        return default


def check_tensor_is_nchw_format_from_ini(ini, tensor_name: str):
    if ini is None:
        return False
    if tensor_name not in ini.custom_tensor_name:
        return ini.output_format == 'NCHW'
    else:
        return ini.custom_tensor_format[ini.custom_tensor_name.index(tensor_name)] == 'NCHW'


def check_tensor_is_int_data_type_from_ini(ini, tensor_name: str):
    if ini is None:
        return True
    if tensor_name not in ini.custom_tensor_name:
        return ini.output_type == 'integer'
    else:
        return ini.custom_data_type[ini.custom_tensor_name.index(tensor_name)] != 'fp32_tensor'


def get_path_from_ini_default(ini):
    if ini is not None:
        return ini.ini_path
    else:
        raise ValueError("Ini must be provided.")


def get_sram_r_byte_pre_cycle(ini):
    if ini is not None:
        return 16 * ini.sram_frequency / ini.dla_frequency * \
            ini.sram_r_efficiency
    else:
        raise ValueError("Ini must be provided and simconfig should be "
                         "valid")


def get_sram_w_byte_pre_cycle(ini):
    if ini is not None:
        return 16 * ini.sram_frequency / ini.dla_frequency * \
            ini.sram_w_efficiency
    else:
        raise ValueError("Ini must be provided and simconfig should be "
                         "valid")

def get_value_align_pow2(value: int) -> int:
    if value == 0:
        return value
    align_value = int(math.pow(2, (math.ceil(math.log(value, 2)))))
    return align_value