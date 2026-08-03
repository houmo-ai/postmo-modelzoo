# Copyright 2025 HOUMO AI
#
# File: bfp.py
# Description:
#   BFP (Binary Floating Point) utility functions for HMATC models.
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
from typing import Union, Literal
import numpy as np


# pylint: disable=too-many-locals
def cast_act_hmfp_data_to_fp_data(
    hmfp_data: np.ndarray,
    pack_format: Union[Literal["g32e8"], Literal["g32e8s11"]],
    pack_axis: int,
):
    if pack_format not in ("g32e8",):
        raise ValueError(f"this format {pack_format} is not supported currently")
    hmfp_data_moved = np.moveaxis(hmfp_data, pack_axis, -1)
    # data must be packed at the last axis
    length_of_packed = hmfp_data_moved.shape[-1]
    num_per_pack = 32
    if length_of_packed % num_per_pack != 0:
        raise ValueError(
            f"axis size {length_of_packed} does not match num per pack {num_per_pack}"
        )
    if hmfp_data.dtype == np.int8:
        indices = [16, 18, 20, 22, 24, 26, 28, 30]
        hmfp_unsigned_dtype = np.uint8
    elif hmfp_data.dtype == np.int16:
        indices = [24, 25, 26, 27, 28, 29, 30, 31]
        hmfp_unsigned_dtype = np.uint16
    else:
        raise ValueError(f"data type {hmfp_data.dtype} is not supported")
    reshaped_data = hmfp_data_moved.reshape(-1, length_of_packed)
    fp_data = []
    for idx_outer in range(reshaped_data.shape[0]):
        exp_of_odd = None  # check the each two group of 32 data has same exp
        flattened_packed_data = reshaped_data[idx_outer]
        for idx_packed in range(0, length_of_packed, num_per_pack):
            chunk_data = flattened_packed_data[idx_packed : idx_packed + num_per_pack]
            man_data = chunk_data.view(hmfp_unsigned_dtype)
            if hmfp_unsigned_dtype == np.uint8:
                man_data = np.array(
                    [
                        ele & 0xFE if i in indices else ele
                        for i, ele in enumerate(man_data)
                    ],
                    hmfp_unsigned_dtype,
                )  # make last bit to be zero
            elif hmfp_unsigned_dtype == np.uint16:
                man_data = np.array(
                    [
                        ele & 0xFFFE if i in indices else ele
                        for i, ele in enumerate(man_data)
                    ],
                    hmfp_unsigned_dtype,
                )  # make last bit to be zero
            man_data = man_data.view(hmfp_data.dtype)
            exp_data = 0
            for index in indices[::-1]:
                exp_data = (exp_data << 1) + (chunk_data[index] & 1)
            fp_data.extend(
                [(2.0 ** (exp_data - 127)) * man_data[i] for i in range(num_per_pack)]
            )
            # check the exp
            if (idx_packed // num_per_pack) % 2 != 0 and exp_of_odd != exp_data:
                raise ValueError(
                    f"exp does not match between two adjacent groups: {exp_of_odd} v.s. {exp_data}"
                )
            exp_of_odd = exp_data if (idx_packed // num_per_pack) % 2 == 0 else None
    float_moved = np.reshape(np.array(fp_data, dtype=np.float16), hmfp_data_moved.shape)
    float_data = np.moveaxis(float_moved, -1, pack_axis)
    return float_data


def cast_unpacked_man_exp_to_fp_data(
    man_data: np.ndarray, exp_data: np.ndarray, pack_axis: int
):
    man_shape = man_data.shape
    man_data_moved = np.moveaxis(man_data, pack_axis, -1)
    exp_data_moved = np.moveaxis(exp_data, pack_axis, -1)
    man_data_moved = man_data_moved.reshape(-1, man_data_moved.shape[-1])
    exp_data_moved = exp_data_moved.reshape(-1, exp_data_moved.shape[-1])
    fp_data = []
    for idx_outer in range(man_data_moved.shape[0]):
        for idx_inner in range(man_data_moved.shape[1]):
            man_data = man_data_moved[idx_outer][idx_inner]
            exp_data = exp_data_moved[idx_outer][0]
            fp_data.append((2.0 ** (exp_data)) * man_data)
    fp_data = np.reshape(np.array(fp_data, dtype=np.float16), man_data_moved.shape)
    fp_data = np.moveaxis(fp_data, -1, pack_axis).reshape(man_shape)
    return fp_data


def _round_to_even_at_bit(value, bit_position):
    """Round-to-even (banker's rounding) at the given bit position.

    Returns 1 if rounding up, 0 otherwise. Operates on a 16-bit integer
    range matching the C++ roundToEvenAtBit<int16_t> implementation.
    """
    total_bits = 16
    if bit_position <= 0:
        return 0
    if bit_position >= total_bits:
        return 1 if value < 0 else 0
    bitmask = (1 << bit_position) - 1
    half_point = 1 << (bit_position - 1)
    truncated_bits = value & bitmask
    if truncated_bits == half_point:
        return 1 if (value & (1 << bit_position)) else 0
    return 1 if truncated_bits > half_point else 0


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def cast_fp_data_to_act_hmfp_data(
    fp_data: np.ndarray,
    pack_format: Union[Literal["g32e8"], Literal["g32e8s11"]],
    pack_axis: int,
    out_dtype=np.int8,
):
    """Convert floating-point data to HMFP (block floating point) packed format.

    Faithfully replicates the C++ convertFPToG32E8 / fillIntParts / fillSharedExpPart
    algorithm for bit-exact results.

    NOTE: The output pack_axis dimension is padded to 64-element alignment
    (hardware block size). Callers should account for this when comparing
    with unpadded data.
    """
    if pack_format not in ("g32e8",):
        raise ValueError(f"this format {pack_format} is not supported currently")

    fp_data_moved = np.moveaxis(fp_data, pack_axis, -1)
    original_shape = fp_data_moved.shape
    length_of_packed = fp_data_moved.shape[-1]
    num_per_pack = 32
    block_size = 64  # hardware block size (Xh2Target::getBFPBlockSize)

    if length_of_packed % num_per_pack != 0:
        raise ValueError(
            f"axis size {length_of_packed} does not match num per pack {num_per_pack}"
        )

    # Pad pack axis to block_size alignment (matching C++ outputPackSize requirement)
    padded_length = ((length_of_packed + block_size - 1) // block_size) * block_size
    if padded_length != length_of_packed:
        pad_width = [(0, 0)] * (len(original_shape) - 1) + [
            (0, padded_length - length_of_packed)
        ]
        fp_data_moved = np.pad(
            fp_data_moved, pad_width, mode="constant", constant_values=0.0
        )
    padded_shape = fp_data_moved.shape

    if out_dtype == np.int8:
        indices = [16, 18, 20, 22, 24, 26, 28, 30]
        unsigned_dtype = np.uint8
        lsb_clear_mask = 0xFE
        exp_position_bitmap = 0x5555000055550000
    elif out_dtype == np.int16:
        indices = [24, 25, 26, 27, 28, 29, 30, 31]
        unsigned_dtype = np.uint16
        lsb_clear_mask = 0xFFFE
        exp_position_bitmap = 0xFF000000FF000000
    else:
        raise ValueError(f"output dtype {out_dtype} is not supported")

    # Pre-compute per-element exp-position flags for a 64-element block
    exp_pos_flags = [bool((exp_position_bitmap >> i) & 1) for i in range(block_size)]

    out_bits = np.iinfo(out_dtype).bits  # 8 or 16
    man_bits = 10  # fp16 mantissa bits
    fp16_exp_offset = 15
    fp32_exp_offset = 127
    extra_bits = (man_bits + 1) - (out_bits - 1)  # int8: 4, int16: -4
    int_digits = out_bits - 1  # int8: 7, int16: 15

    reshaped_fp = fp_data_moved.reshape(-1, padded_length).astype(np.float16)
    output = np.zeros(reshaped_fp.shape, dtype=out_dtype)

    for idx_outer in range(reshaped_fp.shape[0]):
        row = reshaped_fp[idx_outer]

        for blk_start in range(0, padded_length, block_size):
            block_fp16 = row[blk_start : blk_start + block_size].copy()

            # Sanitize inf/nan (matching C++ convertFPToG32E8)
            for i in range(block_size):
                bits = int(block_fp16[i].view(np.uint16))
                e = (bits >> 10) & 0x1F
                m = bits & 0x3FF
                if e == 0x1F:
                    if m != 0:
                        block_fp16[i] = np.float16(0)
                    elif (bits >> 15) & 1:
                        block_fp16[i] = np.finfo(np.float16).min
                    else:
                        block_fp16[i] = np.finfo(np.float16).max

            # getMaxExp<64>: find max biased fp16 exponent across the 64-element block
            target_exp = 0
            for i in range(block_size):
                e = int((block_fp16[i].view(np.uint16) >> 10) & 0x1F)
                target_exp = max(target_exp, e)

            # fillIntParts<64>: quantize each element using bit-level algorithm
            int_block = np.zeros(block_size, dtype=np.int32)
            for i in range(block_size):
                bits = int(block_fp16[i].view(np.uint16))
                sign = (bits >> 15) & 1
                e = (bits >> 10) & 0x1F
                man = bits & 0x3FF

                v = man + ((1 << man_bits) if e != 0 else 0)
                v_c = -v if sign else v

                exp_shift = target_exp - (
                    (e + 1) if (e == 0 and target_exp != 0) else e
                )
                is_exp = exp_pos_flags[i]

                rnd = _round_to_even_at_bit(
                    v_c, extra_bits + exp_shift + (1 if is_exp else 0)
                )

                if extra_bits > 0:
                    v_c >>= extra_bits
                else:
                    v_c <<= -extra_bits
                v_c >>= exp_shift

                # Overflow guard threshold: max abs value representable in
                # the output integer, accounting for the LSB reserved for
                # shared-exponent encoding at exp positions (C++ fillIntParts).
                thr = 0x7F if out_bits == 8 else (0x1FF if is_exp else 0xFF)
                if (v_c & thr) < thr or v_c == -1:
                    v_c += rnd

                int_block[i] = v_c

            # Clip and convert to output dtype
            info = np.iinfo(out_dtype)
            clipped = np.clip(int_block, info.min, info.max).astype(out_dtype)

            # Compute e8 exponent (matching C++ fillIntParts return + FP32_EXP_OFFSET)
            exp_offset = fp16_exp_offset - 1 if target_exp == 0 else fp16_exp_offset
            e8 = int(target_exp - exp_offset - (int_digits - 1) + fp32_exp_offset)
            e8_uint8 = e8 & 0xFF

            # fillSharedExpPart: encode e8 into LSBs of designated elements per 32-block
            for sub_start in range(0, block_size, num_per_pack):
                man_u = (
                    clipped[sub_start : sub_start + num_per_pack]
                    .view(unsigned_dtype)
                    .copy()
                )
                for bit_idx, elem_idx in enumerate(indices):
                    man_u[elem_idx] = (man_u[elem_idx] & lsb_clear_mask) | (
                        (e8_uint8 >> bit_idx) & 1
                    )
                output[
                    idx_outer,
                    blk_start + sub_start : blk_start + sub_start + num_per_pack,
                ] = man_u.view(out_dtype)

    result_moved = output.reshape(padded_shape)
    result = np.moveaxis(result_moved, -1, pack_axis)
    return result
