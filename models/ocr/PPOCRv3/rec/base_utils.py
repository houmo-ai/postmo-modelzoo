#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: base_utils.py
# Description:
#   Basic functions for the PPOCRv3 recognition model example project.
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
import os
import onnx
import math
import torch
import cv2
import numpy as np
from hmatc.utils import logger
from hmatc.utils.utils import *

HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '/usr/local/examples/data/datasets')
HOUMO_TARGET = os.getenv('HOUMO_TARGET', '')
YUV_FORMAT = 'YUV420'
SUB_QUANT_PATH = {
    "xh2": "xhquant"
}
NORAM_DIST = {
    "xh2": 0.99
}
DYNAMIC_RESIZE = {
    "xh2": False
}

MAX_INPUT_SIZE = [1184, 736]

INPUT_DATA_TYPES = [".jpg", ".png", "jpeg", ".JPG", ".JPEG", ".PNG", ".bmp", ".BMP"]

def get_net_input_output_infos(model_path):
    if not os.path.exists(model_path):
        logger.error(f"{model_path} is not found!")
        assert 0
    onnx_model = onnx.load_model(model_path)
    input_infos_list = []
    for idx, net_input in enumerate(onnx_model.graph.input):
        input_name = net_input.name
        #net_input_size = []
        input_shape = [d.dim_value if d.dim_value > 0 else 1 for d in net_input.type.tensor_type.shape.dim]
        input_info = {
            'name': input_name,
            'input_shape': input_shape,
            'dtype': onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(net_input.type.tensor_type.elem_type).name
        }
        input_infos_list.append(input_info)
    output_infos_list = []
    for idx, net_output in enumerate(onnx_model.graph.output):
        output_name = net_output.name
        output_info = {
            'name': output_name,
            'dtype': onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(net_output.type.tensor_type.elem_type).name
        }
        output_infos_list.append(output_info)
    return input_infos_list, output_infos_list

def xh_preprocess(img, net_input_size):
    src_h, src_w, src_c = img.shape
    max_h, max_w = MAX_INPUT_SIZE
    if src_h > max_h or src_w > max_w:
        ratio = min(max_h / src_h, max_w / src_w)
        new_w = int(src_w * ratio)
        new_h = int(src_h * ratio)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        src_h, src_w, _ = img.shape
    dst_map = np.zeros((max_h, max_w, src_c), dtype=np.uint8)
    dst_map[:src_h, :src_w, :] = img
    dst_h, dst_w = net_input_size
    crop_h = src_h if src_h % 2 == 0 else src_h - 1
    crop_w = src_w if src_w % 2 == 0 else src_w - 1
    # crop_h = src_h
    # crop_w = src_w
    dst_ratio = crop_w / float(crop_h)
    if math.ceil(dst_h * dst_ratio) > dst_w:
        resized_w = dst_w
    else:
        resized_w = int(math.ceil(dst_h * dst_ratio)) % 2 + int(math.ceil(dst_h * dst_ratio))
    resized_h = dst_h
    crop_info = [0, 0, crop_h, crop_w, resized_h, resized_w, 0, 0, 0, dst_w - resized_w]
    crop_info = np.array([crop_info], dtype=np.int32)
    dst_map = dst_map.astype(np.float32)
    dst_map = np.transpose(dst_map, (2, 0, 1))
    return torch.from_numpy(dst_map).unsqueeze(0), torch.from_numpy(crop_info)

def onnx_preprocess(img, net_input_size):
    dst_h, dst_w = net_input_size
    src_h, src_w, src_c = img.shape
    env_h = src_h if src_h % 2 == 0 else src_h - 1
    env_w = src_w if src_w % 2 == 0 else src_w - 1
    if env_h < src_h or env_w < src_w:
        img = img[0:env_h, 0:env_w, :]
        src_h, src_w = (env_h, env_w)
    dst_ratio = src_w / float(src_h)
    if math.ceil(dst_h * dst_ratio) > dst_w:
        resized_w = dst_w
    else:
        resized_w = int(math.ceil(dst_h * dst_ratio)) % 2 + int(math.ceil(dst_h * dst_ratio))
    resized_h = dst_h
    resized_img = cv2.resize(img, (resized_w, resized_h), cv2.INTER_LINEAR)
    resized_img = np.transpose(resized_img, (2, 0, 1)) / 255.
    resized_img -= 0.5
    resized_img /= 0.5
    resized_img = resized_img.astype(np.float32)
    dst_img = np.zeros((1, src_c, dst_h, dst_w), dtype=np.float32)
    dst_img[:, :, :, 0:resized_w] = resized_img
    return dst_img

def cosine_distance(data1, data2):
    if data1.shape != data2.shape:
        logger.error(f"[error] shape not equal {data1.shape} vs {data2.shape}")
        return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return cosine_dist
