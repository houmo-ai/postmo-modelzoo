# Copyright (c) 2025 HOUMO AI
#
# File: quant_config.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# Qwen3 models using post-training quantization techniques.
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
res_cfg = dict(dtype="int16",granularity="tensor",calib_metric="minmax")
w_default = dict(dtype="int4", calib_metric="minmax", granularity="dim0")

""""""
emb_quant = dict(
    calib_metric="minmax",
    dtype = "int16",
    granularity="tensor",
)
# emb_quant = res_cfg
input_norm = dict(calib_metric="minmax")

q_proj = dict(
    w_cfg=w_default,
    o_cfg=dict(calib_metric="minmax"),
)
k_proj = dict(
    w_cfg=w_default,
    o_cfg=dict(calib_metric="minmax"),
)
v_proj = dict(
    w_cfg=w_default,
    o_cfg=dict(calib_metric="minmax"),
)
pos_quant = dict(dtype="int8", calib_metric="minmax")
apply_ropeq = dict(calib_metric="minmax",dtype="int8",granularity="tensor")
apply_ropek = dict(calib_metric="minmax",dtype="int8",granularity="tensor")
qkmat = dict(calib_metric="minmax",dtype="int16")
softmax = dict(calib_metric="minmax", dtype="int16")
pvmat = dict(calib_metric="minmax")
o_proj = dict(
    w_cfg=w_default,
    o_cfg = res_cfg # o_cfg=dict(calib_metric="minmax", granularity="dim2", tensor_shape=[1, 1, 3584]),
)
# attn_res_add = dict(
#     calib_metirc="minmax", tensor_shape=[1, 1, 4096], granularity="dim2"
# )
attn_res_add = res_cfg
post_attn_norm = dict(calib_metric="minmax")
gate_proj = dict(
    w_cfg=w_default,
    o_cfg=dict(calib_metric="minmax",dtype="int16"),
)
up_proj = dict(
    w_cfg=w_default,
    o_cfg=dict(calib_metric="minmax",dtype="int16"),
)
silu = dict(calib_metric="minmax",dtype="int16")
mul = dict(calib_metric="minmax",dtype="int16")
down_proj = dict(
    w_cfg=w_default,
    o_cfg=res_cfg #dict(calib_metric="minmax", granularity="dim1", tensor_shape=[1, 4096]),
    #o_cfg =dict(calib_metric="minmax", granularity="dim2", tensor_shape=[ 1, 1, 3584]) #res_cfg
)
mlp_res_add = res_cfg
last_norm = dict(calib_metric="minmax",dtype="int8")  # 可以做最后一个rmsnorm和最后一个linear的smooth操作
lm_head = dict(
    w_cfg = dict(dytpe="int8",granularity="dim0"),
    o_cfg = dict(calib_metric="minmax",dtype="int16")
)
