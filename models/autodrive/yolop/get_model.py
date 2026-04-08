# Copyright 2025 HOUMO AI
#
# File: get_model.py
# Description:
#   Download Yolop model for autonomous driving tasks.
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
import sys
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import argparse
from hmatc.utils.utils import get_file_from_jfrog, get_houmo_version


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def focus2conv(model_path, new_model_path):
    # Load original model
    onnx_model = onnx.load(model_path)
    graph = gs.import_onnx(onnx_model)

    # Assume we replace an existing Conv/ReOrg node
    first_conv_node = None
    for node in graph.nodes:
        if node.name == "Concat_40":
            node.outputs = []
        if node.name == "Conv_41":
            first_conv_node = node

    weight = np.array(
        [
            [[[1, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]]],
            [[[0, 1], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 1], [0, 0]]],
            [[[0, 0], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [1, 0]]],
            [[[0, 0], [0, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 1]]],
        ],
        dtype=np.float32,
    )

    input_tensor = graph.inputs[0]
    weight_const = gs.Constant(name="focus_weight", values=weight)
    output_tensor = gs.Variable(
        name="focus_out", dtype=np.float32, shape=[1, 12, 320, 320]
    )
    first_conv_node.inputs[0] = output_tensor

    focus_node = gs.Node(
        op="Conv",
        name="focus",
        inputs=[input_tensor, weight_const],
        outputs=[output_tensor],
        attrs={
            "kernel_shape": [2, 2],
            "strides": [2, 2],
            "pads": [0, 0, 0, 0],
            "dilations": [1, 1],
            "group": 1,
        },
    )

    graph.nodes.append(focus_node)
    # Clean & save
    graph.cleanup().toposort()
    new_model = gs.export_onnx(graph)
    new_model.ir_version = 8
    onnx.save(new_model, new_model_path)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest="model_type",
        type=str,
        default="hmm",
        choices=["raw", "hmm"],
        help="which model type to get, choise in [raw, hmm]",
    )
    parser.add_argument(
        "--build_model_dir",
        dest="build_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="where to save build_model",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=".",
        help="where to save downloaded model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir

    model_name = "yolop"
    ncore = 1
    batch = 1
    opt_level = "O2"
    version = get_houmo_version()
    target = HOUMO_TARGET
    raw_path = "models/raw/onnx/yolop_640x640.onnx"
    build_path = f"models/{target.lower()}-{version}/{model_name}/{model_name}_{target}_b{batch}_{ncore}core_{opt_level}_{version}.tar.xz"

    if model_type in ["raw"]:
        file_path = get_file_from_jfrog(raw_path, model_dir)
        extract_path = os.path.join(
            os.path.dirname(file_path), "yolop_640x640_clip.onnx"
        )
        onnx.utils.extract_model(
            file_path,
            extract_path,
            input_names=["images"],
            output_names=["1236", "1566", "1896", "drive_area_seg", "lane_line_seg"],
            check_model=True,
        )
        new_model_path = extract_path.replace(".onnx", "_opt.onnx")
        focus2conv(extract_path, new_model_path)
        if not file_path:
            sys.exit(1)

    if model_type in ["hmm"] and not get_file_from_jfrog(
        build_path, model_dir, build_model_dir
    ):
        sys.exit(1)
