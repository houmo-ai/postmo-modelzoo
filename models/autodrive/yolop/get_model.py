import os
import sys
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import argparse
from pathlib import Path
from hmatc.utils.utils import get_file_from_jfrog, get_package_version


HOUMO_TARGET = os.getenv("HOUMO_TARGET", "houmo")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

runtime_version = get_package_version(f"houmo_tcim_runtime_{HOUMO_TARGET}")
runtime_version = runtime_version.split(".dev")[0]


def focus2conv(model_path, new_model_path):
    # 加载原始模型
    onnx_model = onnx.load(model_path)
    graph = gs.import_onnx(onnx_model)

    # 假设我们替换某个已有 Conv/ReOrg 节点
    first_conv_node = None
    for node in graph.nodes:
        if node.name == "/model.0/Concat":
            node.outputs = []
        if node.name == "/model.0/conv/conv/Conv":
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
        name="focus_out", dtype=np.float32, shape=[1, 12, 192, 320]
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
    # 清理 & 保存
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
        default="raw",
        help="which model type to get, choise in [raw, quant, hmm, all]",
    )
    parser.add_argument(
        "--quant_model_dir",
        dest="quant_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="where to save quant_model",
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
        default="",
        help="where to save downloaded model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    quant_model_dir = args.quant_model_dir
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir

    model_name = "yolop"
    ncore = 1
    batch = 1
    opt_level = "O2"
    version = f"v{runtime_version}"
    target = HOUMO_TARGET
    raw_path = f"models/{model_name}/yolop_384x640.onnx"
    quant_path = f"models/{model_name}/hmquant_{model_name}_{target}_{version}.tar.xz"
    build_path = f"models/{model_name}/{model_name}_{target}_b{batch}_{ncore}core_{opt_level}_{version}.tar.xz"

    if model_type in ["raw", "all"]:
        file_path = get_file_from_jfrog(raw_path, model_dir)
        # if file_path:
        #     new_file_path = os.path.join(
        #         os.path.dirname(file_path), "yolop_384x640_focus2conv.onnx"
        #     )
        #     focus2conv(file_path, new_file_path)
        # else:
        #     sys.exit(1)

    if model_type in ["quant", "all"] and not get_file_from_jfrog(
        quant_path, model_dir, quant_model_dir
    ):
        sys.exit(1)

    if model_type in ["hmm", "all"] and not get_file_from_jfrog(
        build_path, model_dir, build_model_dir
    ):
        sys.exit(1)
