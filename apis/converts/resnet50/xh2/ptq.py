import os
import shutil
import torch
import argparse
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
)

HOUMO_MODEL_PATH = os.getenv("HOUMO_MODEL_PATH", "../../../models")
HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", ".")
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "houmo")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default=os.path.join(HOUMO_MODEL_PATH, "resnet50.onnx"),
        help="path to the model path",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default="resnet50",
        help="model name",
    )
    parser.add_argument(
        "--input_shape",
        dest="input_shape",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[1, 3, 224, 224],
        help="new input shape if want change",
    )
    parser.add_argument(
        "--dynamic_resize",
        dest="dynamic_resize",
        action="store_true",
        help="whether to set dynamic crop/resize/pad",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="path to the quanted model dir",
    )
    args = parser.parse_args()
    return args


def calibrate(args=None):
    model_path = args.model_path
    model_name = args.model_name
    output_path = args.model_dir
    input_shape = args.input_shape
    input_name = "input.1"  # onnx输入名字
    output_name = "495"  # onnx输出名字
    quant_type = "w8a8h1_sefp"
    device = "cpu"
    hmonnx_model_path = os.path.join(output_path, f"{model_name}.onnx")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    random_data = torch.randn(input_shape, dtype=torch.float32)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    # 量化以及HMONNX导出
    convert_onnx_to_hmonnx(
        model_path,
        [random_data],
        device_type=DeviceType.XH2a,
        out_hmonnx_file=hmonnx_model_path,
        quant_config=quant_config,
        input_names=[input_name],
        output_names=[output_name],
    )

    print("start save model and generate golden...")
    golden_dir = os.path.join(output_path, "golden")
    if not os.path.exists(golden_dir):
        os.makedirs(golden_dir)
    else:
        shutil.rmtree(golden_dir)
    session = HMONNXGoldenInference(hmonnx_model_path)
    session.to(device)
    session.save_golden = True
    session.golden_dir = golden_dir
    session.step = 0
    # to float16
    session(random_data.half().to(device))
    if os.path.exists(hmonnx_model_path):
        os.remove(hmonnx_model_path)
    shutil.copytree(
        os.path.join(golden_dir, "step_0"),
        output_path,
        dirs_exist_ok=True,
    )
    shutil.rmtree(golden_dir)
    print("save model and generate golden completed.")


if __name__ == "__main__":
    import platform

    arch = platform.machine()
    if arch != "x86_64":
        print(f"[error] hmquant not support platform: {arch}")
        exit(0)
    args = get_args()
    print(args)
    calibrate(args)
