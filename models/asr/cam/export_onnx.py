# Copyright (c) 2025 HOUMO AI
#
# File: export_onnx.py
# Description:
#   CAMPPlus ONNX Export Tool - Export PyTorch model to ONNX format.
#   This script is provided for standalone ONNX export usage.
#   When running full quantization via ptq.py, ONNX export is handled automatically.
#
import argparse
import os

import onnx
import onnxruntime as ort
import torch

from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
OUTPUT_DIR = "./work_dirs/cam/export_fp32"
OUTPUT_ONNX_PATH = os.path.join(OUTPUT_DIR, "onnx/model.onnx")
OUTPUT_SIMPLIFIED_PATH = os.path.join(OUTPUT_DIR, "onnx/model_simplified.onnx")
FEATURE_DIM = 80
DEFAULT_FIXED_FRAMES = 529


def inspect_onnx(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "campplus")
    return model_name


def export_onnx(model_dir, fixed_frames):
    from modelscope.pipelines import pipeline

    sv_pipeline = pipeline(
        "speaker-verification",
        model=model_dir,
        model_revision="v1.0.0",
    )
    embedding_model = sv_pipeline.model.embedding_model.cpu().float().eval()
    dummy_input = torch.randn(1, fixed_frames, FEATURE_DIM)

    os.makedirs(os.path.dirname(OUTPUT_ONNX_PATH), exist_ok=True)

    torch.onnx.export(
        embedding_model,
        dummy_input,
        OUTPUT_ONNX_PATH,
        input_names=["feature"],
        output_names=["embedding"],
        opset_version=14,
    )

    print(f"Exported ONNX model to {OUTPUT_ONNX_PATH}")

    onnx_model = onnx.load(OUTPUT_ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed")
    inspect_onnx(OUTPUT_ONNX_PATH)


def simplify_onnx():
    import onnxsim

    model = onnx.load(OUTPUT_ONNX_PATH)
    model_simplified, check = onnxsim.simplify(model)
    onnx.save(model_simplified, OUTPUT_SIMPLIFIED_PATH)
    print(f"Simplified ONNX model saved to {OUTPUT_SIMPLIFIED_PATH}, check={check}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="model name"
    )
    parser.add_argument(
        "--model-size", type=str, default=None, help="model size"
    )
    parser.add_argument("--step", type=str, default="all", choices=["export", "simplify", "all"])
    parser.add_argument("--fixed-frames", type=int, default=DEFAULT_FIXED_FRAMES)
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    model_name = first_not_none(args.model_name, default_model_name)
    model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(model_name, {}).get(model_size, {})
    model_dir = get_default_model_dir(model_config)

    if args.step in ["export", "all"]:
        export_onnx(model_dir, args.fixed_frames)

    if args.step in ["simplify", "all"]:
        simplify_onnx()