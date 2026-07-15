#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-training quantization for split SigLIP2 encoders.
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
"""Export and quantize split SigLIP2 encoders with the custom model flow.

The script loads a local Hugging Face model, saves tokenizer assets, exports
static normalized vision/text ONNX encoders, then converts them to hmonnx. The
vision encoder uses dynamic resizer preprocessing.
"""

import argparse
import gc
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
VISION_ONNX = "siglip2_large_patch16_256_vision.onnx"
TEXT_ONNX = "siglip2_large_patch16_256_text.onnx"
VISION_NAME = "siglip2_large_patch16_256_vision"
TEXT_NAME = "siglip2_large_patch16_256_text"
IMAGE_SIZE = 256
SEQ_LEN = 64
OPSET_VERSION = 17
DEFAULT_MODEL_DIR = "siglip2-large-patch16-256"
DEFAULT_RESIZER_INPUT_SIZE = [1080, 1920]

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None, help="local Hugging Face model directory"
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="model name for output files"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="model size identifier for output files",
    )
    parser.add_argument(
        "--out_dir",
        default=f"output/{HOUMO_TARGET}/hmquant",
        help="directory for tokenizer, ONNX, and hmonnx artifacts",
    )
    parser.add_argument("--quant_type", default="w8a8_sefp", help="xhquant quant type")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument(
        "--resizer_input_size",
        type=int,
        nargs=2,
        default=DEFAULT_RESIZER_INPUT_SIZE,
        metavar=("H", "W"),
        help="dynamic resizer source height and width",
    )
    parser.add_argument("--skip_vision", action="store_true")
    parser.add_argument("--skip_text", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_target():
    if HOUMO_TARGET != "xh2":
        raise SystemExit(f"Only support HOUMO_TARGET=xh2, got {HOUMO_TARGET!r}")


def validate_model_dir(model_dir):
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Hugging Face model directory not found: {model_dir}")


def output_attr(outputs, *names):
    for name in names:
        if hasattr(outputs, name):
            value = getattr(outputs, name)
            if value is not None:
                return value
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise RuntimeError(f"Cannot find output attribute from {names}")


class SigLIP2VisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        return F.normalize(output_attr(outputs, "pooler_output"), p=2, dim=-1)


class SigLIP2TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.text_model
        self.text_model.config._attn_implementation = "eager"

    def forward(self, input_ids, attention_mask):
        # Passing a 4D mask avoids a Transformers tracing bug in mask creation.
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
            attention_mask = attention_mask[:, None, None, :]

        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(output_attr(outputs, "pooler_output"), p=2, dim=-1)


def external_data_paths(model_path):
    import onnx

    if not model_path.is_file():
        return set()

    paths = set()
    model = onnx.load(str(model_path), load_external_data=False)
    for tensor in model.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location" and item.value:
                paths.add(model_path.parent / item.value)
    return paths


def text_onnx_complete(onnx_path):
    data_path = onnx_path.with_name(f"{onnx_path.name}.data")
    return onnx_path.is_file() and data_path.is_file() and data_path.stat().st_size > 0


def save_single_external_data(model, output_path):
    import onnx

    data_name = f"{output_path.name}.data"
    data_path = output_path.with_name(data_name)
    old_data_paths = external_data_paths(output_path)

    with tempfile.TemporaryDirectory(
        prefix=".siglip2_publish_", dir=output_path.parent
    ) as tmp:
        tmp_dir = Path(tmp)
        tmp_model_path = tmp_dir / output_path.name
        tmp_data_name = f"{tmp_dir.name}.data"
        tmp_data_path = tmp_dir / tmp_data_name

        onnx.save_model(
            model,
            str(tmp_model_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=tmp_data_name,
            size_threshold=0,
            convert_attribute=False,
        )
        if not tmp_data_path.is_file() or tmp_data_path.stat().st_size == 0:
            raise RuntimeError(f"External data file was not created: {tmp_data_path}")

        staged = onnx.load(str(tmp_model_path), load_external_data=False)
        for tensor in staged.graph.initializer:
            for item in tensor.external_data:
                if item.key == "location":
                    item.value = data_name
        tmp_model_path.write_bytes(staged.SerializeToString())

        os.replace(tmp_data_path, data_path)
        os.replace(tmp_model_path, output_path)

    # Use path-based checking after publishing the .onnx.data file. Checking the
    # in-memory model before the data file is available at its final relative
    # location fails for large external-data models.
    onnx.checker.check_model(str(output_path))

    for old_path in old_data_paths:
        if old_path != data_path and old_path.is_file():
            old_path.unlink()


def export_onnx(
    model, output_path, dummy_inputs, input_names, output_names, external_data=False
):
    model.eval().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nExporting ONNX:")
    print(f"  Output:       {output_path}")
    print(f"  Input names:  {input_names}")
    print(f"  Input shapes: {[list(x.shape) for x in dummy_inputs]} (static)")
    print(f"  Output names: {output_names}")
    print(f"  Opset:        {OPSET_VERSION}")

    with tempfile.TemporaryDirectory(prefix="siglip2_onnx_") as tmp:
        export_path = Path(tmp) / output_path.name if external_data else output_path
        torch.onnx.export(
            model,
            tuple(dummy_inputs),
            str(export_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
        )

        if external_data:
            import onnx

            onnx_model = onnx.load(str(export_path), load_external_data=True)
            save_single_external_data(onnx_model, output_path)
            data_path = output_path.with_name(f"{output_path.name}.data")
            print(f"  Graph:        {output_path.stat().st_size / 1024**2:.1f} MB")
            print(
                f"  External data: {data_path} "
                f"({data_path.stat().st_size / 1024**2:.1f} MB)"
            )
        else:
            print(f"  Done:         {output_path.stat().st_size / 1024**2:.1f} MB")


def verify_onnx(output_path, model, input_feed, torch_inputs):
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        print(f"WARNING: Cannot verify ONNX - {exc}")
        return True

    print(f"\nVerification: {output_path}")
    onnx.checker.check_model(str(output_path))
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    for inp in session.get_inputs():
        print(f"  Input:  name={inp.name}, shape={inp.shape}, type={inp.type}")
        if not all(isinstance(dim, int) for dim in inp.shape):
            print(f"[FAIL] Non-static input shape: {inp.name} {inp.shape}")
            return False
        if inp.name not in input_feed:
            print(f"[FAIL] Missing feed for input: {inp.name}")
            return False
        if list(input_feed[inp.name].shape) != list(inp.shape):
            print(f"[FAIL] Feed shape mismatch for {inp.name}")
            return False

    for out in session.get_outputs():
        print(f"  Output: name={out.name}, shape={out.shape}, type={out.type}")
        if not all(isinstance(dim, int) for dim in out.shape):
            print(f"[FAIL] Non-static output shape: {out.name} {out.shape}")
            return False

    ort_out = session.run(None, input_feed)[0]
    with torch.no_grad():
        pt_out = model(*torch_inputs).cpu().numpy()

    diff = abs(pt_out - ort_out).max()
    pt_flat = pt_out.astype("float64").ravel()
    ort_flat = ort_out.astype("float64").ravel()
    cos = float(
        np.dot(pt_flat, ort_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(ort_flat))
    )
    norm_ok = np.allclose(np.linalg.norm(ort_out, axis=-1), 1.0, rtol=0, atol=1e-3)

    print(f"  Max absolute difference: {diff:.6e}")
    print(f"  Cosine similarity:      {cos:.6f}")
    print(f"  Output L2 norm:         {np.linalg.norm(ort_out, axis=-1)}")
    return bool(np.isfinite(cos) and cos > 0.9999 and diff < 1e-4 and norm_ok)


def save_tokenizer(model_dir, out_dir, overwrite=False):
    marker = out_dir / "tokenizer_config.json"
    if marker.is_file() and not overwrite:
        print(f"Reuse existing tokenizer: {out_dir}")
        return

    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    tokenizer.save_pretrained(str(out_dir))
    print(f"Tokenizer saved to {out_dir}")


def export_models(model_dir, onnx_dir, args):
    from transformers import AutoModel

    vision_path = onnx_dir / VISION_ONNX
    text_path = onnx_dir / TEXT_ONNX
    export_vision = not args.skip_vision and (
        args.overwrite or not vision_path.is_file()
    )
    export_text = not args.skip_text and (
        args.overwrite or not text_onnx_complete(text_path)
    )

    if not args.skip_vision and not export_vision:
        print(f"Reuse existing ONNX: {vision_path}")
    if not args.skip_text and not export_text:
        print(f"Reuse existing ONNX: {text_path}")
    if not export_vision and not export_text:
        return vision_path, text_path

    print(f"Loading Hugging Face model from {model_dir}")
    base_model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    base_model.eval()
    torch.manual_seed(42)

    if export_vision:
        vision_model = SigLIP2VisionEncoder(base_model)
        vision_dummy = torch.randn(1, 3, args.image_size, args.image_size)
        export_onnx(
            vision_model,
            vision_path,
            [vision_dummy],
            ["pixel_values"],
            ["image_embeds"],
        )
        if not verify_onnx(
            vision_path,
            vision_model,
            {"pixel_values": vision_dummy.numpy()},
            [vision_dummy],
        ):
            raise RuntimeError(f"ONNX verification failed: {vision_path}")
        del vision_model, vision_dummy

    if export_text:
        text_model = SigLIP2TextEncoder(base_model)
        input_ids = torch.ones(1, args.seq_len, dtype=torch.long)
        attention_mask = torch.ones(1, args.seq_len, dtype=torch.long)
        export_onnx(
            text_model,
            text_path,
            [input_ids, attention_mask],
            ["input_ids", "attention_mask"],
            ["text_embeds"],
            external_data=True,
        )
        if not verify_onnx(
            text_path,
            text_model,
            {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
            [input_ids, attention_mask],
        ):
            raise RuntimeError(f"ONNX verification failed: {text_path}")
        del text_model, input_ids, attention_mask

    del base_model
    gc.collect()
    return vision_path, text_path


def hmonnx_complete(model_path):
    if not model_path.is_file():
        return False
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in external_data_paths(model_path)
    )


def remove_hmonnx(model_path):
    external_paths = external_data_paths(model_path)
    model_path.unlink(missing_ok=True)
    for path in external_paths:
        path.unlink(missing_ok=True)


def quant_config(input_ppc_config, quant_type):
    from xhquant.api import DeviceType, QuantScheme, create_quant_config

    return create_quant_config(
        QuantScheme(
            target_device=DeviceType.XH2a,
            quant_type=quant_type,
            input_ppc_config=input_ppc_config,
            output_enable_fp32=True,
        )
    )


def dynamic_resizer_config(image_size, resizer_input_size):
    from xhquant.api import ResizerScheme

    input_h, input_w = resizer_input_size
    return ResizerScheme(
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        fmt="yuv420",
        int_trans=True,
        crop_size=(input_h, input_w),
        crop_offset=(0, 0),
        pad_size=(0, 0, 0, 0),
        pad_value=0,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        dynamic_crop=True,
        model_inp_fmt="rgb",
    ).to_dict()


def bgr_to_yuv420sp_nchw(bgr):
    bgr = bgr.to(torch.float32)
    blue, green, red = bgr[:, 0:1], bgr[:, 1:2], bgr[:, 2:3]
    y = 0.114 * blue + 0.587 * green + 0.299 * red
    u = 0.5 * blue - 0.331 * green - 0.169 * red + 128
    v = -0.081 * blue - 0.419 * green + 0.5 * red + 128
    uv = torch.stack(
        [
            F.interpolate(u, scale_factor=0.5, mode="bilinear", align_corners=False),
            F.interpolate(v, scale_factor=0.5, mode="bilinear", align_corners=False),
        ],
        dim=-1,
    )
    yuv = torch.cat([y.flatten(1), uv.flatten(1)], dim=1)
    padded = torch.zeros_like(bgr.flatten(1), dtype=torch.float32)
    padded[:, : yuv.shape[1]] = yuv
    return padded.clamp(0, 255).round().to(torch.uint8).view_as(bgr)


def make_vision_inputs(image_size, resizer_input_size):
    input_h, input_w = resizer_input_size
    bgr = torch.randint(0, 256, (1, 3, input_h, input_w), dtype=torch.uint8)
    dyn = torch.tensor(
        [[0, 0, input_h, input_w, image_size, image_size, 0, 0, 0, 0]],
        dtype=torch.int32,
    )
    return [bgr_to_yuv420sp_nchw(bgr), dyn]


def make_text_inputs(seq_len):
    input_ids = torch.ones(1, seq_len, dtype=torch.int32)
    attention_mask = torch.ones(1, seq_len, dtype=torch.int32)
    return [input_ids, attention_mask]


def convert_model(
    onnx_path,
    out_path,
    args,
    input_tensors,
    input_names,
    output_names,
    input_ppc_config,
):
    from xhquant.api import DeviceType, convert_onnx_to_hmonnx, xhquant_init

    xhquant_init()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        remove_hmonnx(out_path)
    if hmonnx_complete(out_path):
        print(f"Skip existing hmonnx: {out_path}")
        return
    if out_path.exists():
        raise RuntimeError(f"Incomplete hmonnx artifacts: {out_path}")

    print(f"Quantizing {onnx_path} -> {out_path}")
    convert_onnx_to_hmonnx(
        str(onnx_path),
        input_tensors,
        device_type=DeviceType.XH2a,
        out_hmonnx_file=str(out_path),
        quant_config=quant_config(input_ppc_config, args.quant_type),
        input_names=input_names,
        output_names=output_names,
    )


def get_model_configs(
    config_path: str,
    config_key: str = "model_configs",
):
    """Get model configs from config.yaml.

    Args:
        config_path (str): Path to config yaml.
        config_key (str): Config key to read, default is ``model_configs``.

    Returns:
        tuple:
            - ``(default_model_size, default_model_name, model_configs)``.
    """
    config = {}
    if not config_path or config_path is None or not os.path.exists(config_path):
        return "", {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    default_model_name = config.get("default_model_name", "")
    default_model_size = config.get("default_model_size", "")
    model_configs = config.get(config_key, {}) or {}

    return default_model_size, default_model_name, model_configs


def first_not_none(*values):
    """Return the first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "siglip2")
    model_size = model_config.get("model_size", "large-patch16-256")
    return f"{model_name}-{model_size}"


def main():
    ensure_target()
    args = parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model_dir = os.path.abspath(
        first_not_none(args.model_dir, get_default_model_dir(model_config))
    )

    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    onnx_dir = out_dir / "onnx"
    vision_dir = out_dir / "vision"
    text_dir = out_dir / "text"

    validate_model_dir(model_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    hf_config_dir = out_dir / "hf_config"
    hf_config_dir.mkdir(parents=True, exist_ok=True)

    save_tokenizer(model_dir, hf_config_dir, args.overwrite)
    vision_onnx, text_onnx = export_models(model_dir, onnx_dir, args)

    if not args.skip_vision:
        if not vision_onnx.is_file():
            raise FileNotFoundError(vision_onnx)
        convert_model(
            vision_onnx,
            vision_dir / f"hmquant_{VISION_NAME}_with_act.onnx",
            args,
            make_vision_inputs(args.image_size, args.resizer_input_size),
            ["pixel_values", "resizer_crop_pixel_values"],
            ["image_embeds"],
            [dynamic_resizer_config(args.image_size, args.resizer_input_size)],
        )

    if not args.skip_text:
        if not text_onnx_complete(text_onnx):
            raise FileNotFoundError(
                f"Incomplete text ONNX artifacts: {text_onnx} and {text_onnx.name}.data"
            )
        convert_model(
            text_onnx,
            text_dir / f"hmquant_{TEXT_NAME}_with_act.onnx",
            args,
            make_text_inputs(args.seq_len),
            ["input_ids", "attention_mask"],
            ["text_embeds"],
            ["float16", "float16"],
        )


if __name__ == "__main__":
    main()
