# Copyright 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Unified quantization script for
#   Z-Image-Turbo model (dit, vae, text_encoder) using post-training
#   quantization techniques.
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

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import os
import argparse
from pathlib import Path
import contextlib
import glob
import json
import shutil
import transformers.modeling_utils as modeling_utils
from diffusers import DiffusionPipeline, ZImagePipeline
import torch

if not hasattr(modeling_utils, "no_init_weights"):

    @contextlib.contextmanager
    def no_init_weights(_enable=True):
        yield

    modeling_utils.no_init_weights = no_init_weights

from xh_model_zoo.xh_llm.models.qwen2_legacy import Qwen2LegacyConvertConfig
from xhquant.api import (
    CacheTensor,
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    get_root_logger,
)
from xhquant.utils import MemoryTracker, TimeProfiler
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Define sub-model names, quantization dtypes, and converter classes
SUB_MODEL_NAMES = ["dit", "vae", "text_encoder"]
SUB_MODEL_QUANT_DTYPE = {
    "dit": torch.float16,
    "vae": torch.bfloat16,
    "text_encoder": torch.bfloat16,
}
SUB_MODEL_CONVERTER = {
    "dit": "xh_model_zoo.xh_llm.models.zimage.dit_converter:Dit_ConverterXH2a",
    "vae": "xh_model_zoo.xh_llm.models.zimage.vae_converter:VAE_ConverterXH2a",
    "text_encoder": "xh_model_zoo.xh_llm.models.zimage:Qwen3LegacyConverterXH2a",
}


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "z-image-turbo")
    model_size = model_config.get("model_size", "6b")
    return f"{model_name}-{model_size}"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # fmt: off
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model", type=str, default=None, help="input hf model path")
    parser.add_argument("--model_name", type=str, default=None, help="output hmonnx model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument("--quant_type", type=str, default=None, help="quant type, default is w8a8h1_sefp")
    parser.add_argument("--work_dir", type=str, default="work_dirs", help="working directory")
    parser.add_argument("--out_dir", type=str, default=os.path.join("output", HOUMO_TARGET, "hmquant"), help="output directory")
    parser.add_argument("--context_length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--input_sequence_length", type=int, default=None, help="input sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="device for quantization (e.g. cuda, cpu)")
    parser.add_argument("--quant_weight", type=str, default=None, help="quant weight path, e.g. gptq or quarot; if empty, use w8a8")
    parser.add_argument("--sub_models", type=str, nargs="+", default=None, choices=SUB_MODEL_NAMES, help="which sub-models to quantize (dit, vae, text_encoder). Default: quantize all three.")
    parser.add_argument("--dump_golden", action="store_true", default=False, help="dump golden data")
    parser.add_argument("--keep_work_dir", action="store_true", help="keep intermediate work_dir after reorganization (default: remove)")
    parser.add_argument("--export_demo_deps", type=str, default="auto", choices=["auto", "only", "none"], help="export demo dependencies: auto exports after successful quantization, only exports deps without quantization, none disables export")
    parser.add_argument("--demo_deps_dir", type=str, default=None, help="directory to store demo dependencies. Default: <out_dir>/hf_config")
    # fmt: on
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    args.input_sequence_length = first_not_none(
        args.input_sequence_length, model_config.get("prefill_length", 256)
    )
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8h1_sefp")
    )
    if args.sub_models is None:
        args.sub_models = SUB_MODEL_NAMES
    return args


# ---------------------------------------------------------------------------
# Demo dependency export
# ---------------------------------------------------------------------------


def get_demo_deps_dir(args) -> Path:
    if args.demo_deps_dir is not None:
        return Path(args.demo_deps_dir)
    return Path(args.out_dir) / "hf_config"


def export_demo_deps(args):
    print(f"Exporting demo dependencies for model: {args.model}")
    output_dir = get_demo_deps_dir(args)
    tokenizer_dir = output_dir / "tokenizer"
    scheduler_dir = output_dir / "scheduler"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )

    pipe.tokenizer.save_pretrained(tokenizer_dir)
    pipe.scheduler.save_config(scheduler_dir)
    torch.save(pipe.transformer.t_embedder.state_dict(), output_dir / "t_embedder.pt")

    transformer_config = {
        "in_channels": pipe.transformer.in_channels,
        "out_channels": pipe.transformer.out_channels,
        "patch_size": pipe.transformer.all_patch_size[0],
        "f_patch_size": pipe.transformer.all_f_patch_size[0],
        "vae_scale_factor": pipe.vae_scale_factor,
    }
    with (output_dir / "transformer_config.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(transformer_config, config_file, ensure_ascii=False, indent=2)

    print(f"Exported demo dependencies to {output_dir}")


# ---------------------------------------------------------------------------
# Golden data helpers (used by text_encoder)
# ---------------------------------------------------------------------------


def is_cuda_device(device) -> bool:
    return torch.device(device).type == "cuda"


def empty_cache_if_cuda(device):
    if is_cuda_device(device):
        torch.cuda.empty_cache()


def dump_golden_data(hmonnx_path, input_args, golden_dir, device):
    model = HMONNXGoldenInference(hmonnx_path)
    model.save_golden = True
    model.exec_device = torch.device(device)
    model.golden_dir = golden_dir
    with torch.no_grad():
        model.forward(*input_args)


def move_golden_data(golden_dir, output_dir):
    npy_files = list(Path(golden_dir).glob("*.npy"))
    if not npy_files:
        return

    output_golden_dir = Path(output_dir) / "step_0"
    output_golden_dir.mkdir(exist_ok=True, parents=True)

    for npy_file in npy_files:
        dst_file = output_golden_dir / npy_file.name
        if dst_file.exists():
            dst_file.unlink()
        shutil.copy2(str(npy_file), dst_file)


def build_prefill_golden_inputs(text_encoder, context_length, input_sequence_length):
    token_embedding = text_encoder.get_input_embeddings()
    device = token_embedding.weight.device
    input_ids = torch.randint(
        0, 1000, (1, input_sequence_length), dtype=torch.long, device=device
    )
    inputs_embeds = token_embedding(input_ids).to(torch.float16)

    num_hidden_layers = text_encoder.config.num_hidden_layers
    head_dim = text_encoder.layers[0].self_attn.head_dim
    num_decoder_layers = num_hidden_layers - 1
    kv_cache_shape = [
        1,
        text_encoder.config.num_key_value_heads,
        context_length,
        head_dim,
    ]

    past_key_caches = []
    past_value_caches = []
    for _ in range(num_decoder_layers):
        past_key_caches.append(
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
        )
        past_value_caches.append(
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
        )

    return (
        inputs_embeds,
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([input_sequence_length], dtype=torch.int32, device=device),
        *past_key_caches,
        *past_value_caches,
    )


def build_quant_context(args, torch_dtype):
    hf_model_dir = args.model
    device = args.device

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=args.quant_type
    )
    config = Qwen2LegacyConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
    )

    pipe = ZImagePipeline.from_pretrained(
        hf_model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    ).to(device)

    return device, config, pipe


def load_converter(path_and_class: str):
    module_name, class_name = path_and_class.split(":", maxsplit=1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def clear_pipe_attrs(pipe, *attr_names):
    for attr_name in attr_names:
        delattr(pipe, attr_name)


# ---------------------------------------------------------------------------
# Quantize functions for each sub-model
# ---------------------------------------------------------------------------


def quantize_dit(args):
    """Quantize the DiT (transformer) sub-model."""
    Dit_ConverterXH2a = load_converter(SUB_MODEL_CONVERTER["dit"])
    device, config, pipe = build_quant_context(args, SUB_MODEL_QUANT_DTYPE["dit"])

    work_dir = Path(args.work_dir) / "dit"
    work_dir.mkdir(exist_ok=True, parents=True)
    if not args.dump_golden:
        (work_dir / "golden" / "zimage_dit-XH2a-w8a8h1_sefp").mkdir(
            exist_ok=True, parents=True
        )

    clear_pipe_attrs(pipe, "text_encoder")
    empty_cache_if_cuda(device)

    logger = get_root_logger()
    with TimeProfiler("dit_convert", logger), MemoryTracker(
        device, "dit_convert", logger
    ):
        Dit_ConverterXH2a(config)._convert(
            pipe.transformer.half(), work_dir, pipe.image_processor.postprocess
        )


def quantize_vae(args):
    """Quantize the VAE sub-model."""
    VAE_ConverterXH2a = load_converter(SUB_MODEL_CONVERTER["vae"])
    device, config, pipe = build_quant_context(args, SUB_MODEL_QUANT_DTYPE["vae"])

    work_dir = Path(args.work_dir) / "vae"
    work_dir.mkdir(exist_ok=True, parents=True)
    if not args.dump_golden:
        (work_dir / "golden" / "zimage_vae-XH2a-w8a8h1_sefp").mkdir(
            exist_ok=True, parents=True
        )

    clear_pipe_attrs(pipe, "text_encoder", "transformer")
    empty_cache_if_cuda(device)

    logger = get_root_logger()
    with TimeProfiler("vae_convert", logger), MemoryTracker(
        device, "vae_convert", logger
    ):
        VAE_ConverterXH2a(config)._convert(
            pipe.vae.half(), work_dir, pipe.image_processor.postprocess
        )


def quantize_text_encoder(args):
    """Quantize the text encoder sub-model."""
    Qwen3LegacyConverterXH2a = load_converter(SUB_MODEL_CONVERTER["text_encoder"])
    device, config, pipe = build_quant_context(
        args, SUB_MODEL_QUANT_DTYPE["text_encoder"]
    )

    work_dir = Path(args.work_dir) / "text_encoder"
    work_dir.mkdir(exist_ok=True, parents=True)

    logger = get_root_logger()
    with TimeProfiler("text_encoder_convert", logger), MemoryTracker(
        device, "text_encoder_convert", logger
    ):
        text_encoder = pipe.text_encoder.half()
        Qwen3LegacyConverterXH2a(config)._convert(text_encoder, work_dir)

    if args.dump_golden:
        with open(work_dir / "meta.json", "r", encoding="utf-8") as meta_file:
            meta_info = json.load(meta_file)
        hmonnx_path = work_dir / meta_info["prefill_onnx"]
        golden_dir = work_dir / "golden" / Path(hmonnx_path).stem
        golden_dir.mkdir(exist_ok=True, parents=True)
        input_args = build_prefill_golden_inputs(
            text_encoder,
            args.context_length,
            args.input_sequence_length,
        )
        dump_golden_data(str(hmonnx_path), input_args, str(golden_dir), device)
        move_golden_data(golden_dir, hmonnx_path.parent)
        meta_info["prefill_golden_dir"] = str(
            (hmonnx_path.parent / "step_0").relative_to(work_dir)
        )
        with open(work_dir / "meta.json", "w", encoding="utf-8") as meta_file:
            json.dump(meta_info, meta_file, indent=4)


# ---------------------------------------------------------------------------
# Reorganize work_dirs into output directory
# ---------------------------------------------------------------------------


def _copy_tree(src: Path, dst: Path):
    """Copy src tree into dst, creating parents as needed."""
    if not src.exists():
        return
    if src.is_dir():
        if dst.is_dir():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(src, dst)


def _move_onnx_artifacts(
    work_sub_dir: Path, out_sub_dir: Path, hmonnx_subdir: str = None
):
    """Move hmonnx onnx + external_data + golden from work dir to output dir.

    For dit/vae, hmonnx files are in work_sub_dir/hmonnx/.
    For text_encoder, hmonnx files are in work_sub_dir/hmonnx/prefill/.

    Args:
        work_sub_dir: Source working directory (e.g., work_dir/dit)
        out_sub_dir: Destination output directory (e.g., out_dir/dit)
        hmonnx_subdir: Subdirectory under hmonnx/ to use (e.g., "prefill" for text_encoder)
    """
    hmonnx_src = work_sub_dir / "hmonnx"
    if hmonnx_subdir:
        hmonnx_src = hmonnx_src / hmonnx_subdir

    if hmonnx_src.exists():
        for item in hmonnx_src.iterdir():
            _copy_tree(item, out_sub_dir / item.name)

    golden_src = work_sub_dir / "golden"
    if golden_src.exists() and any(golden_src.glob("*.npy")):
        move_golden_data(golden_src, out_sub_dir)


def _build_final_onnx_name(args, sub_model_name: str) -> str:
    return (
        f"hmquant_{args.model_name}_{args.model_size}_{args.quant_type}_"
        f"{args.input_sequence_length}_{args.context_length}_{sub_model_name}.onnx"
    )


def _rename_final_onnx_artifact(
    args, out_sub_dir: Path, sub_model_name: str
) -> str | None:
    onnx_files = sorted(out_sub_dir.glob("*.onnx"))
    if not onnx_files:
        return None

    final_name = _build_final_onnx_name(args, sub_model_name)
    final_path = out_sub_dir / final_name
    source_path = next(
        (path for path in onnx_files if not path.name.startswith("hmquant_")),
        onnx_files[0],
    )
    if source_path != final_path:
        if final_path.exists():
            final_path.unlink()
        source_path.rename(final_path)

    for other_path in out_sub_dir.glob("*.onnx"):
        if other_path != final_path:
            other_path.unlink()

    return final_name


def move_to_output(args, failed_models: list = None):
    """Reorganize work_dirs into the canonical output layout.

    Target layout:
        out_dir/
        ├── dit/
        │   ├── <model>.onnx
        │   ├── <model>_external_data
        │   └── step_0/*.npy
        ├── vae/
        │   ├── <model>.onnx
        │   └── step_0/*.npy
        ├── text_encoder/
        │   ├── <model>_prefill.onnx
        │   ├── <model>_prefill_external_data
        │   └── step_0/*.npy
        ├── hf_config/
        │   ├── tokenizer/
        │   ├── scheduler/
        │   ├── t_embedder.pt
        │   └── transformer_config.json
        └── quant_embedding.pt

    Args:
        args: Command line arguments.
        failed_models: List of models that failed quantization.
                       These will be skipped during move.

    """
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if failed_models is None:
        failed_models = []

    sub_model_move_specs = {
        "dit": {
            "work_sub_dir": work_dir / "dit",
            "out_sub_dir": out_dir / "dit",
            "hmonnx_subdir": None,
            "copy_embedding": False,
        },
        "vae": {
            "work_sub_dir": work_dir / "vae",
            "out_sub_dir": out_dir / "vae",
            "hmonnx_subdir": None,
            "copy_embedding": False,
        },
        "text_encoder": {
            "work_sub_dir": work_dir / "text_encoder",
            "out_sub_dir": out_dir / "text_encoder",
            "hmonnx_subdir": "prefill",
            "copy_embedding": True,
        },
    }

    for sub_model_name in SUB_MODEL_NAMES:
        if sub_model_name in failed_models:
            continue

        spec = sub_model_move_specs[sub_model_name]
        work_sub_dir = spec["work_sub_dir"]
        out_sub_dir = spec["out_sub_dir"]
        if not work_sub_dir.exists():
            continue

        print(f"Moving {sub_model_name} artifacts: {work_sub_dir} -> {out_sub_dir}")
        out_sub_dir.mkdir(exist_ok=True, parents=True)
        _move_onnx_artifacts(work_sub_dir, out_sub_dir, spec["hmonnx_subdir"])
        _rename_final_onnx_artifact(args, out_sub_dir, sub_model_name)

        if spec["copy_embedding"]:
            emb_src = work_sub_dir / "token_embedding.pt"
            if emb_src.exists():
                shutil.copy2(emb_src, out_dir / "quant_embedding.pt")

    # Cleanup work_dir only if no failures
    if not failed_models and not args.keep_work_dir and work_dir.exists():
        print(f"Removing intermediate work_dir: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Output reorganized to: {out_dir}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

QUANTIZE_FN_MAP = {
    "dit": quantize_dit,
    "vae": quantize_vae,
    "text_encoder": quantize_text_encoder,
}


if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.export_demo_deps != "none":
        export_demo_deps(args)
        if args.export_demo_deps == "only":
            raise SystemExit(0)

    sub_models = args.sub_models
    print(f"Sub-models to quantize: {sub_models}")

    failed_models = []
    for sub_model in sub_models:
        print(f"\n{'='*60}")
        print(f"Quantizing sub-model: {sub_model}")
        print(f"{'='*60}")
        p = mp.Process(
            target=QUANTIZE_FN_MAP[sub_model],
            args=(args,),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"{sub_model} failed with exit code {p.exitcode}")
            failed_models.append(sub_model)
    # Reorganize work_dirs into output/xh2/hmquant/
    move_to_output(args, failed_models)

    if failed_models:
        print(f"\n=== Quantization failed for: {failed_models}. ===")
    print(f"\n=== All quantization steps completed. ===")
