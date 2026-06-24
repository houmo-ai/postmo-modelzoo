# Copyright 2025 HOUMO AI
#
# File: ptq.py
# Description:
#    MinerU2.5 post-training quantization script
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
import argparse
import os
import re
import json
import shutil
from pathlib import Path
import torch
import copy
from hmatc.utils import logger
from hmatc.utils.utils import first_not_none, get_model_configs
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE
from xhquant.api import Config, xhquant_init
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name
from xhmodel_merak.xh_llm.models.qwen2_vl import XHQwen2VLModel, XHQwen2VLModelConfig

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

MINERU_VISUAL_BUCKETS_MANIFEST = "mineru_visual_buckets.json"
_HMQUANT_DIR_RE = re.compile(r"^hmquant_xh2_.+_\d{8}$")
MODEL_NAME = "mineru2_5_pro_1_2b"


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "mineru2.5-pro")
    model_size = model_config.get("model_size", "2604-1.2b")
    return f"{model_name}-{model_size}"


def _build_cfg_from_model(args) -> Config:
    cfg = dict(
        chip_arch=args.chip_arch,
        model=dict(
            model_type="Qwen2VLForConditionalGeneration",
            hf_model=args.model,
            model_name=MODEL_NAME,
            context_max_length=args.context_length,
            prefill_chunk_length=args.prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type="w8a16h0_ssfp",
                nodes=dict(
                    lm_head=dict(
                        quant_type="w8a16h0_ssfp",
                    )
                ),
            ),
            # 建议使用16-bit激活值matmul，否则attention会有严重的精度问题
            ops=dict(
                MatMul=dict(
                    act_scheme=dict(
                        bits=16,
                        fp_mode="sefp",
                    ),
                    act_schema_2=dict(
                        bits=16,
                        fp_mode="sefp",
                    ),
                )
            ),
            visual_config=dict(
                max_size_w=1036,
                max_size_h=1036,
                patch_size=14,
                temporal_patch_size=2,
                quant_scheme=dict(
                    quant_type="w8a16h0_ssfp",
                    # 建议使用16-bit激活值matmul，否则attention会有严重的精度问题
                    ops=dict(
                        MatMul=dict(
                            act_scheme=dict(
                                bits=16,
                                fp_mode="sefp",
                            ),
                            act_schema_2=dict(
                                bits=16,
                                fp_mode="sefp",
                            ),
                        )
                    ),
                ),
            ),
            only_first_block=False,
        ),
    )
    cfg = format_model_name(cfg)
    return Config(cfg)


def _build_visual_cfg_from_model(args: Path) -> Path | None:
    cfg = dict(
        chip_arch=args.chip_arch,
        model=dict(
            model_type="Qwen2VLForConditionalGeneration_visual",
            hf_model=args.model,
            model_name=f"{MODEL_NAME}_static_bucket",
            max_size_w=1036,
            max_size_h=1036,
            patch_size=14,
            temporal_patch_size=2,
            quant_scheme=dict(
                quant_type="w8a16h0_ssfp",
                # 建议使用16-bit激活值matmul，否则attention会有严重的精度问题
                ops=dict(
                    MatMul=dict(
                        act_scheme=dict(
                            bits=16,
                            fp_mode="sefp",
                        ),
                        act_schema_2=dict(
                            bits=16,
                            fp_mode="sefp",
                        ),
                    )
                ),
            ),
        ),
        visual_buckets=[
            # Medium horizontal content: title, short text, captions, small tables.
            dict(max_size_h=140, max_size_w=392),
            dict(max_size_h=196, max_size_w=560),
            dict(max_size_h=280, max_size_w=784),
            dict(max_size_h=392, max_size_w=1036),
            # Long horizontal content: PPT text strips, formulas, wide table rows.
            dict(max_size_h=112, max_size_w=1792),
            dict(max_size_h=168, max_size_w=1792),
            dict(max_size_h=252, max_size_w=1792),
            dict(max_size_h=392, max_size_w=2044),
            # Non-horizontal content buckets.
            dict(max_size_h=560, max_size_w=560),
            dict(max_size_h=1036, max_size_w=392),
        ],
    )
    # cfg = format_model_name(cfg)
    return Config(cfg)


def move_hmquant_files(output_dir, model_name):
    """Move files from hmquant_xh2_{model}_{quant}_{seq}_{context}_{date} to parent directory."""
    output_path = Path(output_dir) / "hmquant"
    model_dir = output_path / model_name
    if not model_dir.exists():
        logger.warning(f"Output directory not found: {model_dir}")
        return

    hmquant_dirs = [
        item
        for item in model_dir.iterdir()
        if item.is_dir() and _HMQUANT_DIR_RE.match(item.name)
    ]
    if not hmquant_dirs:
        logger.info(f"No hmquant directory found in {model_dir}")
        return

    hmquant_dir = hmquant_dirs[0]
    logger.info(f"Moving files from {hmquant_dir.name} to {output_path.name}/")

    for item in hmquant_dir.iterdir():
        target = output_path / item.name

        if item.is_dir():
            if target.exists():
                logger.warning(f"Overwriting existing directory: {target}")
                shutil.rmtree(target)
            shutil.move(str(item), str(target))
            logger.info(f"  Moved: {item.name}/")
        else:
            if target.exists():
                logger.warning(f"Overwriting existing file: {target}")
                target.unlink()
            shutil.move(str(item), str(target))
            logger.info(f"  Moved: {item.name}")

    shutil.rmtree(model_dir)
    logger.info(f"Cleanup completed, removed: {hmquant_dir.name}")


def _relative_to_export_dir(path: str | Path, exported_dir: Path) -> str:
    path = Path(path)
    exported_dir_abs = exported_dir.resolve()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([exported_dir / path, Path.cwd() / path])
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return candidate.resolve().relative_to(exported_dir_abs).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def _resolve_exported_dir(work_dir: Path, meta_info) -> Path:
    prefill_hmonnx = getattr(meta_info, "prefill_hmonnx", None)
    if prefill_hmonnx:
        matches = [
            path
            for path in work_dir.iterdir()
            if path.is_dir() and (path / str(prefill_hmonnx)).exists()
        ]
        if len(matches) == 1:
            return matches[0]

    meta_files = sorted(
        work_dir.glob("*/golden_meta_info.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if meta_files:
        return meta_files[0].parent
    raise RuntimeError(f"Cannot locate exported HMONNX directory under {work_dir}")


def _export_visual_bucket(
    visual_cfg_model, bucket: tuple[int, int], exported_dir: Path
) -> str:
    max_size_h, max_size_w = bucket
    bucket_dir = exported_dir / "visual_buckets" / f"{max_size_h}x{max_size_w}"

    cfg_model = copy.deepcopy(visual_cfg_model)
    cfg_model.max_size_h = max_size_h
    cfg_model.max_size_w = max_size_w
    cfg_model.model_name = f"{cfg_model.model_name}_{max_size_h}x{max_size_w}"
    model_cfg = AutoLLMConfig.from_pretrained(cfg_model)
    model_cfg.work_dir = str(bucket_dir)
    visual_model = AutoLLMModel.from_pretrained(config=model_cfg)
    logger.info(f"Exporting static visual bucket {bucket} to {bucket_dir}")
    visual_meta = visual_model.export_hmonnx(str(bucket_dir))
    return _relative_to_export_dir(visual_meta.hmonnx, exported_dir)


def _write_visual_bucket_manifest(
    meta_info,
    exported_dir: Path,
    visual_cfg_model,
    fallback_bucket: tuple[int, int],
    buckets: list[tuple[int, int]],
) -> None:
    default_visual_meta = getattr(meta_info, "visual_config", None)
    default_bucket = (
        int(getattr(default_visual_meta, "image_size_h", fallback_bucket[0])),
        int(getattr(default_visual_meta, "image_size_w", fallback_bucket[1])),
    )
    default_hmonnx = getattr(default_visual_meta, "hmonnx", None)
    if default_hmonnx is None:
        raise RuntimeError(
            "Default visual HMONNX path is missing from exported metadata."
        )

    manifest_buckets = []
    for bucket in buckets:
        hmonnx = _export_visual_bucket(visual_cfg_model, bucket, exported_dir)
        manifest_buckets.append(
            {
                "max_size_h": bucket[0],
                "max_size_w": bucket[1],
            }
        )
    manifest = {
        "buckets": manifest_buckets,
        "fallback_bucket": {
            "max_size_h": fallback_bucket[0],
            "max_size_w": fallback_bucket[1],
        },
        "patch_size": int(
            getattr(
                default_visual_meta,
                "patch_size",
                visual_cfg_model.patch_size,
            )
        ),
        "spatial_merge_size": int(
            getattr(
                default_visual_meta,
                "spatial_merge_size",
                SPATIAL_MERGE_SIZE,
            )
        ),
        "temporal_patch_size": int(
            getattr(
                default_visual_meta,
                "temporal_patch_size",
                visual_cfg_model.temporal_patch_size,
            )
        ),
    }
    manifest_path = exported_dir / "mineru_visual_buckets.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"MinerU static visual bucket manifest saved to {manifest_path}")


def quant_mineru2_5(args, device, dtype):
    cfg = _build_cfg_from_model(args)
    model_name = os.path.basename(args.model).lower()
    work_dir = os.path.join(args.out_dir, "hmquant", model_name)
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"Using device: {device}, dtype: {dtype}")
    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Work dir: {work_dir}")

    model_cfg: XHQwen2VLModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model: XHQwen2VLModel = AutoLLMModel.from_pretrained(config=model_cfg)

    meta_info = xh_model.export_hmonnx(str(work_dir))
    exported_dir = _resolve_exported_dir(Path(work_dir), meta_info)
    visual_cfg = _build_visual_cfg_from_model(args)
    fallback_bucket = (
        int(cfg.model.visual_config.max_size_h),
        int(cfg.model.visual_config.max_size_w),
    )
    buckets = [(b["max_size_h"], b["max_size_w"]) for b in visual_cfg.visual_buckets]
    visual_cfg_model = visual_cfg.model
    _write_visual_bucket_manifest(
        meta_info, exported_dir, visual_cfg_model, fallback_bucket, buckets
    )
    # Rename
    move_hmquant_files(args.out_dir, model_name)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="path to the model directory")
    parser.add_argument("--config", dest="config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_name", type=str, default=None, help="model name for output files")
    parser.add_argument("--model_size", type=str, default=None, help="model size identifier for output files")
    parser.add_argument("--out-dir", type=str, default=f"./output/{HOUMO_TARGET}", help="output directory")
    parser.add_argument("--chip-arch", type=str, default="XH2a", choices=["XH2a", "YueHui"])
    parser.add_argument("--context-length", type=int, default=4096, help="max sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--seed", type=int, default=1024, help="random seed for reproducibility")
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model = first_not_none(args.model, get_default_model_dir(model_config))
    # fmt: on

    torch.manual_seed(args.seed)

    xhquant_init(logger=logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    quant_mineru2_5(args, device, dtype)
