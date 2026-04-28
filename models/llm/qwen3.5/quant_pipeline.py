# Copyright (c) 2025 HOUMO AI
#
# File: quant_pipeline.py
# Description:
#   Quantization pipeline functions for ptq.py
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
"""Qwen3.5 VL: rotate + GPTQ (src scripts), then vision + LLM HMONNX export.

Paths (effective ``model_name`` from args, or HF dir stem if unset):
  - Step1–2 under ``work_dir``: ``<model_name>_rotated_fp``, ``<model_name>_gptq_4bit``
    (overridable by --rotated-model-dir / --gptq-model-dir)
  - Step3–4 under ``out_dir``: ``<model_name>/vision_export/...``, ``<model_name>_llm_export/``
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

VISION_CONFIG = "configs/qwen3_5/qwen3_5_instruct_vision_config.py"
LLM_CONFIG = "configs/qwen3_5/qwen3_5_xh2a.py"
MOE_VISION_CONFIG = "configs/qwen3_5_moe/qwen3_5_moe_vision_config.py"

# Packaged ONNX name under ``<HOUMO_TARGET or xh2>/hmquant/{visual,prefill,decoder}/``
HMQUANT_ONNX_NAME = "hmquant_qwen3.5_with_act.onnx"


def _houmo_target_dir() -> str:
    """First path segment under ``--out-dir`` for hmquant; ``HOUMO_TARGET`` env, else ``xh2``."""
    v = (os.environ.get("HOUMO_TARGET") or "").strip()
    return v if v else "xh2"


def _src_dir() -> Path:
    return Path(__file__).resolve().parent / "src"


def msg_output_format(title: str) -> str:
    pad = "*" * 10
    return f"{pad} {title} {pad}"


def _hf_dir_ready(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _run_step(cmd: list[str], cwd: Path, extra_env: dict | None = None) -> None:
    logger.info(msg_output_format("subprocess"))
    logger.info("cwd={} cmd={}", cwd, " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _model_stem(model: str) -> str:
    return Path(model).resolve().name


def _effective_model_name(args) -> str:
    mn = (getattr(args, "model_name", None) or "").strip()
    return mn if mn else _model_stem(args.model)


def _get_size_combinations(args) -> list[tuple[int, int, int]]:
    """Get paired (max_size_w, max_size_h, max_size_t) from args by index."""
    sizes_w = getattr(args, "max_size_w", [448])
    sizes_h = getattr(args, "max_size_h", [448])
    sizes_t = getattr(args, "max_size_t", [2])

    # Validate all lists have the same length
    len_w, len_h, len_t = len(sizes_w), len(sizes_h), len(sizes_t)
    if not (len_w == len_h == len_t):
        raise ValueError(
            f"max_size_w/h/t must have the same number of values, "
            f"got {len_w}/{len_h}/{len_t}"
        )

    return list(zip(sizes_w, sizes_h, sizes_t))


def _paths(args):
    work_root = Path(args.work_dir).resolve()
    name = _effective_model_name(args)
    r_custom = getattr(args, "rotated_model_dir", None)
    g_custom = getattr(args, "gptq_model_dir", None)
    rotated = Path(r_custom).resolve() if r_custom else work_root / f"{name}_rotated_fp"
    gptq = Path(g_custom).resolve() if g_custom else work_root / f"{name}_gptq_4bit"
    return work_root, name, rotated, gptq


def quant_llm(args) -> None:
    """Step1: example_qwen35_vl_rotate_fp.py → Step2: example_qwen35dense.py."""
    if getattr(args, "export_only", False):
        logger.info("export_only: skip quantization (use existing rotated + GPTQ dirs)")
        return

    src = _src_dir()
    if not src.is_dir():
        raise FileNotFoundError(f"Missing src directory: {src}")

    work_root, _name, rotated_dir, gptq_dir = _paths(args)
    work_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    device = getattr(args, "device", "cuda:0")
    trust = getattr(args, "trust_remote_code", True)
    resume = getattr(args, "resume", False)

    if resume and _hf_dir_ready(rotated_dir):
        logger.info("Resume: skip rotation (found {})", rotated_dir)
    else:
        cmd = [
            py,
            str(src / "example_qwen35_vl_rotate_fp.py"),
            "--model",
            str(Path(args.model).resolve()),
            "--out",
            str(rotated_dir),
            "--llm-rotation",
            "hadamard",
            "--vision-rotation",
            "last",
            "--device",
            device,
            "--validate",
            "--seed",
            str(getattr(args, "seed", 1024)),
        ]
        if trust:
            cmd.append("--trust-remote-code")
        _run_step(cmd, src, extra_env={"CUDA_VISIBLE_DEVICES": "0"})

    if resume and _hf_dir_ready(gptq_dir):
        logger.info("Resume: skip GPTQ (found {})", gptq_dir)
    else:
        cmd = [
            py,
            str(src / "example_qwen35dense.py"),
            "--model",
            str(Path(args.model).resolve()),
            "--out",
            str(gptq_dir),
            "--bits",
            "4",
            "--group-size",
            "64",
            "--rotation",
            "hadamard",
            "--nsamples",
            "256",
            "--seqlen",
            "1024",
            "--mse",
            "2.4",
            "--hessian-mse",
            "--device",
            device,
            "--seed",
            str(getattr(args, "seed", 1024)),
        ]
        if getattr(args, "calib_data", None):
            cmd.extend(["--calibration-jsonl", str(Path(args.calib_data).resolve())])
        if trust:
            cmd.append("--trust-remote-code")
        _run_step(cmd, src, extra_env={"CUDA_VISIBLE_DEVICES": "0"})


def quant_llm_moe(args) -> None:
    """MoE GPTQ only: skip rotate_fp and quantize with example_qwen35moe.py."""
    if getattr(args, "export_only", False):
        logger.info("export_only: skip quantization (use existing GPTQ dir)")
        return

    src = _src_dir()
    if not src.is_dir():
        raise FileNotFoundError(f"Missing src directory: {src}")

    work_root, _name, _rotated_dir, gptq_dir = _paths(args)
    work_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    device = getattr(args, "device", "cuda:0")
    resume = getattr(args, "resume", False)

    if resume and _hf_dir_ready(gptq_dir):
        logger.info("Resume: skip GPTQ (found {})", gptq_dir)
    else:
        cmd = [
            py,
            str(src / "example_qwen35moe.py"),
            "--model",
            str(Path(args.model).resolve()),
            "--out",
            str(gptq_dir),
            "--hessian-mse",
            "--moe-routing",
            "bypass",
            "--nsamples",
            "256",
            "--shared-expert-bits",
            "4",
            "--self-attn-bits",
            str(getattr(args, "self_attn_bits", 8)),
            "--expert-bits",
            "4",
            "--device",
            device,
        ]
        _run_step(cmd, src, extra_env={"CUDA_VISIBLE_DEVICES": "0"})


def export_llm(args) -> None:
    """Step3: vision HMONNX; Step4: LLM HMONNX (README defaults)."""
    src = _src_dir()
    _work_root, _name, rotated_dir, gptq_dir = _paths(args)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not _hf_dir_ready(rotated_dir):
        raise FileNotFoundError(f"Rotated model not found: {rotated_dir}")
    if not _hf_dir_ready(gptq_dir):
        raise FileNotFoundError(f"GPTQ model not found: {gptq_dir}")

    py = sys.executable
    model_name = _effective_model_name(args)

    if not getattr(args, "skip_export_vision", False):
        sample_img = getattr(args, "vision_image_path", None)
        if sample_img:
            sample_img = str(Path(sample_img).resolve())
        else:
            default_img = src / "images" / "qwen2_vl_demo.jpeg"
            if not default_img.is_file():
                logger.warning(
                    "Default vision sample image missing at {} (cwd is src/). "
                    "Set ptq --vision-image-path to a JPEG/PNG.",
                    default_img,
                )
        # Step3: match src/README.md (no --valid, no --seed in example)
        size_combinations = _get_size_combinations(args)
        for w, h, t in size_combinations:
            logger.info(
                "Exporting vision with max_size_w={}, max_size_h={}, max_size_t={}",
                w,
                h,
                t,
            )
            cmd_v = [
                py,
                str(src / "qwen3_5_vision_xh2a_export_hmonnx.py"),
                "--config",
                VISION_CONFIG,
                "--hf_model_dir",
                str(rotated_dir),
                "--model_name",
                str(model_name),
                "--output_root",
                str(out_root),
                "--max_size_w",
                str(w),
                "--max_size_h",
                str(h),
                "--max_size_t",
                str(t),
            ]
            if sample_img:
                cmd_v.extend(["--image_path", sample_img])
            if getattr(args, "debug", False):
                cmd_v.append("--debug")
            _run_step(cmd_v, src)
    else:
        logger.info("skip_export_vision: skipped vision HMONNX export")

    if not getattr(args, "skip_export_llm", False):
        llm_work = out_root / f"{model_name}_llm_export"
        # Step4: match src/README.md (--valid, --golden; no conditional toggles)
        cmd_l = [
            py,
            str(src / "qwen3_5_xh2a_export_hmonnx.py"),
            "--config",
            LLM_CONFIG,
            "--hf_model_dir",
            str(gptq_dir),
            "--dtype",
            "fp16",
            "--valid",
            "--golden",
            "--work_dir",
            str(llm_work),
        ]
        if getattr(args, "debug", False):
            cmd_l.append("--debug")
        if getattr(args, "context_length", None):
            cmd_l.extend(["--max_sequence_length", str(args.context_length)])
        if getattr(args, "max_pe_length", None):
            cmd_l.extend(["--max_pe_length", str(args.max_pe_length)])
        _run_step(cmd_l, src)
    else:
        logger.info("skip_export_llm: skipped LLM HMONNX export")


def export_llm_moe(args) -> None:
    """MoE Step3: vision HMONNX; Step4: LLM HMONNX."""
    src = _src_dir()
    _work_root, _name, _rotated_dir, gptq_dir = _paths(args)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not _hf_dir_ready(gptq_dir):
        raise FileNotFoundError(f"GPTQ model not found: {gptq_dir}")

    py = sys.executable
    model_name = _effective_model_name(args)

    if not getattr(args, "skip_export_vision", False):
        sample_img = getattr(args, "vision_image_path", None)
        if sample_img:
            sample_img = str(Path(sample_img).resolve())
        else:
            default_img = src / "images" / "qwen2_vl_demo.jpeg"
            if not default_img.is_file():
                logger.warning(
                    "Default vision sample image missing at {} (cwd is src/). "
                    "Set ptq --vision-image-path to a JPEG/PNG.",
                    default_img,
                )
        size_combinations = _get_size_combinations(args)
        for w, h, t in size_combinations:
            logger.info(
                "Exporting MoE vision with max_size_w={}, max_size_h={}, max_size_t={}",
                w,
                h,
                t,
            )
            cmd_v = [
                py,
                str(src / "qwen3_5_moe_vision_xh2a_export_hmonnx.py"),
                "--config",
                MOE_VISION_CONFIG,
                "--hf_model_dir",
                str(Path(args.model).resolve()),
                "--model_name",
                str(model_name),
                "--output_dir",
                str(out_root),
                "--max_size_w",
                str(w),
                "--max_size_h",
                str(h),
                "--max_size_t",
                str(t),
            ]
            if sample_img:
                cmd_v.extend(["--image_path", sample_img])
            if getattr(args, "debug", False):
                cmd_v.append("--debug")
            _run_step(cmd_v, src)
    else:
        logger.info("skip_export_vision: skipped vision HMONNX export")

    if not getattr(args, "skip_export_llm", False):
        cmd_l = [
            py,
            str(src / "qwen3_5_moe_xh2a_export_hmonnx.py"),
            "--model",
            str(Path(args.model).resolve()),
            "--quant-weight",
            str(gptq_dir),
            "--quant-type",
            "w4a8h0_ssfp",
            "--context-length",
            str(getattr(args, "context_length", 2048)),
            "--input-sequence-length",
            str(getattr(args, "input_sequence_length", 256)),
            "--output-dir",
            str(out_root),
        ]
        if getattr(args, "debug", False):
            cmd_l.append("--debug")
        _run_step(cmd_l, src)
    else:
        logger.info("skip_export_llm: skipped LLM HMONNX export")


def move_models(
    work_dir: Path,
    source: str = "prefill",
    model: str = "prefill",
    target_name: str = "hmquant_qwen3_with_act.onnx",
):
    """Reserved; Qwen3.5 layout is produced under out_dir by export scripts."""
    pass


def _copy_dir_files_rename_onnx(src: Path, dst: Path) -> None:
    """Copy only files from src into dst; ``*.onnx`` → ``HMQUANT_ONNX_NAME``."""
    if not src.is_dir():
        raise FileNotFoundError(f"Expected directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    onnx_seen = 0
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".onnx":
            onnx_seen += 1
            out_name = HMQUANT_ONNX_NAME
        else:
            out_name = p.name
        shutil.copy2(p, dst / out_name)
    if onnx_seen > 1:
        logger.warning(
            "Multiple .onnx files in {}; later copies overwrote {} — check layout",
            src,
            HMQUANT_ONNX_NAME,
        )


def move_llm(args) -> None:
    """Lay out ``<HOUMO_TARGET or xh2>/hmquant`` under --out-dir: visual / prefill / decoder + quant_embedding.pt."""
    out_root = Path(args.out_dir).resolve()
    model_name = _effective_model_name(args)
    target = _houmo_target_dir()
    hmquant = out_root / target / "hmquant"
    prefill_d = hmquant / "prefill"
    decoder_d = hmquant / "decoder"
    prefill_d.mkdir(parents=True, exist_ok=True)
    decoder_d.mkdir(parents=True, exist_ok=True)

    # Find all vision export dirs with size suffixes
    if not getattr(args, "skip_export_vision", False):
        model_parent = out_root / model_name
        found_any = False
        if model_parent.is_dir():
            for vision_export_dir in sorted(model_parent.iterdir()):
                if (
                    vision_export_dir.name.startswith("vision_export_")
                    and vision_export_dir.is_dir()
                ):
                    size_suffix = vision_export_dir.name[len("vision_export_") :]
                    vision_src = vision_export_dir / "vision"
                    if vision_src.is_dir():
                        logger.info(
                            msg_output_format(f"move_llm: visual_{size_suffix}")
                        )
                        visual_d = hmquant / f"visual_{size_suffix}"
                        _copy_dir_files_rename_onnx(vision_src, visual_d)
                        found_any = True
        # Also check the legacy path for backward compatibility
        legacy_vision_src = out_root / model_name / "vision_export" / "vision"
        if legacy_vision_src.is_dir():
            logger.info(msg_output_format("move_llm: visual (legacy)"))
            visual_d = hmquant / "visual"
            _copy_dir_files_rename_onnx(legacy_vision_src, visual_d)
            found_any = True
        if not found_any:
            logger.warning("move_llm: no vision export dirs found")
    else:
        logger.info("move_llm: skip visual (skip_export_vision)")

    llm_export = out_root / f"{model_name}_llm_export"
    emb_src = llm_export / "token_embedding.pt"
    prefill_src = llm_export / "prefill_onnx"
    decode_src = llm_export / "decode_onnx"

    if llm_export.is_dir() and not getattr(args, "skip_export_llm", False):
        logger.info(msg_output_format(f"move_llm: LLM export → {target}/hmquant"))
        if emb_src.is_file():
            shutil.copy2(emb_src, hmquant / "quant_embedding.pt")
        else:
            logger.warning("move_llm: missing {}, skip embedding copy", emb_src)
        if prefill_src.is_dir():
            _copy_dir_files_rename_onnx(prefill_src, prefill_d)
        else:
            logger.warning("move_llm: missing {}, skip prefill", prefill_src)
        if decode_src.is_dir():
            _copy_dir_files_rename_onnx(decode_src, decoder_d)
        else:
            logger.warning("move_llm: missing {}, skip decoder copy", decode_src)
    elif getattr(args, "skip_export_llm", False):
        logger.info("move_llm: skip LLM artifacts (skip_export_llm)")
    else:
        logger.warning("move_llm: {} missing, skip LLM copy layout", llm_export)

    logger.info(msg_output_format("move_llm done"))
    logger.info("hmquant root: {}", hmquant)


def _resolve_moe_llm_export_dir(args, out_root: Path) -> Path:
    candidates = [
        out_root / f"{Path(args.model).resolve().name}_llm_export",
        out_root / f"{_effective_model_name(args)}_llm_export",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def move_llm_moe(args) -> None:
    """Lay out MoE exports into ``<HOUMO_TARGET or xh2>/hmquant`` under --out-dir."""
    out_root = Path(args.out_dir).resolve()
    model_name = _effective_model_name(args)
    target = _houmo_target_dir()
    hmquant = out_root / target / "hmquant"
    prefill_d = hmquant / "prefill"
    decoder_d = hmquant / "decoder"
    prefill_d.mkdir(parents=True, exist_ok=True)
    decoder_d.mkdir(parents=True, exist_ok=True)

    # Find all vision export dirs with size suffixes
    if not getattr(args, "skip_export_vision", False):
        found_any = False
        # Look for dirs like {model_name}_{w}x{h}x{t}
        for candidate in sorted(out_root.iterdir()):
            if candidate.name.startswith(f"{model_name}_") and candidate.is_dir():
                size_suffix = candidate.name[len(f"{model_name}_") :]
                vision_src = candidate / "vision"
                if vision_src.is_dir():
                    logger.info(
                        msg_output_format(f"move_llm_moe: visual_{size_suffix}")
                    )
                    visual_d = hmquant / f"visual_{size_suffix}"
                    _copy_dir_files_rename_onnx(vision_src, visual_d)
                    found_any = True
        # Also check the legacy path for backward compatibility
        legacy_vision_src = out_root / model_name / "vision"
        if legacy_vision_src.is_dir():
            logger.info(msg_output_format("move_llm_moe: visual (legacy)"))
            visual_d = hmquant / "visual"
            _copy_dir_files_rename_onnx(legacy_vision_src, visual_d)
            found_any = True
        if not found_any:
            logger.warning("move_llm_moe: no vision export dirs found")
    else:
        logger.info("move_llm_moe: skip visual (skip_export_vision)")

    llm_export = _resolve_moe_llm_export_dir(args, out_root)
    emb_src_candidates = [
        llm_export / "token_embedding",
        llm_export / "token_embedding.pt",
    ]
    prefill_src = llm_export / "hmonnx" / "prefill"
    decode_src = llm_export / "hmonnx" / "decode"

    if llm_export.is_dir() and not getattr(args, "skip_export_llm", False):
        logger.info(msg_output_format(f"move_llm_moe: LLM export → {target}/hmquant"))
        emb_src = next((p for p in emb_src_candidates if p.is_file()), None)
        if emb_src is not None:
            shutil.copy2(emb_src, hmquant / "quant_embedding.pt")
        else:
            logger.warning(
                "move_llm_moe: missing token embedding in {}, skip embedding copy",
                llm_export,
            )
        if prefill_src.is_dir():
            _copy_dir_files_rename_onnx(prefill_src, prefill_d)
        else:
            logger.warning("move_llm_moe: missing {}, skip prefill", prefill_src)
        if decode_src.is_dir():
            _copy_dir_files_rename_onnx(decode_src, decoder_d)
        else:
            logger.warning("move_llm_moe: missing {}, skip decoder copy", decode_src)
    elif getattr(args, "skip_export_llm", False):
        logger.info("move_llm_moe: skip LLM artifacts (skip_export_llm)")
    else:
        logger.warning("move_llm_moe: {} missing, skip LLM copy layout", llm_export)

    logger.info(msg_output_format("move_llm_moe done"))
    logger.info("hmquant root: {}", hmquant)
