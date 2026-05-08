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
"""Qwen3.5 VL: GPTQ (src scripts), then vision + LLM HMONNX export.

Paths (effective ``model_name`` from args, or HF dir stem if unset):
  - Step1 under ``work_dir``: ``<model_name>_gptq_4bit``
    (overridable by --gptq-model-dir)
  - Step2–3 under ``out_dir``: ``<model_name>/vision_export/...``, ``<model_name>_llm_export/``
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


def _paths(args):
    work_root = Path(args.work_dir).resolve()
    name = _effective_model_name(args)
    r_custom = getattr(args, "rotated_model_dir", None)
    g_custom = getattr(args, "gptq_model_dir", None)
    rotated = Path(r_custom).resolve() if r_custom else work_root / f"{name}_rotated_fp"
    gptq = Path(g_custom).resolve() if g_custom else work_root / f"{name}_gptq_4bit"
    return work_root, name, rotated, gptq


def quant_llm(args) -> None:
    """Step1: example_qwen35dense.py."""
    if getattr(args, "export_only", False):
        logger.info("export_only: skip quantization (use existing GPTQ dir)")
        return

    src = _src_dir()
    if not src.is_dir():
        raise FileNotFoundError(f"Missing src directory: {src}")

    work_root, _name, _rotated_dir, gptq_dir = _paths(args)
    work_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    device = getattr(args, "device", "cuda")
    trust = getattr(args, "trust_remote_code", True)
    resume = getattr(args, "resume", False)

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
            # "--rotation",
            # "hadamard",
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
        _run_step(cmd, src)


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
    device = getattr(args, "device", "cuda")
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
        if getattr(args, "calib_data", None):
            cmd.extend(["--calibration-jsonl", str(Path(args.calib_data).resolve())])
        _run_step(cmd, src)


def export_llm(args) -> None:
    """Step2: vision HMONNX (multiple sizes); Step3: LLM HMONNX (README defaults)."""
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
        # Step2: export for each size combination
        max_size_w_list = getattr(args, "max_size_w", [448])
        max_size_h_list = getattr(args, "max_size_h", [448])
        max_size_t_list = getattr(args, "max_size_t", [2])

        # Ensure all lists have the same length by repeating the last element if needed
        max_len = max(len(max_size_w_list), len(max_size_h_list), len(max_size_t_list))
        while len(max_size_w_list) < max_len:
            max_size_w_list.append(max_size_w_list[-1])
        while len(max_size_h_list) < max_len:
            max_size_h_list.append(max_size_h_list[-1])
        while len(max_size_t_list) < max_len:
            max_size_t_list.append(max_size_t_list[-1])

        for idx, (w, h, t) in enumerate(
            zip(max_size_w_list, max_size_h_list, max_size_t_list)
        ):
            logger.info(
                f"Exporting vision model for size {w}x{h}x{t} ({idx+1}/{max_len})"
            )
            cmd_v = [
                py,
                str(src / "qwen3_5_vision_xh2a_export_hmonnx.py"),
                "--config",
                VISION_CONFIG,
                "--hf_model_dir",
                str(Path(args.model).resolve()),
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
        # Step3: match src/README.md (--valid, --golden; no conditional toggles)
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
        if not getattr(args, "llm_export_full_output_valid", True):
            cmd_l.append("--no-valid_compare_full_output")
        if getattr(args, "context_length", None):
            cmd_l.extend(["--max_sequence_length", str(args.context_length)])
        if getattr(args, "max_pe_length", None):
            cmd_l.extend(["--max_pe_length", str(args.max_pe_length)])
        _run_step(cmd_l, src)
    else:
        logger.info("skip_export_llm: skipped LLM HMONNX export")


def export_llm_moe(args) -> None:
    """MoE Step3: vision HMONNX (multiple sizes); Step4: LLM HMONNX."""
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
        # Export for each size combination
        max_size_w_list = getattr(args, "max_size_w", [448])
        max_size_h_list = getattr(args, "max_size_h", [448])
        max_size_t_list = getattr(args, "max_size_t", [2])

        # Ensure all lists have the same length by repeating the last element if needed
        max_len = max(len(max_size_w_list), len(max_size_h_list), len(max_size_t_list))
        while len(max_size_w_list) < max_len:
            max_size_w_list.append(max_size_w_list[-1])
        while len(max_size_h_list) < max_len:
            max_size_h_list.append(max_size_h_list[-1])
        while len(max_size_t_list) < max_len:
            max_size_t_list.append(max_size_t_list[-1])

        for idx, (w, h, t) in enumerate(
            zip(max_size_w_list, max_size_h_list, max_size_t_list)
        ):
            logger.info(
                f"Exporting MoE vision model for size {w}x{h}x{t} ({idx+1}/{max_len})"
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

    if not getattr(args, "skip_export_vision", False):
        # Find all vision export directories (with size suffix and without)
        model_dir = out_root / model_name
        vision_export_dirs = []
        if model_dir.is_dir():
            # Check for vision_export_* directories
            for p in model_dir.iterdir():
                if p.is_dir() and p.name.startswith("vision_export_"):
                    vision_src_candidate = p / "vision"
                    if vision_src_candidate.is_dir():
                        size_suffix = p.name[len("vision_export_") :]
                        vision_export_dirs.append((size_suffix, vision_src_candidate))
            # Check for the default vision_export directory
            default_vision_src = model_dir / "vision_export" / "vision"
            if default_vision_src.is_dir():
                vision_export_dirs.append(("", default_vision_src))

        if vision_export_dirs:
            logger.info(msg_output_format("move_llm: visual"))
            for size_suffix, vision_src in vision_export_dirs:
                # For multiple sizes, create visual_<size> directories
                if size_suffix:
                    target_visual_d = hmquant / f"visual_{size_suffix}"
                else:
                    target_visual_d = hmquant / "visual"
                target_visual_d.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"  Copying vision {size_suffix or 'default'} to {target_visual_d}"
                )
                _copy_dir_files_rename_onnx(vision_src, target_visual_d)
        else:
            logger.warning(
                "move_llm: no vision export dirs found under {}, skip visual", model_dir
            )
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

    if not getattr(args, "skip_export_vision", False):
        # Find all MoE vision export directories (with size suffix and without)
        vision_export_dirs = []
        # Check for <model_name>_<size> directories
        for p in out_root.iterdir():
            if p.is_dir() and p.name.startswith(f"{model_name}_"):
                vision_src_candidate = p / "vision"
                if vision_src_candidate.is_dir():
                    size_suffix = p.name[len(model_name) + 1 :]
                    vision_export_dirs.append((size_suffix, vision_src_candidate))
        # Check for the default <model_name> directory
        default_vision_src = out_root / model_name / "vision"
        if default_vision_src.is_dir():
            vision_export_dirs.append(("", default_vision_src))

        if vision_export_dirs:
            logger.info(msg_output_format("move_llm_moe: visual"))
            for size_suffix, vision_src in vision_export_dirs:
                # For multiple sizes, create visual_<size> directories
                if size_suffix:
                    target_visual_d = hmquant / f"visual_{size_suffix}"
                else:
                    target_visual_d = hmquant / "visual"
                target_visual_d.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"  Copying vision {size_suffix or 'default'} to {target_visual_d}"
                )
                _copy_dir_files_rename_onnx(vision_src, target_visual_d)
        else:
            logger.warning(
                "move_llm_moe: no vision export dirs found under {}, skip visual",
                out_root,
            )
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
