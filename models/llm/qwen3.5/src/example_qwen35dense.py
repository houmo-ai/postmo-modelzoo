# Copyright (c) 2025 HOUMO AI
#
# File: example_qwen35dense.py
# Description:
#   Example script for Qwen3.5 dense GPTQ quantization and generation validation.
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
import json
import logging
import os
import random
from contextlib import contextmanager

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from qwen35_common import build_calibration_dataset

# ---------------------------------------------------------------------------
# Calibration data helpers
# ---------------------------------------------------------------------------


def get_wikitext2(nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen
    )
    return [example["text"] for example in traindata.select(range(nsamples))]


def get_jsonl_texts(path, nsamples, text_key="text"):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid `{text_key}` at {path}:{line_no}")
            samples.append(text)
            if len(samples) >= nsamples:
                break
    if len(samples) == 0:
        raise ValueError(f"No calibration samples found in {path}")
    return samples


# def build_calibration_dataset(args):
#     if args.calibration_jsonl:
#         calibration_dataset = get_jsonl_texts(
#             path=args.calibration_jsonl,
#             nsamples=args.nsamples,
#             text_key=args.calibration_text_key,
#         )
#         source_name = os.path.basename(args.calibration_jsonl)
#         return calibration_dataset, f"jsonl:{source_name}"

#     return get_wikitext2(nsamples=args.nsamples, seqlen=args.seqlen), "wikitext2"


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def calculate_avg_ppl(model, tokenizer, device=None):
    from gptqmodel.utils.perplexity import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
        device=device,
    )
    all = ppl.calculate(n_ctx=512, n_batch=512)
    return sum(all) / len(all)


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        device = getattr(model, "device", "cpu")
        return torch.device(device)


@contextmanager
def temporary_ppl_device(model, target_device: str | None):
    original_device = _get_model_device(model)
    active_device = str(original_device)
    moved = False
    dispatched = False

    if target_device is None:
        target_device = str(original_device)

    if (
        original_device.type == "cpu"
        and target_device != "cpu"
        and torch.cuda.is_available()
    ):
        if target_device == "auto":
            if torch.cuda.device_count() > 1:
                try:
                    from accelerate import infer_auto_device_map
                    from accelerate.hooks import remove_hook_from_module
                    from accelerate.utils import get_balanced_memory

                    from gptqmodel.utils.model import simple_dispatch_model

                    no_split = list(getattr(model, "_no_split_modules", None) or [])
                    memory_kwargs = {}
                    if no_split:
                        memory_kwargs["no_split_module_classes"] = no_split

                    max_memory = get_balanced_memory(model, **memory_kwargs)
                    device_map = infer_auto_device_map(
                        model, max_memory=max_memory, **memory_kwargs
                    )
                    print(
                        "[ppl] Dispatching model to GPU(s) with accelerate device_map=auto for evaluation ..."
                    )
                    simple_dispatch_model(model, device_map)
                    active_device = next(
                        (
                            dev
                            for dev in device_map.values()
                            if dev not in ("cpu", "disk")
                        ),
                        "cuda",
                    )
                    dispatched = True
                except Exception as exc:
                    target_device = "cuda"
                    print(
                        f"[ppl] Accelerate auto dispatch failed ({exc}); falling back to {target_device}."
                    )
            else:
                target_device = "cuda"

        if not dispatched and target_device != "auto":
            print(f"[ppl] Moving model from cpu to {target_device} for evaluation ...")
            model.to(target_device)
            active_device = target_device
            moved = True

    try:
        yield active_device
    finally:
        if dispatched:
            from accelerate.hooks import remove_hook_from_module

            print(
                "[ppl] Moving accelerate-dispatched model back to cpu after evaluation ..."
            )
            remove_hook_from_module(model, recurse=True)
            model.to("cpu")
            model.hf_device_map = {"": "cpu"}
            torch.cuda.empty_cache()
        elif moved:
            print("[ppl] Moving model back to cpu after evaluation ...")
            model.to("cpu")
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Chat / inference helpers
# ---------------------------------------------------------------------------


def build_chat_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 dense GPTQ quantization + generation validation"
    )
    parser.add_argument("--model", type=str, required=True, help="fp model path")
    parser.add_argument(
        "--out", type=str, required=True, help="quantized model output directory"
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--rotation",
        type=str,
        default=None,
        choices=["hadamard", "block_hadamard", "random"],
        help="optional QuaRot rotation mode",
    )
    parser.add_argument(
        "--check-rotation-ppl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="evaluate wikitext2 perplexity before and after rotation, before quantization starts",
    )
    parser.add_argument(
        "--rotated-out",
        type=str,
        default=None,
        help="optional output dir for the rotated-but-not-yet-quantized fp model",
    )
    parser.add_argument(
        "--stop-after-rotate",
        action="store_true",
        default=False,
        help="apply rotation, optionally save/evaluate it, then exit without quantizing",
    )
    parser.add_argument(
        "--mse",
        type=float,
        default=2.4,
        help="mse use 2.4 to default",
    )
    parser.add_argument(
        "--hessian-mse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Hessian MSE assisted optimization",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument(
        "--seqlen",
        type=int,
        default=1024,
        help="minimum sequence length for wikitext2 filtering",
    )
    parser.add_argument(
        "--calibration-jsonl",
        type=str,
        default=None,
        help="optional JSONL calibration dataset path; overrides wikitext2 when provided",
    )
    parser.add_argument(
        "--calibration-text-key",
        type=str,
        default="text",
        help="JSON field name used to read text from --calibration-jsonl",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="quantization compute device (single GPU)",
    )
    parser.add_argument(
        "--infer-device",
        type=str,
        default=None,
        help="inference device after quantization",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        help="multi-GPU device map strategy; overrides --device/--infer-device when set",
    )
    parser.add_argument(
        "--offload-to-disk", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--offload-path",
        type=str,
        default="./gptqmodel_offload_qwen35dense",
        help="offload path when --offload-to-disk is enabled",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt", type=str, default="请介绍一下 GPTQ 量化的核心思想。"
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        default=False,
        help="skip quantization and directly run generation on --out",
    )
    parser.add_argument(
        "--stop-after-quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after saving quantized model, exit without inference/PPL (--no-stop-after-quantize to run generation + PPL)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if (
        args.check_rotation_ppl or args.rotated_out or args.stop_after_rotate
    ) and not args.rotation:
        raise ValueError(
            "--check-rotation-ppl/--rotated-out/--stop-after-rotate require --rotation to be set."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer_source = args.model
    if not args.skip_quantize:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        calibration_dataset, calibration_source = build_calibration_dataset(args)
        print(
            f"[1/5] Calibration samples ({calibration_source}): {len(calibration_dataset)}"
        )

        quantize_config = QuantizeConfig(
            bits=args.bits,
            group_size=args.group_size,
            rotation=args.rotation,
            hessian_mse=args.hessian_mse,
            mse=args.mse,
            device=args.device,
            offload_to_disk=args.offload_to_disk,
            offload_to_disk_path=args.offload_path if args.offload_to_disk else None,
        )

        print("[2/5] Loading fp model ...")
        load_kwargs = dict(trust_remote_code=args.trust_remote_code)
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        model = GPTQModel.load(args.model, quantize_config, **load_kwargs)

        # -- Rotation artifacts (ppl check / save / early stop) --
        rotation_artifact_requested = any(
            [
                args.check_rotation_ppl,
                args.rotated_out is not None,
                args.stop_after_rotate,
            ]
        )
        if args.rotation and rotation_artifact_requested:
            if args.check_rotation_ppl:
                print(
                    "[rotation] Evaluating fp model PPL on wikitext2 before rotation ..."
                )
                with temporary_ppl_device(model.model, "auto") as ppl_device:
                    fp_ppl_before = calculate_avg_ppl(
                        model.model, tokenizer, device=ppl_device
                    )
                print(
                    f"[rotation] FP model avg PPL before rotation: {fp_ppl_before:.4f}"
                )

            print(f"[rotation] Applying {args.rotation} rotation to fp model ...")
            model.apply_rotation()

            if args.check_rotation_ppl:
                print(
                    "[rotation] Evaluating fp model PPL on wikitext2 after rotation ..."
                )
                with temporary_ppl_device(model.model, "auto") as ppl_device:
                    fp_ppl_after = calculate_avg_ppl(
                        model.model, tokenizer, device=ppl_device
                    )
                print(f"[rotation] FP model avg PPL after rotation: {fp_ppl_after:.4f}")
                print(f"[rotation] PPL delta: {fp_ppl_after - fp_ppl_before:+.6f}")

            if args.rotated_out:
                print(f"[rotation] Saving rotated fp model to: {args.rotated_out}")
                os.makedirs(args.rotated_out, exist_ok=True)
                model.model.save_pretrained(args.rotated_out)
                tokenizer.save_pretrained(args.rotated_out)

            if args.stop_after_rotate:
                print(
                    "[rotation] Stop requested after rotation; skipping quantization."
                )
                return

        print("[3/5] Running GPTQ quantization ...")
        model.quantize(calibration_dataset, batch_size=args.batch_size)
        model.save(args.out)
        print(f"Quantized model saved to: {args.out}")
        if args.stop_after_quantize:
            return
    else:
        print(f"[1/5] Skip quantization, using existing quantized model: {args.out}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.out, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        tokenizer_source = args.out

    # -- Inference --
    infer_device = args.infer_device or args.device
    infer_kwargs = dict(trust_remote_code=args.trust_remote_code)
    if args.device_map:
        infer_kwargs["device_map"] = args.device_map
        print(
            f"[4/5] Loading quantized model with device_map='{args.device_map}' and generating ..."
        )
    else:
        infer_kwargs["device"] = infer_device
        print(f"[4/5] Loading quantized model on {infer_device} and generating ...")
    model = GPTQModel.load(args.out, **infer_kwargs)
    runtime_device = str(model.device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, use_fast=True, trust_remote_code=args.trust_remote_code
        )
    inputs = build_chat_inputs(tokenizer, args.prompt, runtime_device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    answer = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Answer ===")
    print(answer)

    # -- Perplexity evaluation --
    print("[5/5] Evaluating perplexity on wikitext2 ...")
    avg_ppl = calculate_avg_ppl(model, tokenizer)
    print(f"Quantized model avg PPL (wikitext2): {avg_ppl:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
