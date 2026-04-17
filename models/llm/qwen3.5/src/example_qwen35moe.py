#
# Copyright (c) 2025 HOUMO AI
#
# File: example_qwen35moe.py
# Description:
#   Qwen3.5-MoE GPTQ quantization + generation validation.
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
import logging
import os
import random
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import ExpertsRoutingBypass, ExpertsRoutingOverride, MoEConfig


def get_wikitext2(nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen)

    return [example["text"] for example in traindata.select(range(nsamples))]


@torch.inference_mode()
def calculate_avg_ppl(model, tokenizer):
    from gptqmodel.utils.perplexity import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    )

    all = ppl.calculate(n_ctx=512, n_batch=512)

    avg = sum(all) / len(all)

    return avg


def build_chat_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        # Qwen3.5 chat templates may emit a long reasoning preamble unless
        # thinking mode is disabled at template rendering time.
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


def build_moe_config(args):
    if args.moe_routing == "none":
        return None

    if args.moe_routing == "bypass":
        return MoEConfig(routing=ExpertsRoutingBypass(batch_size=args.moe_routing_batch_size))

    return MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok=args.moe_num_experts_per_tok))


def build_dynamic_quant_config(args):
    return {
        r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$": {"bits": args.self_attn_bits},
        r".*\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$": {"bits": args.self_attn_bits},
        r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$": {"bits": args.expert_bits},
        r".*\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$": {"bits": args.shared_expert_bits},
    }


def build_default_output_dir(args):
    model_name = os.path.basename(os.path.normpath(args.model))
    bit_profile = f"attn{args.self_attn_bits}-expert{args.expert_bits}-shared{args.shared_expert_bits}-base{args.bits}"
    time_suffix = datetime.now().strftime("%m%d%H")
    return os.path.join("work_dirs", f"{model_name}-gptq-{bit_profile}-{args.group_size}g_{time_suffix}")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5-MoE GPTQ quantization + generation validation")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3.5-35B-A3B", help="fp model path")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "quantized model output directory, defaults to "
            "work_dirs/<model>-gptq-attn<self-attn>-expert<expert>-shared<shared>-base<bits>-<group_size>g"
        ),
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 8])
    parser.add_argument(
        "--self-attn-bits",
        type=int,
        default=8,
        choices=[2, 3, 4, 5, 8],
        help="bit width for self-attn q/k/v/o projections",
    )
    parser.add_argument(
        "--expert-bits",
        "--non-shared-expert-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 8],
        help="bit width for non-shared MoE experts",
    )
    parser.add_argument(
        "--shared-expert-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 8],
        help="bit width for shared MoE expert",
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=1024, help="minimum sequence length for wikitext2 filtering")
    parser.add_argument("--device", type=str, default="cuda:0", help="quantization device")
    parser.add_argument("--infer-device", type=str, default=None, help="inference device after quantization")
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--offload-path",
        type=str,
        default="./gptqmodel_offload_qwen35moe",
        help="offload path when --offload-to-disk is enabled",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="请介绍一下 GPTQ 量化的核心思想。")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--rotation",
        type=str,
        default=None,
        choices=["random", "hadamard", "block_hadamard"],
        help="可选 QuaRot 旋转模式",
    )
    parser.add_argument(
        "--hessian-mse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Hessian MSE assisted optimization",
    )
    parser.add_argument(
        "--moe-routing",
        type=str,
        default="none",
        choices=["none", "bypass", "override"],
        help="MoE routing mode: none keeps model routing, bypass forwards calibration to all experts, override forces top-k routing",
    )
    parser.add_argument(
        "--moe-routing-batch-size",
        type=int,
        default=None,
        help="Batch size for bypass routing to reduce VRAM pressure",
    )
    parser.add_argument(
        "--moe-num-experts-per-tok",
        type=str,
        default="all",
        help="Override top-k experts per token when --moe-routing=override; accepts a positive integer or 'all'",
    )
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


def main():
    args = parse_args()
    args.out = args.out or build_default_output_dir(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer_source = args.model
    if not args.skip_quantize:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
        calibration_dataset = get_wikitext2(nsamples=args.nsamples, seqlen=args.seqlen)
        print(f"[1/5] Calibration samples (wikitext2): {len(calibration_dataset)}")

        moe_config = build_moe_config(args)
        dynamic_config = build_dynamic_quant_config(args)

        print(
            "[config] bits: "
            f"base={args.bits}, self_attn={args.self_attn_bits}, "
            f"expert={args.expert_bits}, shared_expert={args.shared_expert_bits}"
        )

        quantize_config = QuantizeConfig(
            bits=args.bits,
            dynamic=dynamic_config,
            group_size=args.group_size,
            device=args.device,
            rotation=args.rotation,
            hessian_mse=args.hessian_mse,
            moe=moe_config,
            offload_to_disk=args.offload_to_disk,
            offload_to_disk_path=args.offload_path if args.offload_to_disk else None,
        )

        print("[2/5] Loading fp model ...")
        model = GPTQModel.load(args.model, quantize_config, trust_remote_code=args.trust_remote_code)

        print("[3/5] Running GPTQ quantization ...")
        model.quantize(calibration_dataset, batch_size=args.batch_size)
        model.save(args.out)
        print(f"Quantized model saved to: {args.out}")
        if args.stop_after_quantize:
            return
    else:
        print(f"[1/5] Skip quantization, using existing quantized model: {args.out}")
        tokenizer = AutoTokenizer.from_pretrained(args.out, use_fast=True, trust_remote_code=args.trust_remote_code)
        tokenizer_source = args.out

    infer_device = args.infer_device or args.device
    print(f"[4/5] Loading quantized model on {infer_device} and generating ...")
    model = GPTQModel.load(args.out, device=infer_device, trust_remote_code=args.trust_remote_code)
    runtime_device = str(model.device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=args.trust_remote_code)
    inputs = build_chat_inputs(tokenizer, args.prompt, runtime_device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Answer ===")
    print(answer)

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
