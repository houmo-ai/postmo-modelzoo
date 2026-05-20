# Copyright 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Gemma4 Inference Demo
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
import os
import re
import sys
import math
import time
import argparse
import numpy as np
import torch
from transformers import GemmaTokenizer, Gemma4Processor
from PIL import Image
from loguru import logger

import tcim_lite as tcim
from hmatc.utils.perf_infomations import InferencePerformanceTracker, PERFTYPE
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
HOUMO_EXAMPLES_PATH = os.getenv("HOUMO_EXAMPLES_PATH", ".")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2"

PATCH_SIZE = 16
MAX_SOFT_TOKENS = 280
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


def get_default_tokenizer_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "gemma4")
    model_size = model_config.get("model_size", "26b-a4b")
    return f"{model_name}-{model_size}"


def is_valid_char(cp):
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x0041 <= cp <= 0x005A
        or 0x0061 <= cp <= 0x007A
    )


class HmGemma4:
    def __init__(
        self,
        prefill_path,
        decode_path,
        vit_path,
        embedding_path,
        tokenizer_dir,
        devices,
        max_new_tokens=2048,
    ):
        self.max_new_tokens = max_new_tokens
        self.sliding_window = 1024
        self.image_token_id = 258880
        self.perf_tracker = InferencePerformanceTracker()

        if isinstance(devices, int):
            devices = list(range(devices))
        self.devices = devices

        backend_name = "Xh2HalBackend"
        # Load vision model (optional)
        if vit_path and os.path.isfile(vit_path):
            self.perf_tracker.perf_start(PERFTYPE.VISION_LOAD_TIME)
            dev_mgr0 = tcim.runtime.DevManager(devices, backend_name)
            wm0 = tcim.runtime.WeightManager(dev_mgr0)
            self.vit = tcim.runtime.load(vit_path, option=tcim.runtime.Option(wm0))
            vit_shape = self.vit.get_input_info(self.vit.get_input_name(0)).shape
            vit_out_shape = self.vit.get_output_info(self.vit.get_output_name(0)).shape
            self.vit_num_patches = vit_shape[1]
            self.vit_num_tokens = vit_out_shape[1]
            self.perf_tracker.perf_end(PERFTYPE.VISION_LOAD_TIME)

            grid = int(math.sqrt(self.vit_num_patches))
            self.target_image_size = (grid * PATCH_SIZE, grid * PATCH_SIZE)
            self.upsample_token = self.vit_num_tokens != self.vit_num_patches
            logger.info(
                f"Vision: patches={self.vit_num_patches}, tokens={self.vit_num_tokens}, "
                f"size={self.target_image_size}, upsample={self.upsample_token}"
            )
        else:
            self.vit = None
            self.vit_num_patches = 0
            self.vit_num_tokens = 0
            self.target_image_size = None
            self.upsample_token = False
            logger.warning("Vision model not loaded, text-only mode")

        # Load LLM models
        dev_mgr = tcim.runtime.DevManager(devices, backend_name)
        wm = tcim.runtime.WeightManager(dev_mgr)

        logger.info(f"loading prefill model from {prefill_path}")
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_LOAD_TIME)
        self.prefill = tcim.runtime.load(prefill_path, option=tcim.runtime.Option(wm))
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_LOAD_TIME)
        logger.info("prefill loaded")

        self.prefill_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embed_dim = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.global_mask_w = self.prefill.get_input_info(
            self.prefill.get_input_name(4)
        ).shape[3]
        self.prefill_local_w = self.prefill.get_input_info(
            self.prefill.get_input_name(3)
        ).shape[3]
        self.context_max_length = self.global_mask_w
        logger.info(
            f"Prefill loaded: len={self.prefill_len}, embed_dim={self.embed_dim}"
        )

        # Find and share KV caches
        cache_names = [
            self.prefill.get_input_name(i)
            for i in range(self.prefill.get_num_inputs())
            if "cache" in self.prefill.get_input_name(i).lower()
        ]
        opt = tcim.runtime.Option(wm)
        opt.set_dummy_tensors(cache_names)
        logger.info(f"loading decode model from {decode_path}")
        self.perf_tracker.perf_start(PERFTYPE.DECODE_LOAD_TIME)
        self.decode = tcim.runtime.load(decode_path, option=opt)
        self.perf_tracker.perf_end(PERFTYPE.DECODE_LOAD_TIME)
        logger.info("decode loaded")

        self.decode_local_w = self.decode.get_input_info(
            self.decode.get_input_name(3)
        ).shape[3]
        self.decode_len = self.decode.get_input_info(
            self.decode.get_input_name(0)
        ).shape[1]
        logger.info(f"Decode loaded: len={self.decode_len}")
        logger.info(
            f"global_mask_w={self.global_mask_w}, prefill_local_w={self.prefill_local_w}, decode_local_w={self.decode_local_w}"
        )

        for name in cache_names:
            self.decode.set_input(name, self.prefill.get_dev_input(name))
        self.decode.set_input(
            self.decode.get_input_name(2), np.array([1], dtype="int32")
        )

        # Load tokenizer, processor, embedding
        self.tokenizer: GemmaTokenizer = GemmaTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        self.processor: Gemma4Processor = Gemma4Processor.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )

        if self.vit is not None:
            pool_size = 3 if self.upsample_token else 1
            self.processor.image_processor.max_soft_tokens = MAX_SOFT_TOKENS
            self.processor.image_processor.pooling_kernel_size = pool_size
            self.processor.image_seq_length = (
                MAX_SOFT_TOKENS if self.upsample_token else self.vit_num_patches
            )

        emb = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.embedding = emb["weight"] if isinstance(emb, dict) else emb
        self.embedding = self.embedding.reshape(-1, self.embed_dim).float()

        # Valid mask for vision input
        if self.vit is not None:
            valid_patches = self.vit_num_patches
            self.valid_mask = torch.tensor(
                [True] * valid_patches + [False] * (MAX_SOFT_TOKENS - valid_patches)
            )
        else:
            self.valid_mask = None

        self.perf_tracker.reset_perf_time()

    def _run_vision(self, pixel_values):
        if self.vit is None:
            raise RuntimeError("Vision model not loaded")
        self.perf_tracker.perf_start(PERFTYPE.VISION_INPUT_TIME)
        pv = pixel_values[:, self.valid_mask].half()
        if pv.shape[1] < self.vit_num_patches:
            pv = torch.cat(
                [pv, torch.zeros(1, self.vit_num_patches - pv.shape[1], pv.shape[2])],
                dim=1,
            )
        self.vit.set_input(
            self.vit.get_input_name(0), pv[:, : self.vit_num_patches].numpy()
        )
        self.perf_tracker.perf_end(PERFTYPE.VISION_INPUT_TIME)

        self.perf_tracker.perf_start(PERFTYPE.VISION_INFER_TIME)
        self.vit.run()
        self.vit.sync()
        self.perf_tracker.perf_end(PERFTYPE.VISION_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.VISION_OUTPUT_TIME)
        out = torch.from_numpy(
            self.vit.get_output(self.vit.get_output_name(0)).numpy()
        ).squeeze(0)
        self.perf_tracker.perf_end(PERFTYPE.VISION_OUTPUT_TIME)
        return out

    @staticmethod
    def _aligned(size: int, align: int) -> int:
        return ((size + align - 1) // align) * align

    def _build_masks(self, cur_len, past_len, mask_len=None, mm_types=None):
        q_len = mask_len
        neg = torch.finfo(torch.float16).min

        global_mask = torch.full(
            (1, 1, q_len, self.context_max_length), neg, dtype=torch.float16
        )
        valid_k = min(self.context_max_length, max(1, past_len + cur_len))
        for q in range(cur_len):
            global_mask[0, 0, q, : min(valid_k, past_len + q + 1)] = 0

        sw = self.sliding_window
        slide_ctx = (
            self.context_max_length
            if sw is None
            else min(self.context_max_length, self._aligned(sw + q_len - 1, 16))
        )
        clamped_past = min(past_len, sw - 1) if sw else past_len
        local_mask = torch.full((1, 1, q_len, slide_ctx), neg, dtype=torch.float16)
        for q in range(cur_len):
            causal_end = min(slide_ctx, clamped_past + q + 1)
            sw_start = max(0, clamped_past + q - sw + 1) if sw else 0
            local_mask[0, 0, q, sw_start:causal_end] = 0

        if mm_types is not None and mm_types.numel() > 0:
            mm = mm_types[:cur_len]
            is_vision = (mm == 1) | (mm == 2)
            cache_offset = max(0, past_len - clamped_past)
            group_start = None
            for idx in range(cur_len):
                if is_vision[idx] and group_start is None:
                    group_start = idx
                if group_start is not None and (
                    idx == cur_len - 1 or not is_vision[idx + 1]
                ):
                    group_end = idx + 1
                    abs_start, abs_end = past_len + group_start, past_len + group_end
                    global_mask[0, 0, group_start:group_end, abs_start:abs_end] = 0
                    c_start = max(0, abs_start - cache_offset)
                    c_end = min(slide_ctx, abs_end - cache_offset)
                    if c_start < slide_ctx and c_end > 0:
                        local_mask[0, 0, group_start:group_end, c_start:c_end] = 0
                    group_start = None

        return global_mask.numpy(), local_mask.numpy()

    def chat(self, question="", image_path=None):
        q_text = question or (
            "请详细描述这张图片的内容。" if image_path else "你好，请介绍一下你自己。"
        )
        logger.success(f"question: {q_text}")

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOTAL_TIME)

        # Vision preprocess (image resize)
        if image_path and self.vit is not None:
            self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
            self.perf_tracker.perf_start(PERFTYPE.VISION_PREPROCESS_TIME)
            img = (
                Image.open(image_path)
                .convert("RGB")
                .resize(self.target_image_size, BICUBIC)
            )
            self.perf_tracker.perf_end(PERFTYPE.VISION_PREPROCESS_TIME)
            self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
            content = [{"type": "image", "image": img}]
        else:
            content = []
        content.append({"type": "text", "text": q_text})

        # Tokenize
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_TOKEN_TIME)
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOKEN_TIME)

        input_ids = inputs["input_ids"]
        mm_types = inputs.get("mm_token_type_ids")
        input_len = input_ids.shape[-1]
        logger.info(f"Input length: {input_len}")

        if input_len >= self.context_max_length:
            logger.error(f"Input too long: {input_len}")
            sys.exit(1)

        # Build embeddings with vision
        self.perf_tracker.perf_start(PERFTYPE.PREFILL_EMBED_TIME)
        img_mask = input_ids == self.image_token_id
        llm_ids = input_ids.clone()
        llm_ids[img_mask] = self.tokenizer.pad_token_id or 0
        embeds = self.embedding[llm_ids[0]].unsqueeze(0).to(torch.float16)

        if (
            image_path
            and self.vit is not None
            and img_mask.any()
            and inputs.get("pixel_values") is not None
        ):
            self.perf_tracker.perf_start(PERFTYPE.VISION_TOTAL_TIME)
            img_emb = self._run_vision(inputs["pixel_values"])
            self.perf_tracker.perf_end(PERFTYPE.VISION_TOTAL_TIME)
            logger.info(f"Vision output: {img_emb.shape}")

            expand_mask = img_mask.unsqueeze(-1).expand_as(embeds)
            embeds = embeds.masked_scatter(expand_mask, img_emb)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_EMBED_TIME)

        # Prefill loop
        steps = math.ceil(input_len / self.prefill_len)
        for s in range(steps):
            start, end = s * self.prefill_len, min(
                (s + 1) * self.prefill_len, input_len
            )
            sub_emb = embeds[:, start:end]
            if sub_emb.shape[1] < self.prefill_len:
                sub_emb = torch.cat(
                    [
                        sub_emb,
                        torch.zeros(
                            1, self.prefill_len - sub_emb.shape[1], sub_emb.shape[2]
                        ),
                    ],
                    dim=1,
                )

            chunk_mm = mm_types[:, start:end][0] if mm_types is not None else None
            g_mask, l_mask = self._build_masks(
                end - start, start, self.prefill_len, chunk_mm
            )

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INPUT_TIME)
            self.prefill.set_input(
                self.prefill.get_input_name(0), sub_emb.numpy().astype(np.float16)
            )
            self.prefill.set_input(
                self.prefill.get_input_name(1), np.array([start], dtype="int32")
            )
            self.prefill.set_input(
                self.prefill.get_input_name(2), np.array([end - start], dtype="int32")
            )
            self.prefill.set_input(
                self.prefill.get_input_name(3), l_mask.astype(np.float16)
            )
            self.prefill.set_input(
                self.prefill.get_input_name(4), g_mask.astype(np.float16)
            )
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.PREFILL_INFER_TIME)
            self.prefill.run()
            self.prefill.sync()
            self.perf_tracker.perf_end(PERFTYPE.PREFILL_INFER_TIME)

        self.perf_tracker.perf_start(PERFTYPE.PREFILL_OUTPUT_TIME)
        next_id = (
            self.prefill.get_output(self.prefill.get_output_name(0)).numpy().argmax(-1)
        )
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_OUTPUT_TIME)
        self.perf_tracker.perf_end(PERFTYPE.PREFILL_TOTAL_TIME)
        logger.info(f"Prefill done, first token: {self.tokenizer.decode(next_id[0])}")

        # Decode loop
        eos_ids = (
            {self.tokenizer.eos_token_id}
            if isinstance(self.tokenizer.eos_token_id, int)
            else set(self.tokenizer.eos_token_id)
        )
        eos_ids.add(106)

        logger.success("response:")
        print(f"\033[1;95m{self.tokenizer.decode(next_id[0])}", end="", flush=True)

        history = input_ids[0].tolist() + [next_id[0][0]]
        past_len = input_len
        step = 0
        slide = 10
        skip = 0
        last_resp = self.tokenizer.decode(history[-slide:])

        self.perf_tracker.perf_start(PERFTYPE.DECODE_TOTAL_TIME)
        t0 = time.time()

        while past_len < self.context_max_length and step < self.max_new_tokens:
            self.perf_tracker.perf_start(PERFTYPE.DECODE_EMBED_TIME)
            tok = torch.from_numpy(next_id)
            dec_emb = self.embedding[tok].reshape(1, 1, -1).to(torch.float16)
            # Pad to decode_len
            if self.decode_len > 1:
                dec_emb = torch.cat(
                    [
                        dec_emb,
                        torch.zeros(
                            1, self.decode_len - 1, self.embed_dim, dtype=torch.float16
                        ),
                    ],
                    dim=1,
                )
            self.perf_tracker.perf_end(PERFTYPE.DECODE_EMBED_TIME)

            g_mask, l_mask = self._build_masks(1, past_len, self.decode_len)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_INPUT_TIME)
            self.decode.set_input(
                self.decode.get_input_name(0), dec_emb.numpy().astype(np.float16)
            )
            self.decode.set_input(
                self.decode.get_input_name(1), np.array([past_len], dtype="int32")
            )
            self.decode.set_input(
                self.decode.get_input_name(3), l_mask.astype(np.float16)
            )
            self.decode.set_input(
                self.decode.get_input_name(4), g_mask.astype(np.float16)
            )
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INPUT_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_INFER_TIME)
            self.decode.run()
            self.decode.sync()
            self.perf_tracker.perf_end(PERFTYPE.DECODE_INFER_TIME)

            self.perf_tracker.perf_start(PERFTYPE.DECODE_OUTPUT_TIME)
            next_id = (
                self.decode.get_output(self.decode.get_output_name(0))
                .numpy()
                .astype(np.float32)
                .argmax(-1)
            )
            self.perf_tracker.perf_end(PERFTYPE.DECODE_OUTPUT_TIME)
            tok_id = next_id[0][0]

            if tok_id in eos_ids:
                break

            history.append(tok_id)
            resp = self.tokenizer.decode(history[-(slide + 1) - skip :])[
                len(last_resp) :
            ]
            if resp and is_valid_char(ord(resp[-1])):
                print(resp, end="", flush=True)
                last_resp = self.tokenizer.decode(history[-slide:])
                skip = 0
            else:
                skip += 1

            past_len += 1
            step += 1

        self.perf_tracker.perf_end(PERFTYPE.DECODE_TOTAL_TIME)
        print(f"\033[0m")
        logger.info(f"Decode: {step} tokens in {time.time() - t0:.2f}s")
        self.perf_tracker.set_basic_info(
            1, input_len, step, num_images=1 if image_path else 0
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
    )
    parser.add_argument("--prefill_path", type=str, default=None)
    parser.add_argument("--decode_path", type=str, default=None)
    parser.add_argument("--vit_path", type=str, default=None)
    parser.add_argument("--ndevice", type=int, default=None, help="device number")
    parser.add_argument("--max_size_w", type=int, default=None)
    parser.add_argument("--max_size_h", type=int, default=None)
    parser.add_argument("--question", default="")
    parser.add_argument("--image", default="data/pic/beach.jpeg")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config_path
    )
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.ndevice = first_not_none(args.ndevice, model_config.get("ndevice", 1))
    args.max_size_w = first_not_none(
        args.max_size_w, model_config.get("max_size_w", 448)
    )
    args.max_size_h = first_not_none(
        args.max_size_h, model_config.get("max_size_h", 448)
    )
    if args.tokenizer_dir is None:
        args.tokenizer_dir = get_default_tokenizer_dir(model_config)
    if args.prefill_path is None:
        args.prefill_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_prefill.hmm"
        )
    if args.decode_path is None:
        args.decode_path = os.path.join(
            "output", HOUMO_TARGET, f"{args.model_name}-{args.model_size}_decode.hmm"
        )
    if args.vit_path is None:
        args.vit_path = os.path.join(
            "output",
            HOUMO_TARGET,
            f"{args.model_name}-{args.model_size}_visual_{args.max_size_w}x{args.max_size_h}.hmm",
        )

    if args.ndevice > 1:
        args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    return args


if __name__ == "__main__":
    args = get_args()
    model = HmGemma4(
        args.prefill_path,
        args.decode_path,
        args.vit_path,
        args.embedding_path,
        args.tokenizer_dir,
        list(range(args.ndevice)),
        args.max_new_tokens,
    )
    image_path = args.image
    if image_path and not os.path.isfile(image_path):
        image_path = os.path.join(HOUMO_EXAMPLES_PATH, args.image)
        if not os.path.isfile(image_path):
            logger.warning(f"Image not found: {image_path}, running text-only mode")
            image_path = None
    if image_path and model.vit is None:
        logger.warning("Image provided but vit not loaded, running text-only mode")
        image_path = None
    model.chat(args.question, image_path)
    model.perf_tracker.show_summary()
