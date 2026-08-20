#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 HOUMO AI
#
# File: demo_prefix_caching.py
# Description:
#   Qwen3.5 vision generation with runtime prefix caching.
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
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
IMODELZOO_ROOT = Path(__file__).resolve().parents[4]
HOUMO_EXAMPLES_PATH = Path(os.getenv("HOUMO_EXAMPLES_PATH", str(IMODELZOO_ROOT)))
sys.path.insert(0, str(HOUMO_EXAMPLES_PATH / "utils" / "python"))

from houmo_engine.sampling import GreedySamplingParams

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / HOUMO_TARGET
DEFAULT_IMAGE_PATHS = [str(HOUMO_EXAMPLES_PATH / "data" / "pic" / "beach.jpeg")]
DEFAULT_IMAGE_TOKEN_GEARS = (96, 196, 384, 704, 1536)


def _infer_vision_gear(path: str) -> int | None:
    parts = [Path(path).stem, *[parent.name for parent in Path(path).parents[:2]]]
    for part in parts:
        match = re.search(r"(?:^|[_-])m(\d+)(?:$|[_-])", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _resolve_vision_paths(path_args, model_prefix: str) -> dict[int, str]:
    if path_args is None:
        paths = {}
        candidates = sorted(
            path
            for path in DEFAULT_OUTPUT_DIR.glob(f"**/{model_prefix}_visual_m*.hmm")
            if path.is_file()
        )
        for path in candidates:
            gear = _infer_vision_gear(str(path))
            if gear in DEFAULT_IMAGE_TOKEN_GEARS:
                if gear in paths:
                    raise ValueError(f"duplicate vision gear m{gear}: {paths[gear]} and {path}")
                paths[gear] = str(path)
        if not paths:
            candidates = sorted(
                path
                for path in DEFAULT_OUTPUT_DIR.glob(f"**/{model_prefix}_visual_*.hmm")
                if path.is_file() and _infer_vision_gear(str(path)) is None
            )
            if len(candidates) > 1:
                raise ValueError(f"multiple static vision HMMs found: {candidates}")
            if candidates:
                paths[DEFAULT_IMAGE_TOKEN_GEARS[-1]] = str(candidates[0])
        if not paths:
            fallback = DEFAULT_OUTPUT_DIR / f"{model_prefix}_visual.hmm"
            if fallback.is_file():
                paths[DEFAULT_IMAGE_TOKEN_GEARS[-1]] = str(fallback)
        if not paths:
            raise FileNotFoundError(
                f"no vision HMM found under {DEFAULT_OUTPUT_DIR}; expected "
                f"{model_prefix}_visual_m<gear>.hmm, {model_prefix}_visual_<resolution>.hmm, "
                f"or {model_prefix}_visual.hmm"
            )
    else:
        specs = [item for group in path_args for value in group for item in value.split(",") if item]
        expanded = []
        for spec in specs:
            explicit = re.match(r"^m?(\d+)[=:](.+)$", spec, flags=re.IGNORECASE)
            if explicit:
                expanded.append((int(explicit.group(1)), explicit.group(2).strip()))
            elif Path(spec).is_dir():
                directory_paths = [
                    (_infer_vision_gear(str(path)), str(path))
                    for path in sorted(Path(spec).rglob("*_visual_m*.hmm"))
                ]
                if not directory_paths:
                    static_paths = [
                        path
                        for path in sorted(Path(spec).rglob("*_visual_*.hmm"))
                        if _infer_vision_gear(str(path)) is None
                    ]
                    if not static_paths:
                        static_paths = sorted(Path(spec).rglob("*_visual.hmm"))
                    directory_paths = [(DEFAULT_IMAGE_TOKEN_GEARS[-1], str(path)) for path in static_paths]
                expanded.extend(directory_paths)
            else:
                expanded.append((_infer_vision_gear(spec), spec))
        unresolved = [path for gear, path in expanded if gear is None]
        if unresolved:
            raise ValueError(f"cannot infer gear from vision paths {unresolved}; use <gear>=<path>")
        paths = {}
        for gear, path in expanded:
            if gear not in DEFAULT_IMAGE_TOKEN_GEARS:
                raise ValueError(f"unsupported vision gear m{gear}")
            if gear in paths:
                raise ValueError(f"duplicate vision gear m{gear}")
            paths[gear] = path
    return dict(sorted(paths.items()))
QUESTION_PREFIX = (
    "请仔细观察这张图片。回答时只能依据图片中可以直接观察到的内容；"
    "如果某项信息无法从图片中确定，请明确说明无法确定，不要猜测。回答下"
)
DEFAULT_QUESTIONS = [
    QUESTION_PREFIX + "图片中有哪些主要主体？它们正在进行什么互动？",
    QUESTION_PREFIX + "这张图片拍摄在什么环境中？请列出支持你判断的三个可见线索。",
    QUESTION_PREFIX + "描述女性和狗的外观细节，包括衣着、姿势以及它们身上可见的物品。",
]
TEXT_QUESTION_PREFIX = (
    "你正在参加一个用于验证大语言模型长前缀缓存功能的严格问答测试。"
    "下面是一段所有测试轮次都完全相同的任务说明和作答协议。你必须完整阅读并遵守，"
    "但不得在答案中复述、引用、概括或评论这些说明。"
    "第一，回答必须使用简体中文，不得夹杂英文、拼音、数字、公式、代码、符号或外语缩写。"
    "第二，只能回答整段文字最后提出的那个具体问题，不得回答尚未被询问的相关问题，"
    "不得自行扩展讨论范围，也不得把多个可能答案同时列出。"
    "第三，不要展示思考过程，不要说明判断依据，不要使用诸如因为、所以、首先、其次、"
    "综上所述、一般来说、从技术角度看等引导语。"
    "第四，不要复述用户的问题，不要添加标题、前言、背景、定义、解释、案例、比较、"
    "备注、括号、脚注、免责声明、建议、结论或结束语。"
    "第五，不得使用 Markdown，不得使用项目符号、编号、表格、引用、粗体、代码块、"
    "换行分段或任何排版标记，只允许输出一行纯文本。"
    "第六，答案必须准确、直接、具体，优先选择最具代表性且最常见的一个短语，"
    "不得输出两个或更多并列答案，不得使用或者、以及、同时、等、等等之类的扩展表达。"
    "第七，答案必须严格控制在六个汉字以内；如果完整术语超过六个汉字，应使用业内常见、"
    "含义明确且不会产生歧义的简称，但不得为了缩短答案而使用生造缩写。"
    "第八，如果问题存在多个合理答案，只选择通常被认为最核心、最典型或最主要的一个，"
    "不要解释为什么选择它，也不要补充其他候选项。"
    "第九，如果你认为信息不足，也必须依据通用公开知识给出最可能的标准短语，"
    "不得回答无法确定、视情况而定、需要更多信息或类似回避性内容。"
    "第十，在生成答案之前应在内部检查语言、内容、格式和长度是否同时满足要求，"
    "但不得输出检查过程；如果初稿不符合要求，应在内部修正后只输出最终短语。"
    "本轮测试的固定主题是存算一体技术。存算一体是一类通过缩短数据存储单元与计算单元"
    "之间的物理或逻辑距离，减少传统计算架构中频繁数据搬运的技术方向。相关讨论通常涉及"
    "内存带宽、访问延迟、数据搬运能耗、并行计算、芯片面积、器件一致性、计算精度、"
    "编程模型、软硬件协同以及人工智能推理等方面。这里提供主题范围只是为了限定语境，"
    "并不要求你在答案中解释这些概念，也不代表最后问题一定询问上述全部内容。"
    "请再次确认：你只能回答最后一个具体问题；答案只能是一行简体中文纯文本；"
    "不得复述问题；不得给出理由；不得添加任何标点；不得超过六个汉字；"
    "不得输出多个答案；不得输出本协议中的任何内容。"
    "以上任务说明、主题背景、格式要求和长度约束在每一轮测试中都保持完全一致。"
    "真正需要回答的问题只会出现在这段长前缀之后。现在请严格遵守全部要求，回答："
)
DEFAULT_TEXT_QUESTIONS = [
    TEXT_QUESTION_PREFIX + "它最主要的优势是什么？",
    TEXT_QUESTION_PREFIX + "它最典型的应用是什么？",
    TEXT_QUESTION_PREFIX + "它最主要的挑战是什么？",
]


def _create_prefix_caching_engine():
    import numpy as np
    import torch

    from houmo_engine import Qwen35Engine
    from houmo_engine.core.types import Stage, StageInputs, StageOutputs
    from houmo_engine.sampling import GreedySampler

    class PrefixCachingQwen35Engine(Qwen35Engine):
        """Qwen3.5 Engine extension scoped to this prefix caching demo."""

        def __init__(self, *args, enable_prefix_cache: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_prefix_cache = enable_prefix_cache
            self.prefix_snapshots = {}
            self.cached_input_ids = []
            self.cached_image_paths = ()
            self.cached_image_grid = ()
            self.cached_image_outputs = None

        @staticmethod
        def _matched_prefix_length(previous_ids, current_ids) -> int:
            matched = 0
            for previous, current in zip(previous_ids, current_ids):
                if previous != current:
                    break
                matched += 1
            return matched

        def _snapshot_prefix_state(self):
            state = {}
            for index in range(self.module.prefill.get_num_inputs()):
                name = self.module.prefill.get_input_name(index)
                if "conv_cache" in name or "recurrent_state" in name:
                    state[name] = self.module.prefill.get_dev_input(name).clone()
            return state

        def _restore_prefix_state(self, state) -> None:
            for name, value in state.items():
                value.copy_to(self.module.prefill.get_dev_input(name))

        def _prepare_vision_positions(self, request) -> None:
            positions, self.state.rope_deltas = self.process._vision_positions(
                request.input_ids,
                request.image_grid_thw,
                request.attention_mask,
            )
            request.positions = positions[:, 0, :] + self.state.context_length

        def _vision_with_cache(
            self,
            request,
            image_paths: tuple[str, ...],
            restored_prefix_length: int,
        ) -> None:
            self._prepare_vision_positions(request)
            image_token_id = self.process.processor.image_token_id
            suffix_has_image_tokens = bool(
                (request.input_ids[:, restored_prefix_length:] == image_token_id).any()
            )
            if not suffix_has_image_tokens:
                return

            with self.perf.scope("llm.vision"):
                if (
                    self.enable_prefix_cache
                    and image_paths == self.cached_image_paths
                    and tuple(request.image_grid_thw.flatten().tolist()) == self.cached_image_grid
                    and self.cached_image_outputs is not None
                ):
                    outputs = self.cached_image_outputs
                else:
                    outputs = self.module.run_vision(
                        request.vision_values,
                        request.image_grid_thw,
                    )
                    if self.enable_prefix_cache:
                        image_embeds = outputs.tensors[0].clone()
                        outputs = StageOutputs(tensors=(image_embeds,))
                        self.cached_image_outputs = outputs
                        self.cached_image_grid = tuple(request.image_grid_thw.flatten().tolist())
                self.process.merge_vision(request, outputs, self.state)

        def _prepare_prefill_chunk(self, request, start: int, end: int) -> StageInputs:
            current_length = end - start
            chunk = request.token_embeds[:, start:end]
            padded = torch.zeros(
                1,
                self.prefill_length,
                self.embedding_size,
                dtype=chunk.dtype,
            )
            padded[:, :current_length] = chunk
            if request.uses_vision:
                chunk_positions = request.positions[:, start:end]
                if current_length < self.prefill_length:
                    chunk_positions = torch.cat(
                        [
                            chunk_positions,
                            chunk_positions[:, -1:].expand(
                                -1, self.prefill_length - current_length
                            ),
                        ],
                        dim=1,
                    )
            else:
                chunk_positions = self.process.text_positions(
                    start, self.prefill_length
                )
            return StageInputs(
                tensors=(
                    padded,
                    chunk_positions[0:1],
                    chunk_positions[1:2],
                    chunk_positions[2:3],
                    np.array([start], dtype=np.int32),
                    np.array([current_length], dtype=np.int32),
                    self.process.attention_mask(
                        self.prefill_length, current_length
                    ),
                ),
                metadata={"current_length": current_length},
            )

        def _prefill_with_cache(
            self,
            request,
            matched_prefix_length: int,
            restored_prefix_length: int,
        ):
            input_length = int(request.input_ids.shape[1])
            if input_length >= self.context_max_length:
                raise ValueError("input exceeds model context length")
            if restored_prefix_length >= input_length:
                raise ValueError("cached prefix must be shorter than the complete prompt")

            logits = None
            with self.perf.scope("llm.prefill"):
                start = restored_prefix_length
                while start < input_length:
                    end = min(start + self.prefill_length, input_length)
                    if start < matched_prefix_length < end:
                        end = matched_prefix_length
                    inputs = self._prepare_prefill_chunk(request, start, end)
                    self.module.set_input(Stage.PREFILL, inputs)
                    scope = (
                        "llm.prefix.replay"
                        if end <= matched_prefix_length
                        else "llm.prefix.suffix_prefill"
                    )
                    with self.perf.scope(scope):
                        self.module.run(Stage.PREFILL)
                    logits = self.module.get_output(Stage.PREFILL).tensors[0]
                    if self.enable_prefix_cache:
                        with self.perf.scope("llm.prefix.snapshot"):
                            self.prefix_snapshots[end] = self._snapshot_prefix_state()
                    start = end
                token = int(np.asarray(logits).reshape(-1).argmax())
            self.state.context_length = input_length
            return token, input_length

        def generate(
            self,
            prompt: str,
            *,
            images=None,
            sampling_params=None,
            max_new_tokens: int | None = None,
            **kwargs,
        ):
            del kwargs
            if not prompt:
                raise ValueError("prompt must not be empty")
            if max_new_tokens is not None and max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be greater than zero")
            images = self.process.normalize_images(images)
            image_paths = tuple(str(image) for image in images) if images else ()

            self.perf.reset(preserve_prefixes=("llm.init",))
            self.clear_session()
            if sampling_params is not None:
                self.sampler = GreedySampler(sampling_params)
            self.state.rope_deltas = None
            self.state.generated_ids = []
            self.state.emitted_text = ""
            decode_tokens = 0
            ttft_active = True
            e2e_active = True
            self.perf.start("llm.e2e")
            self.perf.start("llm.ttft")
            try:
                system_prompt = "" if images else "You are a helpful assistant."
                request = self.process.preprocess(prompt, images, system_prompt)
                input_ids = request.input_ids[0].tolist()
                matched_prefix_length = 0
                restored_prefix_length = 0
                if (
                    self.enable_prefix_cache
                    and image_paths == self.cached_image_paths
                    and self.cached_input_ids
                ):
                    matched_prefix_length = self._matched_prefix_length(
                        self.cached_input_ids, input_ids
                    )
                    reusable = [
                        position
                        for position in self.prefix_snapshots
                        if position <= matched_prefix_length
                        and position < len(input_ids)
                    ]
                    restored_prefix_length = max(reusable, default=0)
                    if restored_prefix_length:
                        with self.perf.scope("llm.prefix.restore"):
                            self._restore_prefix_state(
                                self.prefix_snapshots[restored_prefix_length]
                            )
                    self.prefix_snapshots = {
                        position: state
                        for position, state in self.prefix_snapshots.items()
                        if position <= restored_prefix_length
                    }
                elif self.enable_prefix_cache:
                    self.prefix_snapshots = {}

                self.perf.set_metrics(
                    "llm.prefix",
                    matched_tokens=matched_prefix_length,
                    restored_tokens=restored_prefix_length,
                    replay_tokens=matched_prefix_length - restored_prefix_length,
                )
                if request.uses_vision:
                    self._vision_with_cache(
                        request,
                        image_paths,
                        restored_prefix_length,
                    )
                token, input_tokens = self._prefill_with_cache(
                    request,
                    matched_prefix_length,
                    restored_prefix_length,
                )
                if self.enable_prefix_cache:
                    self.cached_input_ids = input_ids
                    self.cached_image_paths = image_paths

                self.state.generated_ids.append(token)
                self.perf.end("llm.ttft")
                ttft_active = False
                delta = self.process.postprocess(self.state)
                if delta:
                    self.perf.end("llm.e2e")
                    e2e_active = False
                    yield delta
                    self.perf.start("llm.e2e")
                    e2e_active = True

                while token not in self.stop_token_ids:
                    if (
                        max_new_tokens is not None
                        and len(self.state.generated_ids) >= max_new_tokens
                    ):
                        break
                    if self.state.context_length >= self.context_max_length:
                        break
                    token = self._decode(token)
                    decode_tokens += 1
                    if token in self.stop_token_ids:
                        break
                    self.state.generated_ids.append(token)
                    delta = self.process.postprocess(self.state)
                    if delta:
                        self.perf.end("llm.e2e")
                        e2e_active = False
                        yield delta
                        self.perf.start("llm.e2e")
                        e2e_active = True

                remainder = self.process.postprocess(self.state, final=True)
                if remainder:
                    self.perf.end("llm.e2e")
                    e2e_active = False
                    yield remainder
                    self.perf.start("llm.e2e")
                    e2e_active = True
                self.perf.set_metrics(
                    "llm",
                    input_tokens=input_tokens - restored_prefix_length,
                    output_tokens=1 + decode_tokens,
                    decode_tokens=decode_tokens,
                    num_images=len(images) if images else 0,
                )
            finally:
                if ttft_active:
                    self.perf.end("llm.ttft")
                if e2e_active:
                    self.perf.end("llm.e2e")

    return PrefixCachingQwen35Engine


class HmQwen35PrefixCaching:
    def __init__(
        self,
        *,
        prefill_path,
        decode_path,
        embedding_path,
        tokenizer_path,
        vision_paths,
        ndevice: int = 1,
        batch: int = 1,
        vision_min_pixels: int = 65536,
        num_position_embeddings: int = 2304,
        visual_rope_cache_length: int = 3072,
        patch_size: int = 16,
        sampling_params: GreedySamplingParams | None = None,
        enable_prefix_cache: bool = True,
        perf: bool = False,
    ):
        engine_type = _create_prefix_caching_engine()
        self.engine = engine_type(
            prefill_path=prefill_path,
            decode_path=decode_path,
            vision_paths=vision_paths,
            embedding_path=embedding_path,
            tokenizer_path=tokenizer_path,
            ndevice=ndevice,
            batch=batch,
            vision_min_pixels=vision_min_pixels,
            num_position_embeddings=num_position_embeddings,
            visual_rope_cache_length=visual_rope_cache_length,
            patch_size=patch_size,
            sampling_params=sampling_params,
            enable_prefix_cache=enable_prefix_cache,
            perf=perf,
        )

    def generate(self, prompt: str, *, images, max_new_tokens: int | None = None):
        yield from self.engine.generate(
            prompt,
            images=images,
            max_new_tokens=max_new_tokens,
        )

    def print_perf(self) -> None:
        self.engine.perf.print_summary()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_name", dest="model_name", type=str, default=None, help="model name"
    )
    parser.add_argument(
        "--model_size", dest="model_size", type=str, default=None, help="model size"
    )
    parser.add_argument(
        "--question",
        dest="questions",
        type=str,
        default=None,
        action="append",
        help="question; repeat to run multiple turns",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=None,
        help="path to the prefill HMM model",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=None,
        help="path to the decode HMM model",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=None,
        help="path to the embedding weights",
    )
    parser.add_argument(
        "--vision_path",
        dest="vision_path",
        nargs="+",
        action="append",
        default=None,
        help="dynamic vision HMMs: directory, files containing _m<gear>, or <gear>=<path>",
    )
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default=None,
        help="path to tokenizer and processor files",
    )
    parser.add_argument(
        "--image_path",
        dest="image_path",
        type=str,
        default=None,
        nargs="+",
        action="extend",
        help="one or more image paths; use None or null for text-only mode",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=None,
        help="number of Houmo devices",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=1,
        help="inference batch size; only 1 is supported",
    )
    parser.add_argument(
        "--vision_min_pixels",
        dest="vision_min_pixels",
        type=int,
        default=65536,
        help="minimum dynamic image pixel budget",
    )
    parser.add_argument(
        "--num_position_embeddings",
        dest="num_position_embeddings",
        type=int,
        default=2304,
        help="number of learned ViT 2D position embeddings",
    )
    parser.add_argument(
        "--visual_rope_cache_length",
        dest="visual_rope_cache_length",
        type=int,
        default=3072,
        help="maximum dynamic visual rotary position",
    )
    parser.add_argument(
        "--patch_size",
        dest="patch_size",
        type=int,
        default=16,
        help="vision patch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="maximum generated tokens per turn",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="sampling temperature",
    )
    parser.add_argument(
        "--topk",
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="top-k logits filtering value",
    )
    parser.add_argument(
        "--topp",
        "--top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="top-p probability filtering value",
    )
    parser.add_argument(
        "--presence-penalty",
        dest="presence_penalty",
        type=float,
        default=1.5,
        help="presence penalty applied to generated tokens",
    )
    parser.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
        help="repetition penalty",
    )
    parser.add_argument(
        "--prefix-cache",
        dest="enable_prefix_cache",
        type=_parse_bool,
        default=True,
        nargs="?",
        const=True,
        help="enable runtime prefix and vision embedding cache",
    )
    parser.add_argument(
        "--perf",
        dest="perf",
        type=_parse_bool,
        default=True,
        nargs="?",
        const=True,
        help="enable performance reporting",
    )
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    import yaml

    with Path(args.config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    args.model_name = args.model_name or config["default_model_name"]
    args.model_size = args.model_size or config["default_model_size"]
    try:
        model_config = config["model_configs"][args.model_name][args.model_size]
    except KeyError as error:
        raise ValueError(
            f"unsupported model configuration: {args.model_name}-{args.model_size}"
        ) from error
    args.ndevice = args.ndevice or int(model_config.get("ndevice", 1))
    prefix = f"{args.model_name}-{args.model_size}"
    args.prefill_path = args.prefill_path or str(
        DEFAULT_OUTPUT_DIR / f"{prefix}_prefill.hmm"
    )
    args.decode_path = args.decode_path or str(
        DEFAULT_OUTPUT_DIR / f"{prefix}_decode.hmm"
    )
    args.embedding_path = args.embedding_path or str(
        DEFAULT_OUTPUT_DIR / "hmquant" / "quant_embedding.pt"
    )
    if args.tokenizer_dir is None:
        repo_ids = model_config.get("modelscope_repo", [])
        tokenizer_name = repo_ids[0].rsplit("/", maxsplit=1)[-1] if repo_ids else prefix
        args.tokenizer_dir = str(MODEL_DIR / tokenizer_name)
    args.vision_paths = _resolve_vision_paths(args.vision_path, prefix)
    if args.ndevice > 1:
        if args.prefill_path.endswith(".hmm"):
            args.prefill_path = args.prefill_path.replace(".hmm", ".hmms")
        if args.decode_path.endswith(".hmm"):
            args.decode_path = args.decode_path.replace(".hmm", ".hmms")
    if args.image_path is None:
        args.image_path = DEFAULT_IMAGE_PATHS
    else:
        normalized_images = [
            image
            for image in args.image_path
            if image.strip().lower() not in {"", "none", "null"}
        ]
        args.image_path = normalized_images or None
    args.questions = args.questions or (
        DEFAULT_QUESTIONS if args.image_path else DEFAULT_TEXT_QUESTIONS
    )
    return args


def main() -> None:
    args = _resolve_args(get_args().parse_args())
    model = HmQwen35PrefixCaching(
        prefill_path=args.prefill_path,
        decode_path=args.decode_path,
        vision_paths=args.vision_paths,
        embedding_path=args.embedding_path,
        tokenizer_path=args.tokenizer_dir,
        ndevice=args.ndevice,
        batch=args.batch,
        vision_min_pixels=args.vision_min_pixels,
        num_position_embeddings=args.num_position_embeddings,
        visual_rope_cache_length=args.visual_rope_cache_length,
        patch_size=args.patch_size,
        sampling_params=GreedySamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
        ),
        enable_prefix_cache=args.enable_prefix_cache,
        perf=args.perf,
    )
    for question in args.questions:
        print(f"\033[1;95m\nQ: {question}\nA: ", end="", flush=True)
        for chunk in model.generate(
            question,
            images=args.image_path,
            max_new_tokens=args.max_new_tokens,
        ):
            print(f"\033[1;95m{chunk}", end="", flush=True)
        print("\033[0m")
        if args.perf:
            model.print_perf()


if __name__ == "__main__":
    main()
