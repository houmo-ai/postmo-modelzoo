#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Offline CT-Transformer punctuation inference demo using a compiled HMM.
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
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from loguru import logger

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_HMM_DIR = Path(__file__).resolve().parent / "output" / HOUMO_TARGET
DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "ct_transformer"
    / "example"
    / "punc_example.txt"
)


def _resolve_input(input_arg: str, model_dir: Path) -> Path:
    """Resolve input file path using only local files."""
    input_path = Path(input_arg)
    if input_path.exists():
        return input_path

    if model_dir.exists():
        candidate = model_dir / "example" / "punc_example.txt"
        if candidate.exists():
            return candidate

    return input_path


def first_not_none(*args: Any) -> Any:
    """Return the first argument that is not None."""
    for arg in args:
        if arg is not None:
            return arg
    return None


def get_model_configs(config_path: str) -> tuple[str, str, dict[str, Any]]:
    """Load model configs from yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    default_model_size = config.get("default_model_size", "")
    default_model_name = config.get("default_model_name", "")
    model_configs = config.get("model_configs", {})
    return default_model_size, default_model_name, model_configs


def get_default_model_dir() -> Path:
    """Return the local directory containing preprocessing metadata.

    The demo deliberately does not use a ModelScope repo id as an implicit
    default: the HMM does not need the original PyTorch weights, but it does
    need a local tokenizer vocabulary and punctuation metadata.
    """
    return Path(__file__).resolve().parent / "ct_transformer"


class _LocalTokenizer:
    """Small ``tokens.json`` tokenizer; no FunASR model construction needed."""

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and all(isinstance(v, int) for v in data.values()):
            self.token_to_id: dict[str, int] = {
                str(k): int(v) for k, v in data.items()
            }
        elif isinstance(data, dict) and all(str(k).isdigit() for k in data):
            self.token_to_id = {str(v): int(k) for k, v in data.items()}
        elif isinstance(data, list):
            self.token_to_id = {str(v): i for i, v in enumerate(data)}
        else:
            raise ValueError(f"Unsupported tokens.json format: {path}")
        self.unk_id: int = next(
            (
                self.token_to_id[key]
                for key in ("<unk>", "<UNK>", "[UNK]")
                if key in self.token_to_id
            ),
            1 if len(self.token_to_id) > 1 else 0,
        )

    def encode(self, tokens: Sequence[str]) -> list[int]:
        """Convert tokens to vocabulary IDs."""
        return [
            self.token_to_id.get(str(token), self.unk_id) for token in tokens
        ]


class _LocalPreprocessor:
    """Load only tokenizer/segmentation/punctuation metadata from disk."""

    def __init__(self, model_dir: Path) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Preprocessing directory not found: {model_dir}. "
                "Pass --model_dir to a local directory containing tokens.json."
            )
        tokens_path = model_dir / "tokens.json"
        if not tokens_path.exists():
            raise FileNotFoundError(
                f"Missing {tokens_path}; tokenizer metadata is required, but "
                "original PyTorch weights are not."
            )
        self.tokenizer = _LocalTokenizer(tokens_path)
        jieba_candidates = (
            model_dir / "jieba_usr_dict",
            model_dir / "jieba_usr_dict.txt",
            model_dir / "seg_dict",
        )
        jieba_usr_dict_path = next(
            (path for path in jieba_candidates if path.exists()),
            None,
        )
        self.jieba_usr_dict: Any | None = None
        if jieba_usr_dict_path is not None:
            try:
                import jieba
                import logging
                jieba.setLogLevel(logging.WARNING)
            except ImportError as exc:
                raise ImportError(
                    "jieba is required to load jieba_usr_dict"
                ) from exc
            jieba.load_userdict(str(jieba_usr_dict_path))
            self.jieba_usr_dict = jieba
        self.punc_list: list[str] = self._read_punc_list(model_dir)
        self.sentence_end_id: int = self._read_sentence_end_id(model_dir)

    @staticmethod
    def _read_metadata(path: Path) -> Any:
        """Read JSON or YAML metadata, returning ``None`` when invalid."""
        try:
            if path.suffix == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @classmethod
    def _read_punc_list(cls, model_dir: Path) -> list[str]:
        """Read the punctuation label list from local metadata."""
        for name in ("punc_list.json", "config.yaml", "configuration.json"):
            path = model_dir / name
            if not path.exists():
                continue
            data = cls._read_metadata(path)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                value = data.get("punc_list") or data.get("punctuation_list")
                if value is None and isinstance(data.get("model_conf"), dict):
                    value = data["model_conf"].get("punc_list")
                if isinstance(value, list):
                    return value
        # Label order of the published punc_ct-transformer model.
        return ["_", "，", "。", "？"]

    @classmethod
    def _read_sentence_end_id(cls, model_dir: Path) -> int:
        """Read the sentence-ending punctuation ID from local metadata."""
        for name in ("sentence_end_id.json", "config.yaml", "configuration.json"):
            path = model_dir / name
            if not path.exists():
                continue
            data = cls._read_metadata(path)
            if isinstance(data, int):
                return data
            if isinstance(data, dict) and isinstance(data.get("sentence_end_id"), int):
                return data["sentence_end_id"]
            if isinstance(data, dict) and isinstance(data.get("model_conf"), dict):
                value = data["model_conf"].get("sentence_end_id")
                if isinstance(value, int):
                    return value
        return 2


def load_text_items(input_path: Path) -> list[str]:
    """Load non-empty text lines from an input file."""
    items: list[str] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(line)
    if not items:
        raise ValueError(f"No valid text lines found in {input_path}")
    return items


def tokenize_to_chunks(
    preprocessor: _LocalPreprocessor,
    text: str,
    split_size: int,
) -> tuple[list[list[str]], list[list[int]]]:
    """Split text and token IDs into aligned mini-sentence chunks."""
    from funasr.models.ct_transformer.utils import split_to_mini_sentence, split_words

    tokenizer = preprocessor.tokenizer
    tokens = split_words(text, jieba_usr_dict=preprocessor.jieba_usr_dict)
    token_ids = tokenizer.encode(tokens)
    mini_sentences = split_to_mini_sentence(tokens, split_size)
    mini_sentences_id = split_to_mini_sentence(token_ids, split_size)
    return mini_sentences, mini_sentences_id


def pad_text_ids(
    token_ids: Sequence[int],
    fixed_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad token IDs to the HMM's fixed input sequence length."""
    token_ids = np.asarray(token_ids, dtype=np.int32)
    if token_ids.shape[0] > fixed_length:
        raise ValueError(
            f"Token length {token_ids.shape[0]} exceeds fixed_length "
            f"{fixed_length}"
        )
    padded = np.zeros((1, fixed_length), dtype=np.int32)
    padded[0, : token_ids.shape[0]] = token_ids
    lengths = np.array([token_ids.shape[0]], dtype=np.int32)
    return padded, lengths


def run_punc_inference_hmm(
    preprocessor: _LocalPreprocessor,
    text: str,
    split_size: int,
    fixed_length: int,
    module: Any,
) -> str:
    """Restore punctuation for one text string using the compiled HMM."""
    mini_sentences, mini_sentences_id = tokenize_to_chunks(
        preprocessor,
        text,
        split_size,
    )
    punc_list = preprocessor.punc_list
    sentence_end_id = preprocessor.sentence_end_id

    cache_sent: list[str] = []
    cache_sent_id = np.array([], dtype=np.int32)
    new_mini_sentence = ""
    cache_pop_trigger_limit = 200

    for mini_sentence_i, sentence_tokens in enumerate(mini_sentences):
        mini_sentence = list(cache_sent) + list(sentence_tokens)
        mini_sentence_id = np.concatenate(
            (cache_sent_id, mini_sentences_id[mini_sentence_i]),
            axis=0,
        )

        padded_text, text_lengths = pad_text_ids(mini_sentence_id, fixed_length)
        model_text_lengths = np.array([fixed_length], dtype=np.int32)

        in_name_0 = module.get_input_name(0)
        input_dtype = module.get_input_info(in_name_0).dtype
        module.set_input(in_name_0, padded_text.astype(input_dtype))
        if module.get_num_inputs() > 1:
            in_name_1 = module.get_input_name(1)
            length_dtype = module.get_input_info(in_name_1).dtype
            module.set_input(
                in_name_1,
                model_text_lengths.astype(length_dtype),
            )

        module.run()
        module.sync()

        out_name = module.get_output_name(0)
        logits_np = module.get_output(out_name).numpy()
        # logits is numpy array: [1, fixed_length, num_puncs]

        # Valid length masking
        valid_logits = logits_np[:, : text_lengths[0], :]
        punctuations = np.argmax(valid_logits, axis=-1).reshape(-1)

        if mini_sentence_i < len(mini_sentences) - 1:
            sentence_end = -1
            last_comma_index = -1
            for index in range(len(punctuations) - 2, 1, -1):
                if punc_list[int(punctuations[index])] in ["。", "？"]:
                    sentence_end = index
                    break
                if last_comma_index < 0 and punc_list[int(punctuations[index])] == "，":
                    last_comma_index = index

            if sentence_end < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                sentence_end = last_comma_index
                punctuations[sentence_end] = sentence_end_id

            cache_sent = mini_sentence[sentence_end + 1 :]
            cache_sent_id = mini_sentence_id[sentence_end + 1 :]
            mini_sentence = mini_sentence[0 : sentence_end + 1]
            punctuations = punctuations[0 : sentence_end + 1]

        words_with_punc: list[str] = []
        for index in range(len(mini_sentence)):
            if (
                index == 0
                or punc_list[int(punctuations[index - 1])] == "。"
                or punc_list[int(punctuations[index - 1])] == "？"
            ) and len(mini_sentence[index][0].encode()) == 1:
                mini_sentence[index] = mini_sentence[index].capitalize()
            if index == 0 and len(mini_sentence[index][0].encode()) == 1:
                mini_sentence[index] = " " + mini_sentence[index]
            if index > 0 and len(mini_sentence[index][0].encode()) == 1 and len(mini_sentence[index - 1][0].encode()) == 1:
                mini_sentence[index] = " " + mini_sentence[index]
            words_with_punc.append(mini_sentence[index])
            if punc_list[int(punctuations[index])] != "_":
                punc_res = punc_list[int(punctuations[index])]
                if len(mini_sentence[index][0].encode()) == 1:
                    if punc_res == "，":
                        punc_res = ","
                    elif punc_res == "。":
                        punc_res = "."
                    elif punc_res == "？":
                        punc_res = "?"
                words_with_punc.append(punc_res)
        new_mini_sentence += "".join(words_with_punc)

        if mini_sentence_i == len(mini_sentences) - 1 and new_mini_sentence:
            if new_mini_sentence[-1] in ["，", "、"]:
                new_mini_sentence = new_mini_sentence[:-1] + "。"
            elif new_mini_sentence[-1] == ",":
                new_mini_sentence = new_mini_sentence[:-1] + "."
            elif new_mini_sentence[-1] not in ["。", "？"] and len(new_mini_sentence[-1].encode()) != 1:
                new_mini_sentence = new_mini_sentence + "。"
            elif new_mini_sentence[-1] not in [".", "?"] and len(new_mini_sentence[-1].encode()) == 1:
                new_mini_sentence = new_mini_sentence + "."

    return new_mini_sentence


def get_args() -> argparse.Namespace:
    """Parse command-line arguments and resolve config-based defaults."""
    # fmt:off
    parser = argparse.ArgumentParser(description="Offline CT-Transformer punctuation inference demo", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="path to config.yaml")
    parser.add_argument("--model_dir", type=Path, default=None, help="local preprocessing metadata directory containing tokens.json; PyTorch weights are not required")
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="input text file")
    parser.add_argument("--split_size", type=int, default=20, help="number of tokens in each mini sentence")
    parser.add_argument("--hmm_dir", type=Path, default=None, help="directory containing the compiled HMM model")
    parser.add_argument("--model_path", type=Path, default=None, help="explicit compiled HMM model path")
    args = parser.parse_args()

    default_model_size, default_model_name, _ = get_model_configs(args.config)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    args.model_dir = first_not_none(args.model_dir, get_default_model_dir())
    args.hmm_dir = first_not_none(args.hmm_dir, DEFAULT_HMM_DIR)
    # fmt:on
    return args


def resolve_hmm_path(
    model_name: str,
    hmm_dir: Path,
    model_path: Path | None = None,
) -> Path:
    """Resolve an explicitly provided HMM path or find it under ``hmm_dir``."""
    if model_path is not None:
        if model_path.is_file():
            return model_path
        raise FileNotFoundError(f"HMM model does not exist: {model_path}")

    direct_path = hmm_dir / f"{model_name}.hmm"
    if direct_path.is_file():
        return direct_path

    if not hmm_dir.is_dir():
        raise FileNotFoundError(f"HMM directory does not exist: {hmm_dir}")

    for subdir in hmm_dir.iterdir():
        if subdir.is_dir():
            candidate = subdir / f"{model_name}.hmm"
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"Cannot find {model_name}.hmm in {hmm_dir} or its direct subdirectories"
    )


def load_hmm_module(model_path: Path) -> tuple[Any, int]:
    """Load the HMM module and return its fixed input sequence length."""
    import tcim_lite

    module = tcim_lite.runtime.load(str(model_path))
    input_name = module.get_input_name(0)
    fixed_length = module.get_input_info(input_name).shape[1]
    return module, fixed_length


def main(args: argparse.Namespace) -> int:
    """Run offline CT-Transformer punctuation inference."""

    t_load0 = time.time()
    try:
        # Match the legacy timing boundary: the first FunASR import belongs to
        # model/tokenizer loading, not to the first inference request.
        importlib.import_module("funasr.models.ct_transformer.utils")
        preprocessor = _LocalPreprocessor(args.model_dir)
    except (FileNotFoundError, ImportError, ValueError, json.JSONDecodeError) as exc:
        logger.error(f"Cannot load local preprocessing metadata: {exc}")
        logger.error(
            "Original PyTorch model weights are not required, but tokens.json "
            "is required."
        )
        return 1

    try:
        model_path = resolve_hmm_path(
            args.model_name,
            args.hmm_dir,
            args.model_path,
        )
        module, fixed_length = load_hmm_module(model_path)
        logger.info(f"Successfully loaded HMM: {model_path}")
        logger.info(
            "Determined static fixed sequence length from HMM model input "
            f"shape: {fixed_length}"
        )

    except Exception as exc:
        logger.error(
            "Cannot initialize TCIM module. Ensure you are on NPU "
            f"environment: {exc}"
        )
        return 1

    t_load1 = time.time()
    try:
        input_path = _resolve_input(args.input, args.model_dir)
        text_items = load_text_items(input_path)
    except (OSError, ValueError) as exc:
        logger.error(f"Cannot load input text: {exc}")
        return 1

    results: list[dict[str, str]] = []
    t_inf_total = 0.0
    for idx, text in enumerate(text_items):
        logger.info(f"Processing sequence [{idx}]")
        t_start = time.time()
        final_text = run_punc_inference_hmm(
            preprocessor,
            text,
            args.split_size,
            fixed_length,
            module,
        )
        t_end = time.time()
        t_inf_total += t_end - t_start
        results.append({"text": final_text})

    logger.success(f"Inference results: {results}")
    logger.success("=" * 100)
    logger.success("                    Model Inference Performance Summary Report")
    logger.success("=" * 100)
    logger.success("Performance Details:")
    logger.success(
        f"  Load Model+Tokenizer   : {(t_load1 - t_load0) * 1000:>7.2f} ms"
    )
    logger.success(
        f"  Total Inference ({len(text_items):<2} seqs) : "
        f"{t_inf_total * 1000:>7.2f} ms"
    )
    if text_items:
        average_time_ms = (t_inf_total * 1000) / len(text_items)
        logger.success(f"  Average Inference/seq  : {average_time_ms:>7.2f} ms")
    logger.success("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(get_args()))
