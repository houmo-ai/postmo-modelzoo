# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   emotion2vec HMM inference demo.
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

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

import numpy as np
import torchaudio
from loguru import logger

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emotion2vec HMM emotion recognition and embedding extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "emotion2vec-plus_large.hmm"),
        help="HMM model path",
    )
    parser.add_argument(
        "--quant-embedding",
        dest="quant_embedding",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="classification weights used for the overall audio result",
    )
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="emotion2vec_plus_large",
        help="directory containing tokens.txt",
    )
    parser.add_argument(
        "--audio",
        dest="audio",
        type=str,
        nargs="+",
        default=[os.path.join("..", "..", "..", "data", "audio", "audio.mp3")],
        help="one or more local audio files",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="results",
        help="model output and summary directory",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=3,
        help="top emotion count",
    )
    parser.add_argument(
        "--chunk_size",
        dest="chunk_size",
        type=float,
        default=16.0,
        help="audio chunk size in seconds",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be greater than 0")
    return args


def load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    waveform, source_rate = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        raise ValueError(f"Audio is empty: {audio_path}")
    waveform = waveform.mean(dim=0)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.numpy().astype(np.float32, copy=False)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    mean = np.mean(waveform, dtype=np.float32)
    variance = np.mean(np.square(waveform - mean, dtype=np.float32), dtype=np.float32)
    return (waveform - mean) / np.sqrt(variance + np.float32(1e-5))


def valid_frame_count(valid_samples: int) -> int:
    length = valid_samples
    conv_kernel_strides = (
        (10, 5),
        (3, 2),
        (3, 2),
        (3, 2),
        (3, 2),
        (2, 2),
        (2, 2),
    )
    for kernel_size, stride in conv_kernel_strides:
        length = (length - kernel_size) // stride + 1
    return max(length, 0)


def output_key(audio_path: str, used_keys: set[str]) -> str:
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename) or "audio"
    key = base
    suffix = 2
    while key in used_keys:
        key = f"{base}_{suffix}"
        suffix += 1
    used_keys.add(key)
    return key


def load_labels(model_dir: str) -> list[str]:
    tokens_path = os.path.join(model_dir, "tokens.txt")
    if not os.path.isfile(tokens_path):
        raise FileNotFoundError(tokens_path)
    with open(tokens_path, "r", encoding="utf-8") as token_file:
        labels = [line.strip() for line in token_file if line.strip()]
    if not labels:
        raise ValueError(f"No emotion labels found in {tokens_path}")
    return labels


def format_classification(probabilities: np.ndarray, labels: list[str], top_k: int) -> dict[str, Any]:
    scores = probabilities.astype(np.float32, copy=False).reshape(-1)
    if scores.shape != (len(labels),):
        raise ValueError(f"Model probabilities and labels do not match: {scores.shape} != ({len(labels)},)")
    ranking = np.argsort(scores)[::-1]
    return {
        "predicted_index": int(ranking[0]),
        "predicted_label": labels[int(ranking[0])],
        "labels": labels,
        "scores": [float(score) for score in scores],
        "top_k": [
            {
                "index": int(index),
                "label": labels[int(index)],
                "score": float(scores[index]),
            }
            for index in ranking[: min(top_k, len(labels))]
        ],
    }


class EmotionClassifier:
    def __init__(self, quant_embedding_path: str, num_labels: int, feature_dim: int):
        if not os.path.isfile(quant_embedding_path):
            raise FileNotFoundError(quant_embedding_path)
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load quant_embedding.pt") from exc

        state_dict = torch.load(quant_embedding_path, map_location="cpu", weights_only=False)
        if "weight" not in state_dict or "bias" not in state_dict:
            raise ValueError(f"Unexpected classification weights: {list(state_dict)}")
        self.weight = state_dict["weight"].detach().float().cpu().numpy()
        self.bias = state_dict["bias"].detach().float().cpu().numpy()
        if self.weight.shape != (num_labels, feature_dim):
            raise ValueError(
                f"Unexpected classification weight shape: {self.weight.shape} != " f"({num_labels}, {feature_dim})"
            )
        if self.bias.shape != (num_labels,):
            raise ValueError(f"Unexpected classification bias shape: {self.bias.shape} != ({num_labels},)")

    def __call__(self, utterance_features: np.ndarray) -> np.ndarray:
        features = utterance_features.astype(np.float32, copy=False)
        if features.ndim != 2 or features.shape[1] != self.weight.shape[1]:
            raise ValueError(f"Unexpected utterance features: {features.shape}")
        utterance_feature = features.mean(axis=0)
        logits = utterance_feature @ self.weight.T + self.bias
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        return probabilities / np.sum(probabilities)


class Emotion2VecHMM:
    INPUT_WAVEFORM = "waveform"
    INPUT_VALID_FRAMES = "valid_frames"
    OUTPUT_FEATURES = "frame_features"
    OUTPUT_MASK = "frame_padding_mask"
    OUTPUT_UTTERANCE = "utterance_feature"
    OUTPUT_PROBABILITIES = "probabilities"

    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        try:
            import tcim_lite
        except ImportError as exc:
            raise ImportError("tcim_lite is required for HMM inference") from exc

        self.model_path = model_path
        self.module = tcim_lite.runtime.load(str(model_path))
        self.input_infos = self._collect_infos("input")
        self.output_infos = self._collect_infos("output")
        self._validate_interface()
        self.window_samples = int(self.input_infos[self.INPUT_WAVEFORM].shape[1])

    def _collect_infos(self, kind: str) -> dict[str, Any]:
        count = getattr(self.module, f"get_num_{kind}s")()
        get_name = getattr(self.module, f"get_{kind}_name")
        get_info = getattr(self.module, f"get_{kind}_info")
        return {get_name(index): get_info(get_name(index)) for index in range(count)}

    def _validate_interface(self) -> None:
        required_inputs = {self.INPUT_WAVEFORM, self.INPUT_VALID_FRAMES}
        required_outputs = {
            self.OUTPUT_FEATURES,
            self.OUTPUT_MASK,
            self.OUTPUT_UTTERANCE,
            self.OUTPUT_PROBABILITIES,
        }
        if not required_inputs.issubset(self.input_infos):
            raise ValueError(f"Unexpected HMM inputs: {sorted(self.input_infos)}")
        if not required_outputs.issubset(self.output_infos):
            raise ValueError(f"Unexpected HMM outputs: {sorted(self.output_infos)}")

        waveform_shape = tuple(self.input_infos[self.INPUT_WAVEFORM].shape)
        feature_shape = tuple(self.output_infos[self.OUTPUT_FEATURES].shape)
        mask_shape = tuple(self.output_infos[self.OUTPUT_MASK].shape)
        utterance_shape = tuple(self.output_infos[self.OUTPUT_UTTERANCE].shape)
        probabilities_shape = tuple(self.output_infos[self.OUTPUT_PROBABILITIES].shape)
        if len(waveform_shape) != 2 or waveform_shape[0] != 1:
            raise ValueError(f"Unexpected waveform shape: {waveform_shape}")
        if len(feature_shape) != 3 or feature_shape[0] != 1 or feature_shape[-1] <= 0:
            raise ValueError(f"Unexpected frame feature shape: {feature_shape}")
        if mask_shape != feature_shape[:2]:
            raise ValueError(f"Unexpected frame padding mask shape: {mask_shape}")
        if utterance_shape != (1, feature_shape[-1]):
            raise ValueError(f"Unexpected utterance feature shape: {utterance_shape}")
        if len(probabilities_shape) != 2 or probabilities_shape[0] != 1:
            raise ValueError(f"Unexpected probabilities shape: {probabilities_shape}")
        self.feature_dim = int(feature_shape[-1])
        self.num_labels = int(probabilities_shape[-1])

    def _run_chunk(self, waveform: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        valid_samples = int(waveform.size)
        padded = np.zeros(self.window_samples, dtype=np.float32)
        padded[:valid_samples] = normalize_waveform(waveform)

        waveform_info = self.input_infos[self.INPUT_WAVEFORM]
        frame_info = self.input_infos[self.INPUT_VALID_FRAMES]
        valid_frames = min(valid_frame_count(valid_samples), self.output_infos[self.OUTPUT_FEATURES].shape[1])
        self.module.set_input(self.INPUT_WAVEFORM, padded[None].astype(waveform_info.dtype))
        self.module.set_input(
            self.INPUT_VALID_FRAMES,
            np.asarray([valid_frames]).astype(frame_info.dtype),
        )

        start = time.perf_counter()
        self.module.run()
        self.module.sync()
        elapsed = time.perf_counter() - start

        features = self.module.get_output(self.OUTPUT_FEATURES).numpy()
        mask = self.module.get_output(self.OUTPUT_MASK).numpy().astype(bool)
        utterance = self.module.get_output(self.OUTPUT_UTTERANCE).numpy()
        probabilities = self.module.get_output(self.OUTPUT_PROBABILITIES).numpy()
        if features.ndim != 3 or mask.ndim != 2 or features.shape[:2] != mask.shape:
            raise ValueError(f"Unexpected frame outputs: {features.shape}, {mask.shape}")
        if utterance.shape != (1, self.feature_dim):
            raise ValueError(f"Unexpected utterance feature: {utterance.shape}")
        if probabilities.shape != (1, self.num_labels):
            raise ValueError(f"Unexpected probabilities: {probabilities.shape}")

        if valid_samples < self.window_samples and not mask.any():
            valid_features = features[0, :valid_frames]
        else:
            valid_features = features[0, ~mask[0]]
        return {
            "frame_features": valid_features,
            "utterance_feature": utterance[0],
            "probabilities": probabilities[0],
        }, elapsed

    def __call__(self, waveform: np.ndarray, chunk_samples: int) -> dict[str, Any]:
        if chunk_samples > self.window_samples:
            raise ValueError(
                f"chunk size cannot exceed the HMM input window: " f"{chunk_samples} > {self.window_samples} samples"
            )
        chunks = []
        infer_seconds = 0.0
        chunk_count = (waveform.size + chunk_samples - 1) // chunk_samples
        for chunk_index, start in enumerate(range(0, waveform.size, chunk_samples), start=1):
            chunk = waveform[start : start + chunk_samples]
            logger.info(f"Processing Chunk {chunk_index}/{chunk_count} " f"({chunk.size / 16000:.3f}s)")
            outputs, elapsed = self._run_chunk(chunk)
            outputs["sample_count"] = int(chunk.size)
            chunks.append(outputs)
            infer_seconds += elapsed

        frame_features = np.concatenate([chunk["frame_features"] for chunk in chunks], axis=0).astype(
            np.float16, copy=False
        )
        if frame_features.ndim != 2 or frame_features.shape[1] != self.feature_dim:
            raise ValueError(f"Unexpected frame features: {frame_features.shape}")
        utterance_features = np.stack([chunk["utterance_feature"] for chunk in chunks], axis=0).astype(
            np.float16, copy=False
        )
        return {
            "frame_features": frame_features,
            "utterance_features": utterance_features,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "chunk_samples": chunk_samples,
            "audio_duration_seconds": waveform.size / 16000,
            "infer_seconds": infer_seconds,
        }


def save_result(
    output_dir: str,
    key: str,
    audio_path: str,
    model_result: dict[str, Any],
    overall_probabilities: np.ndarray,
    labels: list[str],
    top_k: int,
    total_seconds: float,
) -> dict[str, Any]:
    frame_dir = os.path.join(output_dir, "frame")
    utterance_dir = os.path.join(output_dir, "utterance")
    probability_dir = os.path.join(output_dir, "probabilities")
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(utterance_dir, exist_ok=True)
    os.makedirs(probability_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    frame_chunks = []
    utterance_chunks = []
    probability_chunks = []
    chunks = []
    start_sample = 0
    for chunk_index, chunk in enumerate(model_result["chunks"], start=1):
        chunk_name = f"{key}.chunk_{chunk_index:04d}.npy"
        frame_path = os.path.join(frame_dir, chunk_name)
        utterance_path = os.path.join(utterance_dir, chunk_name)
        probability_path = os.path.join(probability_dir, chunk_name)
        frame_features = chunk["frame_features"].astype(np.float16, copy=False)
        utterance_feature = chunk["utterance_feature"].astype(np.float16, copy=False)
        probabilities = chunk["probabilities"].astype(np.float16, copy=False)
        np.save(frame_path, frame_features)
        np.save(utterance_path, utterance_feature)
        np.save(probability_path, probabilities)
        frame_chunks.append(
            {
                "chunk": chunk_index,
                "shape": list(frame_features.shape),
                "dtype": str(frame_features.dtype),
                "path": str(frame_path),
            }
        )
        utterance_chunks.append(
            {
                "chunk": chunk_index,
                "shape": list(utterance_feature.shape),
                "dtype": str(utterance_feature.dtype),
                "path": str(utterance_path),
            }
        )
        probability_chunks.append(
            {
                "chunk": chunk_index,
                "shape": list(probabilities.shape),
                "dtype": str(probabilities.dtype),
                "path": str(probability_path),
            }
        )
        sample_count = chunk["sample_count"]
        chunk_summary_path = os.path.join(summary_dir, f"{key}.chunk_{chunk_index:04d}.json")
        chunk_result = {
            "audio": str(audio_path),
            "chunk": chunk_index,
            "start_seconds": start_sample / 16000,
            "end_seconds": (start_sample + sample_count) / 16000,
            "duration_seconds": sample_count / 16000,
            "classification": format_classification(probabilities, labels, top_k),
            "probabilities": probability_chunks[-1],
            "utterance_feature": utterance_chunks[-1],
            "frame_features": frame_chunks[-1],
            "summary": str(chunk_summary_path),
        }
        with open(chunk_summary_path, "w", encoding="utf-8") as chunk_summary_file:
            json.dump(chunk_result, chunk_summary_file, ensure_ascii=False, indent=2)
        chunks.append(chunk_result)
        start_sample += sample_count

    return {
        "audio": str(audio_path),
        "classification": format_classification(overall_probabilities, labels, top_k),
        "chunks": chunks,
        "frame_features": {
            "shape": list(model_result["frame_features"].shape),
            "dtype": str(model_result["frame_features"].dtype),
            "chunks": frame_chunks,
        },
        "utterance_features": {"chunks": utterance_chunks},
        "probabilities": {"chunks": probability_chunks},
        "runtime": {
            "audio_duration_seconds": model_result["audio_duration_seconds"],
            "chunk_count": model_result["chunk_count"],
            "chunk_size_seconds": model_result["chunk_samples"] / 16000,
            "valid_frame_count": int(model_result["frame_features"].shape[0]),
            "hmm_seconds": model_result["infer_seconds"],
            "average_chunk_seconds": model_result["infer_seconds"] / model_result["chunk_count"],
            "total_seconds": total_seconds,
            "hmm_rtf": model_result["infer_seconds"] / model_result["audio_duration_seconds"],
            "e2e_rtf": total_seconds / model_result["audio_duration_seconds"],
        },
    }


def print_classification(classification: dict[str, Any]) -> None:
    logger.info(f"Predicted emotion: {classification['predicted_label']}")
    logger.info(f"Top-{len(classification['top_k'])} emotions:")
    for rank, item in enumerate(classification["top_k"], start=1):
        logger.info(f"  {rank}. [{item['index']}] {item['label']}: {item['score']:.6f}")


def print_result(key: str, result: dict[str, Any], load_seconds: float) -> None:
    logger.info(f"[{key}] {result['audio']}")
    logger.info("Overall classification (concatenated utterance features):")
    print_classification(result["classification"])
    logger.info(
        f"Frame features: {result['frame_features']['shape']}, "
        f"saved {len(result['frame_features']['chunks'])} chunks -> "
        f"{os.path.dirname(result['frame_features']['chunks'][0]['path'])}"
    )
    runtime = result["runtime"]
    logger.success(f"Audio Duration: {runtime['audio_duration_seconds']:.3f} seconds")
    logger.success(
        f"Chunks: {runtime['chunk_count']} " f"(configured chunk size: {runtime['chunk_size_seconds']:.3f} seconds)"
    )
    logger.success(f"Model Load: {load_seconds * 1000:.3f} ms")
    logger.success(f"HMM Inference: {runtime['hmm_seconds'] * 1000:.3f} ms")
    logger.success(f"Average HMM Cost/Chunk: {runtime['average_chunk_seconds'] * 1000:.3f} ms")
    logger.success(f"E2E Latency: {runtime['total_seconds']:.3f} seconds")
    logger.success(f"HMM RTF (Real-Time Factor): {runtime['hmm_rtf']:.3f}")
    logger.success(f"E2E RTF (Real-Time Factor): {runtime['e2e_rtf']:.3f}")


def main() -> None:
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    load_start = time.perf_counter()
    model = Emotion2VecHMM(args.model_path)
    labels = load_labels(args.tokenizer_dir)
    if len(labels) != model.num_labels:
        raise ValueError(f"Model probabilities and labels do not match: {model.num_labels} != {len(labels)}")
    classifier = EmotionClassifier(args.quant_embedding, model.num_labels, model.feature_dim)
    chunk_samples = int(round(args.chunk_size * 16000))
    if chunk_samples < 1:
        raise ValueError("--chunk_size is too small to contain one audio sample")
    if chunk_samples > model.window_samples:
        parser_window_seconds = model.window_samples / 16000
        raise ValueError(f"--chunk_size cannot exceed the HMM input window of " f"{parser_window_seconds:g} seconds")
    load_seconds = time.perf_counter() - load_start

    summary = {
        "model": str(args.model_path),
        "quant_embedding": str(args.quant_embedding),
        "chunk_size_seconds": chunk_samples / 16000,
        "labels": labels,
        "results": {},
        "runtime": {"load_seconds": load_seconds},
    }
    used_keys: set[str] = set()
    for audio_path in args.audio:
        key = output_key(audio_path, used_keys)
        start = time.perf_counter()
        waveform = load_audio(audio_path)
        model_result = model(waveform, chunk_samples)
        overall_probabilities = classifier(model_result["utterance_features"])
        total_seconds = time.perf_counter() - start
        result = save_result(
            args.output_dir,
            key,
            audio_path,
            model_result,
            overall_probabilities,
            labels,
            args.top_k,
            total_seconds,
        )
        summary["results"][key] = result
        print_result(key, result, load_seconds)

    summary_path = os.path.join(args.output_dir, "Summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    logger.success(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
