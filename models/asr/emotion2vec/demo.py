#!/usr/bin/env python3
"""Run emotion2vec HMM inference and emotion classification."""

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
        "--hmm",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "emotion2vec-plus_large.hmm"),
        help="HMM model path",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "classifier.npz"),
        help="classifier.npz exported by ptq.py",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="emotion2vec_plus_large",
        help="fallback directory containing model.pt and tokens.txt",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        default=[os.path.join("..", "..", "..", "data", "audio", "audio.mp3")],
        help="one or more local audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="embedding and summary output directory",
    )
    parser.add_argument("--top-k", type=int, default=3, help="top emotion count")
    parser.add_argument(
        "--chunk_size",
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


class EmotionClassifier:
    def __init__(self, classifier_path: str, model_dir: str):
        self.weight, self.bias, self.labels, self.source = self._load(classifier_path, model_dir)
        if self.weight.ndim != 2 or self.weight.shape[0] != len(self.labels):
            raise ValueError(f"Unexpected classifier weight shape: {self.weight.shape}")
        if self.bias.shape != (len(self.labels),):
            raise ValueError(f"Unexpected classifier bias shape: {self.bias.shape}")

    @property
    def feature_dim(self) -> int:
        return int(self.weight.shape[1])

    @staticmethod
    def _load(classifier_path: str, model_dir: str) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        if os.path.isfile(classifier_path):
            with np.load(classifier_path, allow_pickle=False) as classifier:
                weight = classifier["weight"].astype(np.float32, copy=False)
                bias = classifier["bias"].astype(np.float32, copy=False)
                labels = [str(label) for label in classifier["labels"].tolist()]
            return weight, bias, labels, classifier_path

        checkpoint_path = os.path.join(model_dir, "model.pt")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        if not os.path.isfile(checkpoint_path) or not os.path.isfile(tokens_path):
            raise FileNotFoundError(f"Neither {classifier_path} nor the fallback classifier files exist")

        import torch

        print(f"Warning: {classifier_path} not found; loading {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        weight = state_dict["proj.weight"].detach().float().cpu().numpy()
        bias = state_dict["proj.bias"].detach().float().cpu().numpy()
        with open(tokens_path, "r", encoding="utf-8") as token_file:
            labels = [line.strip() for line in token_file if line.strip()]
        return weight, bias, labels, checkpoint_path

    def __call__(self, embedding: np.ndarray, top_k: int) -> dict[str, Any]:
        logits = embedding @ self.weight.T + self.bias
        logits -= np.max(logits)
        scores = np.exp(logits)
        scores /= np.sum(scores)
        ranking = np.argsort(scores)[::-1]
        return {
            "predicted_label": self.labels[int(ranking[0])],
            "labels": self.labels,
            "scores": [float(score) for score in scores],
            "top_k": [
                {"label": self.labels[int(index)], "score": float(scores[index])}
                for index in ranking[: min(top_k, len(self.labels))]
            ],
        }


class Emotion2VecHMM:
    INPUT_WAVEFORM = "waveform"
    INPUT_VALID_SAMPLES = "valid_samples"
    OUTPUT_FEATURES = "frame_features"
    OUTPUT_MASK = "frame_padding_mask"

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
        required_inputs = {self.INPUT_WAVEFORM, self.INPUT_VALID_SAMPLES}
        required_outputs = {self.OUTPUT_FEATURES, self.OUTPUT_MASK}
        if not required_inputs.issubset(self.input_infos):
            raise ValueError(f"Unexpected HMM inputs: {sorted(self.input_infos)}")
        if not required_outputs.issubset(self.output_infos):
            raise ValueError(f"Unexpected HMM outputs: {sorted(self.output_infos)}")

        waveform_shape = tuple(self.input_infos[self.INPUT_WAVEFORM].shape)
        feature_shape = tuple(self.output_infos[self.OUTPUT_FEATURES].shape)
        if len(waveform_shape) != 2 or waveform_shape[0] != 1:
            raise ValueError(f"Unexpected waveform shape: {waveform_shape}")
        if len(feature_shape) != 3 or feature_shape[-1] <= 0:
            raise ValueError(f"Unexpected feature shape: {feature_shape}")
        self.feature_dim = int(feature_shape[-1])

    def _run_chunk(self, waveform: np.ndarray) -> tuple[np.ndarray, float]:
        valid_samples = int(waveform.size)
        padded = np.zeros(self.window_samples, dtype=np.float32)
        padded[:valid_samples] = normalize_waveform(waveform)

        waveform_info = self.input_infos[self.INPUT_WAVEFORM]
        length_info = self.input_infos[self.INPUT_VALID_SAMPLES]
        self.module.set_input(self.INPUT_WAVEFORM, padded[None].astype(waveform_info.dtype))
        self.module.set_input(
            self.INPUT_VALID_SAMPLES,
            np.asarray([valid_samples]).astype(length_info.dtype),
        )

        start = time.perf_counter()
        self.module.run()
        self.module.sync()
        elapsed = time.perf_counter() - start

        features = self.module.get_output(self.OUTPUT_FEATURES).numpy()
        mask = self.module.get_output(self.OUTPUT_MASK).numpy().astype(bool)
        if features.ndim != 3 or mask.ndim != 2 or features.shape[:2] != mask.shape:
            raise ValueError(f"Unexpected HMM outputs: {features.shape}, {mask.shape}")

        if valid_samples < self.window_samples and not mask.any():
            frame_count = min(valid_frame_count(valid_samples), features.shape[1])
            return features[0, :frame_count], elapsed
        return features[0, ~mask[0]], elapsed

    def __call__(self, waveform: np.ndarray, chunk_samples: int) -> dict[str, Any]:
        if chunk_samples > self.window_samples:
            raise ValueError(
                f"chunk size cannot exceed the HMM input window: "
                f"{chunk_samples} > {self.window_samples} samples"
            )
        frame_chunks = []
        utterance_chunks = []
        chunk_sample_counts = []
        infer_seconds = 0.0
        chunk_count = (waveform.size + chunk_samples - 1) // chunk_samples
        for chunk_index, start in enumerate(range(0, waveform.size, chunk_samples), start=1):
            chunk = waveform[start : start + chunk_samples]
            logger.info(
                f"Processing Chunk {chunk_index}/{chunk_count} "
                f"({chunk.size / 16000:.3f}s)"
            )
            features, elapsed = self._run_chunk(chunk)
            frame_chunks.append(features)
            utterance_chunks.append(features.astype(np.float32).mean(axis=0))
            chunk_sample_counts.append(int(chunk.size))
            infer_seconds += elapsed

        frame_embedding = np.concatenate(frame_chunks, axis=0).astype(np.float16, copy=False)
        if frame_embedding.ndim != 2 or frame_embedding.shape[1] != self.feature_dim:
            raise ValueError(f"Unexpected frame embedding: {frame_embedding.shape}")
        return {
            "frame_embedding": frame_embedding,
            "utterance_embedding": frame_embedding.astype(np.float32).mean(axis=0),
            "frame_chunks": frame_chunks,
            "utterance_chunks": utterance_chunks,
            "chunk_sample_counts": chunk_sample_counts,
            "chunk_count": len(frame_chunks),
            "chunk_samples": chunk_samples,
            "audio_duration_seconds": waveform.size / 16000,
            "infer_seconds": infer_seconds,
        }


def save_result(
    output_dir: str,
    key: str,
    audio_path: str,
    model_result: dict[str, Any],
    classification: dict[str, Any],
    chunk_classifications: list[dict[str, Any]],
    total_seconds: float,
) -> dict[str, Any]:
    frame_dir = os.path.join(output_dir, "frame")
    utterance_dir = os.path.join(output_dir, "utterance")
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(utterance_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    frame_chunks = []
    utterance_chunks = []
    chunks = []
    start_sample = 0
    for chunk_index, (frame_embedding, utterance_embedding, sample_count, chunk_classification) in enumerate(
        zip(
            model_result["frame_chunks"],
            model_result["utterance_chunks"],
            model_result["chunk_sample_counts"],
            chunk_classifications,
        ),
        start=1,
    ):
        chunk_name = f"{key}.chunk_{chunk_index:04d}.npy"
        frame_path = os.path.join(frame_dir, chunk_name)
        utterance_path = os.path.join(utterance_dir, chunk_name)
        np.save(frame_path, frame_embedding.astype(np.float16, copy=False))
        np.save(utterance_path, utterance_embedding)
        frame_chunks.append(
            {
                "chunk": chunk_index,
                "shape": list(frame_embedding.shape),
                "dtype": "float16",
                "path": str(frame_path),
            }
        )
        utterance_chunks.append(
            {
                "chunk": chunk_index,
                "shape": list(utterance_embedding.shape),
                "dtype": str(utterance_embedding.dtype),
                "path": str(utterance_path),
            }
        )
        chunk_summary_path = os.path.join(summary_dir, f"{key}.chunk_{chunk_index:04d}.json")
        chunk_result = {
            "audio": str(audio_path),
            "chunk": chunk_index,
            "start_seconds": start_sample / 16000,
            "end_seconds": (start_sample + sample_count) / 16000,
            "duration_seconds": sample_count / 16000,
            "classification": chunk_classification,
            "utterance_embedding": utterance_chunks[-1],
            "frame_embedding": frame_chunks[-1],
            "summary": str(chunk_summary_path),
        }
        with open(chunk_summary_path, "w", encoding="utf-8") as chunk_summary_file:
            json.dump(chunk_result, chunk_summary_file, ensure_ascii=False, indent=2)
        chunks.append(chunk_result)
        start_sample += sample_count

    return {
        "audio": str(audio_path),
        "classification": classification,
        "chunks": chunks,
        "utterance_embedding": {
            "shape": list(model_result["utterance_embedding"].shape),
            "dtype": str(model_result["utterance_embedding"].dtype),
            "chunks": utterance_chunks,
        },
        "frame_embedding": {
            "shape": list(model_result["frame_embedding"].shape),
            "dtype": str(model_result["frame_embedding"].dtype),
            "chunks": frame_chunks,
        },
        "runtime": {
            "audio_duration_seconds": model_result["audio_duration_seconds"],
            "chunk_count": model_result["chunk_count"],
            "chunk_size_seconds": model_result["chunk_samples"] / 16000,
            "valid_frame_count": int(model_result["frame_embedding"].shape[0]),
            "hmm_seconds": model_result["infer_seconds"],
            "average_chunk_seconds": model_result["infer_seconds"] / model_result["chunk_count"],
            "total_seconds": total_seconds,
            "hmm_rtf": model_result["infer_seconds"] / model_result["audio_duration_seconds"],
            "e2e_rtf": total_seconds / model_result["audio_duration_seconds"],
        },
    }


def print_result(key: str, result: dict[str, Any], load_seconds: float) -> None:
    logger.info(f"[{key}] {result['audio']}")
    print_classification(result["classification"])
    logger.info(
        f"Utterance embedding: {result['utterance_embedding']['shape']}, "
        f"saved {len(result['utterance_embedding']['chunks'])} chunks -> "
        f"{os.path.dirname(result['utterance_embedding']['chunks'][0]['path'])}"
    )
    logger.info(
        f"Frame embedding: {result['frame_embedding']['shape']}, "
        f"saved {len(result['frame_embedding']['chunks'])} chunks -> "
        f"{os.path.dirname(result['frame_embedding']['chunks'][0]['path'])}"
    )
    runtime = result["runtime"]
    logger.success(f"Audio Duration: {runtime['audio_duration_seconds']:.3f} seconds")
    logger.success(
        f"Chunks: {runtime['chunk_count']} "
        f"(configured chunk size: {runtime['chunk_size_seconds']:.3f} seconds)"
    )
    logger.success(f"Model Load: {load_seconds * 1000:.3f} ms")
    logger.success(f"HMM Inference: {runtime['hmm_seconds'] * 1000:.3f} ms")
    logger.success(f"Average HMM Cost/Chunk: {runtime['average_chunk_seconds'] * 1000:.3f} ms")
    logger.success(f"E2E Latency: {runtime['total_seconds']:.3f} seconds")
    logger.success(f"HMM RTF (Real-Time Factor): {runtime['hmm_rtf']:.3f}")
    logger.success(f"E2E RTF (Real-Time Factor): {runtime['e2e_rtf']:.3f}")


def print_classification(classification: dict[str, Any]) -> None:
    logger.info("All labels and scores:")
    for label, score in zip(classification["labels"], classification["scores"]):
        logger.info(f"  {label}: {score:.6f}")
    logger.info(f"Top-{len(classification['top_k'])} emotions:")
    for rank, item in enumerate(classification["top_k"], start=1):
        logger.info(f"  {rank}. {item['label']}: {item['score']:.6f}")


def main() -> None:
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    load_start = time.perf_counter()
    model = Emotion2VecHMM(args.hmm)
    classifier = EmotionClassifier(args.classifier, args.model_dir)
    chunk_samples = int(round(args.chunk_size * 16000))
    if chunk_samples < 1:
        raise ValueError("--chunk_size is too small to contain one audio sample")
    if chunk_samples > model.window_samples:
        parser_window_seconds = model.window_samples / 16000
        raise ValueError(
            f"--chunk_size cannot exceed the HMM input window of "
            f"{parser_window_seconds:g} seconds"
        )
    if model.feature_dim != classifier.feature_dim:
        raise ValueError(
            f"Model and classifier feature dimensions do not match: " f"{model.feature_dim} != {classifier.feature_dim}"
        )
    load_seconds = time.perf_counter() - load_start

    summary = {
        "model": str(args.hmm),
        "classification_head": str(classifier.source),
        "chunk_size_seconds": chunk_samples / 16000,
        "results": {},
        "runtime": {"load_seconds": load_seconds},
    }
    used_keys: set[str] = set()
    for audio_path in args.audio:
        key = output_key(audio_path, used_keys)
        start = time.perf_counter()
        waveform = load_audio(audio_path)
        model_result = model(waveform, chunk_samples)
        classification = classifier(model_result["utterance_embedding"], args.top_k)
        chunk_classifications = [
            classifier(chunk_embedding, args.top_k)
            for chunk_embedding in model_result["utterance_chunks"]
        ]
        total_seconds = time.perf_counter() - start
        result = save_result(
            args.output_dir,
            key,
            audio_path,
            model_result,
            classification,
            chunk_classifications,
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
