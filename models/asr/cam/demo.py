# Copyright (c) 2025 HOUMO AI
#
# File: demo.py
# Description:
#   CAMPPlus Speaker Verification HMM Inference Demo
#   Input/output format compatible with fp_demo.py
#

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from loguru import logger
from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(FILE_DIR, "config.yaml")
DEFAULT_THRESHOLD = 0.31
DEFAULT_FIXED_FRAMES = 529


class CAMHMMInference:
    def __init__(self, model_path, model_dir, fixed_frames=DEFAULT_FIXED_FRAMES):
        try:
            import tcim_lite
        except ImportError:
            raise ImportError("tcim_lite is required for HMM inference")

        self.module = tcim_lite.runtime.load(model_path)
        self.fixed_frames = fixed_frames

        from modelscope.pipelines import pipeline

        self.feature_extractor = pipeline("speaker-verification", model=model_dir, model_revision="v1.0.0").model

    def _extract_feature(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        feature = self.feature_extractor._SpeakerVerificationCAMPPlus__extract_feature(
            waveform.squeeze(0).unsqueeze(0)
        ).float()
        frames = feature.shape[1]
        if frames < self.fixed_frames:
            feature = F.pad(feature, (0, 0, 0, self.fixed_frames - frames))
        else:
            feature = feature[:, : self.fixed_frames, :]
        return feature

    def _normalize_embedding(self, embedding):
        tensor = torch.from_numpy(embedding) if isinstance(embedding, np.ndarray) else embedding
        return F.normalize(tensor, p=2, dim=1)

    def _cosine_similarity(self, emb1, emb2):
        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def __call__(self, audio_pair, thr=None, output_emb=False):
        """
        Args:
            audio_pair: list of two audio file paths [audio1, audio2]
            thr: similarity threshold (default: 0.31)
            output_emb: whether to output embeddings
        Returns:
            dict with 'similarity', 'threshold', 'same_speaker'
        """
        import time

        threshold = thr if thr is not None else DEFAULT_THRESHOLD

        audio1, audio2 = audio_pair

        t0 = time.time()
        feature1 = self._extract_feature(audio1)
        feature2 = self._extract_feature(audio2)
        t1 = time.time()

        self.module.set_input("feature", feature1.numpy().astype(np.float32))
        t2 = time.time()
        self.module.run()
        self.module.sync()
        t3 = time.time()
        embedding1 = self.module.get_output("embedding").numpy()
        embedding1 = self._normalize_embedding(embedding1).cpu().numpy()
        t4 = time.time()

        self.module.set_input("feature", feature2.numpy().astype(np.float32))
        self.module.run()
        self.module.sync()
        embedding2 = self.module.get_output("embedding").numpy()
        embedding2 = self._normalize_embedding(embedding2).cpu().numpy()
        t5 = time.time()

        similarity = self._cosine_similarity(embedding1[0], embedding2[0])
        is_same = similarity > threshold

        result = {"similarity": similarity, "threshold": threshold, "same_speaker": is_same}

        if output_emb:
            result["embs"] = [embedding1, embedding2]

        logger.success("=" * 100)
        logger.success("                    Model Inference Performance Summary Report")
        logger.success("=" * 100)
        logger.success("Performance Details:")
        logger.success(f"  Extract Features : {(t1 - t0) * 1000:>7.2f} ms")
        logger.success(f"  Set Input 1      : {(t2 - t1) * 1000:>7.2f} ms")
        logger.success(f"  HMM Run 1        : {(t3 - t2) * 1000:>7.2f} ms")
        logger.success(f"  Set Input + Run 2: {(t5 - t4) * 1000:>7.2f} ms")
        logger.success(f"  Total Processing : {(t5 - t0) * 1000:>7.2f} ms")
        logger.success("=" * 100)

        return result


def get_default_assets_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "campplus")
    return model_name


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=str,
        default=None,
        help="model size",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="path to the compiled CAMPPlus HMM model",
    )
    parser.add_argument(
        "--assets-dir",
        "--assets_dir",
        dest="assets_dir",
        type=str,
        default=None,
        help="path to the downloaded ModelScope model and example audio files",
    )
    parser.add_argument(
        "--audio_files",
        dest="audio_files",
        type=str,
        nargs="+",
        default=None,
        help="audio file paths for comparison",
    )
    parser.add_argument(
        "--thr",
        dest="threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="similarity threshold",
    )
    return parser.parse_args()


# Demo usage (same format as fp_demo.py)
if __name__ == "__main__":
    args = get_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    model_name = first_not_none(args.model_name, default_model_name)
    model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(model_name, {}).get(model_size, {})
    assets_dir = Path(
        first_not_none(
            args.assets_dir,
            get_default_assets_dir(model_config),
        )
    ).expanduser()
    model_path = first_not_none(
        args.model_path,
        os.path.join("output", HOUMO_TARGET, "cam_embedding.hmm"),
    )

    print(f"Initializing HMM inference pipeline...")
    print(f"  Model: {model_name}-{model_size}")
    print(f"  Model path: {model_path}")

    # Use example files from the downloaded raw model directory
    examples_dir = assets_dir / "examples"
    speaker1_a_wav = examples_dir / "speaker1_a_cn_16k.wav"
    speaker1_b_wav = examples_dir / "speaker1_b_cn_16k.wav"
    speaker2_a_wav = examples_dir / "speaker2_a_cn_16k.wav"

    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        sv_pipeline = CAMHMMInference(
            model_path=model_path,
            model_dir=str(assets_dir),
        )

        # 相同说话人语音
        print("\n=== Same speaker test ===")
        result = sv_pipeline([speaker1_a_wav, speaker1_b_wav])
        print(result)

        # 不同说话人语音
        print("\n=== Different speaker test ===")
        result = sv_pipeline([speaker1_a_wav, speaker2_a_wav])
        print(result)

        # 可以自定义得分阈值来进行识别，阈值越高，判定为同一人的条件越严格
        print("\n=== Custom threshold test ===")
        result = sv_pipeline([speaker1_a_wav, speaker2_a_wav], thr=args.threshold)
        print(result)

        # 可以传入output_emb参数，输出结果中就会包含提取到的说话人embedding
        print("\n=== With embeddings output ===")
        result = sv_pipeline([speaker1_a_wav, speaker2_a_wav], output_emb=True)
        print("Embedding shapes:", result["embs"][0].shape, result["embs"][1].shape)
        print("Outputs:", {"similarity": result["similarity"], "same_speaker": result["same_speaker"]})

    print(f"\n=== Demo completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ===")
