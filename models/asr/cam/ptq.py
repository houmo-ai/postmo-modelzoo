# Copyright (c) 2025 HOUMO AI
#
# File: ptq.py
# Description:
#   Post-Training Quantization Tool - Python script for quantizing
# CAMPPlus speaker verification models.
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
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio

from hmatc.utils.monitor import ProcessMemoryMonitor
from hmatc.utils.utils import first_not_none, get_model_configs
from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger, xhquant_init
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR / "config.yaml"

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_WORK_DIR = THIS_DIR / "work_dirs" / "cam"
DEFAULT_OUT_DIR = os.path.join("output", HOUMO_TARGET, "hmquant")
FEATURE_DIM = 80
DEFAULT_FIXED_FRAMES = 529
DEFAULT_THRESHOLD = 0.31

torch.manual_seed(42)


def get_default_model_dir(model_config: dict) -> str:
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    model_name = model_config.get("model_name", "cam")
    return model_name


def inspect_onnx(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def simplify_onnx_if_needed(logger, onnx_path: Path, simplified_onnx_path: Path):
    if simplified_onnx_path.exists():
        return

    import onnxsim

    logger.info("Simplifying ONNX model...")
    model = onnx.load(str(onnx_path))
    model_simplified, check = onnxsim.simplify(model)
    onnx.save(model_simplified, str(simplified_onnx_path))
    logger.info(f"Simplified ONNX saved to {simplified_onnx_path}, check={check}")


def export_onnx(model_dir, work_dir, fixed_frames, feat_dim):
    """Export CAM++ model to ONNX format."""
    from modelscope.pipelines import pipeline

    output_dir = Path(work_dir) / "export_fp32" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    simplified_onnx_path = output_dir / "model_simplified.onnx"

    if simplified_onnx_path.exists():
        return onnx_path, simplified_onnx_path

    sv_pipeline = pipeline(
        "speaker-verification",
        model=model_dir,
        model_revision="v1.0.0",
    )
    embedding_model = sv_pipeline.model.embedding_model.cpu().float().eval()
    dummy_input = torch.randn(1, fixed_frames, feat_dim)

    torch.onnx.export(
        embedding_model,
        dummy_input,
        str(onnx_path),
        input_names=["feature"],
        output_names=["embedding"],
        opset_version=14,
    )

    print(f"Exported ONNX model to {onnx_path}")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed")

    import onnxsim
    model_simplified, check = onnxsim.simplify(onnx_model)
    onnx.save(model_simplified, str(simplified_onnx_path))
    print(f"Simplified ONNX saved to {simplified_onnx_path}, check={check}")

    return onnx_path, simplified_onnx_path


def load_feature_extractor(model_dir):
    from modelscope.pipelines import pipeline

    sv_pipeline = pipeline(
        "speaker-verification",
        model=model_dir,
        model_revision="v1.0.0",
    )
    return sv_pipeline.model


def extract_feature(audio_path, feature_extractor, fixed_frames):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    feature = feature_extractor._SpeakerVerificationCAMPPlus__extract_feature(
        waveform.squeeze(0).unsqueeze(0)
    ).float()
    frames = feature.shape[1]
    if frames < fixed_frames:
        feature = F.pad(feature, (0, 0, 0, fixed_frames - frames))
    else:
        feature = feature[:, :fixed_frames, :]
    return feature


def normalize_embedding(embedding):
    tensor = torch.from_numpy(embedding) if isinstance(embedding, np.ndarray) else embedding
    return F.normalize(tensor, p=2, dim=1)


def cosine_similarity(emb1, emb2):
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def run_fp(features, feature_extractor):
    device = next(feature_extractor.embedding_model.parameters()).device
    embeddings = []
    with torch.no_grad():
        for feature in features:
            embedding = feature_extractor.embedding_model(feature.to(device))
            embedding = normalize_embedding(embedding).cpu().numpy()
            embeddings.append(embedding)
    return embeddings


def run_onnx(features, simplified_onnx_path: Path):
    session = ort.InferenceSession(str(simplified_onnx_path), providers=["CPUExecutionProvider"])
    embeddings = []
    for feature in features:
        embedding = session.run(None, {"feature": feature.numpy()})[0]
        embeddings.append(normalize_embedding(embedding).cpu().numpy())
    return embeddings


def run_hmonnx(features, hmonnx_path, exec_device, save_golden_dir=None):
    session = HMONNXInference(str(hmonnx_path))
    session.exec_device = exec_device
    session.to(exec_device)
    if save_golden_dir is not None:
        session.save_golden = True
        session.save_golden_dir = str(save_golden_dir)

    embeddings = []
    with torch.no_grad():
        for feature in features:
            embedding = session.forward(feature.half().to(exec_device))
            if isinstance(embedding, (list, tuple)):
                embedding = embedding[0]
            embeddings.append(normalize_embedding(embedding.float().cpu()).cpu().numpy())
    return embeddings


def report_scores(logger, name, embeddings, threshold):
    sim_same = cosine_similarity(embeddings[0][0], embeddings[1][0])
    sim_diff = cosine_similarity(embeddings[0][0], embeddings[2][0])
    logger.info(f"{name} similarity (same speaker): {sim_same}")
    logger.info(f"{name} similarity (different speaker): {sim_diff}")
    logger.info(f"{name} threshold at {threshold}: same={sim_same > threshold}, diff={sim_diff > threshold}")
    return sim_same, sim_diff


def max_abs_diff(lhs, rhs):
    return float(np.max(np.abs(lhs - rhs)))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="path to config.yaml"
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="output hmonnx model name"
    )
    parser.add_argument(
        "--model-size", type=str, default=None, help="model size"
    )
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--simplified-onnx-path", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--quant-type", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fixed-frames", type=int, default=DEFAULT_FIXED_FRAMES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--save-golden", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--inspect-onnx", action="store_true")

    args = parser.parse_args()

    # Load config.yaml for defaults
    default_model_size, default_model_name, model_configs = get_model_configs(args.config)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8_sefp")
    )
    args.model_dir = get_default_model_dir(model_config)
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    xhquant_init(None, debug=False)
    logger = get_root_logger()
    args.work_dir = Path(args.work_dir)
    args.work_dir.mkdir(exist_ok=True, parents=True)

    with ProcessMemoryMonitor(interval=2, quiet=True) as monitor:
        # Step 1: Export ONNX from raw model
        model_dir_path = args.model_dir
        onnx_path = args.onnx_path or (args.work_dir / "export_fp32" / "onnx" / "model.onnx")
        simplified_onnx_path = args.simplified_onnx_path or (args.work_dir / "export_fp32" / "onnx" / "model_simplified.onnx")

        export_onnx(model_dir_path, args.work_dir, args.fixed_frames, FEATURE_DIM)

        if not simplified_onnx_path.exists():
            raise FileNotFoundError(f"Missing simplified ONNX model: {simplified_onnx_path}")

        if args.inspect_onnx:
            inspect_onnx(str(simplified_onnx_path))

        # Step 2: Quantize
        target_device = DeviceType.XH2a
        quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
        quant_config = create_quant_config(quant_scheme)
        onnx_name = Path(onnx_path).stem
        # Output to output/xh2/hmquant/prefill/ to match build.py default model_dir
        hmonnx_dir = Path(DEFAULT_OUT_DIR) / "prefill"
        hmonnx_dir.mkdir(parents=True, exist_ok=True)
        hmonnx_path = hmonnx_dir / f"{onnx_name}_{target_device}_{args.quant_type}.onnx"

        if not hmonnx_path.exists():
            logger.info(f"Converting ONNX to hmonnx: {hmonnx_path}")
            dummy_input = torch.randn(1, args.fixed_frames, FEATURE_DIM)
            convert_onnx_to_hmonnx(
                str(simplified_onnx_path),
                (dummy_input,),
                target_device,
                str(hmonnx_path),
                quant_config=quant_config,
                input_names=["feature"],
                output_names=["embedding"],
            )

        if args.no_eval and not args.save_golden:
            print(
                f"\n=== Quantization completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
            )
            exit(0)

        # Step 3: Evaluate
        model_dir_path = os.path.join(".", args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir
        example_wavs = [
            os.path.join(model_dir_path, "examples", "speaker1_a_cn_16k.wav"),
            os.path.join(model_dir_path, "examples", "speaker1_b_cn_16k.wav"),
            os.path.join(model_dir_path, "examples", "speaker2_a_cn_16k.wav"),
        ]
        feature_extractor = load_feature_extractor(model_dir_path)
        features = [extract_feature(path, feature_extractor, args.fixed_frames) for path in example_wavs]

        if not args.no_eval:
            fp_embeddings = run_fp(features, feature_extractor)
            onnx_embeddings = run_onnx(features, simplified_onnx_path)
            hmonnx_embeddings = run_hmonnx(features, str(hmonnx_path), args.device)

            logger.info("Evaluation on fixed-length CAM features")
            report_scores(logger, "FP", fp_embeddings, args.threshold)
            report_scores(logger, "ONNX", onnx_embeddings, args.threshold)
            report_scores(logger, "HMONNX", hmonnx_embeddings, args.threshold)

            logger.info(
                "Embedding max abs diff: fp-vs-onnx=%s, fp-vs-hmonnx=%s",
                max_abs_diff(fp_embeddings[0], onnx_embeddings[0]),
                max_abs_diff(fp_embeddings[0], hmonnx_embeddings[0]),
            )

        if args.save_golden:
            golden_dir = hmonnx_dir / f"{onnx_name}_golden"
            run_hmonnx(features, str(hmonnx_path), args.device, save_golden_dir=golden_dir)
            logger.info(f"Golden saved to {golden_dir}")

    print(
        f"\n=== Quantization completed. Peak memory: {monitor.peak_memory_mb:.2f} MB ==="
    )