from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from hmatc.utils.utils import get_model_configs
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def find_configs_merak_dir() -> Path:
    spec = importlib.util.find_spec("xhmodel_merak")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Python package xhmodel_merak not found")

    xhmodel_merak_dir = Path(next(iter(spec.submodule_search_locations)))
    configs_merak_dir = xhmodel_merak_dir.parent / "configs_merak"
    if not configs_merak_dir.is_dir():
        raise FileNotFoundError(f"configs_merak directory not found next to xhmodel_merak: {configs_merak_dir}")
    return configs_merak_dir


def get_workflow_config_path(config_path: str, model_name: str, model_size: str) -> Path:
    _, _, model_configs = get_model_configs(config_path)
    model_config = model_configs.get(model_name, {}).get(model_size, {})
    workflow_config = model_config.get("workflow_config")
    if not workflow_config:
        raise ValueError(f"workflow_config not found for {model_name}/{model_size} in {config_path}")

    workflow_config_path = find_configs_merak_dir() / workflow_config
    if not workflow_config_path.is_file():
        raise FileNotFoundError(f"Workflow config not found: {workflow_config_path}")
    return workflow_config_path


def export_classifier(model_dir: str, output_dir: str) -> Path:
    model_path = Path(model_dir) / "model.pt"
    tokens_path = Path(model_dir) / "tokens.txt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Official checkpoint not found: {model_path}")
    if not tokens_path.is_file():
        raise FileNotFoundError(f"Emotion labels not found: {tokens_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    weight = state_dict["proj.weight"].detach().float().cpu().numpy()
    bias = state_dict["proj.bias"].detach().float().cpu().numpy()
    labels = np.asarray(
        [
            line.strip()
            for line in tokens_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    if weight.shape != (len(labels), 1024) or bias.shape != (len(labels),):
        raise ValueError(
            f"Classifier and labels do not match: weight={weight.shape}, "
            f"bias={bias.shape}, labels={len(labels)}"
        )

    output_path = Path(output_dir) / "classifier.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, weight=weight, bias=bias, labels=labels)
    print(f"Exported classifier: {output_path}")
    return output_path


def export_hmonnx(
    model_dir: str,
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    overwrite: bool = False,
    golden_audio: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)

    workflow = AutoLLMWorkflow.from_config(
        model_dir=model_dir,
        config_path=config_path,
    )
    quant_result = workflow.quant(
        output_dir=f"{output_dir}_quant",
        device=device,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=output_dir,
        device=device,
    )
    if golden_audio is not None:
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device=device,
            input_messages={"audio": golden_audio},
        )
        print(f"golden_dir: {golden_dir}")
    return Path(export_result.work_dir) / "emotion2vec_meta.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the emotion2vec Merak W8A8 export workflow.")
    parser.add_argument("--model-dir", default="emotion2vec_plus_large")
    parser.add_argument("--model-name", default="emotion2vec")
    parser.add_argument("--model-size", default="plus_large")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Dump official PyTorch golden data after HMONNX export.",
    )
    parser.add_argument(
        "--golden-audio",
        default="data/models/emotion2vec_plus_large/example/test.wav",
        help="Audio used by workflow.dump_golden().",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workflow_config_path = get_workflow_config_path(args.config_path, args.model_name, args.model_size)
    meta_path = export_hmonnx(
        model_dir=args.model_dir,
        config_path=str(workflow_config_path),
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.overwrite,
        golden_audio=args.golden_audio if args.dump_golden else None,
    )
    export_classifier(args.model_dir, args.output_dir)
    print(meta_path)


if __name__ == "__main__":
    main()
