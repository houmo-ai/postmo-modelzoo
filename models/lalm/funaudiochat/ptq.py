# Copyright (c) 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   Fun-Audio-Chat post-training quantization and HMONNX export tool.
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

"""Export the six FunAudioChat HMONNX graphs with the Merak workflow."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from hmatc.utils.utils import first_not_none, get_model_configs

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODEL_DIR / "config.yaml"
HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

COMPONENTS = ("audio_tower", "audio_encoder", "qwen3", "audio_decoder")
ARTIFACTS = (
    "audio_tower_hmonnx",
    "audio_encoder_hmonnx",
    "qwen3_prefill_hmonnx",
    "qwen3_decode_hmonnx",
    "audio_decoder_prefill_hmonnx",
    "audio_decoder_decode_hmonnx",
)
WEIGHT_FILES = (
    "quant_audio_embedding.pt",
    "audio_decoder_pre_matching.pt",
)
RELEASE_COMPONENTS = {
    "audio_tower_hmonnx": "audio_tower",
    "audio_encoder_hmonnx": "audio_encoder",
    "qwen3_prefill_hmonnx": "prefill",
    "qwen3_decode_hmonnx": "decode",
    "audio_decoder_prefill_hmonnx": "audio_decoder_prefill",
    "audio_decoder_decode_hmonnx": "audio_decoder_decode",
}


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _configs_merak_dir() -> Path:
    spec = importlib.util.find_spec("xhmodel_merak")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Python package xhmodel_merak not found")
    path = Path(next(iter(spec.submodule_search_locations))).parent / "configs_merak"
    if not path.is_dir():
        raise FileNotFoundError(f"configs_merak directory not found: {path}")
    return path


def _load_config(config_path: Path, model_name: str | None, model_size: str | None) -> tuple[str, str, dict]:
    default_size, default_name, model_configs = get_model_configs(str(config_path))
    resolved_name = first_not_none(model_name, default_name)
    resolved_size = first_not_none(model_size, default_size)
    model_config = model_configs.get(resolved_name, {}).get(resolved_size)
    if model_config is None:
        supported = [f"{name}-{size}" for name, sizes in model_configs.items() for size in sizes]
        raise ValueError(
            f"Unsupported model combination '{resolved_name}-{resolved_size}'. "
            f"Supported models: {', '.join(supported)}"
        )

    return resolved_name, resolved_size, model_config


def _workflow_config_path(workflow_config: str) -> Path:
    path = Path(workflow_config).expanduser()
    if not path.is_absolute():
        path = _configs_merak_dir() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Workflow config not found: {path}")
    return path


def _golden_device(device: str) -> str:
    """Use CUDA for golden inference only when a CUDA compiler is available."""
    requested = str(device)
    if not requested.startswith("cuda"):
        return requested

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")
    if cuda_home and Path(cuda_home, "bin", "nvcc").is_file():
        return requested
    if nvcc:
        return requested

    print(
        "Warning: CUDA Toolkit/nvcc is unavailable; falling back to CPU for golden inference. "
        "HMONNX export will still use the requested CUDA device.",
        flush=True,
    )
    return "cpu"


def _validate_export(export_dir: Path) -> Path:
    meta_path = export_dir / "export_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Export metadata not found: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    enabled = set(metadata.get("components", []))
    missing_components = [name for name in COMPONENTS if name not in enabled]
    artifacts = metadata.get("artifacts") or {}
    missing_artifacts = [name for name in ARTIFACTS if not artifacts.get(name)]
    missing_files = [
        str(export_dir / artifacts[name])
        for name in ARTIFACTS
        if artifacts.get(name) and not (export_dir / artifacts[name]).is_file()
    ]
    missing_weights = [str(export_dir / name) for name in WEIGHT_FILES if not (export_dir / name).is_file()]
    if missing_components or missing_artifacts or missing_files or missing_weights:
        raise FileNotFoundError(
            "Incomplete FunAudioChat export: "
            f"missing components={missing_components}, "
            f"missing artifacts={missing_artifacts}, missing files={missing_files}, "
            f"missing weights={missing_weights}"
        )

    token_embedding = export_dir / "qwen3" / "token_embedding.pt"
    if token_embedding.is_file():
        shutil.copy2(token_embedding, export_dir / "quant_embedding.pt")
        token_embedding.unlink()
    return meta_path


def _release_prefix(
    model_config: dict[str, Any],
    model_name: str,
    quant_type: str,
    context_length: str,
) -> str:
    """Build the default all-lowercase prefix used by the release layout."""
    modelscope_repo = (model_config.get("modelscope_repo") or [model_name])[0]
    modelscope_name = str(modelscope_repo).rsplit("/", maxsplit=1)[-1].lower()
    return f"hmquant_{HOUMO_TARGET}_{modelscope_name}_{quant_type.lower()}_256_{context_length}_{date.today():%Y%m%d}"


def _rename_release_files(component_dir: Path, prefix: str, component: str) -> None:
    """Give the graph and external data their release names."""
    onnx_files = sorted(component_dir.glob("*.onnx"))
    if len(onnx_files) > 1:
        raise RuntimeError(f"Expected one ONNX file in {component_dir}, found {onnx_files}")
    if onnx_files:
        onnx_target = component_dir / f"{prefix}_{component}_with_act.onnx"
        if onnx_files[0] != onnx_target:
            if onnx_target.exists():
                onnx_target.unlink()
            shutil.move(str(onnx_files[0]), str(onnx_target))

    external_files = [
        path
        for path in component_dir.iterdir()
        if path.is_file() and ("external_data" in path.name or path.name.endswith(".data"))
    ]
    if len(external_files) > 1:
        raise RuntimeError(f"Expected one external-data file in {component_dir}, found {external_files}")
    if external_files:
        external_target = component_dir / f"{prefix}_{component}_external_data"
        if not external_target.exists():
            external_target.symlink_to(os.path.relpath(external_files[0], component_dir))

    step_dir = component_dir / "step_0"
    step_dir.mkdir(exist_ok=True)
    golden_dir = component_dir / "golden"
    if golden_dir.is_dir():
        golden_step_dir = golden_dir / "step_0"
        source_dir = golden_step_dir if golden_step_dir.is_dir() else golden_dir
        for path in list(source_dir.iterdir()):
            destination = step_dir / path.name
            if destination.exists() or destination.is_symlink():
                if destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(path), str(destination))
        if golden_step_dir.is_dir():
            golden_step_dir.rmdir()
        golden_dir.rmdir()

    for path in sorted(step_dir.iterdir()):
        if path.is_file() and path.suffix == ".npy" and not path.name.startswith(prefix + "_"):
            marker = f"{component}_"
            suffix = path.name.split(marker, maxsplit=1)[-1] if marker in path.name else path.name
            path.rename(step_dir / f"{prefix}_{component}_{suffix}")

    for name in (f"{prefix}_{component}_with_act.onnx", f"{prefix}_{component}_external_data"):
        target = component_dir / name
        if target.exists():
            link = step_dir / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path("..") / name)


def _organize_release(
    export_dir: Path,
    meta_path: Path,
    release_prefix: str,
) -> Path:
    """Convert Merak's component export into the documented release layout."""
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts") or {}
    moved_sources: set[Path] = set()
    for artifact_name, component in RELEASE_COMPONENTS.items():
        artifact = artifacts.get(artifact_name)
        if not artifact:
            continue
        source = (export_dir / artifact).resolve()
        target_dir = export_dir / component
        if not source.is_file():
            candidates = sorted(target_dir.rglob("*.onnx")) if target_dir.is_dir() else []
            if len(candidates) != 1:
                raise FileNotFoundError(f"Artifact file not found: {source}")
            source = candidates[0].resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Artifact file not found: {source}")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if source not in moved_sources and source != target:
            if target.exists():
                target.unlink()
            shutil.move(str(source), str(target))
            moved_sources.add(source)
        for sibling in list(source.parent.iterdir()):
            if sibling.is_file() and "external_data" in sibling.name and component in sibling.name:
                destination = target_dir / sibling.name
                if destination.exists():
                    destination.unlink()
                shutil.move(str(sibling), str(destination))
        artifacts[artifact_name] = str(Path(component) / target.name)

        source_golden = source.parent / "golden"
        if source_golden.is_dir():
            target_golden = target_dir / "golden"
            target_golden.mkdir(exist_ok=True)
            for golden in source_golden.iterdir():
                if component not in golden.name:
                    continue
                destination = target_golden / golden.name
                if destination.exists():
                    destination.unlink()
                shutil.move(str(golden), str(destination))

    hf_config = export_dir / "qwen3" / "hf_config"
    if hf_config.is_dir():
        shutil.move(str(hf_config), str(export_dir / "hf_config"))

    for artifact_name, component in RELEASE_COMPONENTS.items():
        component_dir = export_dir / component
        if not component_dir.is_dir():
            continue
        _rename_release_files(component_dir, release_prefix, component)
        for key, value in list(artifacts.items()):
            if key == artifact_name:
                artifacts[key] = str(Path(component) / f"{release_prefix}_{component}_with_act.onnx")

    # The token embedding is a release-level asset, not an implementation detail
    # of the qwen3 component.  The audio embedding is already exported at root.
    metadata["artifacts"] = artifacts
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    golden_meta_path = export_dir / "golden_meta_info.json"
    if not golden_meta_path.exists():
        golden_meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    quant_dir = export_dir / "quant"
    if quant_dir.is_dir():
        shutil.rmtree(quant_dir)
    for implementation_dir in ("qwen3", "audio_decoder", "audio_encoder", "audio_tower"):
        path = export_dir / implementation_dir
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    print(f"release_dir: {export_dir} ({release_prefix})", flush=True)
    return export_dir


def _release_workflow(workflow: Any, quant_result: Any, export_result: Any) -> None:
    del workflow, quant_result, export_result
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _patch_xhquant_pooling_for_torch28() -> None:
    """Normalize scalar pooling attributes before xhquant torch.export tracing."""
    try:
        from xhquant.backend.xh2a.ir.operators import pooling as ir_pooling
        from xhquant.quantization.xh2a.qmodules import pooling as q_pooling
    except ImportError:
        return

    original = ir_pooling.torch_ops_xh2a_average_pool2d

    def average_pool2d_compat(
        input_tensor,
        kernel_shape,
        auto_pad,
        strides,
        pads,
        dilations,
        ceil_mode,
        count_include_pad,
    ):
        def pair(value):
            return (value, value) if isinstance(value, int) else tuple(value)

        return original(
            input_tensor,
            pair(kernel_shape),
            auto_pad,
            pair(strides),
            pair(pads),
            pair(dilations),
            ceil_mode,
            count_include_pad,
        )

    ir_pooling.torch_ops_xh2a_average_pool2d = average_pool2d_compat
    q_pooling.torch_ops_xh2a_average_pool2d = average_pool2d_compat


def _export_components(
    *,
    model_dir: Path,
    workflow_path: Path,
    output_dir: Path,
    device: str,
    debug: bool,
    runtime_overrides: dict[str, Any],
) -> Any:
    from xhmodel_merak.workflows import AutoWorkflow

    merged_meta = None
    last_export_result = None
    for component in COMPONENTS:
        overrides = dict(runtime_overrides)
        for name in COMPONENTS:
            overrides[f"export.components.{name}.enabled"] = name == component

        print(f"\n=== Export FunAudioChat component: {component} ===", flush=True)
        workflow = AutoWorkflow.from_config(
            model_dir=str(model_dir),
            config_path=str(workflow_path),
            debug=debug,
        )
        quant_result = workflow.quant(
            output_dir=str(output_dir / "quant" / component),
            device=device,
            config_overrides=overrides,
        )
        export_result = workflow.export(
            quant_result=quant_result,
            output_dir=str(output_dir),
            device=device,
            config_overrides=overrides,
        )
        component_meta = json.loads(
            (Path(export_result.work_dir) / "export_meta_info.json").read_text(encoding="utf-8")
        )
        if merged_meta is None:
            merged_meta = dict(component_meta)
            merged_meta["components"] = list(component_meta.get("components", []))
            merged_meta["artifacts"] = dict(component_meta.get("artifacts", {}))
        else:
            for name in component_meta.get("components", []):
                if name not in merged_meta["components"]:
                    merged_meta["components"].append(name)
            merged_meta["artifacts"].update(component_meta.get("artifacts", {}))

        last_export_result = export_result
        _release_workflow(workflow, quant_result, None)

    if merged_meta is None or last_export_result is None:
        raise RuntimeError("FunAudioChat workflow did not export any component")
    meta_path = output_dir / "export_meta_info.json"
    meta_path.write_text(json.dumps(merged_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    last_export_result.meta = merged_meta
    return last_export_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export FunAudioChat HMONNX graphs with the Merak workflow.")
    parser.add_argument("--config", "--config-path", dest="config_path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-name", "--model_name", dest="model_name", default="funaudiochat")
    parser.add_argument("--model-size", "--model_size", dest="model_size", default="8b")
    parser.add_argument("--model-dir", "--model_dir", dest="model_dir", default="Fun-Audio-Chat-8B")
    parser.add_argument(
        "--audio",
        default=str(Path(os.getenv("HOUMO_EXAMPLES_PATH", MODEL_DIR.parents[2])) / "data" / "audio" / "question.wav"),
        help="reference audio used to build the static audio encoder graph",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", default=os.path.join("output", HOUMO_TARGET, "hmquant")
    )
    parser.add_argument("--device", default="cuda:0", help="device for Merak workflow export (default: %(default)s)")
    parser.add_argument("--system-prompt", dest="system_prompt", default=None)
    parser.add_argument(
        "--context-length",
        "--context_length",
        dest="context_length",
        default="2k",
        help="Context length label used in the release prefix (for example: 2k or 8k).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = _resolve_path(args.config_path, Path.cwd())
    _, _, model_config = _load_config(config_path, args.model_name, args.model_size)

    model_dir = _resolve_path(args.model_dir, Path.cwd())
    audio_path = _resolve_path(args.audio, Path.cwd())
    output_dir = _resolve_path(args.output_dir, Path.cwd())
    device = str(args.device)
    system_prompt = args.system_prompt
    overwrite = args.overwrite
    debug = args.debug
    dump_golden = args.dump_golden

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    _patch_xhquant_pooling_for_torch28()

    # Component enablement, quantization, and export lengths belong to the
    # Merak workflow config. Only machine-local runtime values are overridden.
    overrides: dict[str, Any] = {"export.runtime.audio": str(audio_path)}
    if system_prompt is not None:
        overrides["export.runtime.system_prompt"] = str(system_prompt)

    workflow_path = _workflow_config_path(model_config["workflow_config"])
    export_result = _export_components(
        model_dir=model_dir,
        workflow_path=workflow_path,
        output_dir=output_dir,
        device=device,
        debug=debug,
        runtime_overrides=overrides,
    )
    if dump_golden:
        from xhmodel_merak.workflows import AutoWorkflow

        workflow = AutoWorkflow.from_config(
            model_dir=str(model_dir),
            config_path=str(workflow_path),
            debug=debug,
        )
        golden_dir = workflow.dump_golden(export_result=export_result, device=_golden_device(device))
        print(f"golden_dir: {golden_dir}")
    export_dir = Path(export_result.work_dir).resolve()
    meta_path = _validate_export(export_dir)
    release_prefix = _release_prefix(
        model_config,
        args.model_name,
        model_config.get("quant_type", "wmix_amix"),
        args.context_length,
    )
    _organize_release(export_dir, meta_path, release_prefix)
    print(_validate_export(export_dir))


if __name__ == "__main__":
    main()
