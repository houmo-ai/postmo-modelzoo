# Copyright (c) 2026 HOUMO AI
#
# File: ptq.py
# Description:
#   Quantize Laguna-S-2.1 models and export HMONNX artifacts for XH2.
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

"""Standard Laguna-S-2.1 Merak workflow example.

Run this file from the repository root. Model shape and quantization defaults
stay in YAML; the workflow API only needs paths plus ``QuantResult``.
"""

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

from hmatc.utils.utils import first_not_none, get_model_configs

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _ensure_vendored_autoround_source() -> Path:
    """Prefer the AutoRound source shipped beside the vendored GPTQModel.

    The repository ``env.sh`` adds both ``hmodel/gptqmodel`` and its
    ``third_party/auto-round`` directory to ``PYTHONPATH``.  This helper keeps
    the script usable when that shell setup was not sourced, without relying
    on a separately installed AutoRound version.

    This only resolves the Python source package.  The ARK backend still needs
    its matching compiled ``auto_round_kernel`` extension when that backend is
    actually selected.
    """
    repo_root = Path(__file__).resolve().parents[3]
    gptqmodel_roots: list[Path] = []

    examples_root = os.environ.get("HOUMO_EXAMPLES_PATH")
    if examples_root:
        gptqmodel_roots.append(Path(examples_root) / "hmodel" / "gptqmodel")

    for path_entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not path_entry:
            continue
        path = Path(path_entry).resolve()
        if path.name == "gptqmodel":
            gptqmodel_roots.append(path)
        elif path.name == "auto-round" and path.parent.name == "gptqmodel":
            gptqmodel_roots.append(path.parent)

    gptqmodel_roots.append(repo_root / "hmodel" / "gptqmodel")

    autoround_root = next(
        (
            root / "third_party" / "auto-round"
            for root in gptqmodel_roots
            if (root / "third_party" / "auto-round" / "auto_round" / "__init__.py").is_file()
        ),
        None,
    )
    if autoround_root is None:
        raise ImportError(
            "Vendored AutoRound source was not found. Expected "
            "<gptqmodel>/third_party/auto-round, where <gptqmodel> is derived "
            "from HOUMO_EXAMPLES_PATH or PYTHONPATH."
        )

    autoround_root = autoround_root.resolve()
    loaded_autoround = sys.modules.get("auto_round")
    if loaded_autoround is not None:
        loaded_file = getattr(loaded_autoround, "__file__", None)
        if loaded_file is None or autoround_root not in Path(loaded_file).resolve().parents:
            raise ImportError(
                "An external auto_round package was imported before the vendored "
                f"source could be selected: {loaded_file!r}. Start ptq.py with "
                "the repository env.sh or remove that package from the process."
            )
    elif str(autoround_root) not in sys.path:
        sys.path.insert(0, str(autoround_root))

    importlib.invalidate_caches()
    import auto_round
    from auto_round.inference import backend as autoround_backend
    from transformers.utils.versions import require_version as transformers_require_version

    # The source checkout is intentionally not installed as a distribution,
    # so importlib.metadata cannot satisfy AutoRound's self-requirement such as
    # ``auto-round>=0.5.1``. Treat only that self-requirement as satisfied when
    # the imported package is the verified vendored source. Do not mask
    # ``auto-round-lib``: it is a separate compiled ARK dependency.
    package_file = Path(auto_round.__file__).resolve()
    if autoround_root not in package_file.parents:
        raise ImportError(
            f"AutoRound was not loaded from the vendored source: {package_file}; "
            f"expected it below {autoround_root}."
        )
    if not getattr(autoround_backend, "_xh_vendored_requirement_patch", False):
        def require_version(requirement):
            normalized = str(requirement).lower().replace("_", "-")
            if normalized.startswith("auto-round") and not normalized.startswith("auto-round-lib"):
                return None
            return transformers_require_version(requirement)

        autoround_backend.require_version = require_version
        autoround_backend._xh_vendored_requirement_patch = True
    return autoround_root


def get_default_model_dir(model_config: dict) -> str:
    model_dir = os.path.join(os.path.dirname(__file__), "Laguna-S-2.1")
    if os.path.isdir(model_dir):
        return model_dir
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0].rsplit("/", maxsplit=1)[-1]
    return "Laguna-S-2.1"


def _validate_model_dir(model_dir: str) -> None:
    path = Path(model_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model directory does not look like a HuggingFace checkpoint: {path}; "
            "missing config.json."
        )


def find_configs_merak_dir() -> str:
    spec = importlib.util.find_spec("xhmodel_merak")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Python package xhmodel_merak not found")

    xhmodel_merak_dir = next(iter(spec.submodule_search_locations))
    configs_merak_dir = os.path.join(
        os.path.dirname(xhmodel_merak_dir), "configs_merak"
    )
    if not os.path.isdir(configs_merak_dir):
        raise FileNotFoundError(
            f"configs_merak directory not found next to xhmodel_merak: {configs_merak_dir}"
        )
    return configs_merak_dir


def _remove_output_dir_if_needed(output_dir: str) -> None:
    path = Path(output_dir)
    if path.exists():
        print(f"Output directory already exists, removing it: {path}")
        shutil.rmtree(path)


def _move_hmquant_to_output(export_output_dir: str, target_dir: str) -> None:
    export_dir = Path(export_output_dir)
    hmquant_dirs = sorted(
        path
        for path in export_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"hmquant_{HOUMO_TARGET}_")
    )
    if not hmquant_dirs:
        raise FileNotFoundError(
            f"No hmquant_{HOUMO_TARGET}_* directory found under {export_dir}"
        )
    if len(hmquant_dirs) > 1:
        raise RuntimeError(
            f"Expected one hmquant_{HOUMO_TARGET}_* directory under {export_dir}, found: {hmquant_dirs}"
        )

    source_dir = hmquant_dirs[0]
    target = Path(target_dir)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for item in source_dir.iterdir():
        if item.name == "golden_meta_info.json":
            continue
        shutil.move(str(item), str(target / item.name))
    print(f"hmquant contents moved to: {target}")


def _configure_cuda_device(device: str) -> None:
    import torch

    parsed_device = torch.device(device)
    if parsed_device.type == "cuda" and parsed_device.index is not None:
        torch.cuda.set_device(parsed_device.index)


def _get_available_gpu_device_map() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Laguna GPTQModel quantization requires at least one CUDA device")

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("Laguna GPTQModel quantization found no available CUDA devices")

    device_map = ",".join(str(index) for index in range(device_count - 1, -1, -1))
    rotation_device = "cuda:0"
    return device_map, rotation_device


def _configure_low_memory_export(enabled: bool) -> None:
    from xhmodel_merak.xh_llm.utils import configure_huge_model_export

    configure_huge_model_export(enabled)


def _export_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.context_max_length is not None:
        overrides["export.model.context_max_length"] = args.context_max_length
    if args.quant_type is not None:
        overrides["export.model.quant_scheme.quant_type"] = args.quant_type
        overrides["export.model.quant_scheme.nodes.lm_head.quant_type"] = args.quant_type
    if args.only_first_block:
        overrides["export.model.only_first_block"] = True
    if args.max_layers is not None:
        if args.max_layers <= 0:
            raise ValueError("--max-layers must be greater than zero")
        overrides["export.model.max_layers"] = args.max_layers
    return overrides


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="Run the Laguna-S-2.1 Merak quant/export workflow.")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--model_name", type=str, default=None, help="Model name in config.yaml.")
    parser.add_argument("--model_size", type=str, default=None, help="Model size in config.yaml.")
    parser.add_argument("--model_dir", "--model-dir", type=str, default=None, help="HF model directory; defaults to the local Laguna-S-2.1 link.")
    parser.add_argument("--quant_config_path", dest="quant_config_path", type=str, default=None, help="Explicit workflow YAML path; overrides workflow_config from config.yaml.")
    parser.add_argument("--output_dir", default=f"./output/{HOUMO_TARGET}/hmquant", help="output directory")
    parser.add_argument("--work_dir", type=str, default="./work_dirs", help="work directory")
    parser.add_argument("--device", default="cuda", help="Device for quant/export/golden. Default: cuda")
    parser.add_argument("--golden-device-map", default=None, help="Comma-separated devices for HMONNX golden auto-offload, for example cuda:0,cuda:1.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing export output directories before running.")
    parser.add_argument("--dump-golden", action="store_true", help="Dump golden data after export.")
    parser.add_argument("--export-from-quanted-model", action="store_true", help="Treat --model-dir as an existing GPTQModel/AutoRound checkpoint and export it directly.")
    parser.add_argument("--bits", type=int, default=None, help="quantization bits, set this param to override config.yaml")
    parser.add_argument("--context-max-length", "--context-length", type=int, default=None, help="LLM max context length for export.")
    parser.add_argument("--quant-type", default=None, help="Override export-time HMONNX quantization type.")
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--only-first-block", action="store_true", help="Export only layer 0 for debugging.")
    layer_group.add_argument("--max-layers", type=int, default=None, help="Export the first N decoder layers.")
    parser.add_argument("--disable-low-memory", action="store_true", help="Disable low-memory sparse MoE streaming and use full-memory export.")
    parser.add_argument("--prompt", default="Briefly explain what makes Laguna-S-2.1 useful for coding tasks.")
    parser.add_argument("--debug", action="store_true")
    # fmt: on
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(args.config_path)
    args.model_name = first_not_none(args.model_name, default_model_name)
    args.model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(args.model_name, {}).get(args.model_size, {})
    args.model_config = model_config
    args.model_dir = os.path.abspath(first_not_none(args.model_dir, get_default_model_dir(model_config)))
    _validate_model_dir(args.model_dir)
    if args.quant_config_path is None:
        workflow_config = model_config.get("workflow_config", "")
        args.quant_config_path = os.path.join(find_configs_merak_dir(), workflow_config)
    if not os.path.isfile(args.quant_config_path):
        raise FileNotFoundError(f"Workflow config not found: {args.quant_config_path}")
    return args


def main() -> None:
    args = parse_args()
    autoround_root = _ensure_vendored_autoround_source()
    print(f"using vendored AutoRound source: {autoround_root}")
    _configure_cuda_device(args.device)
    _configure_low_memory_export(not args.disable_low_memory)

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.quant_config_path,
        debug=args.debug,
    )

    quant_output_dir = f"{args.work_dir}/laguna_s_2_1_quantized"
    export_output_dir = f"{args.work_dir}/laguna_s_2_1_export"

    config_overrides = {}
    if args.export_from_quanted_model:
        config_overrides["quant"] = None
    else:
        device_map, rotation_device = _get_available_gpu_device_map()
        config_overrides["quant.runtime.device_map"] = device_map
        config_overrides["quant.runtime.rotation_device"] = rotation_device
        if args.bits is not None:
            config_overrides["quant.bits"] = args.bits
        print(
            f"quant device config_overrides: device_map={device_map}, "
            f"rotation_device={rotation_device}"
        )
    if not args.export_from_quanted_model:
        _remove_output_dir_if_needed(quant_output_dir)
    quant_result = workflow.quant(
        output_dir=quant_output_dir,
        device=args.device,
        config_overrides=config_overrides,
    )
    print(f"quant_result: {quant_result}")

    _remove_output_dir_if_needed(export_output_dir)
    export_overrides = _export_overrides(args)
    print(f"export config_overrides: {export_overrides}")
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=export_output_dir,
        device=args.device,
        config_overrides=export_overrides,
    )
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_meta = workflow.dump_golden(
            export_result=export_result,
            device=args.golden_device_map or args.device,
            input_messages={"text": args.prompt},
        )
        print(f"golden_meta: {golden_meta}")
    _move_hmquant_to_output(export_output_dir, args.output_dir)


if __name__ == "__main__":
    main()
