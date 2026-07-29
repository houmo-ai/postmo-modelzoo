# Copyright 2025 HOUMO AI
#
# File: parser.py
# Description:
#   Build the HMATC CLI argument parser with the current public surface.
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


def build_parser() -> argparse.ArgumentParser:
    """Build the HMATC CLI argument parser with the current public surface."""
    # fmt: off
    target = os.environ.get("HOUMO_TARGET")

    parent_config = argparse.ArgumentParser(add_help=False)
    parent_target = argparse.ArgumentParser(add_help=False)
    parent_onnx = argparse.ArgumentParser(add_help=False)
    parent_hmonnx = argparse.ArgumentParser(add_help=False)
    parent_device_id = argparse.ArgumentParser(add_help=False)
    parent_cuda = argparse.ArgumentParser(add_help=False)
    parent_model_cfg = argparse.ArgumentParser(add_help=False)
    parent_layers = argparse.ArgumentParser(add_help=False)
    parent_log = argparse.ArgumentParser(add_help=False)

    common_group = parent_target.add_argument_group("Common options")
    common_group.add_argument("--target", "-t", type=str, required=target != "xh2", choices=("xh2",), default=target, help="Target chip platform")

    log_group = parent_log.add_argument_group("Logging options")
    log_group.add_argument("--log_level", type=str, required=False, default="INFO", choices=("DEBUG", "INFO", "WARN", "ERROR", "FATAL"), help="Logging level")

    config_group = parent_config.add_argument_group("Config-driven mode")
    config_group.add_argument("--config", "-c", type=str, required=False, help="YAML config file path")

    backend_group = parent_onnx.add_argument_group("Config-driven backend selection")
    backend_group.add_argument("--onnx", action="store_true", help="Use ONNX backend instead of chip backend")

    hidden_backend_group = parent_hmonnx.add_argument_group("Config-driven backend selection")
    hidden_backend_group.add_argument("--hmonnx", action="store_true", help=argparse.SUPPRESS)

    runtime_group = parent_device_id.add_argument_group("Runtime options")
    runtime_group.add_argument("--device_id", type=int, required=False, default=0, help="Chip device id for inference")

    quant_runtime_group = parent_cuda.add_argument_group("Quantization runtime options")
    quant_runtime_group.add_argument("--cuda", action="store_true", help="Enable CUDA quantization")

    build_override_group = parent_model_cfg.add_argument_group("Config build overrides")
    build_override_group.add_argument("--batch", "-b", type=int, required=False, default=None, help="Override build batch")
    build_override_group.add_argument("--ncore", "-nc", type=int, required=False, default=None, choices=(1, 2), help="Override IPU core count")
    build_override_group.add_argument("--opt_level", type=int, required=False, default=None, choices=(0, 1, 2), help="Override build optimization level")
    build_override_group.add_argument("--roi_num", type=int, required=False, default=None, help="Override ROI count")

    golden_debug_group = parent_layers.add_argument_group("Golden/debug options")
    golden_debug_group.add_argument("--layers", action="store_true", help="Generate or check per-layer outputs")

    parser = argparse.ArgumentParser(description="HouMo Model Assist Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="quant build compare perf demo eval benchmark check gen golden")

    quant_parser = subparsers.add_parser("quant", parents=[parent_target, parent_config, parent_cuda, parent_log], help="Quantize a model from config")
    build_cmd_parser = subparsers.add_parser("build", parents=[parent_target, parent_model_cfg, parent_device_id, parent_log], help="Build from config or hmonnx")
    compare_parser = subparsers.add_parser("compare", parents=[parent_target, parent_config, parent_model_cfg, parent_device_id, parent_log], help="Compare ONNX / hmquant / chip outputs")
    perf_parser = subparsers.add_parser("perf", parents=[parent_target, parent_model_cfg, parent_device_id, parent_log], help="Measure model performance")
    demo_parser = subparsers.add_parser("demo", parents=[parent_target, parent_config, parent_onnx, parent_hmonnx, parent_model_cfg, parent_device_id, parent_log], help="Run config-driven model demo")
    evaluate_parser = subparsers.add_parser(
        "eval",
        parents=[parent_target, parent_config, parent_onnx, parent_hmonnx, parent_model_cfg, parent_device_id, parent_log],
        help="Run config-driven eval or large-model EvalScope eval",
        description=(
            "Run model evaluation in one of two modes:\n"
            "  1. Config-driven small-model eval: hmatc eval -c config.yml [--onnx]\n"
            "  2. Large-model EvalScope eval: hmatc eval --model MODEL --model-dir DIR --dataset DATASET [...]\n\n"
            "The two modes are mutually exclusive. --model-dir is only used by the large-model EvalScope path."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    benchmark_parser = subparsers.add_parser("benchmark", parents=[parent_target, parent_config, parent_device_id, parent_cuda, parent_log], help="Run config-driven benchmark")
    check_parser = subparsers.add_parser("check", parents=[parent_target, parent_layers, parent_device_id, parent_log], help="Check golden data from config or hmm")
    gen_parser = subparsers.add_parser("gen", parents=[parent_target, parent_log], help="Generate default config from ONNX")
    golden_parser = subparsers.add_parser("golden", parents=[parent_target, parent_layers, parent_cuda, parent_log], help="Generate golden data from hmonnx")

    # quant
    quant_hidden_group = quant_parser.add_argument_group("Hidden compatibility options")
    quant_hidden_group.add_argument("--quant_type", type=str, default="w8a8h1_sefp", help=argparse.SUPPRESS)
    quant_hidden_group.add_argument("--enable_layernorm2rmsnorm", action="store_true", help=argparse.SUPPRESS)

    # build
    build_mode_group = build_cmd_parser.add_argument_group("Input mode (choose one)")
    build_exclusive_group = build_mode_group.add_mutually_exclusive_group(required=True)
    build_exclusive_group.add_argument("--config", "-c", type=str, help="Build from YAML config file")
    build_exclusive_group.add_argument("--hmonnx", type=str, help="Build directly from hmonnx file")

    build_output_group = build_cmd_parser.add_argument_group("Direct hmonnx output options")
    build_output_group.add_argument("--hmm_name", type=str, default="model", help="Output hmodel name (only for --hmonnx mode)")
    build_output_group.add_argument("--output", "-o", type=str, default="output", help="Output directory (only for --hmonnx mode)")

    build_llm_group = build_cmd_parser.add_argument_group("LLM build options")
    build_llm_group.add_argument("--flash_attn", type=int, default=0, choices=[0, 1, 2], help="Flash attention optimization: 0=off, 1=graph level, 2=operator level")
    build_llm_group.add_argument("--llm_opt", action="store_true", help="Enable LLM optimization")
    build_llm_group.add_argument("--enable_xh2_stable_output", action="store_true", help="Enable XH2 stable output (prefill faster, decode slower)")
    build_llm_group.add_argument("--llm_batch", type=int, default=1, help="LLM batch size, cannot be set together with --batch (only for --hmonnx mode)")
    build_llm_group.add_argument("--context_length", type=int, default=None, help="Maximum context length for LLM")
    build_llm_group.add_argument("--prefill_length", type=int, default=None, help="Prefill input sequence length (LLM prefill mode)")
    build_llm_group.add_argument("--ndevice", type=int, default=1, choices=[1, 2, 4], help="Number of devices for multi-device inference")
    build_llm_group.add_argument("--is_prefill", action="store_true", help="Build prefill model for LLM")
    build_llm_group.add_argument("--enable_common_subgraph", action="store_true", help="Enable common subgraph")
    build_llm_group.add_argument("--subgraph_repeat_hint", type=int, default=20, help="Hint for number of repeat blocks in the model")

    build_compile_group = build_cmd_parser.add_argument_group("Build execution options")
    build_compile_group.add_argument("--profile", action="store_true", required=False, help="Enable profile")
    build_compile_group.add_argument("--skip_mlir_compile", action="store_true", help="Skip MLIR compile")
    build_compile_group.add_argument("--dump_compiled_mlir", action="store_true", default=False, help="Dump compiled MLIR")
    build_compile_group.add_argument("--skip_check", action="store_true", help="Skip golden check after build")
    build_compile_group.add_argument("--jobs", "-j", type=int, default=None, help="Number of parallel build jobs")
    build_compile_group.add_argument("--upload_dir_name", type=str, help=argparse.SUPPRESS)
    build_compile_group.add_argument("--file_prefix", type=str, help=argparse.SUPPRESS)
    build_compile_group.add_argument("--upload", action="store_true", help=argparse.SUPPRESS)

    # compare
    compare_data_group = compare_parser.add_argument_group("Compare input data")
    compare_data_group.add_argument("--data_path", "-d", type=str, required=True, help="Input data path, image or npz")

    # perf
    perf_mode_group = perf_parser.add_argument_group("Input mode (choose one)")
    perf_exclusive_group = perf_mode_group.add_mutually_exclusive_group(required=True)
    perf_exclusive_group.add_argument("--config", "-c", type=str, help="Run perf from YAML config file")
    perf_exclusive_group.add_argument("--model", "-m", type=str, help="Run perf directly from model path")

    perf_run_group = perf_parser.add_argument_group("Performance run options")
    perf_run_group.add_argument("--warmup", "-wn", type=int, default=1, required=False, help="Warmup iteration count")
    perf_run_group.add_argument("--sample", "-sn", type=int, required=False, default=1, help="Sample iteration count")
    perf_run_group.add_argument("--loop_num", "-ln", type=int, required=False, default=1, help="Loop count per sample")
    perf_run_group.add_argument("--thread", "-tn", type=int, required=False, default=1, help="Thread count")
    perf_run_group.add_argument("--stream", type=int, required=False, default=0, help="Stream count")
    perf_run_group.add_argument("--infer-only", action="store_true", default=False, help="Only perform inference, without data IO")

    # check golden
    check_mode_group = check_parser.add_argument_group("Input mode (choose one)")
    check_exclusive_group = check_mode_group.add_mutually_exclusive_group(required=True)
    check_exclusive_group.add_argument("--config", "-c", type=str, help="Check golden from YAML config file")
    check_exclusive_group.add_argument("--hmm", type=str, help="Check golden directly from hmm file")
    check_data_group = check_parser.add_argument_group("Direct hmm golden options")
    check_data_group.add_argument("--golden", type=str, required=False, help="Golden data directory required with --hmm")

    # gen default config.yaml
    gen_input_group = gen_parser.add_argument_group("Config generation options")
    gen_input_group.add_argument("--onnx", type=str, required=True, help="Input ONNX model path")
    gen_input_group.add_argument("--output", type=str, required=False, default="config.yml", help="Output config YAML path")

    # gen golden
    golden_input_group = golden_parser.add_argument_group("Golden generation options")
    golden_input_group.add_argument("--hmonnx", type=str, required=True, help="Input hmonnx file path")
    golden_input_group.add_argument("--output", type=str, required=True, help="Output golden data directory")
    golden_input_group.add_argument("--data_path", type=str, required=False, help="Optional npz input data path")

    # large-model eval
    llm_eval_group = evaluate_parser.add_argument_group("Large-model EvalScope mode")
    llm_eval_group.add_argument(
        "--model",
        type=str,
        help="Large-model implementation script (.py) or Python module name",
    )
    llm_eval_group.add_argument(
        "--model-dir",
        type=str,
        help="Large-model artifact directory, isolated from config model.model_path",
    )
    llm_eval_group.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help="One or more EvalScope dataset names/paths, e.g. --dataset mmlu gsm8k",
    )
    llm_eval_group.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Large-model evaluation output directory",
    )
    llm_eval_group.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples to evaluate, 0 means full dataset",
    )
    llm_eval_group.add_argument(
        "--model-args",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra large-model args. Repeat for multiple values.",
    )
    # fmt: on
    return parser
