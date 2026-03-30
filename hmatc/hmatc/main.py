# Copyright 2025 HOUMO AI
#
# File: main.py
# Description:
#     Main entry point for the HMATC tool.
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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import os
import sys
import argparse
import json
import logging
import time
import torch
from ._version import __build_time__, __commit__, __version__
from .base.base_exec import BaseExec
from .utils import logger
from .utils.check import check_cfg
from .utils.logging_format import LoggingFormatter
from .utils.gen_default_config import generate_default_config
from .utils.utils import (
    read_yaml_to_dict,
    set_random_seed,
)
from .utils.benchmark import run_benchmark
from .utils.result_manager import save_result


def set_logger(op, log_dir, filename):
    """
    Set up a logger that writes to a timestamped log file in the specified directory.

    Args:
        op (str): Operation name used in the log file name
        log_dir (str): Directory where the log file will be saved
        filename (str): Base name of the log file
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(LoggingFormatter())
    logger.addHandler(file_handler)


def main():
    """
    Main entry point for the HMATC tool. Parses command-line arguments and executes the appropriate subcommand.
    """
    # fmt: off
    # Parent parser
    target = os.environ.get("HOUMO_TARGET")
    parent_config = argparse.ArgumentParser(add_help=False)
    parent_target = argparse.ArgumentParser(add_help=False)
    parent_onnx = argparse.ArgumentParser(add_help=False)
    parent_hmonnx = argparse.ArgumentParser(add_help=False)
    parent_device_id = argparse.ArgumentParser(add_help=False)
    parent_cuda = argparse.ArgumentParser(add_help=False)
    parent_model_cfg = argparse.ArgumentParser(add_help=False)
    parent_layers = argparse.ArgumentParser(add_help=False)

    # model config
    parent_model_cfg.add_argument("--batch", type=int, required=False, default=1, help="Specify a build batch")
    parent_model_cfg.add_argument("--ncore", type=int, required=False, default=1, choices=(1, 2, 4) if target == "xh1" else (1, 2), help="Specify a ncore")
    parent_model_cfg.add_argument("--opt_level", type=int, required=False, default=2, choices=(0, 1, 2), help="Specify a opt_level")
    parent_model_cfg.add_argument("--roi_num", type=int, required=False, default=1, help="Specify a roi_num")

    parent_config.add_argument("--config", "-c", type=str, required=True, help="config file path")
    parent_target.add_argument("--target", "-t", type=str, required=target not in ["xh1", "xh2"], choices=("xh1", "xh2"), default=target, help="Specify a chip target")
    parent_device_id.add_argument("--device_id", type=int, required=False, default=0, help="Specify a device, running inference on chip")
    parent_hmonnx.add_argument("--hmonnx", action="store_true", help=argparse.SUPPRESS)
    parent_onnx.add_argument("--onnx", action="store_true", help="Specify onnx model as the backend")
    parent_cuda.add_argument("--cuda", action="store_true", help="Enable cuda quantization")
    parent_layers.add_argument("--layers", action="store_true", help="Generate model layers output")

    # main parser
    parser = argparse.ArgumentParser(description="HouMo Model Assist Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="qunat build compare perf demo eval")
    # subparsers
    quant_parser = subparsers.add_parser("quant", parents=[parent_target, parent_config, parent_cuda], help="Quantize a model")
    build_parser = subparsers.add_parser("build", parents=[parent_target, parent_model_cfg, parent_device_id], help="Build a model")
    compare_parser = subparsers.add_parser("compare", parents=[parent_target, parent_config, parent_model_cfg, parent_device_id], help="Compare onnx/hmquant/chip")
    perf_parser = subparsers.add_parser("perf", parents=[parent_target, parent_model_cfg, parent_device_id], help="Test model performance")
    demo_parser = subparsers.add_parser("demo", parents=[parent_target, parent_config, parent_onnx, parent_hmonnx, parent_model_cfg, parent_device_id], help="Run model demo")
    evaluate_parser = subparsers.add_parser("eval", parents=[parent_target, parent_config, parent_onnx, parent_hmonnx, parent_model_cfg, parent_device_id], help="Run model evaluate")
    benchmark_parser = subparsers.add_parser("benchmark", parents=[parent_target, parent_config, parent_device_id, parent_cuda], help="Run model benchmark")
    check_parser = subparsers.add_parser("check", parents=[parent_target, parent_layers, parent_device_id], help="Check model golden")
    gen_parser = subparsers.add_parser("gen", parents=[parent_target], help="Generate default config.yaml")
    golden_parser = subparsers.add_parser("golden", parents=[parent_target, parent_layers, parent_cuda], help="Generate golden data")
    
    # quant
    quant_parser.add_argument("--quant_type", type=str, default="w8a8h1_sefp", help=argparse.SUPPRESS)
    quant_parser.add_argument("--enable_layernorm2rmsnorm", action="store_true", help=argparse.SUPPRESS)
    
    # build
    build_parser.add_argument("--profile", action="store_true", required=False, help="Enable profile")
    build_exclusive_group = build_parser.add_mutually_exclusive_group(required=True)
    build_exclusive_group.add_argument("--config", "-c", type=str, help="Specify config file path")
    build_exclusive_group.add_argument("--hmonnx", type=str, help="Specify hmonnx file path")
    build_parser.add_argument("--hmm_name", type=str, help="Specify hmodel name" if "--hmonnx" in sys.argv else argparse.SUPPRESS)
    build_parser.add_argument("--output", "-o", type=str, default="output", help="Specify output path" if "--hmonnx" in sys.argv else argparse.SUPPRESS)
    build_parser.add_argument("--flash_attn", type=int, default=0, choices=[0, 1, 2], help="flash attention optimization")
    build_parser.add_argument("--llm_opt", action="store_true", help="enable llm optimization")
    build_parser.add_argument("--enable_common_subgraph", action="store_true", help="enable common subgraph")
    build_parser.add_argument("--skip_mlir_compile", action="store_true", help="skip mlir compile")
    build_parser.add_argument("--subgraph_repeat_hint", type=int, default=20, help="A hint for number of repeat blocks in the model")
    build_parser.add_argument("--upload_dir_name", type=str, help=argparse.SUPPRESS)
    build_parser.add_argument("--file_prefix", type=str, help=argparse.SUPPRESS)
    build_parser.add_argument("--skip_check", action="store_true", help="Skip check golden after build")
    build_parser.add_argument("--upload", action="store_true", help=argparse.SUPPRESS)
        
    # compare
    compare_parser.add_argument("--data_path", "-d", type=str, required=True, help="Specify a data path, image or npz")
    
    # perf
    perf_exclusive_group = perf_parser.add_mutually_exclusive_group(required=True)
    perf_exclusive_group.add_argument("--config", "-c", type=str, help="Specify config file path")
    perf_exclusive_group.add_argument("--model", "-m", type=str, help="Specify model path")
    perf_parser.add_argument("--warmup", "-wn", type=int, default=1, required=False, help="Specify warmup num")
    perf_parser.add_argument("--sample", "-sn", type=int, required=False, default=1, help="Specify sample num")
    perf_parser.add_argument("--loop_num", "-ln", type=int, required=False, default=1, help="Specify loop num")
    perf_parser.add_argument("--thread", "-tn", type=int, required=False, default=1, help="Specify thread num")
    perf_parser.add_argument("--stream", type=int, required=False, default=0, help="Specify stream num")

    # check golden
    check_exclusive_group = check_parser.add_mutually_exclusive_group(required=True)
    check_exclusive_group.add_argument("--config", "-c", type=str, help="Specify config file path")
    check_exclusive_group.add_argument("--hmm", type=str, help="Specify hmm file path")
    check_parser.add_argument("--golden", type=str, required="--hmm" in sys.argv, help="Specify golden data path")
    
    # gen default config.yaml
    gen_parser.add_argument("--onnx", type=str, required=True, help="Specify a onnx")
    gen_parser.add_argument("--output", type=str, required=False, default="config.yml", help="Specify a config.yml")
    
    # gen golden
    golden_parser.add_argument("--hmonnx", type=str, required=True, help="Specify a hmonnx file")
    golden_parser.add_argument("--output", type=str, required=True, help="Specify a output")
    golden_parser.add_argument("--data_path", type=str, required=False, help="Specify a npz file")

    args = parser.parse_args()
    # print version info
    logger.info(f"Hmatc version: {__version__}, commit: {__commit__}, build time: {__build_time__}")
    # fmt: on

    # Set random seed
    set_random_seed(1234)
    # command
    current_command = args.command

    if current_command == "quant" and args.enable_layernorm2rmsnorm:
        logger.info("ENABLE_LAYERNORM2RMSNORM = 1")
        os.environ["ENABLE_LAYERNORM2RMSNORM"] = "1"

    # Generate config
    if current_command == "gen":
        generate_default_config(args.onnx, args.output_path)
        logger.info(f"Generate default config done, and save to {args.output_path}")
        return

    # Process batch model benchmark
    if current_command == "benchmark":
        run_benchmark(args.config, args.target, args.device_id, args.cuda)
        return

    # Directly specify model for perf can skip config file
    if current_command == "perf" and args.model is not None:
        new_res_info = BaseExec.model_perf(
            args.model,
            args.warmup,
            args.sample,
            args.loop_num,
            args.thread,
            args.stream,
            devices=[args.device_id],
        )
        # Save result to model directory
        model_dir = os.path.dirname(args.model)
        result_path = os.path.join(model_dir, "result.yml")
        save_result(result_path, new_res_info, "", target)
        return
    # Directly build from hmonnx
    if current_command == "build" and args.hmonnx is not None:
        if target == "xh1":
            raise NotImplementedError("xh1 not support build from hmonnx yet")
        elif target == "xh2":
            from .exec.xh2_exec import Xh2Exec as Exec
        Exec.build_from_hmonnx(
            args.hmonnx,
            args.hmm_name,
            args.output,
            args.ncore,
            args.opt_level,
            args.batch,
            args.profile,
            args.roi_num,
            args.flash_attn,
            args.llm_opt,
            args.enable_common_subgraph,
            args.skip_mlir_compile,
            args.subgraph_repeat_hint,
        )
        return
    # Check golden
    if current_command == "check" and args.hmm is not None:
        if target == "xh1":
            raise NotImplementedError("xh1 not support check golden from hmm")
        elif target == "xh2":
            from .exec.xh2_exec import Xh2Exec as Exec
        Exec.check_golden_from_hmm(
            args.hmm, args.golden, enable_layers=False, device_id=args.device_id
        )
        return

    # Generate golden
    if current_command == "golden":
        if target == "xh1":
            raise NotImplementedError("xh1 not support check golden from hmm")
        elif target == "xh2":
            from .exec.xh2_exec import Xh2Exec as Exec
        Exec.gen_golden(
            args.hmonnx,
            args.output,
            data_path=args.data_path,
            enable_layers=args.layers,
        )
        return

    target = args.target
    # set_logger(current_command, "log", "config")
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        logger.error("Config file not found")
        return
    cfg = read_yaml_to_dict(cfg_path)
    if not check_cfg(cfg):
        logger.error("Config file error")
        return

    # Update command line arguments to config file
    cfg["target"] = target
    if current_command in ["build", "perf", "compare", "demo", "eval"]:
        batch = args.batch
        ncore = args.ncore
        opt_level = args.opt_level
        roi_num = args.roi_num
        if batch < 1:
            logger.error("Batch must be greater than 0")
            return
        if batch > 1:
            cfg["build"]["batch"] = batch
        if batch > 1 and roi_num != 1 and target == "xh1":
            logger.error("batch > 1, roi_num must be == 1")
            return
        if batch == 1 and roi_num < 1 and target == "xh1":
            logger.error("batch == 1, roi_num must be >= 1")
            return
        if roi_num > 1 and target == "xh1":
            cfg["build"]["roi_num"] = roi_num
        if opt_level is not None:
            cfg["build"]["opt_level"] = opt_level
        if ncore is not None:
            if ncore == 4 and target == "xh2":
                logger.error("ncore == 4, target must be xh1")
                return
            cfg["build"]["ncore"] = ncore
        # Add upload_dir_name for upload directory
        if hasattr(args, "upload_dir_name") and args.upload_dir_name is not None:
            cfg["build"]["upload_dir_name"] = args.upload_dir_name
        # Add file_prefix for compressed file name
        if hasattr(args, "file_prefix") and args.file_prefix is not None:
            cfg["build"]["file_prefix"] = args.file_prefix
    logger.info(f"\n{json.dumps(cfg, indent=2, sort_keys=False)}")

    hm_exec = None
    if target == "xh1":
        from .exec.xh1_exec import Xh1Exec

        hm_exec = Xh1Exec(cfg)
    elif target == "xh2":
        from .exec.xh2_exec import Xh2Exec

        hm_exec = Xh2Exec(cfg)
    else:
        logger.error(f"Not support target: {target}")
        return

    new_res_info = dict()
    backend = target
    # Update onnx backend
    if current_command in ["demo", "eval"]:
        if args.onnx:
            backend = "onnx"
        elif args.hmonnx:
            backend = "hmonnx"
    # Execute the corresponding command
    if current_command == "quant":
        if args.cuda and target == "xh1" and torch.cuda.is_available():
            hm_exec.device = "cuda"
        new_res_info = hm_exec.quantize()
    elif current_command == "build":
        hm_exec.enable_upload = args.upload
        new_res_info = hm_exec.build(
            enable_profile=args.profile,
            upload_dir_name=getattr(args, "upload_dir_name", None),
            file_prefix=getattr(args, "file_prefix", None),
        )
        logger.info(f"Build {hm_exec.model_name} done.")
        # Merge check_golden outputs into build result (skip if --skip_check)
        if not args.skip_check:
            check_result = hm_exec.check_golden(args.device_id)
            if "outputs" in check_result and "build" in new_res_info:
                new_res_info["build"]["outputs"] = check_result["outputs"]
    elif current_command == "check":
        new_res_info = hm_exec.check_golden(args.device_id, args.layers)
    elif current_command == "compare":
        data_path = args.data_path
        if not os.path.exists(data_path):
            # If not exists, look in HOUMO_DATASETS_PATH environment variable
            HOUMO_DATASETS_PATH = os.environ.get(
                "HOUMO_DATASETS_PATH", "/usr/local/src/houmo-modelzoo/data/datasets"
            )
            data_path = os.path.join(HOUMO_DATASETS_PATH, data_path)
            if not os.path.exists(data_path):
                logger.error(f"{data_path} or {args.data_path} not exists.")
                exit(-1)
        new_res_info = hm_exec.compare(data_path, args.device_id)
    elif current_command == "perf":
        new_res_info = hm_exec.model_perf(
            hm_exec.hmm_path,
            args.warmup,
            args.sample,
            args.loop_num,
            args.thread,
            args.stream,
            devices=[args.device_id],
        )
    elif current_command == "demo":
        new_res_info = hm_exec.demo(backend=backend, device_id=args.device_id)
    elif current_command == "eval":
        new_res_info = hm_exec.evaluate(backend=backend, device_id=args.device_id)
    else:
        raise NotImplementedError

    # Save result to {save_dir}/{target}/result.yml
    if hm_exec and hasattr(hm_exec, "save_dir") and hm_exec.save_dir:
        result_path = os.path.join(hm_exec.save_dir, target, "result.yml")
        save_result(result_path, new_res_info, hm_exec.model_name, target)


if __name__ == "__main__":
    main()
