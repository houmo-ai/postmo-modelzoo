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
import json
import logging
import time
import psutil
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
from .cli.parser import build_parser
from .cli.resolve import resolve_command_request


def set_logger(op, log_dir, filename, log_level=logging.INFO):
    """
    Set up a logger that writes to a timestamped log file in the specified directory.

    Args:
        op (str): Operation name used in the log file name
        log_dir (str): Directory where the log file will be saved
        filename (str): Base name of the log file
        log_level (int): Logging level for both console and file (default: logging.INFO)
    """
    # Set logger level (affects all handlers)
    logger.setLevel(log_level)

    # Set console handler level
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(log_level)

    # Add file handler
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(LoggingFormatter())
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)


def main():
    """
    Main entry point for the HMATC tool. Parses command-line arguments and executes the appropriate subcommand.
    """
    parser = build_parser()
    args = parser.parse_args()
    # Parse log level
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "ERROR": logging.ERROR,
        "FATAL": logging.FATAL,
    }
    log_level = log_level_map.get(args.log_level, logging.INFO)
    logger.setLevel(log_level)

    # print version info
    logger.info(
        f"Hmatc version: {__version__}, commit: {__commit__}, build time: {__build_time__}"
    )

    # Set random seed
    set_random_seed(1234)
    # command
    current_command = args.command
    request = resolve_command_request(args, parser)
    if request.kind == "eval.llm":
        from .lm_eval import run_lm_eval

        return run_lm_eval(args)
    if current_command == "quant" and args.enable_layernorm2rmsnorm:
        logger.info("ENABLE_LAYERNORM2RMSNORM = 1")
        os.environ["ENABLE_LAYERNORM2RMSNORM"] = "1"
    # Generate config
    if request.kind == "gen.onnx":
        generate_default_config(args.onnx, args.output)
        logger.info(f"Generate default config done, and save to {args.output}")
        return
    # Process batch model benchmark
    if current_command == "benchmark":
        run_benchmark(args.config, args.device_id)
        return
    # Directly specify model for perf can skip config file
    if request.kind == "perf.model":
        BaseExec.model_perf(
            args.model,
            args.warmup,
            args.sample,
            args.loop_num,
            args.thread,
            args.stream,
            1 if args.batch is None else args.batch,
            args.infer_only,
            devices=[args.device_id],
        )
        return
    # Directly build from hmonnx
    if request.kind == "build.hmonnx":
        # Check subgraph_repeat_hint only works with enable_common_subgraph
        if args.subgraph_repeat_hint != 20 and not args.enable_common_subgraph:
            logger.warning(
                "subgraph_repeat_hint only takes effect when enable_common_subgraph is enabled. "
                "Use --enable_common_subgraph to enable this feature."
            )
        # Check flash_attn requires context_length >= 2048
        if (
            args.flash_attn > 0
            and args.context_length is not None
            and args.context_length < 2048
        ):
            logger.warning(
                "Flash attention disabled: context_length < 2048. "
                "Set --context_length >= 2048 to enable flash attention."
            )
            args.flash_attn = 0
        from .exec.xh2_exec import Xh2Exec as Exec

        Exec.build_from_hmonnx(
            hmonnx=args.hmonnx,
            hmm_name=str(args.hmm_name).lower(),
            output=args.output,
            ncore=1 if args.ncore is None else args.ncore,
            opt_level=2 if args.opt_level is None else args.opt_level,
            batch=1 if args.batch is None else args.batch,
            llm_batch=args.llm_batch,
            enable_profile=args.profile,
            roi_num=1 if args.roi_num is None else args.roi_num,
            flash_attn=args.flash_attn,
            llm_opt=args.llm_opt,
            enable_xh2_stable_output=args.enable_xh2_stable_output,
            context_length=args.context_length,
            prefill_length=args.prefill_length,
            ndevice=args.ndevice,
            is_prefill=args.is_prefill,
            enable_common_subgraph=args.enable_common_subgraph,
            skip_mlir_compile=args.skip_mlir_compile,
            subgraph_repeat_hint=args.subgraph_repeat_hint,
            dump_compiled_mlir=args.dump_compiled_mlir,
            parallel_jobs=(
                psutil.cpu_count(logical=False) if args.jobs is None else args.jobs
            ),
        )
        return
    # Check golden
    if request.kind == "check.hmm":
        from .exec.xh2_exec import Xh2Exec as Exec

        Exec.check_golden_from_hmm(
            args.hmm,
            args.golden,
            device_id=args.device_id,
        )
        return
    # Generate golden
    if request.kind == "golden.hmonnx":
        from .exec.xh2_exec import Xh2Exec as Exec

        Exec.gen_golden(
            args.hmonnx,
            args.output,
            data_path=args.data_path,
            enable_layers=args.layers,
        )
        return
    # Config
    target = args.target
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        logger.fatal("Config file not found")
    cfg = read_yaml_to_dict(cfg_path)
    if cfg is None:
        logger.fatal("Config file is empty")
    cfg_version = cfg.get("version", 1)
    if cfg_version not in [1, 2]:
        logger.fatal("Unsupported config version")
    # Large model
    if cfg_version == 2:
        from .lm_runner import lm_main

        lm_main(args, cfg)
        return
    # ONNX
    if not check_cfg(cfg):
        logger.fatal("Config file error")
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(cfg_path))

    # Update command line arguments to config file
    cfg["target"] = target
    if current_command in ["build", "perf", "compare", "demo", "eval"]:
        batch = args.batch
        ncore = args.ncore
        opt_level = args.opt_level
        roi_num = args.roi_num
        if batch is not None:
            if batch < 1:
                logger.fatal("Batch must be greater than 0")
            elif batch > 1 and roi_num != 1:
                logger.fatal("batch > 1, roi_num must be == 1")
            cfg["build"]["batch"] = batch
        batch = cfg["build"].get("batch", 1)
        if roi_num is not None:
            if roi_num < 1:
                logger.fatal("roi_num must be >= 1")
            elif roi_num > 1 and batch != 1:
                logger.fatal("roi_num > 1, batch must be == 1")
            cfg["build"]["roi_num"] = roi_num
        if opt_level is not None:
            cfg["build"]["opt_level"] = opt_level
        if ncore is not None:
            cfg["build"]["ncore"] = ncore
        if hasattr(args, "jobs") and args.jobs is not None:
            cfg["build"]["parallel_jobs"] = args.jobs
        # Add upload_dir_name for upload directory
        if hasattr(args, "upload_dir_name") and args.upload_dir_name is not None:
            cfg["build"]["upload_dir_name"] = args.upload_dir_name
        # Add file_prefix for compressed file name
        if hasattr(args, "file_prefix") and args.file_prefix is not None:
            cfg["build"]["file_prefix"] = args.file_prefix

    # Set logger with fixed log_dir: ${save_dir}/${target}/logs
    save_dir = cfg.get("model", {}).get("save_dir", "output")
    log_dir = os.path.join(save_dir, target, "logs")
    set_logger(current_command, log_dir, "hmatc", log_level)

    logger.info(f"\n{json.dumps(cfg, indent=2, sort_keys=False)}")

    hm_exec = None
    from .exec.xh2_exec import Xh2Exec

    hm_exec = Xh2Exec(cfg)

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
            HOUMO_DATASETS_PATH = os.environ.get("HOUMO_DATASETS_PATH", "")
            data_path = os.path.join(HOUMO_DATASETS_PATH, data_path)
            if not os.path.exists(data_path):
                logger.fatal(f"{data_path} or {args.data_path} not exists.")
        new_res_info = hm_exec.compare(data_path, args.device_id)
    elif current_command == "perf":
        new_res_info = hm_exec.model_perf(
            hm_exec.hmm_path,
            args.warmup,
            args.sample,
            args.loop_num,
            args.thread,
            args.stream,
            hm_exec.build_batch,
            args.infer_only,
            devices=[args.device_id],
        )
    elif current_command == "demo":
        new_res_info = hm_exec.demo(backend=backend, device_id=args.device_id)
    elif current_command == "eval":
        new_res_info = hm_exec.evaluate(backend=backend, device_id=args.device_id)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
