# Copyright 2025 HOUMO AI
#
# File: execute_compilation.py
# Description:
#   Execute LLM model compilation flow.
#
#   This script provides command-line interface for compiling LLM models with various configurations.
#   It handles model compilation, optimization, and post-processing steps including optional model
#   stripping for shared weight removal.
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

import os
import glob
import shutil
import argparse
import logging
from compiler_utils import setup_logging, execute_cmd


script_dir = os.path.dirname(os.path.abspath(__file__))
setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the compilation process."""
    parser = argparse.ArgumentParser(description="Compile LLMs")
    parser.add_argument(
        "task_id",
        nargs="?",
        default=None,
        help="Task ID (optional positional argument)",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        type=str,
        help="(required) model path, example: models/llm/qwen3",
    )
    parser.add_argument(
        "-qm",
        "--quant_model_path",
        type=str,
        default="",
        help="Quantized model path",
    )
    parser.add_argument(
        "-cl",
        "--context_length",
        type=int,
        default=0,
        help="Context length, default is 0.",
    )
    parser.add_argument(
        "-dn",
        "--device_num",
        type=int,
        default=-1,
        help="The number of device, default is -1.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=0,
        help="batch number, default is 0.",
    )
    parser.add_argument(
        "-cn",
        "--core_num",
        type=int,
        default=1,
        help="The number of core, default is 1.",
    )
    parser.add_argument(
        "-j",
        "--j",
        type=int,
        default=0,
        help="build parallel jobs.",
    )
    parser.add_argument(
        "--flash_attention",
        type=int,
        choices=[0, 1, 2, 3],
        help="flash attention optimization",
    )
    parser.add_argument(
        "-r",
        "--result_dir",
        type=str,
        default="./",
        help="The path for storing the results.",
    )
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="./execute_compilation_log.log",
        help="The path of log.",
    )
    parser.add_argument(
        "--strip",
        type=str,
        default="off",
        choices=["off", "overwrite", "copy"],
        help="Strip shared weights from the last input model",
    )

    args = parser.parse_args()
    return args


def _check_golden(dir_path: str) -> bool:
    """
    Check if golden files exist in the specified directory.

    Args:
        dir_path (str): Directory path to check for golden files

    Returns:
        bool: True if both decoder and prefill golden files exist, False otherwise
    """
    import glob

    decoder_golden = list(glob.glob(dir_path + "/decoder/hmquant_*.npy"))
    prefill_golden = list(glob.glob(dir_path + "/prefill/hmquant_*.npy"))

    if len(decoder_golden) > 0 and len(prefill_golden) > 0:
        return True
    return False


def _generate_cmds(args) -> list:
    """
    Generate compilation command line based on input arguments.

    Args:
        args: Parsed command line arguments containing model configuration

    Returns:
        list: List of command line arguments for the build process
    """

    model_name = "deepseek" if "deepseek" in args.model_name else args.model_name

    # stage = "all" if _check_golden(args.quant_model_path) else "build"
    stage = "build"
    cmds = [
        "python3",
        "build.py",
        "--stage",
        stage,
        "--model_dir",
        args.quant_model_path,
        "--model_name",
        model_name,
        "--ncore",
        str(args.core_num),
        "--output_dir",
        args.result_dir,
    ]

    if args.batch > 0:
        cmds += ["--batch", str(args.batch)]
    if args.context_length > 0:
        cmds += ["--context_length", str(args.context_length)]
    if args.device_num >= 0:
        cmds += ["--ndevice", str(args.device_num)]
    if args.j > 0:
        cmds += ["--j", str(args.j)]
    if args.flash_attention and args.flash_attention in [0, 1, 2, 3]:
        cmds += ["--flash_attention", str(args.flash_attention)]

    return cmds


def _strip_models(output_dir: str, strip: str) -> None:
    """
    Use hmmstrip command line to strip hmm models by removing shared weights.

    Args:
        output_dir (str): Directory containing the compiled HMM model files
        strip (str): Strip mode - "copy" (backup original), "overwrite" (replace in place), or "off"
    """

    hmm_files = glob.glob(os.path.join(output_dir, "*.hmm")) + glob.glob(
        os.path.join(output_dir, "*.hmms")
    )
    strip_cmds = []
    for hmm_path in hmm_files:
        strip_cmd = [
            "hmmstrip",
            "--strip",
        ]
        if "_prefill" not in hmm_path:
            continue

        decode_hmm_path = hmm_path.replace("_prefill", "_decode")
        if not os.path.exists(decode_hmm_path):
            decode_hmm_path = hmm_path.replace("_prefill", "_decoder")
            if not os.path.exists(decode_hmm_path):
                continue

        if strip == "copy":
            ori_backup = f"{decode_hmm_path}.ori"
            shutil.copy2(decode_hmm_path, ori_backup)
            logger.info(f"Copy {decode_hmm_path} -> {ori_backup}")
        strip_cmd += [
            "-o",
            decode_hmm_path,
            "-i",
            hmm_path,
            decode_hmm_path,
        ]
        strip_cmds.append(strip_cmd)
    for cmd in strip_cmds:
        logger.info(f"Ready to strip hmm: {cmd}")
        ret = execute_cmd(cmd, args.log_file)


def main(args) -> int:
    """
    Main function to execute the model compilation process.

    This function orchestrates the complete compilation pipeline including:
    - Setting up the working environment
    - Generating compilation commands
    - Executing the build process
    - Post-processing steps including copying embedding files
    - Optional model stripping for shared weights

    Args:
        args: Parsed command line arguments containing model configuration

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    logger.info(
        "Model name: %s, Model path: %s, Quant model path: %s, Context length: %d, "
        "Batch: %d, Device num: %d, Core Num: %d, J: %d, Flash attention: %s, HmmStrip: %s, "
        "Result Dir: %s",
        args.model_name,
        args.model_path,
        args.quant_model_path,
        args.context_length,
        args.batch,
        args.device_num,
        args.core_num,
        args.j,
        args.flash_attention,
        args.strip,
        args.result_dir,
    )

    model_dir = f"{script_dir}/../../" + args.model_path
    os.chdir(model_dir)
    logger.info("Current dir: %s", os.getcwd())

    output_dir = args.result_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except FileNotFoundError as e:
        logger.error(f"error create folder failed: {e}")
        return -1

    os.environ["HDPL_PLATFORM"] = "ISIM"

    cmds = _generate_cmds(args)
    ret = execute_cmd(cmds, args.log_file)
    if not ret:
        return 1

    pt_pattern = f"{args.quant_model_path}/*.pt"
    embed_files = glob.glob(pt_pattern)
    if embed_files and len(embed_files) > 0:
        embed_dir = f"{output_dir}/hmquant"
        os.makedirs(embed_dir, exist_ok=True)
        cp_cmds = ["cp", "-a"] + embed_files + [f"{embed_dir}/"]
        execute_cmd(
            cp_cmds,
            args.log_file,
        )

    HOUMO_PATH = os.getenv("HOUMO_PATH", None)
    strip = args.strip
    if (
        strip in ["overwrite", "copy"]
        and HOUMO_PATH is not None
        and os.path.exists(f"{HOUMO_PATH}/bin/hmmstrip")
    ):
        _strip_models(output_dir, strip)

    return 0


if __name__ == "__main__":
    args = parse_args()

    ret = main(args)
    logger.info("Ret: %d", ret)

    exit(ret)
