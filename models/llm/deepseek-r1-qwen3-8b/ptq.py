import torch, argparse, os
import os.path as osp
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '')
HOUMO_TARGET = os.getenv('HOUMO_TARGET', '')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1',""):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import shutil
from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen2 import Qwen2ConvertConfig

from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

def parse_args():
    parser = argparse.ArgumentParser(description="Quant DeepSeek-R1-Qwen3")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--context-length", type=int, default=8192, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    return args

def main(args):
    hf_model_path = args.model
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    config = Qwen2ConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
    )

    prefix = f"xh2"
    work_dir = Path("output") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "Qwen3ForCausalLM_legacy", config, str(work_dir))
    shutil.move(work_dir / "hmonnx/prefill", work_dir / "hmquant/prefill")
    shutil.move(work_dir / "hmquant/prefill/DeepSeek-R1-0528-Qwen3-8B-XH2a-8k-w8a8_sefp_prefill.onnx", work_dir / "hmquant/prefill/hmquant_deepseek_with_act.onnx")
    shutil.move(work_dir / "hmonnx/decode", work_dir / "hmquant/decoder")
    shutil.move(work_dir / "hmquant/decoder/DeepSeek-R1-0528-Qwen3-8B-XH2a-8k-w8a8_sefp_decode.onnx", work_dir / "hmquant/decoder/hmquant_deepseek_with_act.onnx")
    shutil.move(work_dir / "token_embedding.pt", work_dir / "hmquant/quant_embedding.pt")
    shutil.rmtree(work_dir / "hmonnx")
    shutil.rmtree(work_dir / "hf_config")

if __name__ == "__main__":
    args = parse_args()
    main(args)