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

if HOUMO_TARGET == 'xh1':
    import hmquant.llm.llm_utils  as utils
    from hmquant.llm.llm_api import QwenQuantPipline

    def parse_args():
        parser = argparse.ArgumentParser(description="Quant Qwen3")
        parser.add_argument("--model", type=str, default="qwen3-14b", help="path to model")
        # 1. quant
        parser.add_argument("--quant_config", type=str, default="quant_config.py")
        parser.add_argument("--n_calib", type=int, default=16)
        # 2. export
        parser.add_argument("--model_name", type=str, default="qwen3")
        parser.add_argument("--save_path",type=str,default=f"output/{HOUMO_TARGET}/hmquant")
        parser.add_argument("--prefill_shape", type=int, nargs='+', default=[4, 64], help="List of integers for prefill shape")
        parser.add_argument("--cache_len", type=int, default=8192)
        parser.add_argument("--multi_batch",action="store_true",default=False,help="weather use multi batch for export")
        # 3. others
        parser.add_argument("--wikitext_local",type=str,default=os.path.join(HOUMO_DATASETS_PATH,"wikitext-2-raw-v1"),help="if has local wikitext, set it here")
        parser.add_argument("--eval_ppl",action="store_true",default=False)

        """  args below are for debug, please not used """
        parser.add_argument("--blocks", default=36, type=int)
        parser.add_argument("--decoder_shape", type=int, nargs='+', default=[1, 1], help="List of integers for decoder shape")
        parser.add_argument("--gptq",type=str2bool,default=False,help="weather use gptq to quant weight") # boost precision
        parser.add_argument("--cache_2_input",type=str2bool,default=True)
        parser.add_argument("--rotate_ov",type=str2bool,default=True,help="weather rotate o_proj and v_proj")
        parser.add_argument("--rotate_pre_rope",type=str2bool,default=False,help="weather rotate acts before rope")
        parser.add_argument("--rotate_post_rope",type=str2bool,default=False,help="weather rotate acts after rope")
        parser.add_argument("--use_klt",type=str2bool,default=True,help="weather use klt for rotation")
        parser.add_argument("--compile_mode",type=str2bool,default=False,help="weather show convert err")
        """  args above are for debug, please not used """
        args = parser.parse_args()
        if args.multi_batch:
            args.decoder_shape = [4,1]
        else:
            args.decoder_shape = [1,1]
        return args

    def main(args):
        args = parse_args()
        model, tokenizer = AutoModelForCausalLM.from_pretrained(args.model), AutoTokenizer.from_pretrained(args.model)
        quant_pipline = QwenQuantPipline()
        # 1. quant model
        qmodel = quant_pipline.quant_llm(model, tokenizer, args=args)
        if args.eval_ppl:
            utils.eval_ppl(qmodel, tokenizer, disk_file=args.wikitext_local)
            ques_res = qmodel.stream_chat(tokenizer,"hello")
        # 2. export model
        quant_pipline.export_llm(qmodel, tokenizer, args)

        # 3. chat
        if not args.cache_2_input:
            while True:
                prompt = input("\n你的问题：")
                prompt = prompt.replace("\\n", "\n")
                quant_pipline.chat(prompt,args)

        # 4. generate golden
        quant_pipline.generate_golden(args, save_path=args.save_path, model_name=args.model_name)

elif HOUMO_TARGET == 'xh2':
    import shutil
    from xh_model_zoo.xh_llm import LLMConverter
    from xh_model_zoo.xh_llm.models.qwen2 import Qwen2ConvertConfig

    from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
    from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
    from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

    def parse_args():
        parser = argparse.ArgumentParser(description="Quant Qwen3")
        parser.add_argument("--debug", action="store_true", help="debug mode")
        parser.add_argument("--model", type=str, default="qwen3-14b")
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
        shutil.move(work_dir / "hmquant/prefill/qwen3-14b-XH2a-8k-w8a8_sefp_prefill.onnx", work_dir / "hmquant/prefill/hmquant_qwen3_with_act.onnx")
        shutil.move(work_dir / "hmonnx/decode", work_dir / "hmquant/decoder")
        shutil.move(work_dir / "hmquant/decoder/qwen3-14b-XH2a-8k-w8a8_sefp_decode.onnx", work_dir / "hmquant/decoder/hmquant_qwen3_with_act.onnx")
        shutil.move(work_dir / "token_embedding.pt", work_dir / "hmquant/quant_embedding.pt")
        shutil.rmtree(work_dir / "hmonnx")
        shutil.rmtree(work_dir / "hf_config")

if __name__ == "__main__":
    args = parse_args()
    main(args)