import torch, argparse,os
import hmquant.llm.llm_utils  as utils
from hmquant.llm.quant_pipeline.qwen_quant_pipline import QwenQuantPipline
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASETS_PATH = os.getenv('DATASETS_PATH', '')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1',""):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Quant Qwen2")
    parser.add_argument("--model", type=str, default="qwen2-7b-instruct-hf", help="path to model")
    # 1. quant
    parser.add_argument("--quant_config", type=str, default="quant_config.py")
    parser.add_argument("--n_calib", type=int, default=16)
    # 2. export
    parser.add_argument("--model_name", type=str, default="qwen2")
    parser.add_argument("--save_path",type=str,default="output/houmo/result")
    parser.add_argument("--prefill_shape", type=int, nargs='+', default=[4, 32], help="List of integers for prefill shape")
    parser.add_argument("--cache_len", type=int, default=4096)
    parser.add_argument("--multi_batch",action="store_true",default=False,help="weather use multi batch for export")
    # 3. others
    parser.add_argument("--wikitext_local",type=str,default="demo/ptq/qwen/wikitext-2-raw-v1",help="if has local wikitext, set it here")
    parser.add_argument("--eval_ppl",action="store_true",default=False)

    """  args below are for debug, please not used """
    parser.add_argument("--blocks", default=28, type=int)
    parser.add_argument("--decoder_shape", type=int, nargs='+', default=[1, 1], help="List of integers for decoder shape")
    parser.add_argument("--gptq",type=str2bool,default=True,help="weather use gptq to quant weight") # boost precision
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


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = AutoModelForCausalLM.from_pretrained(args.model), AutoTokenizer.from_pretrained(args.model)
    quant_pipline = QwenQuantPipline()
    # 1. quant model
    qmodel = quant_pipline.quant_llm(model, tokenizer, args=args)
    if args.eval_ppl:
        utils.eval_ppl(qmodel, tokenizer, disk_file=args.wikitext_local)
        ques_res = qmodel.stream_chat(tokenizer,"hello")
    # 2. export model
    quant_pipline.export_llm(qmodel, tokenizer, args, save_path=args.save_path,model_name=args.model_name)
