import torch
import shutil
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import os
import os.path as osp
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen2 import Qwen2ConvertConfig
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title

def quant_llm(args):
    out_dir = Path(args.work_dir)
    hf_model_dir = args.model

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    cfg_name = model_name
    if not args.skip_quarot:
        cfg_name += "_quarot"
    if not args.skip_gptq:
        cfg_name += "_gptq"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    config = AutoConfig.from_pretrained(hf_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    # # for qwen2-0.5b 1.5b & 3b
    # if args.tie_embed_head:
    #     config.tie_word_embeddings = False

    native_model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

    native_model.eval()
    native_model.to(dtype)

    quant_methods = []
    if not args.skip_quarot:
        torch.cuda.reset_peak_memory_stats()
        quant_methods.append("quarot")
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        if not os.path.exists(filename):
            from xh_model_zoo.xh_llm.quarot.quantizer_utils import quarot

            logger.info(msg_output_format("Start quarot quantization"))
            native_model = quarot(native_model, device=device)
            logger.info(msg_output_format("End quarot quantization"))

            state_dict = native_model.state_dict()
            logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
            # torch.save(state_dict, filename)
            save_safetensors_file(state_dict, filename)
            logger.info(f"Save checkpoint to: {filename}")
        else:
            state_dict = load_safetensors_file(filename)
            native_model.load_state_dict(state_dict)
            logger.info(msg_output_format(f"Load state_dict from {filename}"))

        consumption = torch.cuda.max_memory_allocated()
        unit = "B"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "k"
            if consumption > 1024:
                consumption = consumption / 1024
                unit = "M"
            consumption = round(consumption, 2)
        logger.info(f"GPU memory cost for export {consumption}{unit}")

    if not args.skip_gptq:
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

        gptq_config = dict(
            calib_dataset="wikitext2",
            calib_samples=128,
            seqlen=2048,
            w_clip=True,
            w_bits=args.w_bits,
            w_asym=False,
            w_groupsize=64,
            percdamp=0.01,
            act_order=False,
            int8_down_proj=False,
            heading_gptq=True,
        )

        torch.cuda.reset_peak_memory_stats()
        logger.info(msg_output_format("Start gptq quantization"))
        quant_methods.append("gptq")
        layers_cache_dir = work_dir / "layers_cache"
        layers_cache_dir.mkdir(exist_ok=True, parents=True)

        native_model = gptq(
            native_model,
            args=args,
            model_name=hf_model_dir,
            **gptq_config,
            device=device,
            layers_cache_dir=str(layers_cache_dir),
        )
        logger.info(msg_output_format("End gptq quantization"))

        consumption = torch.cuda.max_memory_allocated()
        unit = "B"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "k"
            if consumption > 1024:
                consumption = consumption / 1024
                unit = "M"
            consumption = round(consumption, 2)
        logger.info(f"GPU memory cost for export {consumption}{unit}")

    if len(quant_methods) != 0:
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        state_dict = native_model.state_dict()
        # del native_model
        for k in tqdm(state_dict):
            paths = k.split(".")
            v = state_dict[k]
            if paths[-1] == "quant_weight":
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
            else:
                v = v.to(torch.float16)

            state_dict[k] = v
        logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
        save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")

def export_llm(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    model_name = os.path.basename(args.model)
    model_dir = os.path.join(args.work_dir, "{}_quarot_gptq".format(model_name))
    quant_weight = os.path.join(model_dir, "quarot_gptq-state-dict.safetensors")
    config = Qwen2ConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=quant_weight,
    )

    prefix = f"{model_name}-{target_device}-{args.context_length//1024}k-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        if "qwen3" in args.model.lower():
            architecture = "Qwen3ForCausalLM_legacy"
        elif "qwen2" in args.model.lower():
            architecture = "Qwen2ForCausalLM_legacy"
        else:
            raise ValueError("Unsupported model architecture") 
        LLMConverter.from_pretrained(hf_model_path, architecture, config, str(work_dir))

def move_models(work_dir: Path, source: str = "prefill", model : str = "prefill", target_name: str = "hmquant_qwen2.5_with_act.onnx"):
    source_dir = work_dir / "hmquant/{}".format(source)
    matched_files = list(source_dir.glob("*{}.onnx".format(model)))
    
    if not matched_files:
        raise FileNotFoundError(f"未找到匹配的onnx文件于 {source_dir}")
    
    target_path = source_dir / target_name
    if target_path.exists():
        target_path.unlink()
    
    shutil.move(matched_files[0], target_path)
    return target_path

def format_number(n):
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}m"
    elif n >= 1024:
        return f"{n // 1024}k"
    else:
        return f"0k"

def move_llm(args):
    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    model_name = os.path.basename(args.model)
    hmm_model_dir = "{}-XH2a-{}-{}".format(model_name, format_number(args.context_length), args.quant_type)
    logger.info(msg_output_format("Start move from {} to {}").format(work_dir / hmm_model_dir, dest_dir))
    shutil.move(work_dir / hmm_model_dir / "hmonnx/prefill", dest_dir / "hmquant/prefill")
    move_models(dest_dir, "prefill")
    shutil.move(work_dir / hmm_model_dir / "hmonnx/decode", dest_dir / "hmquant/decoder")
    move_models(dest_dir, "decoder", "decoder")
    shutil.move(work_dir / hmm_model_dir / "token_embedding.pt", dest_dir / "hmquant/quant_embedding.pt")
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir)