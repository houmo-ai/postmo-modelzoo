import os
import os.path as osp
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from loguru import logger
from qwen_vl_utils import process_vision_info
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3_VLConvertConfig, VisualConfig

from xhquant.api import (
    DeviceType,
    xhquant_init,
    QuantScheme,
    get_root_logger,
)  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def robust_file_copy(src_file, dst_path):
    try:
        # Check if source file exists
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Error: Source file '{src_file}' does not exist.")

        dst_dir = os.path.dirname(dst_path)
        os.makedirs(
            dst_dir, exist_ok=True
        )  # Create target directory if it doesn't exist

        # Perform copy operation
        shutil.copy2(src_file, dst_path)
        print(f"Operation successful! File copied to: {dst_path}")

    except PermissionError:
        print(
            f"Error: Permission denied, cannot write to target location '{dst_path}'."
        )
    except OSError as e:
        print(f"Error: Operating system error - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def quant_llm(args):
    out_dir = Path(args.work_dir)
    hf_model_dir = args.model

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    cfg_name = model_name
    cfg_name += "_quarot"
    cfg_name += "_gptq"

    cfg_name += f"_transformers-{transformers.__version__}"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    config = AutoConfig.from_pretrained(hf_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    native_model: nn.Module = AutoModelForImageTextToText.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

        native_model.config.text_config.tie_word_embeddings = False

    native_model.eval()
    native_model.to(dtype)

    processor = AutoProcessor.from_pretrained(hf_model_dir)

    quant_methods = []

    torch.cuda.reset_peak_memory_stats()
    quant_methods.append("quarot")
    quant_name = "_".join(quant_methods)
    filename = work_dir / f"{quant_name}-state-dict.safetensors"

    if not os.path.exists(filename):
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import quarot

        logger.info(msg_output_format("Start quarot quantization"))
        native_model = quarot(native_model, device=device)
        logger.info(msg_output_format("End quarot quantization"))
        native_model.to(torch.float16)
        state_dict = native_model.state_dict()
        logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
        # torch.save(state_dict, filename)
        save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")
    else:
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import rotation_utils

        rotation_utils.fuse_layer_norms(native_model)
        state_dict = load_safetensors_file(filename)
        native_model.to(torch.float16)
        native_model.load_state_dict(state_dict)
        logger.info(msg_output_format(f"Load state_dict from {filename}"))

    # if args.demo:
    #     demo(native_model, processor, args.image_dir)

    from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

    gptq_config = dict(
        calib_dataset=args.calib_dataset,
        calib_samples=args.calib_samples,
        seqlen=2048,
        w_clip=True,
        w_bits=args.w_bits,
        w_asym=False,
        w_groupsize=64,
        percdamp=0.01,
        act_order=False,
        int8_down_proj=False,
        heading_gptq=args.heading_gptq,
        w_head_bits=args.w_head_bits,
    )

    torch.cuda.reset_peak_memory_stats()
    logger.info(msg_output_format("Start gptq quantization"))
    quant_methods.append("gptq")
    quant_methods.append(f"use_hession_mse_{args.use_hession_mse}")
    quant_methods.append(f"calib_samples_{args.calib_samples}")
    quant_methods.append(f"heading_gptq_{args.heading_gptq}")
    layers_cache_dir = work_dir / "layers_cache"
    layers_cache_dir.mkdir(exist_ok=True, parents=True)

    native_model = gptq(
        native_model,
        args=args,
        model_name=hf_model_dir,
        **gptq_config,
        device=device,
        processor=processor,
        data_files=args.data_files,
        is_qwen3_vl=True,
        use_hession_mse=args.use_hession_mse,
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


def find_files_pathlib(directory):
    path = Path(directory)
    target_files = [
        f.name
        for f in path.iterdir()
        if f.is_file() and "gptq" in f.name and "quarot" in f.name
    ]
    return target_files


def export_llm(args):
    from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3VLForConditionalGeneration

    robust_file_copy("../../../data/pic/beach.jpeg", "data/images/qwen2_vl_demo.jpeg")
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    ops = dict(
        MatMul=dict(
            act_scheme=dict(
                bits=8,
                fp_mode="sefp",
            ),
            act_schema_2=dict(
                bits=16,
                fp_mode="sefp",
            ),
        )
    )
    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=quant_type, ops=ops
    )
    model_name = os.path.basename(args.model)
    model_dir = os.path.join(args.work_dir, "{}_quarot_gptq".format(model_name))
    quant_weight_name = find_files_pathlib(
        "{}_transformers-{}".format(model_dir, transformers.__version__)
    )[0]
    quant_weight = os.path.join(
        "{}_transformers-{}".format(model_dir, transformers.__version__),
        quant_weight_name,
    )

    native_model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        # config=config,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

    native_model.eval()
    native_model.to(torch.bfloat16)

    processor = AutoProcessor.from_pretrained(hf_model_path)

    # demo(native_model, processor)
    config = Qwen3_VLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        quant_weight=quant_weight,
        gptqmodel_cfg=args.use_gptqmodel,
        max_pe_length=args.max_pe_length,
        visual_config=VisualConfig(
            image_max_size_h=args.image_max_size_h,
            image_max_size_w=args.image_max_size_w,
            image_max_size_t=args.image_max_size_t,
            temporal_patch_size=args.temporal_patch_size,
            patch_size=args.patch_size,
            sample_image_path=args.sample_image_path,
        ),
    )

    prefix = f"{model_name}-{target_device}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(
            hf_model_path, "Qwen3VLForConditionalGeneration", config, work_dir
        )


def simple_batch_rename(directory, old_prefix, new_prefix):
    """
    Simple batch rename without dry-run mode [2,6](@ref)
    """
    for filename in os.listdir(directory):
        if filename.startswith(old_prefix):
            old_path = os.path.join(directory, filename)
            # Only process files, not directories [7](@ref)
            if os.path.isfile(old_path):
                new_filename = new_prefix + filename[len(old_prefix) :]
                new_path = os.path.join(directory, new_filename)

                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")


def clean_directory(directory):
    """Remove all subfolders but keep files in the directory."""
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            shutil.rmtree(path)


def ensure_and_clean_dir(dir_path):
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"Cleared existing directory: {dir_path}")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def move_llm(args):
    work_dir = Path(args.work_dir)
    dest_dir = Path(args.out_dir)
    model_name = os.path.basename(args.model)
    hmm_model_dir = "{}-XH2a".format(model_name)
    hmm_model_prefix = hmm_model_dir + "-{}".format(args.quant_type)
    logger.info(
        msg_output_format("Start move from {} to {}").format(
            work_dir / hmm_model_dir, dest_dir
        )
    )
    ensure_and_clean_dir(dest_dir / "hmquant")
    shutil.move(
        work_dir / hmm_model_dir / "token_embedding.pt",
        dest_dir / "hmquant/quant_embedding.pt",
    )
    shutil.move(
        work_dir / hmm_model_dir / "golden/{}_vision".format(hmm_model_prefix),
        dest_dir / "hmquant/visual",
    )
    clean_directory("{}/hmquant/visual".format(dest_dir))
    simple_batch_rename(
        "{}/hmquant/visual".format(dest_dir),
        "hmquant_{}_vision".format(hmm_model_prefix),
        "hmquant_{}".format(args.model_name),
    )
    shutil.move(
        work_dir / hmm_model_dir / "golden/{}-llm-prefill".format(hmm_model_prefix),
        dest_dir / "hmquant/prefill",
    )
    clean_directory("{}/hmquant/prefill".format(dest_dir))
    simple_batch_rename(
        "{}/hmquant/prefill".format(dest_dir),
        "hmquant_{}-llm-prefill".format(hmm_model_prefix),
        "hmquant_{}".format(args.model_name),
    )
    shutil.move(
        work_dir / hmm_model_dir / "golden/{}-llm-decode".format(hmm_model_prefix),
        dest_dir / "hmquant/decoder",
    )
    clean_directory("{}/hmquant/decoder".format(dest_dir))
    simple_batch_rename(
        "{}/hmquant/decoder".format(dest_dir),
        "hmquant_{}-llm-decode".format(hmm_model_prefix),
        "hmquant_{}".format(args.model_name),
    )
    logger.info(msg_output_format("Start remove work_dir: {}".format(work_dir)))
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree("data", ignore_errors=True)
