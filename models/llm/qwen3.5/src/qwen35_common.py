# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from gptqmodel import GPTQModel
from gptqmodel.quantization.rotation.mtp import rotate_qwen3_5_mtp


FULL_VISION_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_wikitext2(nsamples: int, seqlen: int):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen
    )
    return [example["text"] for example in traindata.select(range(nsamples))]


def get_jsonl_texts(path: str, nsamples: int, text_key: str = "text"):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid `{text_key}` at {path}:{line_no}")
            samples.append(text)
            if len(samples) >= nsamples:
                break
    if not samples:
        raise ValueError(f"No calibration samples found in {path}")
    return samples


def text_to_chat_calibration(samples):
    return [
        [{"role": "user", "content": [{"type": "text", "text": sample}]}]
        for sample in samples
    ]


def build_calibration_dataset(args):
    if getattr(args, "calibration_jsonl", None):
        ds = get_jsonl_texts(args.calibration_jsonl, args.nsamples, args.calibration_text_key)
        source = f"jsonl:{os.path.basename(args.calibration_jsonl)}"
    else:
        ds = get_wikitext2(args.nsamples, args.seqlen)
        source = "wikitext2"
    if has_vision_model(args.model, args.trust_remote_code):
        return text_to_chat_calibration(ds), f"{source}:text-chat"
    return ds, source


def load_config(model_path: str, trust_remote_code: bool):
    return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def model_type_of(model_path: str, trust_remote_code: bool) -> str:
    return str(getattr(load_config(model_path, trust_remote_code), "model_type", "")).lower()


def text_num_hidden_layers(model_path: str, trust_remote_code: bool) -> int:
    cfg = load_config(model_path, trust_remote_code)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "num_hidden_layers", None) is not None:
        return int(text_cfg.num_hidden_layers)
    if getattr(cfg, "num_hidden_layers", None) is not None:
        return int(cfg.num_hidden_layers)
    raise ValueError(f"Cannot resolve num_hidden_layers from {model_path}")


def layer_limit_dynamic_skips(model_path: str, trust_remote_code: bool, max_quant_layers: Optional[int]):
    if max_quant_layers is None:
        return {}
    num_layers = text_num_hidden_layers(model_path, trust_remote_code)
    if max_quant_layers < 0 or max_quant_layers > num_layers:
        raise ValueError(f"--max-quant-layers must be between 0 and {num_layers}, got {max_quant_layers}")
    return {
        rf"-:.*\.layers\.{idx}\..*": {}
        for idx in range(max_quant_layers, num_layers)
    }


def is_moe_model(model_path: str, trust_remote_code: bool) -> bool:
    cfg = load_config(model_path, trust_remote_code)
    cfgs = [cfg]
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        cfgs.append(text_cfg)
    for c in cfgs:
        model_type = str(getattr(c, "model_type", "")).lower()
        if "moe" in model_type:
            return True
        if getattr(c, "num_experts", None) and int(c.num_experts) > 1:
            return True
        if getattr(c, "moe_intermediate_size", None):
            return True
    return False


def has_vision_model(model_path: str, trust_remote_code: bool) -> bool:
    cfg = load_config(model_path, trust_remote_code)
    return model_type_of(model_path, trust_remote_code) in FULL_VISION_MODEL_TYPES and hasattr(cfg, "vision_config")


def resolve_rotation_vision(model_path: str, rotation: Optional[str], trust_remote_code: bool) -> Optional[str]:
    return rotation if rotation and has_vision_model(model_path, trust_remote_code) else None


@torch.inference_mode()
def calculate_avg_ppl(model, tokenizer, device=None, nsamples: int = 8, seqlen: int = 512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(x["text"] for x in dataset if x["text"].strip())
    tokens = tokenizer(text, return_tensors="pt").input_ids
    available = max(0, (tokens.shape[1] - 1) // seqlen)
    nsamples = min(nsamples, available)
    if nsamples <= 0:
        raise ValueError("Not enough tokens to evaluate perplexity.")

    try:
        embed_device = model.get_input_embeddings().weight.device
        if embed_device.type != "meta":
            device = embed_device
    except AttributeError:
        pass

    losses = []
    for i in range(nsamples):
        input_ids = tokens[:, i * seqlen:(i + 1) * seqlen].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        losses.append(outputs.loss.detach().float().cpu())

    mean_loss = torch.stack(losses).mean()
    return torch.exp(mean_loss).item()


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(getattr(model, "device", "cpu"))


@contextmanager
def temporary_ppl_device(model, target_device: Optional[str]):
    original_device = _get_model_device(model)
    active_device = str(original_device)
    moved = False

    if target_device is None:
        target_device = str(original_device)
    if target_device == "auto":
        target_device = "cuda:0" if torch.cuda.is_available() else str(original_device)

    if original_device.type == "cpu" and target_device != "cpu" and torch.cuda.is_available():
        print(f"[ppl] Moving model from cpu to {target_device} for evaluation ...")
        model.to(target_device)
        active_device = target_device
        moved = True

    try:
        yield active_device
    finally:
        if moved:
            print("[ppl] Moving model back to cpu after evaluation ...")
            model.to("cpu")
            torch.cuda.empty_cache()


def ensure_processor_saved(src_model_dir: str, dst_model_dir: str, trust_remote_code: bool) -> None:
    if not has_vision_model(src_model_dir, trust_remote_code):
        return
    processor = AutoProcessor.from_pretrained(src_model_dir, trust_remote_code=trust_remote_code)
    processor.save_pretrained(dst_model_dir)


def rotate_mtp_after_save(src_model_dir: str, dst_model_dir: str, gptq_model, is_moe: bool, rotation: Optional[str]) -> Optional[str]:
    if not rotation:
        return None
    q_block = getattr(gptq_model, "_rotation_q_block", None)
    if q_block is None:
        raise RuntimeError("LLM rotation was requested but no Q_block was exposed for MTP rotation.")
    return rotate_qwen3_5_mtp(
        src_model_dir=src_model_dir,
        dst_model_dir=dst_model_dir,
        Q_block=q_block,
        is_moe=is_moe,
    )


def ensure_visual_tensors_present(src_model_dir: str, dst_model_dir: str) -> None:
    src_idx = os.path.join(src_model_dir, "model.safetensors.index.json")
    if not os.path.exists(src_idx):
        return
    with open(src_idx, "r", encoding="utf-8") as f:
        src_wm = json.load(f).get("weight_map", {})

    visual_keys = sorted(k for k in src_wm if k.startswith("visual.") or k.startswith("model.visual."))
    if not visual_keys:
        return

    dst_idx_path = os.path.join(dst_model_dir, "model.safetensors.index.json")
    dst_idx = {"metadata": {}, "weight_map": {}}
    if os.path.exists(dst_idx_path):
        with open(dst_idx_path, "r", encoding="utf-8") as f:
            dst_idx = json.load(f)
    else:
        single = os.path.join(dst_model_dir, "model.safetensors")
        if os.path.exists(single):
            with safe_open(single, framework="pt", device="cpu") as f:
                dst_idx["weight_map"] = {k: "model.safetensors" for k in f.keys()}

    dst_wm = dst_idx.setdefault("weight_map", {})
    missing = [k for k in visual_keys if k not in dst_wm]
    if not missing:
        print(f"[visual] all {len(visual_keys)} visual tensors already present in {dst_model_dir}")
        return

    print(f"[visual] {len(missing)}/{len(visual_keys)} visual tensors missing — copying passthrough sidecar")
    sidecar_name = "visual_passthrough.safetensors"
    sidecar_path = os.path.join(dst_model_dir, sidecar_name)
    state = {}
    by_shard = {}
    for key in missing:
        by_shard.setdefault(src_wm[key], []).append(key)
    for shard_name, keys in by_shard.items():
        shard_path = os.path.join(src_model_dir, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                state[key] = f.get_tensor(key)

    save_file(state, sidecar_path, metadata={"format": "pt", "produced_by": "qwen35_common.visual_passthrough"})
    for key in missing:
        dst_wm[key] = sidecar_name
    meta = dst_idx.setdefault("metadata", {})
    try:
        meta["total_size"] = int(meta.get("total_size", 0)) + os.path.getsize(sidecar_path)
    except OSError:
        pass
    with open(dst_idx_path, "w", encoding="utf-8") as f:
        json.dump(dst_idx, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[visual] wrote {len(missing)} tensors to {sidecar_name}; index.json patched")


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[dtype_name]


def is_gptq_quantized_dir(model_dir: str) -> bool:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    quant_config = config.get("quantization_config")
    return isinstance(quant_config, dict) and str(quant_config.get("quant_method", "")).lower() == "gptq"


def build_vision_demo_inputs(processor, prompt: str, image_size: int, seed: int):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], videos=None, padding=True, return_tensors="pt")


def move_batch_to_device(batch, device: str, float_dtype: torch.dtype):
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=float_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


@torch.inference_mode()
def vision_demo_outputs(model_dir: str, batch, device: str, torch_dtype, trust_remote_code: bool, max_new_tokens: int):
    wrapper = None
    if is_gptq_quantized_dir(model_dir):
        load_kwargs = {"device_map": "auto"} if device == "auto" else {"device": device}
        wrapper = GPTQModel.load(model_dir, trust_remote_code=trust_remote_code, **load_kwargs)
        model = wrapper.model
    else:
        load_kwargs = {"device_map": "auto"} if device == "auto" else {"device_map": "cpu"}
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            **load_kwargs,
        )
    model.eval()
    forward_dtype = next(model.parameters()).dtype
    if device != "auto":
        model.to(device=device)
        runtime_device = device
    else:
        try:
            runtime_device = str(model.get_input_embeddings().weight.device)
        except AttributeError:
            runtime_device = str(next(model.parameters()).device)
    batch = move_batch_to_device(batch, device=runtime_device, float_dtype=forward_dtype)
    logits = model(**batch, use_cache=False).logits.detach().float().cpu()
    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    ).detach().cpu()
    if device != "auto" and wrapper is None:
        model.to("cpu")
    del wrapper
    del model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits, generated


def _processor_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _decode_generated_suffix(processor, generated: torch.Tensor, prompt_len: int) -> str:
    tokenizer = _processor_tokenizer(processor)
    output_ids = generated[0, prompt_len:].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _format_first_token_diff(processor, ref_generated: torch.Tensor, test_generated: torch.Tensor, prompt_len: int) -> str:
    tokenizer = _processor_tokenizer(processor)
    ref_suffix = ref_generated[0, prompt_len:].tolist()
    test_suffix = test_generated[0, prompt_len:].tolist()
    for idx, (ref_id, test_id) in enumerate(zip(ref_suffix, test_suffix)):
        if ref_id != test_id:
            ref_piece = tokenizer.decode([ref_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            test_piece = tokenizer.decode([test_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return (
                f"first_diff_generated_token={idx}: "
                f"ref_id={ref_id} ref={ref_piece!r}, "
                f"test_id={test_id} test={test_piece!r}"
            )
    if len(ref_suffix) != len(test_suffix):
        return f"generated_length_diff: ref={len(ref_suffix)}, test={len(test_suffix)}"
    return "no generated token diff"


def validate_vision_demo_pair(
    ref_model_dir: str,
    test_model_dir: str,
    *,
    device: str,
    dtype: str,
    prompt: str,
    image_size: int,
    seed: int,
    trust_remote_code: bool,
    atol: float,
    rtol: float,
    generation_tokens: int = 256,
    require_equal: bool = True,
) -> None:
    if not has_vision_model(ref_model_dir, trust_remote_code):
        print("[vision] source model has no vision tower; skipping vision demo")
        return
    torch_dtype = resolve_torch_dtype(dtype)
    processor = AutoProcessor.from_pretrained(ref_model_dir, trust_remote_code=trust_remote_code)
    batch = build_vision_demo_inputs(processor, prompt, image_size, seed)
    ref_logits, ref_generated = vision_demo_outputs(ref_model_dir, batch, device, torch_dtype, trust_remote_code, generation_tokens)
    test_logits, test_generated = vision_demo_outputs(test_model_dir, batch, device, torch_dtype, trust_remote_code, generation_tokens)
    max_abs = (test_logits - ref_logits).abs().max().item()
    mean_abs = (test_logits - ref_logits).abs().mean().item()
    last_token_equal = bool(torch.equal(test_logits[:, -1].argmax(dim=-1), ref_logits[:, -1].argmax(dim=-1)))
    prompt_len = batch["input_ids"].shape[1]
    generation_equal = bool(torch.equal(test_generated[:, prompt_len:], ref_generated[:, prompt_len:]))
    print(
        f"[vision] max_abs_diff={max_abs:.6e}, mean_abs_diff={mean_abs:.6e}, "
        f"last_token_equal={last_token_equal}, generation_equal={generation_equal}"
    )
    ref_text = _decode_generated_suffix(processor, ref_generated, prompt_len)
    test_text = _decode_generated_suffix(processor, test_generated, prompt_len)
    token_diff = _format_first_token_diff(processor, ref_generated, test_generated, prompt_len)
    print(f"[vision] {token_diff}")
    print(f"[vision] ref_decoded ={ref_text!r}")
    print(f"[vision] test_decoded={test_text!r}")
    if require_equal and not torch.allclose(test_logits, ref_logits, atol=atol, rtol=rtol) and not generation_equal:
        raise AssertionError(
            "Vision demo mismatch after rotation: "
            f"max_abs_diff={max_abs:.6e}, mean_abs_diff={mean_abs:.6e}, "
            f"last_token_equal={last_token_equal}, generation_equal={generation_equal}, "
            f"{token_diff}, ref_decoded={ref_text!r}, test_decoded={test_text!r}"
        )
    if not require_equal and not test_text.strip():
        raise AssertionError(
            "Quantized vision demo produced empty output: "
            f"max_abs_diff={max_abs:.6e}, mean_abs_diff={mean_abs:.6e}, "
            f"last_token_equal={last_token_equal}, generation_equal={generation_equal}, "
            f"{token_diff}"
        )
