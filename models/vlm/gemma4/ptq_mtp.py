# fmt: off
import os
import gc
import json
import torch
from pathlib import Path
from xhquant.api import Config
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhmodel_merak.xh_llm import format_model_name, support_llm_model_types
from xhmodel_merak.xh_llm.types import LLMModelState, ModelSwitcher
from xhmodel_merak.xh_llm.utils import unfold_args
from xhmodel_merak.xh_llm.models.gemma4_moe import XHGemma4MoeWithMaskConfig, XHGemma4MoeWithMaskModel
from xhmodel_merak.utils import calculate_file_md5
from xhquant.api import to_export_graph, to_export_hmonnx_v2, ptq_quantize
from xhquant.api import ConfigDict, PrecisionMode
from gemma4_moe_mtp_common import XHGemma4AssistantDraftModel
from hmatc.utils import logger
from ptq_e import move_hmquant_files
import shutil


def _build_cfg_from_model(args):
    cfg = dict(
        chip_arch="XH2a",
        model=dict(
            model_type="Gemma4ForConditionalGeneration_with_mask",
            hf_model=f"{args.model}-gptq-4bit",
            fallback_hf_model=args.model,
            model_name=f"{args.model_name}-{args.model_size}",
            context_max_length=args.context_length,
            prefill_chunk_length=args.prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type="w4a8h1_sefp",
                ops={},
            ),
            visual_config=dict(
                model_type="Gemma4ForConditionalGeneration_visual",
                hf_model=f"{args.model}-gptq-4bit",
                model_name=f"{args.model_name}-{args.model_size}",
                max_size_w=args.max_size_w,
                max_size_h=args.max_size_h,
                upsample_token=False,
                fuse_norm=True,
                quant_scheme=dict(
                    quant_type="w8a8h1_sefp",
                    ops={},
                ),
            ),
        ),
        mtp=dict(
            assistant_model=args.assistant_model,
            assistant_quant_type="w8a8h1_sefp",
            num_draft_tokens=4,
        ),
    )
    cfg = format_model_name(cfg)
    return Config(cfg)


def _release_model_memory(xh_model) -> None:
    if xh_model is None:
        return

    try:
        kvcache_mixin = xh_model.get_kvcache_mixin()
        kvcache_mixin.clear_kv_cache()
        kvcache_mixin.clear_other_cache()
    except Exception:
        pass

    attrs_to_clear = (
        "_inference_model",
        "_exported_model",
        "_quanted_model",
        "_frontend_model",
        "_wrap_model",
        "_data_processor",
        "hf_compatible_model",
    )
    for attr_name in attrs_to_clear:
        if hasattr(xh_model, attr_name):
            setattr(xh_model, attr_name, None)

    for sub_model in getattr(xh_model, "_models", {}).values():
        for attr_name in attrs_to_clear:
            if hasattr(sub_model, attr_name):
                setattr(sub_model, attr_name, None)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del xh_model
        
        
def _drop_visual_model(xh_model) -> None:
    if hasattr(xh_model, "visual"):
        try:
            delattr(xh_model, "visual")
        except AttributeError:
            pass
    if hasattr(xh_model, "_models") and isinstance(xh_model._models, dict):
        xh_model._models.pop("visual", None)
        
        
def _load_target_model(cfg):
    model_cfg: XHGemma4MoeWithMaskConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHGemma4MoeWithMaskConfig", f"Expected model config type XHGemma4MoeWithMaskConfig, but got {type(model_cfg).__name__}"
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")

    xh_model: XHGemma4MoeWithMaskModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHGemma4MoeWithMaskModel", f"Expected model type XHGemma4MoeWithMaskModel, but got {type(xh_model).__name__}"
    _drop_visual_model(xh_model)
    return xh_model


def _find_export_dir(work_dir: Path) -> Path:
    export_dirs = sorted(path for path in work_dir.glob("hmquant_*") if path.is_dir())
    if len(export_dirs) != 1:
        raise RuntimeError(f"Expected exactly one hmquant export dir under {work_dir}, got {export_dirs}")
    return export_dirs[0]


def _repair_dangling_clone_edges(onnx_file: str | Path) -> bool:
    import onnx
    from onnx import TensorProto

    onnx_file = Path(onnx_file)
    model = onnx.load_model(str(onnx_file), load_external_data=False)

    produced = {value.name for value in model.graph.input}
    produced |= {init.name for init in model.graph.initializer}
    produced |= {sparse.name for sparse in model.graph.sparse_initializer}
    for node in model.graph.node:
        produced.update(name for name in node.output if name)

    missing_inputs = {input_name for node in model.graph.node for input_name in node.input if input_name and input_name not in produced}

    output_counts: dict[str, int] = {}
    for node in model.graph.node:
        for output_name in node.output:
            if output_name:
                output_counts[output_name] = output_counts.get(output_name, 0) + 1

    repaired_edges: list[tuple[str, str]] = []
    for node in model.graph.node:
        if node.op_type != "Transpose" or not node.name.startswith("node_") or len(node.output) != 1:
            continue
        expected_output = node.name.removeprefix("node_")
        actual_output = node.output[0]
        if expected_output in missing_inputs and output_counts.get(actual_output, 0) > 1:
            node.output[0] = expected_output
            repaired_edges.append((actual_output, expected_output))

    produced = {value.name for value in model.graph.input}
    produced |= {init.name for init in model.graph.initializer}
    produced |= {sparse.name for sparse in model.graph.sparse_initializer}
    for node in model.graph.node:
        produced.update(name for name in node.output if name)

    output_names = {output.name for output in model.graph.output}
    added_hidden_output = False
    if "last_hidden_state" not in output_names:
        lm_head_node = next((node for node in model.graph.node if node.op_type == "Linear" and len(node.input) >= 3 and node.input[1] == "lm_head.qweight" and node.input[2] == "lm_head.scale_or_exp"), None)
        logits_output = next((output for output in model.graph.output if output.name == "logits"), None)
        if lm_head_node is not None and logits_output is not None:
            hidden_name = lm_head_node.input[0]
            hidden_output = onnx.helper.make_node(
                "Identity",
                [hidden_name],
                ["last_hidden_state"],
                name="node_mtp_last_hidden_state",
            )
            model.graph.node.append(hidden_output)
            logits_shape = logits_output.type.tensor_type.shape.dim
            in_features = next((onnx.helper.get_attribute_value(attr) for attr in lm_head_node.attribute if attr.name == "in_features"), None)
            hidden_shape = [
                logits_shape[0].dim_value or logits_shape[0].dim_param,
                logits_shape[1].dim_value or logits_shape[1].dim_param,
                int(in_features),
            ]
            model.graph.output.append(onnx.helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT16, hidden_shape))
            added_hidden_output = True

    added_lm_head = False
    if "logits" in output_names and "logits" not in produced and "last_hidden_state" in produced:
        hidden_output = next((output for output in model.graph.output if output.name == "last_hidden_state"), None)
        logits_output = next((output for output in model.graph.output if output.name == "logits"), None)
        qweight = next((init for init in model.graph.initializer if init.name == "lm_head.qweight"), None)
        scale_or_exp = next((init for init in model.graph.initializer if init.name == "lm_head.scale_or_exp"),None)
        if hidden_output is not None and logits_output is not None and qweight is not None and scale_or_exp is not None:
            hidden_shape = hidden_output.type.tensor_type.shape.dim
            logits_shape = logits_output.type.tensor_type.shape.dim
            if len(hidden_shape) == 3 and len(logits_shape) == 3:
                logits_shape[0].dim_value = hidden_shape[0].dim_value
                logits_shape[1].dim_value = hidden_shape[1].dim_value
                logits_shape[2].dim_value = int(qweight.dims[-1])
            logits_node = onnx.helper.make_node(
                "Linear",
                ["last_hidden_state", "lm_head.qweight", "lm_head.scale_or_exp"],
                ["logits"],
                name="node_mtp_lm_head",
                domain="ai.houmo.xh2a",
                hmfp_psum_dtype=b"fp24",
                hmfp_psum_round_mode=b"trunc",
                hmfp_weight_hidden_bit=1,
                output_dtype=b"float16",
                hmfp_weight_rounding=b"RNE",
                have_bias=0,
                in_features=int(hidden_shape[-1].dim_value),
                out_features=int(qweight.dims[-1]),
                mode=b"sefp",
                hmfp_act_exp_bit=5,
                hmfp_act_man_bit=8,
                hmfp_act_rounding=b"RNE",
                hmfp_act_nshare=64,
                hmfp_act_hidden_bit=1,
                hmfp_weight_exp_bit=5,
                hmfp_weight_nshare=64,
                hmfp_weight_man_bit=8,
            )
            model.graph.node.append(logits_node)
            added_lm_head = True
        else:
            logger.warning(
                f"Cannot add lm_head logits producer in {onnx_file}: "
                f"hidden_output={hidden_output is not None}, logits_output={logits_output is not None}, "
                f"qweight={qweight is not None}, scale_or_exp={scale_or_exp is not None}"
            )

    if not repaired_edges and not added_lm_head and not added_hidden_output:
        if missing_inputs:
            logger.warning(f"No dangling clone edges repaired in {onnx_file}, missing inputs: {sorted(missing_inputs)[:8]}")
        return False

    onnx.save_model(model, str(onnx_file))
    logger.info(
        f"Repaired target HMONNX outputs in {onnx_file}: "
        f"dangling_edges={repaired_edges}, added_hidden_output={added_hidden_output}, added_lm_head={added_lm_head}"
    )
    return True


def _repair_target_hmonnx_files(export_dir: Path) -> None:
    base_meta_path = export_dir / "golden_meta_info.json"
    if not base_meta_path.exists():
        return
    with open(base_meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    changed = False
    for hmonnx_key, md5_key in (("prefill_hmonnx", "prefill_hmonnx_md5"), ("decode_hmonnx", "decode_hmonnx_md5")):
        hmonnx_path = export_dir / meta[hmonnx_key]
        if _repair_dangling_clone_edges(hmonnx_path):
            meta[md5_key] = calculate_file_md5(str(hmonnx_path))
            changed = True

    if changed:
        with open(base_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)
            

def _load_exported_text_config(export_dir: Path, cfg) -> dict:
    candidate_paths = [export_dir / "hf_config" / "config.json", Path(cfg.hf_model) / "config.json"]
    fallback_hf_model = getattr(cfg, "fallback_hf_model", None)
    if fallback_hf_model:
        candidate_paths.append(Path(fallback_hf_model) / "config.json")
    for config_path in candidate_paths:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                hf_config = json.load(f)
            text_config = dict(hf_config.get("text_config", hf_config))
            for token_key in ("image_token_id", "audio_token_id", "video_token_id"):
                if token_key not in text_config and token_key in hf_config:
                    text_config[token_key] = hf_config[token_key]
            return text_config
    raise FileNotFoundError(f"Cannot find Gemma4 config.json for rebuilding meta under {export_dir}")
    

def _resolve_decode_quanted_model(xh_model):
    if xh_model._state != LLMModelState.QUANTED_ALIGNED:
        xh_model.to_quanted_aligned()
    xh_model._quanted_model.fixed()
    if isinstance(xh_model._quanted_model, ModelSwitcher):
        decode_quanted_model = xh_model._quanted_model.decode
    else:
        decode_quanted_model = xh_model._quanted_model
    if not decode_quanted_model.is_fixed():
        raise ValueError("decode_quanted_model model is not fixed, Please call `fixed` first.")
    decode_quanted_model.to("cpu")
    return decode_quanted_model

         
def _export_target_hmonnx(xh_model, work_dir: Path) -> Path:
    xh_model.export_hmonnx(str(work_dir))
    export_dir = _find_export_dir(work_dir)
    _repair_target_hmonnx_files(export_dir)
    return export_dir


def _export_target_verify_hmonnx(xh_model: XHGemma4MoeWithMaskModel, export_dir: Path, cfg) -> tuple[str, str]:
    num_draft_tokens = int(cfg.mtp.num_draft_tokens)
    verify_length = num_draft_tokens + 1
    export_model_name = export_dir.name
    verify_dir = export_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    text_config = _load_exported_text_config(export_dir, cfg.model)
    eos_token_id = text_config.get("eos_token_id", 1)
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    xh_model.pad_token_id = int(text_config.get("pad_token_id") or eos_token_id or 1)

    xh_model._llm_prefill = False
    xh_model.set_input_sequence_length(verify_length)

    original_get_decode_dummy_inputs = xh_model.get_decode_dummy_inputs
    
    def get_verify_decode_dummy_inputs():
        input_ids = torch.randint(0, 100, (1, verify_length), dtype=torch.long)
        past_seq_length = xh_model.config.prefill_chunk_length
        return {"input_ids": input_ids, "past_seq_length": past_seq_length}
    
    xh_model.get_decode_dummy_inputs = get_verify_decode_dummy_inputs
    
    try:
        decode_quanted_model = _resolve_decode_quanted_model(xh_model)
    finally:
        xh_model.get_decode_dummy_inputs = original_get_decode_dummy_inputs
        
    with xh_model.get_kvcache_mixin().kv_cache_scope(device="meta"):
        xh_model.set_decode()
        xh_model.set_input_sequence_length(verify_length)
        data_processor = xh_model.get_data_preprocessor()
        data_processor.input_sequence_length = verify_length
        inputs = data_processor(get_verify_decode_dummy_inputs())
        inputs = unfold_args(inputs)
        verify_exported_model = to_export_graph(decode_quanted_model, inputs)
        verify_hmonnx_file = str(verify_dir / f"{export_model_name}_verify.onnx")
        export_cfg = xh_model.get_export_cfg()
        verify_hmonnx_file = to_export_hmonnx_v2(
            verify_exported_model,
            inputs,
            verify_hmonnx_file,
            export_cfg,
            normalize_onnx_name=True,
        )  

    verify_path = Path(verify_hmonnx_file)
    _repair_dangling_clone_edges(verify_path)
    verify_rel = str(verify_path.relative_to(export_dir))
    verify_md5 = calculate_file_md5(verify_hmonnx_file)
    logger.info(f"Target verify HMONNX exported to {verify_hmonnx_file}")
    return verify_rel, verify_md5


def _export_assistant_draft(cfg: Config, export_dir: Path) -> str:
    mtp_cfg = cfg.mtp
    draft_onnx_dir = export_dir / "draft_onnx"
    draft_onnx_dir.mkdir(parents=True, exist_ok=True)

    assistant_model = XHGemma4AssistantDraftModel(
        assistant_model_dir=mtp_cfg.assistant_model,
        target_model_dir=cfg.model.hf_model,
        wrap_cfg=ConfigDict(
            input_sequence_length=1,
            max_sequence_length=cfg.model.context_max_length,
            dtype="float16",
        ),
        quant_config=ConfigDict(quant_type=mtp_cfg.assistant_quant_type),
    )
    assistant_model.init_wrap_model()
    dummy_data = assistant_model.prepare_inputs(None)
    assistant_model.convert_to_fronted_graph(dummy_data)
    assistant_model.convert_to_quant_graph(cfg.chip_arch)
    ptq_quantize(
        assistant_model.quanted_model,
        [assistant_model.prepare_inputs(None)],
        PrecisionMode.ALIGNED,
        [torch.device("cpu")],
    )
    assistant_model.convert_to_export_graph(dummy_data)
    onnx_file = assistant_model.to_export_onnx(dummy_data, str(draft_onnx_dir), prefix="gemma4_assistant_decode")[0]
    assistant_model.release_exported_model()
    assistant_model.release_quanted_model()
    assistant_model.release_frontend_model()
    assistant_model.release_wraped_model()
    logger.info(f"Assistant draft ONNX exported to {onnx_file}")
    return onnx_file


def quant_mtp(args, device):
    model_name = os.path.basename(args.model).lower()
    model_name = f"{model_name}-mtp"
    work_dir = os.path.join(args.out_dir, "hmquant", model_name)
    os.makedirs(work_dir, exist_ok=True)
    work_dir = Path(work_dir)
    logger.info("Quantizing MTP model...")
    cfg = _build_cfg_from_model(args)
    logger.info(f"Model configuration: {cfg.pretty_text}")

    # Target model
    target_model = _load_target_model(cfg)
    target_model.wrap_cfg.output_hidden_states_for_export = True
    export_dir = _export_target_hmonnx(target_model, work_dir)
    logger.info("Releasing target export model before verify export")
    _release_model_memory(target_model)
    
    # Verify model
    logger.info("Loading target model for verify export")
    verify_model = _load_target_model(cfg)
    verify_model.wrap_cfg.num_logits_to_keep = 0
    verify_model.wrap_cfg.output_hidden_states_for_export = True
    verify_hmonnx, verify_hmonnx_md5 = _export_target_verify_hmonnx(verify_model, export_dir, cfg)
    logger.info("Releasing target verify model before assistant export")
    _release_model_memory(verify_model)
    
    # Assistant model
    base_meta_path = export_dir / "golden_meta_info.json"
    draft_onnx_file = _export_assistant_draft(cfg, export_dir)
    
    # Move to hmquant
    move_hmquant_files(args.out_dir, model_name)
    
    # Rename draft
    # hmquant_xh2_gemma4-26b-a4b_w8a8_256_2k_20260527_draft_with_act.onnx 
    # hmquant_xh2_gemma4-26b-a4b_w4a8_256_2k_20260527_draft_external_data
    old_draft_onnx_ = os.path.join(args.out_dir, "hmquant", "draft_onnx", "gemma4_assistant_decode.onnx")
    new_draft_onnx = os.path.join(args.out_dir, "hmquant", "draft_onnx", "hmquant_gemma4_assistant_decode.onnx")
    shutil.move(old_draft_onnx_, new_draft_onnx)
    old_draft_data = os.path.join(args.out_dir, "hmquant", "draft_onnx", "gemma4_assistant_decode_external_data")
    new_draft_data = os.path.join(args.out_dir, "hmquant", "draft_onnx", "hmquant_gemma4_assistant_decode_external_data")
    shutil.move(old_draft_data, new_draft_data)
