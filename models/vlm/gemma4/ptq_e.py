import os
import re
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoConfig
from hmatc.utils import logger
from xhquant.api import Config, DeviceType, QuantSchema
from xhquant.api import convert_onnx_to_hmonnx, create_quant_config
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name

_HMQUANT_DIR_RE = re.compile(r"^hmquant_xh2_.+_\d{8}$")


def _build_e_cfg_from_model(args):
    cfg = dict(
        chip_arch="XH2a",
        model=dict(
            model_type="Gemma4ForConditionalGeneration",
            hf_model=args.model,
            model_name=Path(args.model).name,
            context_max_length=args.context_length,
            prefill_chunk_length=args.prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type="w8a8h1_sefp",
            ),
            visual_config=dict(
                export_mode="compact",
                max_size_w=args.max_size_w,
                max_size_h=args.max_size_h,
            ),
            audio_config=dict(
                sampling_rate=args.audio_sampling_rate,
            ),
        ),
    )
    cfg = format_model_name(cfg)
    return Config(cfg)


@dataclass
class TextConfig:
    vocab_size_per_layer_input = 262144
    num_hidden_layers = 35
    hidden_size = 1536
    hidden_size_per_layer_input = 256
    pad_token_id = 0
    rms_norm_eps = 1e-6


class PerLayerInputRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normed * self.weight


class PerLayerInputBuilder(nn.Module):
    def __init__(
        self,
        num_hidden_layers=35,
        hidden_size=1536,
        hidden_size_per_layer_input=256,
        vocab_size_per_layer_input=262144,
        pad_token_id=0,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.embed_scale = hidden_size_per_layer_input**0.5
        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_model_projection_scale = hidden_size**-0.5
        self.embed_tokens_per_layer = nn.Embedding(
            vocab_size_per_layer_input,
            num_hidden_layers * hidden_size_per_layer_input,
            pad_token_id,
        )
        self.per_layer_model_projection = nn.Linear(
            hidden_size, num_hidden_layers * hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection_norm = PerLayerInputRMSNorm(
            hidden_size_per_layer_input, eps=rms_norm_eps
        )

    @classmethod
    def from_artifact(cls, artifact_path: str, text_config: TextConfig = None):
        if text_config is None:
            text_config = TextConfig()
        saved = torch.load(artifact_path, map_location="cpu", weights_only=False)
        state_dict = saved.get("state_dict", saved)
        builder = cls(
            num_hidden_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            hidden_size_per_layer_input=text_config.hidden_size_per_layer_input,
            vocab_size_per_layer_input=text_config.vocab_size_per_layer_input,
            pad_token_id=text_config.pad_token_id,
            rms_norm_eps=text_config.rms_norm_eps,
        )
        builder.load_state_dict(state_dict)
        builder.to(dtype=next(iter(state_dict.values())).dtype)
        builder.per_layer_model_projection.to(torch.float32)
        builder.per_layer_projection_norm.to(torch.float32)
        return builder.eval()

    @staticmethod
    def save_tokens_per_layer_embedding(
        embed_tokens_per_layer: nn.Embedding, embedding_path: str
    ):
        torch.save(embed_tokens_per_layer.state_dict(), embedding_path)

    def forward(self, pli: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # b, s = input_ids.shape
        # pli: torch.Tensor = (
        #     self.embed_tokens_per_layer(input_ids).to(dtype) * self.embed_scale
        # )  # seq_len, num_hidden_layers * hidden_size_per_layer_input
        b, s, _ = pli.shape
        dtype = self.per_layer_model_projection.weight.dtype
        pli *= self.embed_scale
        pli = (
            pli.reshape(b, s, self.num_hidden_layers, self.hidden_size_per_layer_input)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        proj = (
            self.per_layer_model_projection(inputs_embeds.to(dtype))
            * self.per_layer_model_projection_scale
        )
        proj = (
            proj.reshape(b, s, self.num_hidden_layers, self.hidden_size_per_layer_input)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        proj = self.per_layer_projection_norm(proj)
        return ((proj + pli) * self.per_layer_input_scale).to(inputs_embeds.dtype)


def export_plib_hmonnx(args):
    text_config = AutoConfig.from_pretrained(args.model).get_text_config()
    per_layer_input_builder_path = os.path.join(
        args.out_dir,
        "hmquant",
        "per_layer_input_builder.pt",
    )
    per_layer_input_builder = PerLayerInputBuilder.from_artifact(
        per_layer_input_builder_path, text_config=text_config
    )
    per_layer_input_builder.save_tokens_per_layer_embedding(
        per_layer_input_builder.embed_tokens_per_layer,
        os.path.join(args.out_dir, "hmquant", "embed_tokens_per_layer.pt"),
    )
    p_pli = torch.zeros(
        (
            1,
            args.prefill_chunk_length,
            text_config.num_hidden_layers * text_config.hidden_size_per_layer_input,
        ),
        dtype=torch.float32,
    )
    p_input_embeds = torch.zeros(
        (1, args.prefill_chunk_length, text_config.hidden_size), dtype=torch.float32
    )
    d_pli = torch.zeros(
        (1, 1, text_config.num_hidden_layers * text_config.hidden_size_per_layer_input),
        dtype=torch.float32,
    )
    d_input_embeds = torch.zeros((1, 1, text_config.hidden_size), dtype=torch.float32)

    plib_path = os.path.join(args.out_dir, "hmquant", "plib", "onnx")
    os.makedirs(plib_path, exist_ok=True)
    torch.onnx.export(
        per_layer_input_builder,
        (p_pli, p_input_embeds),
        os.path.join(plib_path, "plib_prefill.onnx"),
        input_names=["pli", "input_embeds"],
        output_names=["output"],
        opset_version=17,
        export_modules_as_functions=False,
    )
    logger.info("Plib prefill ONNX model exported successfully.")
    torch.onnx.export(
        per_layer_input_builder,
        (d_pli, d_input_embeds),
        os.path.join(plib_path, "plib_decode.onnx"),
        input_names=["pli", "input_embeds"],
        output_names=["output"],
        opset_version=17,
        export_modules_as_functions=False,
    )
    logger.info("Plib decode ONNX model exported successfully.")
    # export hmonnx
    convert_onnx_to_hmonnx(
        os.path.join(plib_path, "plib_prefill.onnx"),
        (p_pli, p_input_embeds),
        device_type=DeviceType.XH2a,
        out_hmonnx_file=os.path.join(
            args.out_dir, "hmquant", "plib", "hmquant_plib_prefill.onnx"
        ),
        input_names=["plib", "input_embeds"],
        quant_config=create_quant_config(QuantSchema(quant_type="w8a8h1_sefp")),
    )
    logger.info("Plib prefill HMONNX model exported successfully.")
    convert_onnx_to_hmonnx(
        os.path.join(plib_path, "plib_decode.onnx"),
        (d_pli, d_input_embeds),
        device_type=DeviceType.XH2a,
        out_hmonnx_file=os.path.join(
            args.out_dir, "hmquant", "plib", "hmquant_plib_decode.onnx"
        ),
        input_names=["plib", "input_embeds"],
        quant_config=create_quant_config(QuantSchema(quant_type="w8a8h1_sefp")),
    )
    logger.info("Plib decode HMONNX model exported successfully.")


def move_hmquant_files(output_dir: str | Path, model_name: str) -> None:
    """Move files from hmquant_xh2_{model}_{quant}_{seq}_{context}_{date} to parent directory."""
    output_path = Path(output_dir) / "hmquant"
    model_dir = output_path / model_name
    if not model_dir.exists():
        logger.warning(f"Output directory not found: {model_dir}")
        return

    hmquant_dirs = [
        item
        for item in model_dir.iterdir()
        if item.is_dir() and _HMQUANT_DIR_RE.match(item.name)
    ]
    if not hmquant_dirs:
        logger.info(f"No hmquant directory found in {model_dir}")
        return

    hmquant_dir = hmquant_dirs[0]
    logger.info(f"Moving files from {hmquant_dir.name} to {output_path.name}/")

    for item in hmquant_dir.iterdir():
        target = output_path / item.name

        if item.is_dir():
            if target.exists():
                logger.warning(
                    f"\033[33mOverwriting existing directory: {target}\033[0m"
                )
                shutil.rmtree(target)
            shutil.move(str(item), str(target))
            logger.info(f"  Moved: {item.name}/")
        else:
            if target.exists():
                logger.warning(f"\033[33mOverwriting existing file: {target}\033[0m")
                target.unlink()
            shutil.move(str(item), str(target))
            logger.info(f"  Moved: {item.name}")

    shutil.rmtree(model_dir)
    logger.info(f"Cleanup completed, removed: {hmquant_dir.name}")


def quant_e(args, device, dtype):
    cfg = _build_e_cfg_from_model(args)
    model_name = os.path.basename(args.model).lower()
    work_dir = os.path.join(args.out_dir, "hmquant", model_name)
    os.makedirs(work_dir, exist_ok=True)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = work_dir
    xh_model.export_hmonnx(work_dir)
    move_hmquant_files(args.out_dir, model_name)
    # export PerLayerInputBuilder models
    export_plib_hmonnx(args)
