from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import xhquant.nn as xhnn
from safetensors.torch import load_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextModel

from xh_model_zoo.xh_llm.models.base_model import BaseModel
from xhquant.api import ConfigDict, get_xhquant_logger
from xhquant.nn import RMSNorm as XHRMSNorm
from xhquant.nn import Rope as XHRope

DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    key = str(dtype_name).strip().lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return DTYPE_MAP[key]


def aligned(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


def load_json_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_layer_type_to_last_index(layer_types: list[str]) -> dict[str, int]:
    return {layer_type: index for index, layer_type in enumerate(layer_types)}


def _compute_rotary_cache(
    inv_freq: torch.Tensor, attention_scaling: float, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(
        max_seq_len, device=inv_freq.device, dtype=torch.float32
    ).view(max_seq_len, 1)
    freqs = positions * inv_freq.float().view(1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=inv_freq.dtype), sin.to(dtype=inv_freq.dtype)


def _convert_gemma4_rmsnorm(hf_norm: nn.Module) -> XHRMSNorm:
    """Convert a HF ``Gemma4RMSNorm`` (which scales by ``1 + weight``) into an
    ``xhquant.nn.RMSNorm`` whose graph fuses to a single RMSNorm op. The
    ``(1 + weight)`` factor is baked into the new weight so we drop the extra
    ``Add`` / ``Mul`` from the exported graph.
    """
    if not isinstance(hf_norm, Gemma4RMSNorm):
        # Already converted or a custom op — return as-is.
        if isinstance(hf_norm, XHRMSNorm):
            return hf_norm
        raise TypeError(f"Expected Gemma4RMSNorm, got {type(hf_norm).__name__}")
    hidden_size = int(hf_norm.weight.shape[0])
    eps = float(getattr(hf_norm, "eps", 1e-6))
    new_norm = XHRMSNorm(hidden_size, eps)
    with torch.no_grad():
        new_norm.weight.data.copy_(
            hf_norm.weight.data.detach().to(new_norm.weight.dtype) + 1.0
        )
    new_norm.weight.requires_grad_(False)
    return new_norm


class Gemma4AssistantSelfAttention(nn.Module):
    def __init__(self, hf_attn: nn.Module, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.config = hf_attn.config
        self.q_proj = hf_attn.q_proj
        # NOTE: keep HF ``Gemma4RMSNorm`` here. It is swapped to ``xhnn.RMSNorm``
        # by ``Gemma4AssistantDraftModule._convert_norms_for_export`` *after*
        # the checkpoint is loaded so the bake of ``weight = 1 + hf_weight``
        # uses the trained weights rather than the randomly-initialised ones.
        self.q_norm = hf_attn.q_norm
        self.o_proj = hf_attn.o_proj
        default_head_dim = (
            self.config.global_head_dim
            if layer_type == "full_attention"
            else self.config.head_dim
        )
        self.head_dim = getattr(hf_attn, "head_dim", default_head_dim)
        self.num_attention_heads = int(self.config.num_attention_heads)
        if layer_type == "full_attention":
            self.num_key_value_heads = int(
                getattr(
                    self.config,
                    "num_global_key_value_heads",
                    self.config.num_key_value_heads,
                )
            )
        else:
            self.num_key_value_heads = int(self.config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        # Gemma4 MTP assistant attention always uses scaling == 1.0 (mirrors
        # vLLM ``Gemma4MTPAttention`` which hard-codes ``self.scaling = 1.0``).
        # HF ``Gemma4Attention`` defaults to ``head_dim**-0.5`` — using that
        # value here would scale attn logits by ~0.06 vs the trained 1.0,
        # collapsing the draft-model accept rate. Avoid emitting a redundant
        # ``Mul`` op into the graph by also skipping the multiplication at
        # runtime when scaling is 1.0.
        self.scaling = 1.0
        self.attn_logit_softcapping = getattr(
            hf_attn.config, "attn_logit_softcapping", None
        )
        # Use xhquant fused rope op so the graph contains a single Rope node
        # instead of slice/neg/concat/mul/add patterns.
        self.rope = XHRope()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        shared_key_cache: torch.Tensor,
        shared_value_cache: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]

        cos, sin = position_embeddings
        # Project Q and place heads on dim 1 before rope so cos/sin (shape
        # ``(1, 1, T, D)``) broadcasts directly without an extra unsqueeze.
        query_states = (
            self.q_proj(hidden_states)
            .reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        query_states = self.q_norm(query_states)
        query_states = self.rope(query_states, cos, sin)

        # K branch: transpose first, then repeat_interleave on the head axis so
        # the front-end can fuse ``repeat_interleave + matmul`` into a single
        # groupmatmul (matching the main Gemma4 attention path).
        key_states = shared_key_cache.transpose(2, 3)
        if self.num_key_value_groups != 1:
            key_states = torch.repeat_interleave(
                key_states, self.num_key_value_groups, dim=1
            )
        attn_weights = torch.matmul(query_states, key_states)
        if self.scaling != 1.0:
            attn_weights = attn_weights * self.scaling

        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # V branch: same pattern as the main Gemma4 attention.
        value_states = shared_value_cache
        if self.num_key_value_groups != 1:
            value_states = torch.repeat_interleave(
                value_states, self.num_key_value_groups, dim=1
            )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
        )
        return self.o_proj(attn_output)


class Gemma4AssistantDecoderLayer(nn.Module):
    def __init__(self, hf_layer: nn.Module, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = Gemma4AssistantSelfAttention(hf_layer.self_attn, layer_type)
        # Keep HF Gemma4RMSNorm instances here; they are converted to xhquant
        # RMSNorm by ``Gemma4AssistantDraftModule._convert_norms_for_export``
        # after the safetensors checkpoint is loaded.
        self.input_layernorm = hf_layer.input_layernorm
        self.post_attention_layernorm = hf_layer.post_attention_layernorm
        self.pre_feedforward_layernorm = hf_layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = hf_layer.post_feedforward_layernorm
        self.mlp = hf_layer.mlp
        self.register_buffer("layer_scalar", hf_layer.layer_scalar.detach().clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        shared_key_cache: torch.Tensor,
        shared_value_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_key_cache=shared_key_cache,
            shared_value_cache=shared_value_cache,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states * self.layer_scalar.to(dtype=hidden_states.dtype)
        return hidden_states


class Gemma4AssistantBackbone(nn.Module):
    def __init__(
        self,
        text_model: Gemma4TextModel,
        max_position_embeddings: int,
        input_sequence_length: int,
    ):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens
        self.layers = nn.ModuleList(
            [
                Gemma4AssistantDecoderLayer(layer, text_model.config.layer_types[index])
                for index, layer in enumerate(
                    text_model.layers[: text_model.config.num_hidden_layers]
                )
            ]
        )
        # Convert the final HF Gemma4RMSNorm too so the trailing RMSNorm in the
        # graph is a single fused op rather than the decomposed pow/mul chain.
        # The actual swap to xhquant RMSNorm happens in
        # ``Gemma4AssistantDraftModule._convert_norms_for_export`` after weights
        # are loaded.
        self.norm = text_model.norm
        self.layer_types = list(text_model.config.layer_types)
        self.max_position_embeddings = max_position_embeddings
        self.input_sequence_length = int(input_sequence_length)

        rotary_emb = text_model.rotary_emb
        for layer_type in ("sliding_attention", "full_attention"):
            inv_freq = getattr(rotary_emb, f"{layer_type}_inv_freq", None)
            attention_scaling = getattr(
                rotary_emb, f"{layer_type}_attention_scaling", None
            )
            if inv_freq is None or attention_scaling is None:
                continue
            cos, sin = _compute_rotary_cache(
                inv_freq, attention_scaling, max_position_embeddings
            )
            # Store caches as ``(1, 1, max_seq, head_dim)`` so DynamicSlice can
            # slice along axis=2 and the result broadcasts directly with the
            # transposed Q of shape ``(B, num_heads, T, head_dim)``.
            cos_buf = cos.unsqueeze(0).unsqueeze(0)
            sin_buf = sin.unsqueeze(0).unsqueeze(0)
            self.register_buffer(f"{layer_type}_cos_cached", cos_buf, persistent=False)
            self.register_buffer(f"{layer_type}_sin_cached", sin_buf, persistent=False)

        # Use xhquant DynamicSlice driven by ``past_seq_length`` instead of
        # ``cos_cache[position_ids]`` (which would emit a GatherND op).
        self.sliding_cos_slice = xhnn.DynamicSlice(
            [self.input_sequence_length], [2], [1]
        )
        self.sliding_sin_slice = xhnn.DynamicSlice(
            [self.input_sequence_length], [2], [1]
        )
        self.full_cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        self.full_sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])

    def _get_position_embeddings(
        self, past_seq_length: torch.Tensor, layer_type: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_cache = getattr(self, f"{layer_type}_cos_cached")
        sin_cache = getattr(self, f"{layer_type}_sin_cached")
        if layer_type == "full_attention":
            cos = self.full_cos_slice(cos_cache, past_seq_length)
            sin = self.full_sin_slice(sin_cache, past_seq_length)
        else:
            cos = self.sliding_cos_slice(cos_cache, past_seq_length)
            sin = self.sliding_sin_slice(sin_cache, past_seq_length)
        return cos, sin

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_seq_length: torch.Tensor,
        local_attention_mask: torch.Tensor | None,
        global_attention_mask: torch.Tensor | None,
        shared_key_cache_sliding: torch.Tensor,
        shared_value_cache_sliding: torch.Tensor,
        shared_key_cache_full: torch.Tensor,
        shared_value_cache_full: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        # Pre-compute the two layer-type position embeddings once per forward.
        full_pos = self._get_position_embeddings(past_seq_length, "full_attention")
        sliding_pos = self._get_position_embeddings(
            past_seq_length, "sliding_attention"
        )
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                attention_mask = global_attention_mask
                shared_key_cache = shared_key_cache_full
                shared_value_cache = shared_value_cache_full
                position_embeddings = full_pos
            else:
                attention_mask = local_attention_mask
                shared_key_cache = shared_key_cache_sliding
                shared_value_cache = shared_value_cache_sliding
                position_embeddings = sliding_pos
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                shared_key_cache=shared_key_cache,
                shared_value_cache=shared_value_cache,
            )
        return self.norm(hidden_states)


class Gemma4AssistantDraftModule(nn.Module):
    def __init__(
        self,
        assistant_model_dir: str,
        target_model_dir: str,
        max_position_embeddings: int,
        input_sequence_length: int = 1,
    ):
        super().__init__()
        assistant_config_dict = load_json_file(
            Path(assistant_model_dir) / "config.json"
        )
        target_config_dict = load_json_file(Path(target_model_dir) / "config.json")

        self.assistant_model_dir = str(Path(assistant_model_dir).resolve())
        self.target_model_dir = str(Path(target_model_dir).resolve())
        self.assistant_config_dict = assistant_config_dict
        self.target_config_dict = target_config_dict

        text_config = Gemma4TextConfig(**assistant_config_dict["text_config"])
        self.text_config = text_config
        self.backbone_hidden_size = int(assistant_config_dict["backbone_hidden_size"])
        self.use_ordered_embeddings = bool(
            assistant_config_dict.get("use_ordered_embeddings", False)
        )
        if self.use_ordered_embeddings:
            raise ValueError(
                "Ordered assistant embeddings are not supported in this export path yet."
            )

        # The local transformers build in this workspace predates Gemma4 assistant
        # support and cannot instantiate an all-shared-KV Gemma4TextModel.
        # Build a structurally equivalent backbone with local K/V modules present,
        # then ignore those weights in our custom assistant attention path.
        text_model_config = Gemma4TextConfig(**assistant_config_dict["text_config"])
        text_model_config.num_kv_shared_layers = 0
        text_model = Gemma4TextModel(text_model_config)
        self.model = Gemma4AssistantBackbone(
            text_model,
            max_position_embeddings=max_position_embeddings,
            input_sequence_length=input_sequence_length,
        )
        self.pre_projection = nn.Linear(
            2 * self.backbone_hidden_size, text_config.hidden_size, bias=False
        )
        self.post_projection = nn.Linear(
            text_config.hidden_size, self.backbone_hidden_size, bias=False
        )
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.lm_head.weight = self.model.embed_tokens.weight

        self._load_weights(Path(assistant_model_dir) / "model.safetensors")
        # After weights are loaded, fold every HF ``Gemma4RMSNorm`` into a
        # single fused ``xhquant.nn.RMSNorm`` (baking ``1 + weight``) so the
        # exported graph emits one RMSNorm op per layer instead of the
        # pow/reduce/add/mul decomposition. This also removes the redundant
        # trailing ``Mul`` (issue #4).
        self._convert_norms_for_export()

    def _convert_norms_for_export(self) -> None:
        backbone = self.model
        for layer in backbone.layers:
            attn = layer.self_attn
            if isinstance(attn.q_norm, Gemma4RMSNorm):
                attn.q_norm = _convert_gemma4_rmsnorm(attn.q_norm)
            for attr in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                module = getattr(layer, attr, None)
                if isinstance(module, Gemma4RMSNorm):
                    setattr(layer, attr, _convert_gemma4_rmsnorm(module))
        if isinstance(backbone.norm, Gemma4RMSNorm):
            backbone.norm = _convert_gemma4_rmsnorm(backbone.norm)
        target_dtype = self.lm_head.weight.dtype
        self.to(dtype=target_dtype)

    def _load_weights(self, weight_path: Path) -> None:
        state_dict = load_file(str(weight_path), device="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "lm_head.weight",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.k_norm",
            "model.layers.0.self_attn.v_norm",
            "model.layers.1.self_attn.k_proj",
            "model.layers.1.self_attn.v_proj",
            "model.layers.1.self_attn.k_norm",
            "model.layers.1.self_attn.v_norm",
            "model.layers.2.self_attn.k_proj",
            "model.layers.2.self_attn.v_proj",
            "model.layers.2.self_attn.k_norm",
            "model.layers.2.self_attn.v_norm",
            "model.layers.3.self_attn.k_proj",
            "model.layers.3.self_attn.v_proj",
            "model.layers.3.self_attn.k_norm",
            "model.layers.3.self_attn.v_norm",
        )
        bad_missing = [
            key for key in missing_keys if not key.startswith(allowed_missing_prefixes)
        ]
        if bad_missing or unexpected_keys:
            raise RuntimeError(
                "Unexpected assistant checkpoint mismatch: "
                f"missing={bad_missing}, unexpected={unexpected_keys}"
            )
        logger = get_xhquant_logger()
        logger.info(f"Loaded Gemma4 assistant weights from {weight_path}")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_seq_length: torch.Tensor,
        local_attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor,
        shared_key_cache_sliding: torch.Tensor,
        shared_value_cache_sliding: torch.Tensor,
        shared_key_cache_full: torch.Tensor,
        shared_value_cache_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.pre_projection(inputs_embeds)
        hidden_states = self.model(
            inputs_embeds=hidden_states,
            past_seq_length=past_seq_length,
            local_attention_mask=local_attention_mask,
            global_attention_mask=global_attention_mask,
            shared_key_cache_sliding=shared_key_cache_sliding,
            shared_value_cache_sliding=shared_value_cache_sliding,
            shared_key_cache_full=shared_key_cache_full,
            shared_value_cache_full=shared_value_cache_full,
        )
        logits = self.lm_head(hidden_states)
        assistant_hidden_state = self.post_projection(hidden_states)
        return logits, assistant_hidden_state


class XHGemma4AssistantDraftModel(BaseModel):
    def __init__(
        self,
        assistant_model_dir: str,
        target_model_dir: str,
        wrap_cfg: ConfigDict,
        quant_config: ConfigDict,
        frontend_type: str = "TorchFX",
        export_cfg: ConfigDict | None = None,
    ):
        self.assistant_model_dir = str(Path(assistant_model_dir).resolve())
        self.target_model_dir = str(Path(target_model_dir).resolve())
        export_cfg = export_cfg or ConfigDict(
            input_names=[
                "inputs_embeds",
                "past_seq_length",
                "local_attention_mask",
                "global_attention_mask",
                "shared_key_cache_sliding",
                "shared_value_cache_sliding",
                "shared_key_cache_full",
                "shared_value_cache_full",
            ],
            output_names=["logits", "assistant_hidden_state"],
        )
        super().__init__(
            hf_model=self.assistant_model_dir,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=True,
            export_cfg=export_cfg,
        )

    def get_tokenizer(self, **kwargs):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.target_model_dir, trust_remote_code=True, **kwargs
        )

    def init_wrap_model(self, hf_model=None):
        del hf_model
        max_position_embeddings = int(self.wrap_cfg.get("max_sequence_length", 2048))
        input_sequence_length = int(self.wrap_cfg.get("input_sequence_length", 1))
        self._wrap_model = Gemma4AssistantDraftModule(
            assistant_model_dir=self.assistant_model_dir,
            target_model_dir=self.target_model_dir,
            max_position_embeddings=max_position_embeddings,
            input_sequence_length=input_sequence_length,
        )
        self._wrap_model = self._wrap_model.to(
            dtype=resolve_torch_dtype(self.wrap_cfg.get("dtype", "float16"))
        )
        self._wrap_model.eval()
        return self._wrap_model

    def prepare_inputs(self, data=None):
        if self._wrap_model is None:
            self.init_wrap_model()
        target_text_cfg = self._wrap_model.target_config_dict["text_config"]
        input_sequence_length = int(self.wrap_cfg.get("input_sequence_length", 1))
        max_sequence_length = int(self.wrap_cfg.get("max_sequence_length", 2048))
        hidden_size = int(self._wrap_model.backbone_hidden_size)
        dtype = resolve_torch_dtype(self.wrap_cfg.get("dtype", "float16"))
        local_window = aligned(
            int(target_text_cfg.get("sliding_window", 1024))
            + input_sequence_length
            - 1,
            16,
        )
        full_heads = int(
            target_text_cfg.get(
                "num_global_key_value_heads", target_text_cfg["num_key_value_heads"]
            )
        )
        full_head_dim = int(
            target_text_cfg.get("global_head_dim", target_text_cfg["head_dim"])
        )

        if data is None:
            data = dict(
                inputs_embeds=torch.zeros(
                    (1, input_sequence_length, hidden_size * 2), dtype=dtype
                ),
                # ``past_seq_length`` is the starting offset for the cos/sin
                # DynamicSlice. Shape ``(1,)`` int32 matches the main Gemma4
                # ``_Gemma4TextModel`` contract.
                past_seq_length=torch.zeros((1,), dtype=torch.int32),
                local_attention_mask=torch.zeros(
                    (1, 1, input_sequence_length, local_window), dtype=dtype
                ),
                global_attention_mask=torch.zeros(
                    (1, 1, input_sequence_length, max_sequence_length), dtype=dtype
                ),
                shared_key_cache_sliding=torch.zeros(
                    (
                        1,
                        int(target_text_cfg["num_key_value_heads"]),
                        local_window,
                        int(target_text_cfg["head_dim"]),
                    ),
                    dtype=dtype,
                ),
                shared_value_cache_sliding=torch.zeros(
                    (
                        1,
                        int(target_text_cfg["num_key_value_heads"]),
                        local_window,
                        int(target_text_cfg["head_dim"]),
                    ),
                    dtype=dtype,
                ),
                shared_key_cache_full=torch.zeros(
                    (1, full_heads, max_sequence_length, full_head_dim),
                    dtype=dtype,
                ),
                shared_value_cache_full=torch.zeros(
                    (1, full_heads, max_sequence_length, full_head_dim),
                    dtype=dtype,
                ),
            )

        return (
            data["inputs_embeds"],
            data["past_seq_length"],
            data["local_attention_mask"],
            data["global_attention_mask"],
            data["shared_key_cache_sliding"],
            data["shared_value_cache_sliding"],
            data["shared_key_cache_full"],
            data["shared_value_cache_full"],
        )

    @property
    def need_quant(self):
        return True

    def get_empty_hf_model(self, device_map="cpu", **kwargs):
        del device_map, kwargs
        return None

    def get_hf_model(self, device_map="cpu", **kwargs):
        del device_map, kwargs
        return None
