# SPDX-License-Identifier: Apache-2.0
_base_ = [
    "../_base_/xh2a_base.py",
]

frontend_type = "TorchFX"

quant_config = dict(
    inputs=dict(
        inputs_embeds=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        time_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int64"),
            )
        ),
        hight_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int64"),
            )
        ),
        width_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int64"),
            )
        ),
        past_seq_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        current_input_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        linear_attn_mask=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
    ),
    w_schema=dict(
        fp_mode="ssfp",
        hidden_bit=False,
        bits=8,
    ),
    act_schema=dict(
        fp_mode="sefp",
        hidden_bit=True,
        bits=8,
    ),
)

release = dict(
    xh_version="xh2",
    modelscope_name="qwen3_5",
)

model = dict(
    type="XHQwen3_5Model",
    wrap_cfg=dict(
        batch_size=1,
        max_pe_length=262144,
        max_sequence_length=2048,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        linear_attention_mode="auto",
        linear_chunk_size=64,
        support_long_context_over_fp16_limit=False,
        kv_cache=dict(
            cache_axis=2,
        ),
        enable_rope=True,
        enable_auto_offload=True,
        auto_offload_max_memory=None,
    ),
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "time_position_ids",
            "hight_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
            "linear_attn_mask",
        ],
        output_names=["logits"],
    ),
)
