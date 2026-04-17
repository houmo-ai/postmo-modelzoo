# SPDX-License-Identifier: Apache-2.0
_base_ = ["../_base_/xh2a_base.py"]
trace_type = "TorchFX"

quant_config = dict(
    inputs=dict(
        pixel_values=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
    )
)

model = dict(
    type="XHQwen3_5VisionModel",
    wrap_cfg=dict(
        max_sequence_length=2048,
        max_size_w=448,
        max_size_h=448,
        max_size_t=2,
        temporal_patch_size=2,
        patch_size=16,
    ),
    quant_config=quant_config,
)