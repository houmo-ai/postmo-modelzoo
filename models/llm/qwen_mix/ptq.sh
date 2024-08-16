#!/bin/bash
set -e

MODEL_DIR=/data01/datasets/qwen1.5-7b-chat-hf
HMLLMQUANT_DIR=/usr/local/src/hmllmquant

export PYTHONPATH=:$HMLLMQUANT_DIR:$PYTHONPATH

mkdir -p weights

batch=1

# download wikitext2 cache if you can't download it directly
# wget $MODEL_PATH/models/qwen/wikitext2_cache.zip

python3 $HMLLMQUANT_DIR/scripts/generate_smooth_f.py --model $MODEL_DIR \
        --seed=2024 --rotate --a_bits=8 --k_bits=8 \
        --v_bits=8 --w_bits=4 --w_clip True --w_gptq  \
        --act_bits=8 --attn_bits=8 --down_bits=16 \
        --rotate_pre_rope --rotate_post_rope \
        --online_hadamard None --rdtype float32 \
        --rotate_ov --fully_quant --bsz $batch \
        --a_dynamic_method pertensor \
        --save_qmodel_path weights/qwen1.5-7b-chat_w4a8_rope.pt

python3 $HMLLMQUANT_DIR/scripts/quant_export.py --model $MODEL_DIR --smooth_f weights/qwen1.5-7b-chat_w4a8_rope.pt  --config $HMLLMQUANT_DIR/configs/mix_v1.py --cache_len 2048 --model_name qwen --output_path output/$HOUMO_TARGET/result