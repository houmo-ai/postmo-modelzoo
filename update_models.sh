#!/bin/bash
set -e

target=$HOUMO_TARGET
user=$1
pwd=$2
date=20250625

UPDATE_HMQUANT() {
    dir=$1
    name=$2
    size=$3
    hmquant=hmquant_${target}_${name}_${size}_${date}.zip
    pushd $dir/output/$target/hmquant
    echo "zip -r ${hmquant} * -x hmquant_${name}_with_act/*"
    zip -r ${hmquant} * -x hmquant_${name}_with_act/*
    echo "curl -u${user}:${pwd} -T ${hmquant} $HOUMO_MODELZOO_URL/models/$name/${hmquant}"
    curl -u${user}:${pwd} -T ${hmquant} $HOUMO_MODELZOO_URL/models/$name/${hmquant}
    popd
}

UPDATE_HMQUANT_LM() {
    dir=$1
    name=$2
    size=$3
    hmquant=hmquant_${target}_${name}_${size}_${date}.zip
    pushd $dir/output/$target/hmquant
    echo "zip -r ${hmquant} * -x prefill/hmquant_${name}_with_act/* -x decoder/hmquant_${name}_with_act/*"
    zip -r ${hmquant} * -x prefill/hmquant_${name}_with_act/* -x decoder/hmquant_${name}_with_act/*
    echo "curl -u${user}:${pwd} -T ${hmquant} $HOUMO_MODELZOO_URL/models/$name/${hmquant}"
    curl -u${user}:${pwd} -T ${hmquant} $HOUMO_MODELZOO_URL/models/$name/${hmquant}
    popd
}

UPDATE_HMM_LM() {
    dir=$1
    name=$2
    size=$3
    ncore=$4
    hmm=hmm_${target}_${name}_${size}_${ncore}_${date}.zip
    pushd $dir/output/$target
    echo "zip -r ${hmm} *.hmm *.hmms hmquant/quant_embedding.pt"
    zip -r ${hmm} *.hmm *.hmms hmquant/quant_embedding.pt
    echo "curl -u${user}:${pwd} -T ${hmm} $HOUMO_MODELZOO_URL/models/$name/${hmm}"
    curl -u${user}:${pwd} -T ${hmm} $HOUMO_MODELZOO_URL/models/$name/${hmm}
    popd
}

# UPDATE_HMQUANT "models/backbone/resnet50" "resnet50"
# UPDATE_HMQUANT "models/backbone/mobilenetv2" "mobilenetv2"
# UPDATE_HMQUANT "models/backbone/efficientnet" "efficientnet"
# UPDATE_HMQUANT "models/detection/yolov3" "yolov3"
# UPDATE_HMQUANT "models/detection/yolov5s" "yolov5s"
# UPDATE_HMQUANT "models/detection/yolov8m" "yolov8m"
# UPDATE_HMQUANT "models/autodrive/yolop" "yolop"
# UPDATE_HMQUANT "models/asr/wenet" "wenet"
# UPDATE_HMQUANT "models/diffusion/sdxl" "sdxl"

# UPDATE_HMQUANT_LM "models/llm/qwen2.5" "qwen2.5" "7b_256_8k"
# UPDATE_HMQUANT_LM "models/llm/deepseek" "deepseek" "8b_256_8k"
# UPDATE_HMQUANT_LM "models/llm/qwen3" "qwen3" "8b_256_2k"
# UPDATE_HMQUANT_LM "models/llm/qwen3-14b" "qwen3" "14b_256_2k"

# UPDATE_HMM_LM "models/llm/qwen2.5" "qwen2.5" "7b_256_8k" "4cores"
# UPDATE_HMM_LM "models/llm/deepseek" "deepseek" "8b_256_8k" "4cores"
# UPDATE_HMM_LM "models/llm/qwen3" "qwen3" "8b_256_8k" "4cores"

# UPDATE_HMM_LM "models/llm/qwen3" "qwen3" "8b_256_8k" "2cores"
# UPDATE_HMM_LM "models/llm/qwen3" "qwen3" "8b_256_2k" "2cores"
# UPDATE_HMM_LM "models/llm/qwen3-14b" "qwen3" "14b_256_8k" "2cores"
# UPDATE_HMM_LM "models/llm/qwen3-14b" "qwen3" "14b_256_2k" "2cores"
# UPDATE_HMM_LM "models/llm/qwen3-32b" "qwen3" "32b_256_2k" "2cores"
