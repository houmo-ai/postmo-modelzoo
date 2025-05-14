#!/bin/bash
set -e

target=$HOUMO_TARGET
user=$1
pwd=$2
date=20250514

UPDATE_HMQUANT() {
    dir=$1
    name=$2
    size=$3
    hmquant=hmquant_${name}${size}_${date}.zip
    pushd $dir/$name/output/$target/hmquant
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
    hmquant=hmquant_${name}_${size}_${date}.zip
    pushd $dir/$name/output/$target/hmquant
    echo "zip -r ${hmquant} * -x prefill/hmquant_${name}_with_act/* -x decoder/hmquant_${name}_with_act/*"
    zip -r ${hmquant} * -x prefill/hmquant_${name}_with_act/* -x decoder/hmquant_${name}_with_act/*
    echo "curl -u${user}:${pwd} -T ${hmm} $HOUMO_MODELZOO_URL/models/$name/${hmm}"
    curl -u${user}:${pwd} -T ${hmquant} $HOUMO_MODELZOO_URL/models/$name/${hmquant}
    popd
}

UPDATE_HMM_LM() {
    dir=$1
    name=$2
    size=$3
    ncore=$4
    hmm=hmm_${name}_${size}_${ncore}_${date}.zip
    pushd $dir/$name/output/$target
    echo "zip -r ${hmm} *.hmm hmquant/quant_embedding.pt"
    zip -r ${hmm} *.hmm hmquant/quant_embedding.pt
    echo "curl -u${user}:${pwd} -T ${hmm} $HOUMO_MODELZOO_URL/models/$name/${hmm}"
    curl -u${user}:${pwd} -T ${hmm} $HOUMO_MODELZOO_URL/models/$name/${hmm}
    popd
}

UPDATE_HMQUANT "models/backbone" "resnet50"
UPDATE_HMQUANT "models/backbone" "mobilenetv2"
UPDATE_HMQUANT "models/backbone" "efficientnet"
UPDATE_HMQUANT "models/detection" "yolov3"
UPDATE_HMQUANT "models/detection" "yolov5s"
UPDATE_HMQUANT "models/detection" "yolov8m"
UPDATE_HMQUANT "models/autodrive" "yolop"
UPDATE_HMQUANT "models/asr" "wenet"
UPDATE_HMQUANT "models/diffusion" "sdxl"

UPDATE_HMQUANT_LM "models/llm" "qwen2.5" "256_4k"
UPDATE_HMQUANT_LM "models/llm" "deepseek" "256_4k"
UPDATE_HMQUANT_LM "models/llm" "qwen3" "256_8k"

UPDATE_HMM_LM "models/llm" "qwen2.5" "256_4k" "4cores"
UPDATE_HMM_LM "models/llm" "deepseek" "256_4k" "4cores"
UPDATE_HMM_LM "models/llm" "qwen3" "256_8k" "4cores"
