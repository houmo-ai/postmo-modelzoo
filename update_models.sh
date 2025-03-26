#!/bin/bash
set -e

target=$HOUMO_TARGET
user=$1
pwd=$2
date=20250317

UPDATE_HMQUANT() {
    dir=$1
    name=$2
    size=$3
    hmquant=hmquant_${name}${size}_${date}.zip
    pushd $dir/$name/output/$target/hmquant
    zip -r ${hmquant} * -x hmquant_${name}_with_act/*
    curl -u${user}:${pwd} -T ${hmquant} $MODELZOO_URL/models/$name/${hmquant}
    popd
}

UPDATE_HMQUANT_LM() {
    dir=$1
    name=$2
    size=$3
    hmquant=hmquant_${name}${size}_${date}.zip
    pushd $dir/$name/output/$target/hmquant
    zip -r ${hmquant} * -x prefill/hmquant_${name}_with_act/* -x decoder/hmquant_${name}_with_act/*
    curl -u${user}:${pwd} -T ${hmquant} $MODELZOO_URL/models/$name/${hmquant}
    popd
}

UPDATE_HMM_LM() {
    dir=$1
    name=$2
    size=$3
    ncore=$4
    hmm=hmm_${name}${size}${ncore}_${date}.zip
    pushd $dir/$name/output/$target
    zip -r ${hmm} *.hmm hmquant/quant_embedding.pt
    curl -u${user}:${pwd} -T ${hmm} $MODELZOO_URL/models/$name/${hmm}
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

UPDATE_HMQUANT_LM "models/llm" "qwen2" "_256_4096"
UPDATE_HMQUANT_LM "models/llm" "qwen2.5" "_256_4096"
UPDATE_HMQUANT_LM "models/llm" "deepseek" "_256_4096"

UPDATE_HMM_LM "models/llm" "qwen2" "_256_4096" "_4cores"
UPDATE_HMM_LM "models/llm" "qwen2.5" "_256_4096" "_4cores"
UPDATE_HMM_LM "models/llm" "deepseek" "_256_4096" "_4cores"
