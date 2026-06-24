#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}/../../../models"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

STEP="demo"
SKIP_DOWNLOAD="false"
MODEL_CONFIGS=("qwen3:0.6b" "qwen3.5:2b" "qwen3-vl:4b" "glm-asr:nano-2512" "qwen3-asr:1.7b" "whisper:large-v3-turbo")
NDEVICE=1
parse_args "$@"

if [[ -n "${MODEL_NAME}" && -n "${MODEL_SIZE}" ]]; then
    MODEL_CONFIGS=("${MODEL_NAME}:${MODEL_SIZE}")
fi

cd "${SCRIPT_DIR}"

if is_asic; then
    check_step_python_packages || exit 0
else
    echo "Demo only support xh2 platform, skip demo."
    exit 0
fi

if should_run_step "demo"; then
    for config in "${MODEL_CONFIGS[@]}"; do
        MODEL_NAME="${config%%:*}"
        MODEL_SIZE="${config##*:}"
        echo "Processing model: ${MODEL_NAME}-${MODEL_SIZE}"

        if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
            echo "Download pre-compiled model."
            python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" \
             --extract_dir "models/${MODEL_NAME}-${MODEL_SIZE}"
        fi
        echo "Convert embedding file"
        if [[ "${MODEL_NAME}" != *"whisper"* ]]; then
            python3 "${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py" \
            --path "models/${MODEL_NAME}-${MODEL_SIZE}/hmquant/quant_embedding.pt"
        fi
    done
    ./build_linux.sh > /dev/null 2>&1
    cd ${SCRIPT_DIR}
    ctest --test-dir build --output-on-failure
    for config in "${MODEL_CONFIGS[@]}"; do
        MODEL_NAME="${config%%:*}"
        MODEL_SIZE="${config##*:}"
        echo "Running inference for model: ${MODEL_NAME}-${MODEL_SIZE}"
        if [[ $config == "qwen3:0.6b" ]]; then
            ./bin/sample_infer --model qwen3_llm \
            --prefill "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
            --decode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
            --embedding "models/${MODEL_NAME}-${MODEL_SIZE}/hmquant/quant_embedding.bin" \
            --tokenizer "tokenizers/${MODEL_NAME}-${MODEL_SIZE}" \
            --prompt "介绍下你自己"
        fi
        if [[ $config == "qwen3.5:2b" ]]; then
            ./bin/sample_infer --model qwen35_mllm \
            --prefill "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
            --decode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
            --vision "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_visual_448x448x2.hmm" \
            --embedding "models/${MODEL_NAME}-${MODEL_SIZE}/hmquant/quant_embedding.bin" \
            --tokenizer "tokenizers/${MODEL_NAME}-${MODEL_SIZE}" \
            --prompt "介绍下图片" \
            --image "tests/data/a.png"
            ./bin/sample_infer --model qwen35_mllm \
            --prefill "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
            --decode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
            --vision "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_visual_896x896x2.hmm" \
            --embedding "models/${MODEL_NAME}-${MODEL_SIZE}/hmquant/quant_embedding.bin" \
            --tokenizer "tokenizers/${MODEL_NAME}-${MODEL_SIZE}" \
            --prompt "介绍下图片" \
            --image "tests/data/b.jpg"
        fi
        if [[ $config == "qwen3-vl:4b" ]]; then
            ./bin/sample_infer --model qwen3_vlm \
            --prefill "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
            --decode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
            --vision "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_visual_448x448x2.hmm" \
            --embedding "models/${MODEL_NAME}-${MODEL_SIZE}/hmquant/quant_embedding.bin" \
            --tokenizer "tokenizers/${MODEL_NAME}-${MODEL_SIZE}" \
            --prompt "分别介绍下两个图片" \
            --image "tests/data/a.png" --image "tests/data/b.jpg"
        fi
        if [[ $config == "glm-asr:nano-2512" ]]; then
            ./bin/sample_glm_asr \
            --encode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_encode.hmm" \
            --prefill "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
            --decode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
            --embedding "models/${MODEL_NAME}-${MODEL_SIZE}/hmquant/quant_embedding.bin" \
            --tokenizer "tokenizers/${MODEL_NAME}-${MODEL_SIZE}" \
            --audio "tests/data/long_audio.mp3"
        fi
        if [[ $config == "qwen3-asr:1.7b" ]]; then
            ./bin/sample_qwen3_asr \
            --encode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_encode.hmm" \
            --prefill "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
            --decode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
            --embedding "models/${MODEL_NAME}-${MODEL_SIZE}/hmquant/quant_embedding.bin" \
            --tokenizer "tokenizers/${MODEL_NAME}-${MODEL_SIZE}" \
            --audio "tests/data/long_audio.mp3"
        fi
        if [[ $config == "whisper:large-v3-turbo" ]]; then
            ./bin/sample_whisper_asr \
            --encode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_encode.hmm" \
            --prefill "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
            --decode "models/${MODEL_NAME}-${MODEL_SIZE}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
            --tokenizer "tokenizers/${MODEL_NAME}-${MODEL_SIZE}" \
            --audio "tests/data/long_audio.mp3"
        fi
    done
fi