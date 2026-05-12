#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -e

# Optional: JPEG/PNG for vision HMONNX export if src/images/qwen2_vl_demo.jpeg is missing.
# export VISION_IMAGE=/path/to/sample.jpg

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

STEP="demo"
SKIP_DOWNLOAD="false"
MODEL_NAME="qwen3.6"
MODEL_SIZE="35b-a3b"
RAW_HF_DIR="${SCRIPT_DIR}/${MODEL_NAME}"
PERF_CONFIG="config.yaml"
if [[ "${MODEL_NAME}" == "qwen3.6" ]]; then
    PERF_CONFIG="config.qwen3.6.yaml"
fi

parse_args "$@"

case "${MODEL_NAME}" in
    qwen3.5|qwen3.6)
        ;;
    *)
        echo "Error: Unsupported model name '${MODEL_NAME}', support: qwen3.5, qwen3.6." >&2
        exit 1
        ;;
esac

case "${MODEL_SIZE}" in
    0.8b|2b|4b|9b|27b|35b-a3b)
        ;;
    *)
        echo "Error: Unsupported model size '${MODEL_SIZE}', support: 0.8b, 2b, 4b, 9b, 27b, 35b-a3b." >&2
        exit 1
        ;;
esac

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3.5_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
    gptq_requirements="${HOUMO_EXAMPLES_PATH}/hmodel/gptqmodel/requirements.txt"
    if [ -f "${gptq_requirements}" ]; then
        pip3 install -r "${gptq_requirements}"
    fi
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model (${MODEL_NAME} ${MODEL_SIZE}) → ${RAW_HF_DIR}."
        python3 get_model.py --type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Start model quantization (${MODEL_NAME} ${MODEL_SIZE})."
    PTQ_ARGS=(--model "${RAW_HF_DIR}")
    if [[ -n "${VISION_IMAGE}" ]]; then
        PTQ_ARGS+=(--vision-image-path "${VISION_IMAGE}")
    fi
    python3 ptq.py "${PTQ_ARGS[@]}"
fi

if should_run_step "build"; then
    echo "Start model compilation (${MODEL_NAME} ${MODEL_SIZE})." 
    python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model (${MODEL_NAME} ${MODEL_SIZE})."
        python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Execute demo."
    if [[ "${MODEL_NAME}" == "qwen3.5" ]]; then
        python3 demo.py \
            --tokenizer_dir "${RAW_HF_DIR}" \
            --prefill_path "output/${HOUMO_TARGET}/${MODEL_NAME}_prefill.hmm" \
            --decode_path "output/${HOUMO_TARGET}/${MODEL_NAME}_decode.hmm" \
            --vision_path "output/${HOUMO_TARGET}/${MODEL_NAME}_visual_896x896x2.hmm" \
            --max_size_w 896 \
            --max_size_h 896

    else
        python3 demo.py
    fi
    echo "Execute performance case."
    python3 "${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py" --path "output/${HOUMO_TARGET}/hmquant/quant_embedding.pt"
    llm_perf -c "${PERF_CONFIG}"
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
