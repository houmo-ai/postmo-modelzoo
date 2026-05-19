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
NDEVICE=1
MTP="false"

parse_args "$@"

case "${MODEL_NAME}:${MODEL_SIZE}" in
    qwen3.5:0.8b|qwen3.5:2b|qwen3.5:4b|qwen3.5:9b|qwen3.6:27b|qwen3.6:35b-a3b)
        ;;
    *)
        echo "Error: Unsupported model combination '${MODEL_NAME}-${MODEL_SIZE}'." >&2
        echo "       qwen3.5 supports: 0.8b, 2b, 4b, 9b" >&2
        echo "       qwen3.6 supports: 27b, 35b-a3b" >&2
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
        echo "Download raw model (${MODEL_NAME}-${MODEL_SIZE})."
        python3 get_model.py --type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Start model quantization (${MODEL_NAME}-${MODEL_SIZE})."
    python3 ptq.py --model-name "${MODEL_NAME}" --model-size "${MODEL_SIZE}"
fi

if should_run_step "build"; then
    echo "Start model compilation (${MODEL_NAME}-${MODEL_SIZE})."
    if [ "${MTP}" = "true" ]; then
        python3 build_mtp.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    else
        python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model (${MODEL_NAME}-${MODEL_SIZE})."
        GET_MODEL_ARGS=(--type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}")
        if [ "${MTP}" = "true" ]; then
            GET_MODEL_ARGS+=(--mtp)
        fi
        python3 get_model.py "${GET_MODEL_ARGS[@]}"
    fi
    echo "Execute demo."
    if [ "${MTP}" = "true" ]; then
        python3 demo_mtp.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    else
        python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"

        if command -v llm_perf &>/dev/null; then
            echo "Execute performance case (${MODEL_NAME}-${MODEL_SIZE})."
            python3 "${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py" --path "output/${HOUMO_TARGET}/hmquant/quant_embedding.pt"
            devices_param=$(get_devices_param "${NDEVICE}")
            if [[ "${NDEVICE}" -gt 1 ]]; then
                model_suffix="hmms"
            else
                model_suffix="hmm"
            fi
            llm_perf --model_name "${MODEL_NAME}-${MODEL_SIZE}" --devices "${devices_param}" \
                --input 256,1024,2048 --output 256,256,256 --loop 1 --batch 1 \
                --prefill "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.${model_suffix}" \
                --decode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.${model_suffix}" \
                --visual "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_visual_896x896x2.hmm" \
                --embedding "output/${HOUMO_TARGET}/hmquant/quant_embedding.bin"
        fi
    fi
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
