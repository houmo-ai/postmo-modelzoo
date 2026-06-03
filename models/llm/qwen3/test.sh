#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

STEP="demo"
SKIP_DOWNLOAD="false"
MODEL_NAME="qwen3"
MODEL_SIZE="8b"
NDEVICE=1
parse_args "$@"

case "${MODEL_SIZE}" in
    0.6b|1.7b|4b|8b|14b)
        ;;
    *)
        echo "Error: Unsupported model size '${MODEL_SIZE}', support: 0.6b, 1.7b, 8b, 14b." >&2
        exit 1
        ;;
esac

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model (size: ${MODEL_NAME}-${MODEL_SIZE})."
        python3 get_model.py --type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Start model quantization (size: ${MODEL_NAME}-${MODEL_SIZE})."
    python3 ptq.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
fi

if should_run_step "build"; then
    echo "Start model compilation (size: ${MODEL_NAME}-${MODEL_SIZE})."
    python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download precompiled model (size: ${MODEL_NAME}-${MODEL_SIZE})."
        python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Execute demo (size: ${MODEL_SIZE})."
    python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"

    if command -v llm_perf &>/dev/null; then
        echo "Execute performance case (size: ${MODEL_NAME}-${MODEL_SIZE})."
        python3 "${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py" --path output/${HOUMO_TARGET}/hmquant/quant_embedding.pt
        devices_param=$(get_devices_param "${NDEVICE}")
        if [[ "${NDEVICE}" -gt 1 ]]; then
            model_suffix="hmms"
        else
            model_suffix="hmm"
        fi
        llm_perf --model_name "${MODEL_NAME}-${MODEL_SIZE}" --devices "${devices_param}" \
            --input 256,1024,2048 --output 256,256,256 --loop 1 --batch 1 \
            --prefill output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.${model_suffix} \
            --decode output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.${model_suffix} \
            --embedding output/${HOUMO_TARGET}/hmquant/quant_embedding.bin
    fi
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
