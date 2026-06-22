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
MODEL_NAME="gpt-oss"
MODEL_SIZE="20b"
NDEVICE=1
parse_args "$@"

cd "${SCRIPT_DIR}"

check_step_python_packages || exit 1

if should_run_step "build"; then
    echo "Start model compilation."
    python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."
        python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"
    fi
    echo "Execute demo."
    python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"

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
            --input 256 --output 256 --loop 1 --batch 1 \
            --prefill "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.${model_suffix}" \
            --decode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.${model_suffix}" \
            --embedding "output/${HOUMO_TARGET}/hmquant/quant_embedding.bin"
    fi
fi
