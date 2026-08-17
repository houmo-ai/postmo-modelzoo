#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

STEP="demo"
SKIP_DOWNLOAD="false"
MODEL_NAME="minicpm"
MODEL_SIZE="v-4.5"
NDEVICE=1
parse_args "$@"
cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="minicpm_venv"
setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} quant build"
check_step_python_packages || exit 1

if should_run_step "quant"; then
    check_gpu require || exit 1
    if ! should_skip_download; then
        python3 get_model.py --type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    python3 ptq.py
fi

if should_run_step "build"; then
    python3 build.py \
        --model_name "${MODEL_NAME}" \
        --model_size "${MODEL_SIZE}" \
        --ndevice "${NDEVICE}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."
        python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Execute python demo."
    python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"

    python3 "${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py" --path "output/${HOUMO_TARGET}/hmquant/quant_embedding.pt"
    if command -v llm_perf &>/dev/null; then
        echo "Execute performance case (${MODEL_NAME}-${MODEL_SIZE})."
        cd "${SCRIPT_DIR}"
        if [[ "${NDEVICE}" -gt 1 ]]; then
            model_suffix="hmms"
        else
            model_suffix="hmm"
        fi
        devices_param=$(get_devices_param "${NDEVICE}")
        llm_perf --model_name "${MODEL_NAME}-${MODEL_SIZE}" --devices "${devices_param}" \
            --input 256,1024,2048 --output 256,256,256 --loop 1 --batch 1 \
            --prefill "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.${model_suffix}" \
            --decode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.${model_suffix}" \
            --visual "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_visual_1x.hmm" \
            --embedding "output/${HOUMO_TARGET}/hmquant/quant_embedding.bin"
        llm_perf --model_name "${MODEL_NAME}-${MODEL_SIZE}" --devices "${devices_param}" \
            --input 256,1024,2048 --output 256,256,256 --loop 1 --batch 1 \
            --prefill "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.${model_suffix}" \
            --decode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.${model_suffix}" \
            --visual "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_visual_6x.hmm" \
            --embedding "output/${HOUMO_TARGET}/hmquant/quant_embedding.bin"
    fi

fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
