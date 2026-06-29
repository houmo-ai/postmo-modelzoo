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
MODEL_NAME="gemma4"
MODEL_SIZE="e2b"
NDEVICE=1
MTP="false"
for arg in "$@"; do
    if [ "$arg" = "--mtp" ]; then
        MTP="true"
        break
    fi
done
parse_args "$@"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="gemma4_venv"
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
        echo "Download raw model."
        if [ "${MTP}" = "true" ]; then
            python3 get_model.py --type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --mtp
        else
            python3 get_model.py --type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
        fi
    fi
    if [ "${MTP}" = "true" ]; then
        echo "Start model quantization with MTP."
        ASSISTANT_SIZE=$(echo "${MODEL_SIZE}" | tr '[:lower:]' '[:upper:]')
        python3 ptq.py --model-name "${MODEL_NAME}" --model-size "${MODEL_SIZE}" --assistant-model "${SCRIPT_DIR}/gemma-4-${ASSISTANT_SIZE}-it-assistant"
    else
        echo "Start model quantization."
        python3 ptq.py --model-name "${MODEL_NAME}" --model-size "${MODEL_SIZE}"
    fi
fi

if should_run_step "build"; then
    echo "Start model compilation."
    if [ "${MTP}" = "true" ]; then
        python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}" --mtp
    else
        python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"
    fi
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."
        if [ "${MTP}" = "true" ]; then
            python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --mtp
        else
            python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
        fi
    fi
    echo "Execute python demo."
    if [ "${MTP}" = "true" ]; then
        python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}" --mtp
    else
        python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"
    fi
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
