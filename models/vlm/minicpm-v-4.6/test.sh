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
MODEL_SIZE="v-4.6"
NDEVICE=1

parse_args "$@"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="minicpm_venv"
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
        GET_MODEL_ARGS=(
            --type raw
            --model_name "${MODEL_NAME}"
            --model_size "${MODEL_SIZE}"
        )
        python3 get_model.py "${GET_MODEL_ARGS[@]}"
    fi
    echo "Start model quantization (${MODEL_NAME}-${MODEL_SIZE})."
    PTQ_ARGS=(
        --model-name "${MODEL_NAME}"
        --model-size "${MODEL_SIZE}"
    )
    python3 ptq.py "${PTQ_ARGS[@]}"
fi

if should_run_step "build"; then
    echo "Start model compilation (${MODEL_NAME}-${MODEL_SIZE})."
    BUILD_ARGS=(
        --model_name "${MODEL_NAME}"
        --model_size "${MODEL_SIZE}"
        --ndevice "${NDEVICE}"
    )
    python3 build.py "${BUILD_ARGS[@]}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model (${MODEL_NAME}-${MODEL_SIZE})."
        GET_MODEL_ARGS=(
            --type hmm
            --model_name "${MODEL_NAME}"
            --model_size "${MODEL_SIZE}"
        )
        python3 get_model.py "${GET_MODEL_ARGS[@]}"
    fi
    echo "Execute Python demo (${MODEL_NAME}-${MODEL_SIZE})."
    DEMO_ARGS=(
        --model_name "${MODEL_NAME}"
        --model_size "${MODEL_SIZE}"
        --ndevice "${NDEVICE}"
    )
    DEMO_ARGS+=("${SYSTEM_PROMPT_ARGS[@]}")
    python3 demo.py "${DEMO_ARGS[@]}"
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
