#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
if [[ ! -f "${MODELS_DIR}/test_common.sh" ]]; then
    echo "Error: test_common.sh not found." >&2
    exit 1
fi
source "${MODELS_DIR}/test_common.sh"

STEP="demo"
SKIP_DOWNLOAD="false"
MODEL_NAME="ornith1.0"
MODEL_SIZE="35b"
NDEVICE=1
CONFIG_PATH="${SCRIPT_DIR}/config.yaml"
MODEL_DIR="${SCRIPT_DIR}/Ornith-1.0-35B"

parse_args "$@"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="ornith_venv"
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
    if ! should_skip_download && [[ ! -d "${MODEL_DIR}" ]]; then
        echo "Download raw model deepreinforce-ai/Ornith-1.0-35B."
        GET_MODEL_ARGS=(
            --config "${CONFIG_PATH}"
            --type raw
            --model_name "${MODEL_NAME}"
            --model_size "${MODEL_SIZE}"
        )
        python3 get_model.py "${GET_MODEL_ARGS[@]}"
    fi

    PTQ_ARGS=(
        --config "${CONFIG_PATH}"
        --model_name "${MODEL_NAME}"
        --model_size "${MODEL_SIZE}"
        --model_dir "${MODEL_DIR}"
    )
    if [[ -n "${QUANT_TYPE}" && "${QUANT_TYPE}" == w4* ]]; then
        PTQ_ARGS+=(--bits 4)
    fi
    python3 ptq.py "${PTQ_ARGS[@]}"
fi

if should_run_step "build"; then
    BUILD_ARGS=(
        --config "${CONFIG_PATH}"
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
        --config "${CONFIG_PATH}"
        --model_name "${MODEL_NAME}"
        --model_size "${MODEL_SIZE}"
        --ndevice "${NDEVICE}"
    )
    DEMO_ARGS+=("${SYSTEM_PROMPT_ARGS[@]}")
    python3 demo.py "${DEMO_ARGS[@]}"
fi
