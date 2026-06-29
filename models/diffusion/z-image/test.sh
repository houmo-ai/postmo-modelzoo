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
MODEL_NAME="z-image-turbo"
MODEL_SIZE="6b"
parse_args "$@"

check_houmo_target "xh2"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="zimage_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
    sudo apt-get update
    sudo apt-get install -y python3-tk
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model (${MODEL_NAME}-${MODEL_SIZE})."
        python3 get_model.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi

    echo "Start model quantization (${MODEL_NAME}-${MODEL_SIZE})."
    if [ -n "${QUANT_TYPE}" ]; then
        PTQ_ARGS+=(--quant_type "${QUANT_TYPE}")
    fi
    python3 ptq.py "${PTQ_ARGS[@]}"
fi

if should_run_step "build"; then
    echo "Start model compilation (${MODEL_NAME}-${MODEL_SIZE})."
    BUILD_ARGS=(--model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --enable_common_subgraph)
    if [ -n "${CONTEXT_LENGTH}" ]; then
        BUILD_ARGS+=(--context_length "${CONTEXT_LENGTH}")
    fi
    python3 build.py "${BUILD_ARGS[@]}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model (${MODEL_NAME}-${MODEL_SIZE})."
        GET_MODEL_ARGS=(--type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}")
        if [ -n "${QUANT_TYPE}" ]; then
            GET_MODEL_ARGS+=(--quant_type "${QUANT_TYPE}")
        fi
        python3 get_model.py "${GET_MODEL_ARGS[@]}"
    fi

    echo "Execute demo."
    python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
