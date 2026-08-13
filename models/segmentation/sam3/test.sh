#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

STEP="demo"
SKIP_DOWNLOAD="false"
MODEL_NAME="sam3"
MODEL_SIZE="0.8b"
VENV_DIR="sam3_venv"
parse_args "$@"

check_houmo_target "xh2"
cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
    setup_python_venv "${VENV_DIR}" "${SCRIPT_DIR}/requirements.txt" "${VENV_DIR} SAM3"
fi

cleanup() {
    if [[ "${TEST_VENV_ACTIVE:-0}" -eq 1 ]]; then
        cleanup_python_venv "${VENV_DIR}"
    fi
}
trap cleanup EXIT

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if should_skip_download; then
        echo "Skip raw SAM3 model download."
    else
        echo "Download raw SAM3 model (${MODEL_NAME}-${MODEL_SIZE})."
        python3 get_model.py \
            --type raw \
            --model_name "${MODEL_NAME}" \
            --model_size "${MODEL_SIZE}"
    fi

    echo "Start SAM3 quantization (${MODEL_NAME}-${MODEL_SIZE})."
    python3 ptq.py \
        --model_name "${MODEL_NAME}" \
        --model_size "${MODEL_SIZE}"
fi

if should_run_step "build"; then
    echo "Start SAM3 compilation (${MODEL_NAME}-${MODEL_SIZE})."
    python3 build.py \
        --model_name "${MODEL_NAME}" \
        --model_size "${MODEL_SIZE}"
fi

if should_run_step "demo"; then
    if [[ "${STEP}" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled SAM3 model (${MODEL_NAME}-${MODEL_SIZE})."
        python3 get_model.py \
            --type hmm \
            --model_name "${MODEL_NAME}" \
            --model_size "${MODEL_SIZE}"
    fi

    echo "Execute SAM3 demo (${MODEL_NAME}-${MODEL_SIZE})."
    python3 demo.py \
        --model_name "${MODEL_NAME}" \
        --model_size "${MODEL_SIZE}"
fi
