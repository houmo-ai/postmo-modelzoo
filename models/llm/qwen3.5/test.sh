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
MODEL_SIZE="9b"
parse_args "$@"

case "${MODEL_SIZE}" in
    2b|4b|9b|27b|35b-a3b)
        ;;
    *)
        echo "Error: Unsupported model size '${MODEL_SIZE}', support: 2b, 4b, 9b, 27b, 35b-a3b." >&2
        exit 1
        ;;
esac

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3.5_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model (size: ${MODEL_SIZE})."
        python3 get_model.py --type raw --model_size ${MODEL_SIZE}
    fi
    echo "Start model quantization (size: ${MODEL_SIZE})."
    python3 ptq.py --model_size ${MODEL_SIZE}
fi

if should_run_step "build"; then
    echo "Start model compilation (size: ${MODEL_SIZE})."
    python3 build.py --model_size ${MODEL_SIZE}
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model (size: ${MODEL_SIZE})."
        python3 get_model.py --type hmm --model_size ${MODEL_SIZE}
    fi
    echo "Execute demo."
    python3 demo.py
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
