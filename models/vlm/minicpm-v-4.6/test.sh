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
MODEL_NAME="minicpm"
MODEL_SIZE="v-4.6"
NDEVICE=1

parse_args "$@"

if [[ "${STEP}" == "all" ]]; then
    # No local PTQ flow is available, so the complete supported flow uses HMMs.
    STEP="demo"
elif should_run_step "quant"; then
    echo "Error: MiniCPM-V 4.6 does not support the quant step; use build, demo, or all." >&2
    exit 1
fi

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

if should_run_step "build"; then
    echo "Start model compilation."

    python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" --ndevice "${NDEVICE}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."

        python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Execute python demo."
    demo_args=(
        --model_name "${MODEL_NAME}"
        --model_size "${MODEL_SIZE}"
        --ndevice "${NDEVICE}"
    )
    demo_args+=("${SYSTEM_PROMPT_ARGS[@]}")
    python3 demo.py "${demo_args[@]}"
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
