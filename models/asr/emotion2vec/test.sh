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
MODEL_NAME="emotion2vec"
MODEL_SIZE="plus_large"
parse_args "$@"

check_houmo_target "xh2"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="emotion2vec_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

export PYTHONPATH="/hmdd/3rdparty/xh2modelzoo${PYTHONPATH:+:${PYTHONPATH}}"

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model (${MODEL_NAME}-${MODEL_SIZE})."
        python3 get_model.py --type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Start model quantization (${MODEL_NAME}-${MODEL_SIZE})."
    python3 ptq.py --model-name "${MODEL_NAME}" --model-size "${MODEL_SIZE}"
fi

if should_run_step "build"; then
    echo "Start model compilation (${MODEL_NAME}-${MODEL_SIZE})."
    python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model (${MODEL_NAME}-${MODEL_SIZE})."
        python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Execute python demo (${MODEL_NAME}-${MODEL_SIZE})."
    python3 demo.py \
        --hmm "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}.hmm" \
        --classifier "output/${HOUMO_TARGET}/hmquant/classifier.npz" \
        --model-dir "${SCRIPT_DIR}/emotion2vec_plus_large" \
        --audio "${HOUMO_EXAMPLES_PATH}/data/audio/audio.mp3" \
        --output-dir "${SCRIPT_DIR}/results"
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
