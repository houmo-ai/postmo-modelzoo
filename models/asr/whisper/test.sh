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
MODEL_NAME="whisper"
MODEL_SIZE="medium" # medium or large-v3-turbo

parse_args "$@"

case "${MODEL_NAME}-${MODEL_SIZE}" in
    whisper-medium|whisper-large-v3-turbo)
        ;;
    *)
        echo "Error: Unsupported model '${MODEL_NAME}-${MODEL_SIZE}'." >&2
        echo "       Supported models: whisper-medium, whisper-large-v3-turbo" >&2
        exit 1
        ;;
esac

check_houmo_target "xh2"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="whisper_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

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
    python3 ptq.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
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
    python3 demo.py --audio audio.mp3 --tokenizer_path "${MODEL_NAME}-${MODEL_SIZE}" --language "auto" \
    --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"

    echo "Execute cpp demo (${MODEL_NAME}-${MODEL_SIZE})."
    cd cpp && ./build_linux.sh && cd ..
    ./bin/demo \
    --encode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_encode.hmm" \
    --prefill "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
    --decode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
    --tokenizer "${MODEL_NAME}-${MODEL_SIZE}" \
    --audio "${HOUMO_EXAMPLES_PATH}/data/audio/audio.mp3"
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
