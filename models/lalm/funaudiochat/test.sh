#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

show_help() {
    cat <<'EOF'
Usage: test.sh [options]
Options:
  -s, --step              Steps: quant, build, demo, all. Default: demo.
  -size, --model_size     Model size. Default: 8b.
  -name, --model_name     Model name. Default: funaudiochat.
  --ndevice               Number of devices. Default: 1.
  --demo_mode             Demo mode. Choices: s2t, s2s, e2e.
                          If omitted, run s2t, s2s, and e2e in order.
  --skip_download         Skip model downloads. Raw model is required for quant; HMM model is used for demo-only runs.
  --system_prompt         System prompt passed to demo.py.
  -h, --help              Show this help message.
EOF
    exit 0
}

STEP="demo"
SKIP_DOWNLOAD="false"
MODEL_NAME="funaudiochat"
MODEL_SIZE="8b"
NDEVICE=1

parse_args "$@"

check_houmo_target "xh2"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="funaudiochat_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} quant build demo"
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    check_gpu require || exit 1
    if ! should_skip_download; then
        echo "Download raw FunAudioChat model."
        python3 get_model.py \
            --type raw \
            --model_name "${MODEL_NAME}" \
            --model_size "${MODEL_SIZE}" \
            --ndevice "${NDEVICE}"
    fi
    echo "Export FunAudioChat HMONNX graphs and golden data."
    python3 ptq.py \
        --model-name "${MODEL_NAME}" \
        --model-size "${MODEL_SIZE}" \
        --model-dir "Fun-Audio-Chat-8B" \
        --audio "${HOUMO_EXAMPLES_PATH}/data/audio/question.wav"
fi

if should_run_step "demo" && ! should_skip_download; then
    echo "Download pre-compiled model (${MODEL_NAME}-${MODEL_SIZE})."
    python3 get_model.py \
        --type hmm \
        --model_name "${MODEL_NAME}" \
        --model_size "${MODEL_SIZE}" \
        --ndevice "${NDEVICE}"
fi

if should_run_step "build"; then
    echo "Build FunAudioChat HMM graphs."
    python3 build.py \
        --model_dir "output/${HOUMO_TARGET}/hmquant" \
        --output_dir "output/${HOUMO_TARGET}" \
        --ndevice "${NDEVICE}" \
        --enable_common_subgraph
fi

if should_run_step "demo"; then
    if [[ -n "${DEMO_MODE}" ]]; then
        demo_modes=("${DEMO_MODE}")
    else
        demo_modes=(s2t s2s e2e)
    fi

    for demo_mode in "${demo_modes[@]}"; do
        demo_args=(--stage "${demo_mode}" --ndevice "${NDEVICE}")
        demo_args+=("${SYSTEM_PROMPT_ARGS[@]}")

        echo "Execute ${demo_mode} demo."
        python3 demo.py "${demo_args[@]}"
    done
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
