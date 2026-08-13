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
  -s, --step              Step to run. Only demo is supported. Default: demo.
  -size, --model_size     Model size. Default: 8b.
  -name, --model_name     Model name. Default: funaudiochat.
  --ndevice               Number of devices. Default: 1.
  --demo_mode             Demo mode. Choices: s2t, s2s, e2e.
                          If omitted, run s2t, s2s, and e2e in order.
  --skip_download         Skip pre-compiled model download.
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

if [[ "${STEP}" != "demo" ]]; then
    echo "Error: Fun-Audio-Chat test.sh currently only supports the demo step." >&2
    exit 1
fi

check_houmo_target "xh2"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="funaudiochat_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

check_step_python_packages || exit 1

if ! should_skip_download; then
    echo "Download pre-compiled model (${MODEL_NAME}-${MODEL_SIZE})."
    python3 get_model.py \
        --type hmm \
        --model_name "${MODEL_NAME}" \
        --model_size "${MODEL_SIZE}" \
        --ndevice "${NDEVICE}"
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
