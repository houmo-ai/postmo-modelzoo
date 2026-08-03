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
MODEL_NAME="qwen3-tts"
MODEL_SIZE="0.6b-customvoice"
DEMO_MODE="oneshot"

PARSE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --demo_mode)
            if [[ $# -ge 2 && "$2" != -* ]]; then
                DEMO_MODE="$2"
                shift 2
            else
                shift
            fi
            ;;
        *)
            PARSE_ARGS+=("$1")
            shift
            ;;
    esac
done
parse_args "${PARSE_ARGS[@]}"

if [[ "${DEMO_MODE}" != "oneshot" && "${DEMO_MODE}" != "streaming" ]]; then
    echo "Error: Unsupported demo_mode '${DEMO_MODE}', support: oneshot, streaming." >&2
    exit 1
fi

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3_tts_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} example"
fi

check_step_python_packages || exit 1

COMMON_MODEL_ARGS=(--model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}")
COMMON_DEMO_ARGS=(--mode "${DEMO_MODE}")

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model."
        python3 get_model.py --type raw "${COMMON_MODEL_ARGS[@]}"
    fi
    echo "Start model quantization."
    python3 ptq.py "${COMMON_MODEL_ARGS[@]}"
fi

if should_run_step "build"; then
    echo "Start model compilation."
    python3 build.py --enable_common_subgraph "${COMMON_MODEL_ARGS[@]}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."
        python3 get_model.py --type hmm "${COMMON_MODEL_ARGS[@]}"
    fi

    sudo apt update
    sudo apt install sox libsox-dev -y

    echo "Execute demo."
    cd "${SCRIPT_DIR}"
    if [[ "${MODEL_SIZE}" == "0.6b-base" ]]; then
        python3 demo_base.py "${COMMON_MODEL_ARGS[@]}" "${COMMON_DEMO_ARGS[@]}"
    elif [[ "${MODEL_SIZE}" == "0.6b-customvoice" || "${MODEL_SIZE}" == "1.7b-customvoice" ]]; then
        python3 demo.py "${COMMON_MODEL_ARGS[@]}" "${COMMON_DEMO_ARGS[@]}"
        python3 python/demo.py "${COMMON_MODEL_ARGS[@]}" "${COMMON_DEMO_ARGS[@]}"

        echo "Build C++ streaming demo."
        "${SCRIPT_DIR}/cpp/build_linux.sh"

        echo "Execute C++ streaming demo."
        "${SCRIPT_DIR}/bin/qwen3_tts_text_demo" \
            "${COMMON_MODEL_ARGS[@]}" \
            --mode streaming \
            --output_wav "${SCRIPT_DIR}/output_custom_voice_cpp.wav"
    else
        echo "Unsupported MODEL_SIZE for demo step: ${MODEL_SIZE}" >&2
        echo "Supported values: 0.6b-base, 0.6b-customvoice, 1.7b-customvoice" >&2
        exit 1
    fi
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi

echo "✅ Script execution completed successfully."
exit 0
