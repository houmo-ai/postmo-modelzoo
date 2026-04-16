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
MODEL_SIZE="0.6b"
MODEL_NAME="qwen3_asr"
parse_args "$@"

check_houmo_target "xh2"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3-asr"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

check_step_python_packages || exit 1

# Determine processor directory, PTQ model name, and demo script based on MODEL_NAME and MODEL_SIZE
PROCESSOR_DIR="Qwen3-ASR-0.6B"
PTQ_MODEL="Qwen3-ASR-0.6B"
DEMO_SCRIPT="demo_asr.py"
case "$MODEL_NAME" in
    qwen3_asr)
        case "$MODEL_SIZE" in
            0.6b)
                ;;
            1.7b)
                PROCESSOR_DIR="Qwen3-ASR-1.7B"
                PTQ_MODEL="Qwen3-ASR-1.7B"
                ;;
            *)
                echo "Error: Unsupported model_size '$MODEL_SIZE' for model '$MODEL_NAME'" >&2
                echo "Supported combinations: qwen3_asr with 0.6b or 1.7b; qwen3_forcealigner with 0.6b"
                exit 1
                ;;
        esac
        ;;
    qwen3_forcealigner)
        case "$MODEL_SIZE" in
            0.6b)
                PROCESSOR_DIR="Qwen3-ForcedAligner-0.6B"
                PTQ_MODEL="Qwen3-ForcedAligner-0.6B"
                DEMO_SCRIPT="demo_forcealigner.py"
                ;;
            *)
                echo "Error: Unsupported model_size '$MODEL_SIZE' for model '$MODEL_NAME'" >&2
                echo "Supported combinations: qwen3_asr with 0.6b or 1.7b; qwen3_forcealigner with 0.6b"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Error: Unknown model '$MODEL_NAME'" >&2
        echo "Supported model_name values: qwen3_asr, qwen3_forcealigner"
        exit 1
        ;;
esac

echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Step: ${STEP}"
echo "=============================================="

if should_run_step "quant"; then
    if ! should_skip_download; then
        echo "Download raw model."
        python3 get_model.py --type raw --model_size ${MODEL_SIZE} --model_name ${MODEL_NAME}
    fi
    echo "Start model quantization."
    python3 ptq.py --model ${PTQ_MODEL} --model_name ${MODEL_NAME}
fi

if should_run_step "build"; then
    echo "Start model compilation."
    python3 build.py --model_name ${MODEL_NAME}
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."
        python3 get_model.py --type hmm --model_size ${MODEL_SIZE} --model_name ${MODEL_NAME}
    fi
    echo "Execute demo ${DEMO_SCRIPT}"
    if [ "$DEMO_SCRIPT" = "demo_asr.py" ]; then
        python3 demo_asr.py --processor_dir ${PROCESSOR_DIR}
    else
        python3 demo_forcealigner.py --processor_dir ${PROCESSOR_DIR}
    fi
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi