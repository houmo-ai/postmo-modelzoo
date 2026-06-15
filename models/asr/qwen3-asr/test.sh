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
MODEL_NAME="qwen3-asr"
parse_args "$@"

check_houmo_target "xh2"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3_asr_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

check_step_python_packages || exit 1

DEMO_SCRIPT="demo_asr.py"
case "$MODEL_NAME" in
    qwen3-asr)
        case "$MODEL_SIZE" in
            0.6b)
                ;;
            1.7b)
                ;;
            *)
                echo "Error: Unsupported model_size '$MODEL_SIZE' for model '$MODEL_NAME'" >&2
                echo "Supported combinations: qwen3-asr with 0.6b or 1.7b; qwen3-forcealigner with 0.6b"
                exit 1
                ;;
        esac
        ;;
    qwen3-forcealigner)
        case "$MODEL_SIZE" in
            0.6b)
                DEMO_SCRIPT="demo_forcealigner.py"
                ;;
            *)
                echo "Error: Unsupported model_size '$MODEL_SIZE' for model '$MODEL_NAME'" >&2
                echo "Supported combinations: qwen3-asr with 0.6b or 1.7b; qwen3-forcealigner with 0.6b"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Error: Unknown model '$MODEL_NAME'" >&2
        echo "Supported model_name values: qwen3-asr, qwen3-forcealigner"
        exit 1
        ;;
esac

echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Step: ${STEP}"
echo "=============================================="

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model."
        python3 get_model.py --type raw --model_size "${MODEL_SIZE}" --model_name "${MODEL_NAME}"
    fi
    echo "Start model quantization."
    python3 ptq.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
fi

if should_run_step "build"; then
    echo "Start model compilation."
    python3 build.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."
        python3 get_model.py --type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Execute demo ${DEMO_SCRIPT}"
    if [ "$DEMO_SCRIPT" = "demo_asr.py" ]; then
        python3 demo_asr.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
        python3 "${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py" --path "output/${HOUMO_TARGET}/hmquant/quant_embedding.pt"
        if command -v llm_perf &>/dev/null; then
            echo "Execute performance case (${MODEL_NAME}-${MODEL_SIZE})."
            cd "${SCRIPT_DIR}"
            devices_param=$(get_devices_param "${NDEVICE}")
            if [ "$MODEL_SIZE" = "0.6b" ]; then
                tokenizer_path="Qwen3-ASR-0.6B/tokenizer.json"
            elif [ "$MODEL_SIZE" = "1.7b" ]; then
                tokenizer_path="Qwen3-ASR-1.7B/tokenizer.json"
            fi
            llm_perf --encode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_encode.hmm" \
                --prefill "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.hmm" \
                --decode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.hmm" \
                --embedding "output/${HOUMO_TARGET}/hmquant/quant_embedding.bin" \
                --tokenizer "${tokenizer_path}" \
                --audio "${HOUMO_EXAMPLES_PATH}/data/audio/audio.mp3"
        fi
    else
        python3 demo_forcealigner.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi