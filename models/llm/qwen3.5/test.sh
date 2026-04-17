#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Optional: JPEG/PNG for vision HMONNX export if src/images/qwen2_vl_demo.jpeg is missing.
# export VISION_IMAGE=/path/to/sample.jpg
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

PERF_CONFIG="config.yaml"
RAW_HF_DIR="${SCRIPT_DIR}/qwen3.5"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3.5_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
    pip3 install -r ../../../hmodel/gptqmodel/requirements.txt
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model (size: ${MODEL_SIZE}) → ${RAW_HF_DIR}."
        python3 get_model.py --type raw --model_size ${MODEL_SIZE}
    fi
    echo "Start model quantization (size: ${MODEL_SIZE})."
    PTQ_ARGS="--model ${RAW_HF_DIR}"
    if [[ -n "${VISION_IMAGE}" ]]; then
        PTQ_ARGS+=" --vision-image-path ${VISION_IMAGE}"
    fi
    python3 ptq.py ${PTQ_ARGS}
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
    echo "Execute performance case."
    python3 ../../../tools/llm_perf/convert_embed.py --path "output/xh2/hmquant/quant_embedding.pt"
    llm_perf -c "${PERF_CONFIG}"
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
