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
MODEL_NAME="qwen3.6"
MODEL_SIZE="35b-a3b"
NDEVICE=1
MTP="false"
LOAD_MODE=""

parse_args "$@"

if [ "${LORA}" = "true" ] && { [ "${MODEL_NAME}" != "qwen3.6" ] || [ "${MODEL_SIZE}" != "35b-a3b" ]; }; then
    echo "Error: LoRA mode is only provided for qwen3.6-35b-a3b." >&2
    exit 1
fi

case "${MODEL_NAME}:${MODEL_SIZE}" in
    qwen3.5:0.8b)
        ;;
    qwen3.5:2b)
        ;;
    qwen3.5:4b)
        ;;
    qwen3.5:9b)
        ;;
    qwen3.5:122b-a10b)
        LOAD_MODE="--LazyMode"
        ;;
    qwen3.6:27b)
        ;;
    qwen3.6:35b-a3b)
        LOAD_MODE="--LazyMode"
        ;;
    *)
        echo "Error: Unsupported model combination '${MODEL_NAME}-${MODEL_SIZE}'." >&2
        echo "       qwen3.5 supports: 0.8b, 2b, 4b, 9b, 122b-a10b" >&2
        echo "       qwen3.6 supports: 27b, 35b-a3b" >&2
        exit 1
        ;;
esac

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3.5_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
    gptq_requirements="${HOUMO_EXAMPLES_PATH}/hmodel/gptqmodel/requirements.txt"
    if [ -f "${gptq_requirements}" ]; then
        pip3 install -r "${gptq_requirements}"
    fi
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model (${MODEL_NAME}-${MODEL_SIZE})."
        GET_MODEL_ARGS=(--type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}")
        if [ -n "${QUANT_TYPE}" ]; then
            GET_MODEL_ARGS+=(--quant_type "${QUANT_TYPE}")
        fi
        if [ "${LORA}" = "true" ]; then
            GET_MODEL_ARGS+=(--lora)
        fi
        python3 get_model.py "${GET_MODEL_ARGS[@]}"
    fi
    echo "Start model quantization (${MODEL_NAME}-${MODEL_SIZE})."
    PTQ_ARGS=(--model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}")
    if [ "${LORA}" = "true" ]; then
        PTQ_ARGS+=(--lora)
    fi
    if [ "${MTP}" = "true" ]; then
        PTQ_ARGS+=(--mtp)
    fi
    python3 ptq.py "${PTQ_ARGS[@]}"
fi

if should_run_step "build"; then
    echo "Start model compilation (${MODEL_NAME}-${MODEL_SIZE})."
    BUILD_ARGS=(--model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}")
    if [ "${LORA}" = "true" ]; then
        BUILD_ARGS+=(--lora)
    fi
    if [ "${MTP}" = "true" ]; then
        BUILD_ARGS+=(--mtp)
    fi
    python3 build.py "${BUILD_ARGS[@]}"
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model (${MODEL_NAME}-${MODEL_SIZE})."
        GET_MODEL_ARGS=(--type hmm --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}")
        if [ "${MTP}" = "true" ]; then
            GET_MODEL_ARGS+=(--mtp)
        fi
        if [ "${LORA}" = "true" ]; then
            GET_MODEL_ARGS+=(--lora)
        fi
        if [ -n "${QUANT_TYPE}" ]; then
            GET_MODEL_ARGS+=(--quant_type "${QUANT_TYPE}")
        fi
        python3 get_model.py "${GET_MODEL_ARGS[@]}"
    fi

    echo "Execute demo."
    demo_args=(--model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}")
    if [ "${MTP}" = "true" ]; then

        demo_args+=("${SYSTEM_PROMPT_ARGS[@]}")
        python3 python/demo_mtp.py "${demo_args[@]}"
    else
        if [ "${LORA}" = "true" ]; then
            demo_args+=(--lora)
        fi
        demo_args+=("${SYSTEM_PROMPT_ARGS[@]}")
        python3 python/demo.py "${demo_args[@]}"
        if [ "${LORA}" = "true" ]; then
            echo "Skip prefix caching, cpp demo, and llm_perf in LoRA mode."
        else
            python3 python/demo_prefix_caching.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"

            python3 "${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py" --path "output/${HOUMO_TARGET}/hmquant/quant_embedding.pt"
            echo "Execute cpp demo."
            cd cpp && ./build_linux.sh && cd ..
            if [[ "${NDEVICE}" -eq 1 ]]; then
                ./bin/demo --config config.yaml \
                    --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}" \
                    --prompt "描述这些图片" \
                    --image_path "${HOUMO_EXAMPLES_PATH}/data/pic/beach.jpeg"
            fi
            if command -v llm_perf &>/dev/null; then
                echo "Execute performance case (${MODEL_NAME}-${MODEL_SIZE})."
                visual_prefix="output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_visual"
                visual_model_path=""
                for gear in 1536 704 384 196 96; do
                    if [ -f "${visual_prefix}_m${gear}.hmm" ]; then
                        visual_model_path="${visual_prefix}_m${gear}.hmm"
                        break
                    fi
                done
                if [ -z "${visual_model_path}" ]; then
                    for path in "${visual_prefix}"_*.hmm; do
                        if [ -f "${path}" ]; then
                            visual_model_path="${path}"
                            break
                        fi
                    done
                fi
                if [ -z "${visual_model_path}" ] && [ -f "${visual_prefix}.hmm" ]; then
                    visual_model_path="${visual_prefix}.hmm"
                fi
                if [ -z "${visual_model_path}" ]; then
                    echo "Error: No dynamic visual model found for '${MODEL_NAME}-${MODEL_SIZE}'." >&2
                    exit 1
                fi
                devices_param=$(get_devices_param "${NDEVICE}")
                if [[ "${NDEVICE}" -gt 1 ]]; then
                    model_suffix="hmms"
                else
                    model_suffix="hmm"
                fi
                llm_perf --model_name "${MODEL_NAME}-${MODEL_SIZE}" \
                    --devices "${devices_param}" \
                    --input 256,1024,2048 \
                    --output 256,256,256 \
                    --loop 1 --batch 1 ${LOAD_MODE} \
                    --prefill "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.${model_suffix}" \
                    --decode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.${model_suffix}" \
                    --visual "${visual_model_path}" \
                    --embedding "output/${HOUMO_TARGET}/hmquant/quant_embedding.bin"
            fi
        fi
    fi
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
