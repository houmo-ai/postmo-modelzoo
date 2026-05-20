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
MODEL_NAME="qwen3-vl"
MODEL_SIZE="8b"
NDEVICE=1
parse_args "$@"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwenvl"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    if ! should_skip_download; then
        echo "Download raw model."
        python3 get_model.py --type raw --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    fi
    echo "Downloading dataset for quantization."
    mkdir -p data/calib_data/LMUData
    link_subdirs "${HOUMO_EXAMPLES_PATH}/data/datasets/LMUData" "data/calib_data/LMUData"
    echo "Start model quantization."
    python3 ptq.py --model-name "${MODEL_NAME}" --model-size "${MODEL_SIZE}" \
        --data_files ${HOUMO_EXAMPLES_PATH}/hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json \
            ${HOUMO_EXAMPLES_PATH}/hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json \
            ${HOUMO_EXAMPLES_PATH}/hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json \
            ${HOUMO_EXAMPLES_PATH}/hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json \
        --calib-samples 64 --calib_dataset vllm_custom_data
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
    echo "Execute python demo."
    python3 demo.py --model_name "${MODEL_NAME}" --model_size "${MODEL_SIZE}"
    
    python3 "${HOUMO_EXAMPLES_PATH}/tools/llm_perf/convert_embed.py" --path "output/${HOUMO_TARGET}/hmquant/quant_embedding.pt"
    if command -v llm_perf &>/dev/null; then
        echo "Execute performance case (${MODEL_NAME}-${MODEL_SIZE})."
        cd "${SCRIPT_DIR}"
        if [[ "${NDEVICE}" -gt 1 ]]; then
            model_suffix="hmms"
        else
            model_suffix="hmm"
        fi
        devices_param=$(get_devices_param "${NDEVICE}")
        llm_perf --model_name "${MODEL_NAME}-${MODEL_SIZE}" --devices "${devices_param}" \
            --input 256,1024,2048 --output 256,256,256 --loop 1 --batch 1 \
            --prefill "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_prefill.${model_suffix}" \
            --decode "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_decode.${model_suffix}" \
            --visual "output/${HOUMO_TARGET}/${MODEL_NAME}-${MODEL_SIZE}_visual_448x448x2.hmm" \
            --embedding "output/${HOUMO_TARGET}/hmquant/quant_embedding.bin"
    fi

    echo "Execute cpp demo."
    cd cpp && ./build.sh && cd ..
    ./bin/example_cxx_qwen3_vl --image ${HOUMO_EXAMPLES_PATH}/data/pic/beach.jpeg
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi