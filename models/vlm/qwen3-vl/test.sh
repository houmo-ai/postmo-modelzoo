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
        python3 get_model.py --type raw
    fi
    echo "Downloading dataset for quantization."
    mkdir -p data/calib_data/LMUData
    link_subdirs "../../../data/datasets/LMUData" "data/calib_data/LMUData"
    echo "Start model quantization."
    python3 ptq.py --data_files ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json  ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib-samples 64 --calib_dataset vllm_custom_data
fi

if should_run_step "build"; then
    echo "Start model compilation."
    python3 build.py
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."
        python3 get_model.py --type hmm
    fi
    echo "Execute python demo."
    python3 demo.py

    echo "Execute performance case."
    cd "${SCRIPT_DIR}"
    python3 ../../../tools/llm_perf/convert_embed.py --path output/xh2/hmquant/quant_embedding.pt
    llm_perf -c config.yaml

    echo "Execute cpp demo."
    cd cpp && ./build.sh && cd ..
    ./bin/example_cxx_qwen3_vl --image ../../../data/pic/beach.jpeg
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi