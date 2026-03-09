#!/usr/bin/env bash
set -e

STEP="all"
MODEL_TYPE="precompiled"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step         execution step, default is all, support: all, demo, build."
    echo "  -t, --model_type   The method for getting the compiled model, default is precompiled, support: precompiled, compile."
    echo "  -h, --help         help information"
    exit 0
}

link_subdirs() {
    if [ $# -ne 2 ]; then
        echo "link_subdirs <source_dir> <target_dir>"
        return 1
    fi

    local src_dir="$1"
    local dest_dir="$2"

    if [ ! -d "$src_dir" ]; then
        echo "error: $src_dir unavailable!"
        return 1
    fi

    mkdir -p "$dest_dir"

    for subdir in "$src_dir"/*/; do
        [ -d "$subdir" ] || continue
        local subdir_name=$(basename "$subdir")
        local link_path="$dest_dir/$subdir_name"
        if [ -e "$link_path" ]; then
            rm -rf "$link_path"
        fi

        local abs_subdir=$(realpath "$subdir")
        ln -s "$abs_subdir" "$link_path"
    done
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--step)
            STEP="$2"
            shift 2
        ;;
        -t|--model_type)
            MODEL_TYPE="$2"
            shift 2
        ;;
        -h|--help)
            show_help
        ;;
        *)
            echo "Error: Unknown parameter '$1'" >&2
            show_help
        ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

VENV_FLAG=0
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    VENV_FLAG=1
fi

dir_path="qwenvl"
if [[ "$VENV_FLAG" -eq "1" ]]; then
    echo "⚠ Create python3 venv for ${dir_path} demo."
    PY_EXE=$(command -v python3)
    SITE_PACKAGES=$($PY_EXE -c "import site; print(site.getsitepackages()[0])")
    if [[ $PY_EXE == */opt/venv* ]]; then
        virtualenv --python=$PY_EXE --extra-search-dir=$SITE_PACKAGES $dir_path
        VENV_PYTHON="${dir_path}/bin/python3"
        VENV_SITE=$(${VENV_PYTHON} -c "import site; print(site.getsitepackages()[0])")
        echo "export ORIGINAL_PYTHONPATH=\$PYTHONPATH" >> $dir_path/bin/activate  # 保存原始值
        echo "export PYTHONPATH=${VENV_SITE}:${SITE_PACKAGES}:\$ORIGINAL_PYTHONPATH" >> $dir_path/bin/activate
        echo "export PYTHONPATH=\$ORIGINAL_PYTHONPATH" >> $dir_path/bin/deactivate  # 恢复外部原始值
        echo "unset ORIGINAL_PYTHONPATH" >> $dir_path/bin/deactivate  # 清除临时变量
        sed -i 's/include-system-site-packages = true/include-system-site-packages = false/g' $dir_path/pyvenv.cfg
    else
        virtualenv --python=$PY_EXE --system-site-packages $dir_path
    fi
    source $dir_path/bin/activate
    pip3 install -r requirements.txt
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "build" ]; then
    if [[ "$MODEL_TYPE" == "precompiled" ]]; then
        echo "Download precompiled model."
        python3 get_model.py --type hmm
    else
        FOUND_GPU=0
        echo "Checking GPU..."
        if command -v nvidia-smi &> /dev/null; then
            gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | wc -l)
            if [ $gpu_count -gt 0 ]; then
                FOUND_GPU=1
                echo "Found NVIDIA GPU, count: ${gpu_count}"
            else
                echo "⚠ Not found NVIDIA GPU device."
            fi
        else
            echo "⚠ Not install NVIDIA GPU driver."
        fi

        PACKAGE_PATTERN=hmquant
        FOUND_PACKAGE=0
        echo "================================"
        echo "Checking python3 package: $PACKAGE_PATTERN"
        if command -v python3 &>/dev/null && command -v pip3 &>/dev/null; then
            if pip3 list --format=columns 2>/dev/null | grep -E "^$PACKAGE_PATTERN" >/dev/null 2>&1; then
                echo "✓ Found python3 package: $PACKAGE_PATTERN"
                pip3 list --format=columns 2>/dev/null | grep -E "^$PACKAGE_PATTERN" | while read -r line; do
                    echo "  - $line"
                done
                FOUND_PACKAGE=1
            else
                echo "✗ Not found package: $PACKAGE_PATTERN"
            fi
        else
            echo "⚠ Not found python3 or pip3."
            exit 0
        fi

        if [[ "$FOUND_PACKAGE" -eq 1 && "$FOUND_GPU" -eq 1 ]]; then
            echo "Start to quant and compile model."
            python3 get_model.py --type raw
            mkdir -p data/calib_data/LMUData
            link_subdirs "../../../data/datasets/LMUData" "data/calib_data/LMUData"
            python3 ptq.py --data_files ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json  ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json ../../../hmodel/xh2/data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib-samples 64 --calib_dataset vllm_custom_data
            python3 build.py
        else
            echo "✗ Not support model quantization and compilation."
            exit 1
        fi
    fi
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "demo" ]; then
    echo "Execute demo."
    python3 demo.py
    python3 ../../../tools/llm_perf/convert_embed.py --path output/xh2/hmquant/quant_embedding.pt --type vllm
    llm_perf -c ../../../tools/llm_perf/configs/qwen3_vl_config.json
fi

if [[ "$VENV_FLAG" -eq "1" ]]; then
    deactivate
    rm -rf $dir_path
fi