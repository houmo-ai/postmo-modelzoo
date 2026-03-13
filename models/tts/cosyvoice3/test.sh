#!/usr/bin/env bash
set -e

STEP="demo"
MODEL_TYPE="precompiled"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step         execution step, default is demo, support: all, build, demo."
    echo "  -t, --model_type   The method for getting the compiled model, default is precompiled, support: precompiled, compile."
    echo "  -h, --help         help information"
    exit 0
}

check_gpu() {
    local gpu_count

    echo "Checking GPU..."
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "Error: NVIDIA driver (nvidia-smi) not found; GPU is required." >&2
        return 1
    fi

    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${gpu_count}" -gt 0 ]; then
        echo "Found NVIDIA GPU, count: ${gpu_count}"
        return 0
    fi

    echo "Error: No NVIDIA GPU device detected; GPU is required." >&2
    return 1
}

check_python_package() {
    local package_pattern="$1"
    local found=0

    echo "================================"
    echo "Checking python3 package: $package_pattern"

    # check python3 and pip3 existence
    if ! command -v python3 &>/dev/null || ! command -v pip3 &>/dev/null; then
        echo "⚠ Not found python3 or pip3 in PATH."
        return 2
    fi

    # check package existence
    if pip3 list --format=columns 2>/dev/null | grep -E "^$package_pattern" >/dev/null 2>&1; then
        echo "✓ Found python3 package: $package_pattern"
        pip3 list --format=columns 2>/dev/null | grep -E "^$package_pattern" | while read -r line; do
            echo "  - $line"
        done
    else
        echo "✗ Not found package: $package_pattern"
        found=1
    fi

    return $found
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

dir_path="cosyvoice3_venv"
if [[ "$VENV_FLAG" -eq "1" ]]; then
    echo "⚠ Create python3 venv - ${dir_path} for cosyvoice3 demo."
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


if [[ "$MODEL_TYPE" == "precompiled" ]]; then
    echo "[CosyVoice3 Bash] Download precompiled models..."
    python3 get_model.py --type hmm
else
    echo "[CosyVoice3 Bash] Download raw models..."
    python3 get_model.py --type raw
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "build" ]; then
    if ! check_gpu; then
        exit 1
    fi

    quant_pkg_status=0
    check_python_package "hmquant"
    quant_pkg_status=$?
    if [ -z "$quant_pkg_status" ] || [ "$quant_pkg_status" -ne 0 ]; then
        echo "Error: Required package 'hmquant' not found or environment error, exit." >&2
        exit 1
    fi

    build_pkg_status=0
    check_python_package "houmo-tcim"
    build_pkg_status=$?
    if [ -z "$build_pkg_status" ] || [ "$build_pkg_status" -ne 0 ]; then
        echo "Error: Required package 'houmo-tcim' not found or environment error, exit." >&2
        exit 1
    fi

    echo "[CosyVoice3 Bash] Quant models..."
    python3 ptq.py
    echo "[CosyVoice3 Bash] Compile models..."
    python3 build.py
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "demo" ]; then
    echo "[CosyVoice3 Bash] Execute demo..."
    python3 demo.py
fi

if [[ "$VENV_FLAG" -eq "1" ]]; then
    deactivate
    rm -rf $dir_path
fi

echo "[CosyVoice3 Bash] ✅ Script execution completed successfully."
exit 0