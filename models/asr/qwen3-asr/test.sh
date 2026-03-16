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

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

VENV_FLAG=0
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    VENV_FLAG=1
fi

dir_path="qwen3-asr"
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
exit 0
if [ "$STEP" = "all" ] || [ "$STEP" = "quant" ]; then
    if [[ "$MODEL_TYPE" == "precompiled" ]]; then
        echo "Download precompiled model."
        python3 get_model.py --type hmm
    else
        if [[ "$MODEL_TYPE" == "compile" ]]; then
            echo "Down raw model for Quant and Compile."
            python3 get_model.py --type raw
            echo "Start Quant Model."
            python3 ptq.py
        else
            echo "✗ Only support using precompiled and compile."
            exit 1
        fi
    fi
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "build" ]; then
    if [[ "$MODEL_TYPE" == "precompiled" ]]; then
        echo "Using download precompiled model, skip build."
    else
        if [[ "$MODEL_TYPE" == "compile" ]]; then
            echo "Compile model."
            python3 build.py
        else
            echo "✗ Only support using precompiled and compile."
            exit 1
        fi
    fi
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "demo" ]; then
    echo "Execute demo."
    python3 demo.py
fi

if [[ "$VENV_FLAG" -eq "1" ]]; then
    deactivate
    rm -rf $dir_path
fi