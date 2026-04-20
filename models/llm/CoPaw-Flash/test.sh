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
dir_path="copawflash_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! check_gpu require; then
        exit 1
    fi

    BASE_DIR=$HOUMO_EXAMPLES_PATH
    if [ -z "$BASE_DIR" ]; then
        BASE_DIR=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")
        echo "No environment variables were found; we have backtracked to the example root directory: $BASE_DIR"
    else
        echo "Discover environment variables: $BASE_DIR"
    fi

    if ! should_skip_download; then
        echo "Download raw model."
        python3 get_model.py --type raw
    fi

    if [[ ! -d "$dir_path/lib/python3.12/site-packages/distutils" ]]; then
        ln -s /opt/venv/houmo/lib/python3.12/site-packages/setuptools/_distutils $dir_path/lib/python3.12/site-packages/distutils
    fi
    GPTQ_CALIB_PATH="$BASE_DIR/hmodel/gptqmodel/gptqmodel/quantization/calibration"
    if [ ! -d "$GPTQ_CALIB_PATH" ]; then
        echo "Error: Directory $GPTQ_CALIB_PATH does not exist."
        exit 1
    fi
    find "$GPTQ_CALIB_PATH" -type d | while read -r dir; do
        INIT_FILE="$dir/__init__.py"
        if [ ! -f "$INIT_FILE" ]; then
            touch "$INIT_FILE"
        fi
    done
    cd $BASE_DIR/hmodel/gptqmodel
    python3 setup.py install
    cd $SCRIPT_DIR

    echo "Start model quantization."
    python3 ptq.py --gptqmodel
fi

if should_run_step "build"; then
    echo "Start model compilation."
    python3 build.py
fi

if should_run_step "demo"; then
    if [[ "$STEP" == "demo" ]] && ! should_skip_download; then
        echo "Download precompiled model."
        python3 get_model.py --type hmm
    fi
    echo "Execute demo."
    python3 demo.py
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
