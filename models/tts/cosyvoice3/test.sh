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

dir_path="cosyvoice3_venv"
TEST_VENV_ACTIVE=0
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} for cosyvoice3 demo"

    arch=$(uname -m)
    if [[ "$arch" == "aarch64" ]]; then
        pip3 install scikit-learn==1.3.0 --no-binary scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
        export PYTHONPATH="${dir_path}/lib/python3.9/site-packages:$PYTHONPATH"
        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
    fi
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
    echo "Start model quantization."
    python3 ptq.py
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
    echo "Execute demo."
    python3 demo.py
    cd cpp
    ./build.sh
    python3 scripts/convert_embeddings.py
    cd ..
    export LD_LIBRARY_PATH=$PWD/bin:$LD_LIBRARY_PATH
    ./bin/cosyvoice3-demo
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi