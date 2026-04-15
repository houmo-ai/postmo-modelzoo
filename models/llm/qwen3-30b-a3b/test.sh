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

check_houmo_target "xh2"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="qwen3moe_venv"
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

    if [[ ! -d "$dir_path/lib/python3.12/site-packages/distutils" && "$VENV_FLAG" -eq "1" ]]; then
        ln -s /opt/venv/houmo/lib/python3.12/site-packages/setuptools/_distutils \
            $dir_path/lib/python3.12/site-packages/distutils
    fi
    pip3 install gptqmodel-5.4.4-py3-none-any.whl

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
    echo "Execute performance case."
    python3 ../../../tools/llm_perf/convert_embed.py --path output/xh2/hmquant/quant_embedding.pt
    llm_perf -c config.yaml
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi
