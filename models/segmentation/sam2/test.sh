#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

STEP="all"
SKIP_DOWNLOAD="false"
MODEL_NAME="sam2"
NDEVICE=1
DEMO_MODE="2"
BACKEND="hmm"
parse_args "$@"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="sam2_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} demo"
fi

check_step_python_packages || exit 1

HAS_ASIC=0
if is_asic; then
    HAS_ASIC=1
else
    echo "[warn] No ASIC detected, skip compare, perf and demo steps."
fi

if should_run_step "quant" || should_run_step "build"; then
    if should_skip_download; then
        echo "Skip raw model download."
    else
        echo "Download raw model (${MODEL_NAME})."
        python3 get_model.py --type raw
    fi
fi

if should_run_step "quant"; then
    echo "Start model quantization (${MODEL_NAME})."
    hmatc quant -c encoder.yml
    hmatc quant -c decoder.yml
fi

if should_run_step "build"; then
    echo "Start model compilation (${MODEL_NAME})."
    hmatc build -c encoder.yml
    hmatc build -c decoder.yml

    if [[ "${HAS_ASIC}" -eq "1" ]]; then
        # hmatc compare -c encoder.yml --data_path coco2017/val2017/000000000139.jpg

        echo "Start model benchmark (${MODEL_NAME})."
        hmatc perf -c encoder.yml -sn 100 -tn 1
        hmatc perf -c decoder.yml -sn 100 -tn 1
    else
        echo "[warn] No ASIC detected, skip compare and perf."
    fi
fi

if should_run_step "demo"; then
    if [[ "${HAS_ASIC}" -ne "1" ]]; then
        echo "[warn] No ASIC detected, skip demo."
    else
        if [[ "${STEP}" == "demo" ]] && ! should_skip_download; then
            echo "Download pre-compiled model (${MODEL_NAME})."
            python3 get_model.py --type hmm
        fi
        echo "Execute python demo (${MODEL_NAME})."
        python3 demo.py --backend "${BACKEND}" --mode "${DEMO_MODE}"
    fi
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi