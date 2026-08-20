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
HOUMO_TARGET="${HOUMO_TARGET:-xh2}"

cd "${SCRIPT_DIR}"

TEST_VENV_ACTIVE=0
dir_path="paddleocr_v5_venv"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "${dir_path} example"
fi

check_step_python_packages || exit 1

if should_run_step "quant" || should_run_step "build"; then
    if should_skip_download; then
        echo "Skip raw model download."
    else
        echo "Download raw model."
        python3 get_model.py --type raw
    fi
fi

if should_run_step "quant" || should_run_step "build"; then
    echo "Start PaddleOCR V5 HMONNX quantization."
    hmatc quant -c det.yml
    hmatc quant -c rec.yml
fi

if should_run_step "build"; then
    echo "Start model compilation."
    hmatc build -c det.yml
    hmatc build -c rec.yml
fi

if should_run_step "demo"; then
    if should_skip_download; then
        echo "Skip HMM model download."
    else
        echo "Download HMM model."
        python3 get_model.py --type hmm
    fi

    echo "Start PaddleOCR V5 HMM demo."
    python3 demo.py --image "${SCRIPT_DIR}/ocr.jpeg" --perf
fi

if [[ "${TEST_VENV_ACTIVE:-0}" -eq "1" ]]; then
    cleanup_python_venv "${dir_path}"
fi

echo "✅ Script execution completed successfully."
exit 0
