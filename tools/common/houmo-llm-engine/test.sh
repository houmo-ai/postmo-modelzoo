#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${HOUMO_EXAMPLES_PATH}/models"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

STEP="demo"
SKIP_DOWNLOAD="false"
parse_args "$@"

cd "${SCRIPT_DIR}"

if is_asic; then
    check_step_python_packages || exit 0
else
    echo "Demo only support xh2 platform, skip demo."
    exit 0
fi

if should_run_step "demo"; then
    modelscope download Qwen/Qwen3-0.6B --local_dir tokenizers/qwen3-0.6b --exclude *.safetensors
    ./build_linux.sh > /dev/null 2>&1
    cd ${SCRIPT_DIR}
    ctest --test-dir build --output-on-failure
fi