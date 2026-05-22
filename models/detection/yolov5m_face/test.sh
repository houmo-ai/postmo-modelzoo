#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
source "${MODELS_DIR}/test_common.sh"

cd "${SCRIPT_DIR}"

FOUND_PACKAGE=0
if check_python_package "hmquant"; then
    FOUND_PACKAGE=1
else
    package_status=$?
    if [ "${package_status}" -eq 2 ]; then
        exit 0
    fi
fi

python3 get_model.py --type raw
if [ $FOUND_PACKAGE -eq 0 ]; then
    python3 get_model.py --type hmm
else
    hmatc quant   -c config.yml
    hmatc build   -c config.yml
    hmatc compare -c config.yml --data_path widerface/WIDER_val/images/8--Election_Campain/8_Election_Campain_Election_Campaign_8_118.jpg
fi
hmatc perf    -c config.yml -wn 1 -sn 1 -tn 1
hmatc demo    -c config.yml
hmatc demo    -c config.yml --onnx
hmatc eval    -c config.yml
hmatc eval    -c config.yml --onnx
