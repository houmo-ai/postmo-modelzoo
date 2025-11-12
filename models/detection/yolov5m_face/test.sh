#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

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
