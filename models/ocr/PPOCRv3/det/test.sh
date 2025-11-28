#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [[ ! "xh1 xh2" =~ (^|[[:space:]])"$houmo_target"($|[[:space:]]) ]]; then
    echo "Only supports HOUMO_TARGET as xh1 or xh2."
    exit 1
fi

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
    hmatc compare -c config.yml --data_path "CCPD2020/ccpd_green/val/0196354166667-93_258-296&451_528&537-528&537_305&516_296&451_522&465-0_0_3_24_25_33_29_31-124-19.jpg"
fi
hmatc perf    -c config.yml -wn 10 -sn 1000 -tn 1
pip3 install -r requirements.txt
hmatc demo    -c config.yml
hmatc demo    -c config.yml --onnx
hmatc eval    -c config.yml
hmatc eval    -c config.yml --onnx
