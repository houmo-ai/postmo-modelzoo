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
    python3 quant_compile.py --compile
    python3 compare.py
fi
python3 perf.py ./output/${HOUMO_TARGET}/ppocrv3_rec_${HOUMO_TARGET}_b1_1core_O2.hmm

pip3 install -r requirements.txt
python3 run_model.py --model_path ./output/${HOUMO_TARGET}/ppocrv3_rec_${HOUMO_TARGET}_b1_1core_O2.hmm \
--data_path CCPD2020/PPOCR/val/crop_imgs \
--infer_mode demo \
--num 10
python3 run_model.py --model_path ./output/${HOUMO_TARGET}/ppocrv3_rec_${HOUMO_TARGET}_b1_1core_O2.hmm \
--data_path CCPD2020 \
--infer_mode eval
python3 run_model.py --model_path ./paddleocr_rec-sim.onnx \
--data_path CCPD2020/PPOCR/val/crop_imgs \
--infer_mode demo \
--num 10
python3 run_model.py --model_path ./paddleocr_rec-sim.onnx \
--data_path CCPD2020 \
--infer_mode eval
