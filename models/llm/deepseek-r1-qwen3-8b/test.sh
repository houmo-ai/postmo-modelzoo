#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

FOUND_GPU=0
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | wc -l)
    if [ $gpu_count -gt 0 ]; then
        FOUND_GPU=1
        echo "Found NVIDIA GPU, count: ${gpu_count}"
    else
        echo "⚠ Not found NVIDIA GPU device."
    fi
else
    echo "⚠ Not install NVIDIA GPU driver."
fi

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

if [ $FOUND_PACKAGE -eq 0 ] || [ $FOUND_GPU -eq 0 ]; then
    python3 get_model.py --type hmm
else
    python3 get_model.py --type raw
    python3 ptq.py
    python3 build.py
fi
python3 demo.py