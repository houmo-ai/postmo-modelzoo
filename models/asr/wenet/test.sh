#!/usr/bin/env bash
set -e

STEP="all"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step     execution step, default is all, support: build, demo."
    echo "  -h, --help     help information"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--step)
            STEP="$2"
            shift 2
        ;;
        -h|--help)
            show_help
        ;;
        *)
            echo "Error: Unknown parameter '$1'" >&2
            show_help
        ;;
    esac
done

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh1" ]; then
    echo "Only supports HOUMO_TARGET as xh1."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

VENV_FLAG=0
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    VENV_FLAG=1
fi

if [[ "$VENV_FLAG" -eq 1 ]]; then
    echo "⚠ Create python3.9 venv for wenet demo."
    virtualenv --python=python3.9 --system-site-packages wenet_demo
    source wenet_demo/bin/activate
    pip3 install -r requirements.txt
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "build" ]; then
    PACKAGE_PATTERN=houmo-tcim-xh1
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
    if [[ "$FOUND_PACKAGE" -eq 1 ]]; then
        echo "Start to compile model."
        python3 get_model.py --type quant
        python3 build.py
    else
        echo "✗ Not support model compilation."
    fi
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "demo" ]; then
    target_dir="${SCRIPT_DIR}/output/${HOUMO_TARGET}"
    if [ -d "$target_dir" ] && ls "$target_dir"/*.hmm >/dev/null 2>&1; then
        echo "Execute demo using precompiled model."
        python3 demo.py
    else
        echo "⚠ No precompiled models were detected in the ${target_dir} directory."
    fi
fi

if [[ "$VENV_FLAG" -eq 1 ]]; then
    deactivate
    rm -rf wenet_demo
fi