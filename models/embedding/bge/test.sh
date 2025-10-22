#!/usr/bin/env bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

STEP="demo"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step     execution step, default is demo, support: demo, build."
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

if [[ "$STEP" == "build" ]]; then
    arch=$(uname -m)
    if [ "$arch" = "x86_64" ]; then
        echo "Start to compile model."
        python3 get_model.py --type raw
        python3 ptq.py
	python3 build.py
    else
        echo "✗ Not support model compilation."
    fi
elif [[ "$STEP" == "demo" ]]; then
    echo "Execute demo using precompiled model."
    python3 get_model.py --type hmm
    python3 demo.py #--model_dir ./onnx --model_type onnx
else
    echo "✗ Unknown step ${STEP}."
fi
