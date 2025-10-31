#!/usr/bin/env bash
set -e

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
    if [[ "$FOUND_PACKAGE" -eq 1 && "$FOUND_GPU" -eq 1 ]]; then
        echo "Start to quant and compile model."
        pip3 install transformers==4.56.0
        python3 get_model.py --type raw
        python3 ptq.py
        python3 build.py
    else
        echo "✗ Not support model quantization and compilation."
    fi
elif [[ "$STEP" == "demo" ]]; then
    echo "Execute demo using precompiled model."
    python3 get_model.py --type hmm
    pip3 install transformers==4.51.0
    python3 demo.py
else
    echo "✗ Unknown step ${STEP}."
fi