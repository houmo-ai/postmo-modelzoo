#!/usr/bin/env bash
set -e

STEP="all"
MODEL_TYPE="precompiled"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step         execution step, default is all, support: all, demo, build."
    echo "  -t, --model_type   The method for getting the compiled model, default is precompiled, support: precompiled, compile."
    echo "  -h, --help         help information"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--step)
            STEP="$2"
            shift 2
        ;;
        -t|--model_type)
            MODEL_TYPE="$2"
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

VENV_FLAG=0
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    VENV_FLAG=1
fi

if [[ "$VENV_FLAG" -eq 1 ]]; then
    echo "⚠ Create python3.12 venv for qwenvl demo."
    dir_path="qwenvl"
    if [ ! -d "$dir_path" ]; then
        virtualenv-clone /opt/venv/houmo/ qwenvl
    fi
    source qwenvl/bin/activate
    pip3 install -r requirements.txt
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "build" ]; then
    if [[ "$MODEL_TYPE" == "precompiled" ]]; then
        echo "Download precompiled models."
        python3 get_model.py --type hmm
    else
        echo "Only supports precompiled models."
        exit 1
    fi
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "demo" ]; then
    echo "Execute demo."
    python3 demo.py
fi

if [[ "$VENV_FLAG" -eq 1 ]]; then
    deactivate
    rm -rf qwenvl
fi