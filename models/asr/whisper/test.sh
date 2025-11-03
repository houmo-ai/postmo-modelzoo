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

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

if [[ "$STEP" == "demo" ]]; then
    echo "Execute demo using precompiled model."
    pip3 install torch==2.8.0
    pip3 install torchcodec=0.7.0
    pip3 install torchvision==0.23.0
    apt update && apt install ffmpeg
    python3 get_model.py --type hmm
    python3 demo.py
else
    echo "✗ Unknown step ${STEP}."
fi