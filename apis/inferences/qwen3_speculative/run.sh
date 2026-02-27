#!/usr/bin/env bash
set -e

STEP="all"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step         execution step, default is all, support: all, get_model, demo."
    echo "  -h, --help         help information"
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
cd "${SCRIPT_DIR}" || exit 1

arch=$(uname -m)
if [ "$arch" = "aarch64" ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "get_model" ]; then
    python3 get_model.py
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "demo" ]; then
    python3 demo.py --fast
fi
