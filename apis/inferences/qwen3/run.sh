#!/usr/bin/env bash
set -e

STEP="all"
DEMO_TYPE="normal"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step         execution step, default is all, support: all, get_model, demo."
    echo "  -t, --demo_type    the type of the demo to be executed, default is normal, support: normal, prefix_caching."
    echo "  -h, --help         help information"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--step)
            STEP="$2"
            shift 2
        ;;
        -t|--demo_type)
            DEMO_TYPE="$2"
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
    if [ "$DEMO_TYPE" = "normal" ]; then
        echo "Execute Qwen3 Normal Demo."
        python3 demo.py
    elif [ "$DEMO_TYPE" = "prefix_caching" ]; then
        echo "Execute Qwen3 Prefix Caching Demo."
        python3 demo_prefix_caching.py
    else
        echo "Unknown demo type: $DEMO_TYPE"
        exit 1
    fi
    # c++ example
    ./run_linux.sh
fi
