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

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

VENV_FLAG=0
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    VENV_FLAG=1
fi

if [[ "$VENV_FLAG" -eq 1 ]]; then
    PY=$(command -v python3)
    echo "$PY"

    if [[ "$PY" == /usr/bin/* || "$PY" == /bin/* ]]; then
        echo "⚠ Create python3 venv for minicpmo demo."
        virtualenv --python=$PY --system-site-packages minicpmo_venv
        source minicpmo_venv/bin/activate
    fi
    pip3 install -r requirements.txt
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "build" ]; then
    if [[ "$MODEL_TYPE" == "precompiled" ]]; then
        echo "Download precompiled model."
        python3 get_model.py --type hmm
    else
        PACKAGE_PATTERN=houmo-tcim-xh2
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

        if [[ "$FOUND_PACKAGE" -eq 1 && "$FOUND_GPU" -eq 1 ]]; then
            echo "Start to compile model."
            python3 get_model.py --type quant
            python3 build.py
        else
            echo "✗ Not support model compilation."
            exit 1
        fi
    fi
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "demo" ]; then
    echo "Execute demo."
    python3 demo.py
fi
