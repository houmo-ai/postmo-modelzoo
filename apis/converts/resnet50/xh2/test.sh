#!/usr/bin/env bash
set -e

STEP="all"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step     execution step, default is all, support: quant, build, all."
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

if [[ "$STEP" == "all" || "$STEP" == "quant" ]]; then
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

    if [ $FOUND_PACKAGE -eq 0 ]; then
        echo "⚠ Not found hmquant."
        exit 1
    fi

    python3 ../get_model.py
    python3 ptq.py
fi

if [[ "$STEP" == "all" || "$STEP" == "build" ]]; then
    python3 build.py
fi
