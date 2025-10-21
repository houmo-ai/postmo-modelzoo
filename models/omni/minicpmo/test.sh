#!/usr/bin/env bash
houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
    python3 get_model.py --type quant
    python3 compile.py
else
    python3 get_model.py --type hmm
fi
python3 demo.py