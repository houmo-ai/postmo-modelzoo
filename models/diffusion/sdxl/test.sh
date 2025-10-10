#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh1" ]; then
    echo "Only supports HOUMO_TARGET as xh1."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

pip3 install -r requirements.txt

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
    python3 get_model.py --type quant
    python3 build.py
else
    python3 get_model.py --type hmm
fi
python3 demo.py