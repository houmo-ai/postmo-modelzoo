#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

arch=$(uname -m)
if [ "$arch" = "aarch64" ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
fi

# get test model
python3 get_model.py
# python example
python3 demo.py
