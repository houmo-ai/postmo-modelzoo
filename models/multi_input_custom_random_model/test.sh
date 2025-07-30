#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
  python3 export_onnx.py
  python3 gen_data.py
  hmexec quant   -c config.yml
  hmexec build   -c config.yml
  hmexec compare -c config.yml -t xh1 --data_path data/0.npz
  hmexec perf    -c config.yml -wn 1 -sn 1 -tn 1
fi
