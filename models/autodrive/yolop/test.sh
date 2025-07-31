#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
  python3 get_model.py
  hmexec quant   -c config.yml
  hmexec build   -c config.yml
  hmexec compare -c config.yml -t xh1 --data_path bdd100k/images/100k/val/b1c9c847-3bda4659.jpg
  hmexec perf    -c config.yml -wn 1 -sn 1 -tn 1
  hmexec demo    -c config.yml
  hmexec demo    -c config.yml --onnx
  # hmexec eval    -c config.yml
  # hmexec eval    -c config.yml --onnx
fi
