#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
  python3 get_model.py
  hmatc quant   -c config.yml
  hmatc build   -c config.yml
  hmatc compare -c config.yml --data_path bdd100k/images/100k/val/b1c9c847-3bda4659.jpg
  hmatc perf    -c config.yml -wn 1 -sn 1 -tn 1
  hmatc demo    -c config.yml
  hmatc demo    -c config.yml --onnx
  # hmatc eval    -c config.yml
  # hmatc eval    -c config.yml --onnx
fi
