#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh1" ]; then
    echo "Only supports HOUMO_TARGET as xh1."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
  python3 get_model.py
  hmatc quant   -c config.yml
  hmatc build   -c config.yml
  hmatc compare -c config.yml --data_path "CCPD2020/ccpd_green/val/0196354166667-93_258-296&451_528&537-528&537_305&516_296&451_522&465-0_0_3_24_25_33_29_31-124-19.jpg"
  hmatc perf    -c config.yml -wn 1 -sn 1 -tn 1
  hmatc demo    -c config.yml
  hmatc demo    -c config.yml --onnx
  hmatc eval    -c config.yml
  hmatc eval    -c config.yml --onnx
fi
