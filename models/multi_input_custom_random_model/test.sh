#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh1" ]; then
    echo "Only supports HOUMO_TARGET as xh1."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

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

python3 export_onnx.py
python3 gen_data.py
if [ $FOUND_PACKAGE -eq 0 ]; then
  python3 get_model.py --type build
else
  hmatc quant   -c config.yml
  hmatc build   -c config.yml
  hmatc compare -c config.yml --data_path data/0.npz
fi
hmatc perf     -c config.yml -wn 1 -sn 1 -tn 1