#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

if [ "$HDPL_HOST" == "AARCH64" ]; then
  python3 get_model.py --type quant
else
  python3 get_model.py --type raw
  python3 ptq.py
fi
python3 build.py