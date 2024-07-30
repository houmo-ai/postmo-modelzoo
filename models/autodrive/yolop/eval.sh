#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

if [ "$HDPL_HOST" == "AARCH64" ]; then
  python3 get_model.py --type quant
else
  python3 get_model.py --type raw
  hmquant.sh
fi
hmbuild.sh
hmdemo.sh
