#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

export HDPL_PLATFORM=ISIM
python3 calibrate.py
python3 compile.py "$@"
