#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

python3 get_model.py
hmquant.sh
hmbuild.sh
hmdemo.sh
hmeval.sh
