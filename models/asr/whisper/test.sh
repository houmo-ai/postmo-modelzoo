#!/usr/bin/env bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

export HF_ENDPOINT=https://hf-mirror.com

python3 get_model.py --type hmm

python3 demo.py