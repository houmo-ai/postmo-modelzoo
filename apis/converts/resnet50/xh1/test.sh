#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

action="${1:-all}"

declare -A action_map=(
  ["quant"]=1
  ["build"]=2
  ["all"]=9
)

action_num="${action_map[$action]:-0}"

if [[ $action_num > 0 ]]; then
  python3 ../get_model.py
  python3 ptq.py
fi
if [[ $action_num > 1 ]]; then
  python3 build.py
fi
