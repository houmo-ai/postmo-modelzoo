#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)


cd "${SCRIPT_PATH}"

# For bash 4.4+, must not be in posix mode, may use temporary files
perf_scripts=()
while IFS='' read -r line; do perf_scripts+=("$line"); done < <(find . -type f -name "perf.sh")
for script in "${perf_scripts[@]}"; do
  echo "Run ${script}"
  bash "${script}"
done

# For bash 4.4+, must not be in posix mode, may use temporary files
eval_scripts=()
while IFS='' read -r line; do eval_scripts+=("$line"); done < <(find . -type f -name "eval.sh")
for script in "${eval_scripts[@]}"; do
  echo "Run ${script}"
  bash "${script}"
done
