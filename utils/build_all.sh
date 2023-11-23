#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"

scripts=()
while IFS='' read -r line; do scripts+=("$line"); done < <(find . -type f -name "build.sh")
for script in "${scripts[@]}"; do
  echo "Run ${script}"
  bash "${script}"
done