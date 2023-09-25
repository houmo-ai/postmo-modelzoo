#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

cd prepare_model
./run.sh
cd ..

cd compile_model
./run.sh
cd ..

cd inference_model
./run.sh
