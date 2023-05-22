#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

# shellcheck source=cv/resnet50/env.sh
source ./env.sh


cd prepare_model
./run.sh
cd ..

cd compile_model
./run.sh
cd ..

cd inferece_model
./run.sh
