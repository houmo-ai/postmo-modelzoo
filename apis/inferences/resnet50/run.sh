#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}" || exit 1

# get test model
python3 get_model.py

# python example
python3 resnet50.py

# c++ example
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$SCRIPT_DIR -DCMAKE_BUILD_TYPE=Release ..
make -j
make install

cd $SCRIPT_DIR
./example_resnet50

echo "resnet50 run.sh end."