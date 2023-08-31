#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

python3 train_resnet.py --data $DATASETS_PATH/imagenet --arch resnet50 --batch-size 8 --lr 1e-5
