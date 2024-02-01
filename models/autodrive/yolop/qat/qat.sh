#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ ! -d $MODELZOO_PATH/hmodel/thirdparty/YOLOP ]]; then
  git clone https://github.com/hustvl/YOLOP $MODELZOO_PATH/hmodel/thirdparty/YOLOP
fi

export PYTHONPATH=$MODELZOO_PATH/hmodel/thirdparty/YOLOP:$PYTHONPATH

pip3 install yacs tensorboardX seaborn prefetch_generator
pip3 install protobuf==3.20

cd $SCRIPT_DIR
python3 qat.py