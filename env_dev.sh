#!/usr/bin/env bash
MODELZOO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd)
export MODELZOO_PATH
HMASSIST_PATH=$MODELZOO_PATH/hmassist
export HMASSIST_PATH

# set hal library log level
export HM800_HAL_CONSOLE_LEVEL=0

# main path
export HOUMO_PATH=/usr/local/houmo
export TCIM_PATH=$(python3 -c "import tvm; print(tvm.__path__[0])")
export QUANTOOL_PATH=$MODELZOO_PATH/../../quantool

# paths for c/c++ compiling
export TCIM_INC_PATH=$TCIM_PATH/include
export TCIM_LIB_PATH=$TCIM_PATH
export HDPL_INC_PATH=$HOUMO_PATH/include
export HDPL_LIB_PATH=$HOUMO_PATH/lib
export CLANG_PATH=$HOUMO_PATH
export HDPL_PATH=$HOUMO_PATH

# paths for runtime
export LD_LIBRARY_PATH=$HDPL_LIB_PATH:$TCIM_LIB_PATH
export PYTHONPATH=$TCIM_PATH:$MODELZOO_PATH:$QUANTOOL_PATH
export PATH=$HOUMO_PATH/bin:$HMASSIST_PATH:$PATH

# data and model path
if [[ -z $DATASETS_PATH ]]; then
  export DATASETS_PATH=$MODELZOO_PATH/data/datasets
fi
if [[ -z $MODEL_PATH ]]; then
  export MODEL_PATH=$MODELZOO_PATH/data/models
fi

# default platform is isim
if [[ -z $HDPL_PLATFORM ]]; then
  export HDPL_PLATFORM=ISIM
fi

echo "[Please check the following path. Unset the environment variable if you want to use the default path!]"
echo "HOUMO_PATH is $HOUMO_PATH"
echo "TCIM_PATH is $TCIM_PATH"
echo "QUANTOOL_PATH is $QUANTOOL_PATH"
echo "MODELZOO_PATH is $MODELZOO_PATH"
echo "DATASETS_PATH is $DATASETS_PATH"
echo "MODEL_PATH is $MODEL_PATH"
echo "PYTHONPATH is $PYTHONPATH"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
echo "PATH is $PATH"
echo "HDPL_PLATFORM is $HDPL_PLATFORM"