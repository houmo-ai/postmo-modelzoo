#!/usr/bin/env bash
MODELZOO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd)
export MODELZOO_PATH
HMASSIST_PATH=$MODELZOO_PATH/hmassist
export HMASSIST_PATH

pip3 install PrettyTable

# set hal library log level
export HM800_HAL_CONSOLE_LEVEL=0

export CMAKE_CONFIG_PATH=$MODELZOO_PATH/develop.cmake

# main path
export QUANTOOL_PATH=$MODELZOO_PATH/../../quantool
HDPL_TOOLCHAIN_ITVM_INSTALL=$TVM_ROOT/build/install

# paths for c/c++ compiling
export TCIM_INC_PATH=$HDPL_TOOLCHAIN_ITVM_INSTALL/include
export TCIM_LIB_PATH=$HDPL_TOOLCHAIN_ITVM_INSTALL/lib
export IDNNL_INC_PATH=$IDNNL_PATH/include
export IDNNL_LIB_PATH=$IDNNL_PATH/lib
export HDPL_INC_PATH=$HDPL_PATH/include
export HDPL_LIB_PATH=$HDPL_PATH/lib
export CLANG_LIB_PATH=$CLANG_PATH/lib

# paths for runtime
export PYTHONPATH=$MODELZOO_PATH:$QUANTOOL_PATH:$PYTHONPATH
export PATH=$HMASSIST_PATH:$PATH

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