#!/usr/bin/env bash

MODELZOO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd)
export MODELZOO_PATH
HMASSIST_PATH=$MODELZOO_PATH/hmassist
export HMASSIST_PATH

if [[ -z $HOUMO_TARGET ]]; then
  export HOUMO_TARGET=houmo
fi
export CMAKE_CONFIG_PATH=$MODELZOO_PATH/release.cmake
if [[ -z $MODELZOO_URL ]]; then
  export MODELZOO_URL=http://139.224.0.199:8082/artifactory/houmo/release
fi

# set hal library log level
export HM800_HAL_CONSOLE_LEVEL=0
export HDPL_API_TIMEOUT=30000

# main path
if [[ -z $HOUMO_PATH ]]; then
  HOUMO_PATH=/usr/local/houmo
  export HOUMO_PATH
fi

if [[ -z $TCIM_PATH ]]; then
  TCIM_PATH=$(python3 -c "import tvm; print(tvm.__path__[0])")
  export TCIM_PATH
fi

# paths for c/c++ compiling
export TCIM_INC_PATH=$TCIM_LITE/include
export TCIM_LIB_PATH=$TCIM_LITE/lib
export HDPL_INC_PATH=$HOUMO_PATH/include
export HDPL_LIB_PATH=$HOUMO_PATH/lib

# paths for runtime
export LD_LIBRARY_PATH=$HDPL_LIB_PATH:$TCIM_LIB_PATH:$LD_LIBRARY_PATH
export PYTHONPATH=$HMASSIST_PATH:$TCIM_LITE/python:$PYTHONPATH
export PATH=$HOUMO_PATH/bin:$HMASSIST_PATH:$PATH

# data and model path
if [[ -z $DATASETS_PATH ]]; then
  export DATASETS_PATH=$MODELZOO_PATH/data/datasets
fi
if [[ -z $MODEL_PATH ]]; then
  export MODEL_PATH=$MODELZOO_PATH/models
fi

# use asic if detected
if [[ -z $HDPL_PLATFORM ]]; then
  if ls /dev/ | grep -q 'hm_host_pcie'; then
    export HDPL_PLATFORM=ASIC
  else
    export HDPL_PLATFORM=ISIM
  fi
fi

echo "[Please check the following path. Unset the environment variable if you want to use the default path!]"
echo "HOUMO_TARGET is $HOUMO_TARGET"
echo "HOUMO_PATH is $HOUMO_PATH"
echo "TCIM_PATH is $TCIM_PATH"
echo "TCIM_LITE is $TCIM_LITE"
echo "QUANTOOL_PATH is $QUANTOOL_PATH"
echo "MODELZOO_PATH is $MODELZOO_PATH"
echo "DATASETS_PATH is $DATASETS_PATH"
echo "MODEL_PATH is $MODEL_PATH"
echo "PYTHONPATH is $PYTHONPATH"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
echo "PATH is $PATH"
echo "HDPL_PLATFORM is $HDPL_PLATFORM"
