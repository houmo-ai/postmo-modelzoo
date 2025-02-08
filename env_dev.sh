#!/usr/bin/env bash

# install requirements
sudo pip3 install -r requirements.txt
sudo pip3 uninstall -y houmo-tcim2

# common define
PRINT_GREEN() { echo -e "\033[1;32m$@\033[0m"; }
PRINT_YELLOW() { echo -e "\033[1;33m$@\033[0m"; }

__dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export MODELZOO_PATH=${__dir}
export HMASSIST_PATH=$MODELZOO_PATH/hmassist
export CMAKE_CONFIG_PATH=$MODELZOO_PATH/develop.cmake

if [[ -z $HOUMO_TARGET ]]; then
  export HOUMO_TARGET=xh1
fi
if [[ -z $MODELZOO_URL ]]; then
  export MODELZOO_URL=http://10.10.1.53:8082/artifactory/toolchain/release
fi

# main path
unset HOUMO_PATH
export HMCC_PATH=/develop02/yan.cao/hmcc
if [[ -z $TCIM_RUNTIME_PATH ]]; then
  export TCIM_RUNTIME_PATH=$HMCC_PATH/builds/tcim_lite
fi

# paths for build
export HMCC_SOURCE_PATH=$HMCC_PATH
export HMCC_BUILD_PATH=$HMCC_PATH/builds/debugo3_clang
export HDPLCC_PATH=$HMCC_PATH/builds/hdpl_cc
export HDPL_LIB_PATH=$HMCC_PATH/builds/hdpl_lib
export ISIM_PATH=$HMCC_PATH/builds/isim
export HAL_DRIVER_PATH=$HMCC_PATH/builds/hal_driver
export PYTHONPATH=$HMCC_BUILD_PATH/tools/hmcc/python_packages/hmcc:$HMCC_PATH/compiler/python:$PYTHONPATH
export PATH=$HMCC_BUILD_PATH/bin:$HMCC_PATH/compiler/tools/hdpl-compile:$HDPLCC_PATH/bin:$PATH

# paths for runtime
export PYTHONPATH=$TCIM_RUNTIME_PATH/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HMCC_BUILD_PATH/lib:$TCIM_RUNTIME_PATH/lib:$HDPL_LIB_PATH/lib:$ISIM_PATH/lib:$LD_LIBRARY_PATH

# paths for hmassist
export PYTHONPATH=$HMASSIST_PATH:$PYTHONPATH
export PATH=$HMASSIST_PATH:$PATH

# data and model path
if [[ -z $DATASETS_PATH ]]; then
  export DATASETS_PATH=$MODELZOO_PATH/data/datasets
fi
if [[ -z $MODEL_PATH ]]; then
  CI_MODEL_PATH=/data02/modelzoo_ci/models
  if test -d $CI_MODEL_PATH; then
    export MODEL_PATH=$CI_MODEL_PATH
  else
    export MODEL_PATH=$MODELZOO_PATH/models
  fi
fi

# use asic if detected
if [[ -z $HDPL_PLATFORM ]]; then
  if ls /dev/ | grep -q 'hmcl_client_mgr'; then
    export HDPL_PLATFORM=ASIC
  else
    export HDPL_PLATFORM=ISIM
  fi
fi

PRINT_YELLOW "[Please check the following path. Unset the env and source again if you want to use the default path!]"
PRINT_GREEN "HOUMO_TARGET is $HOUMO_TARGET"
PRINT_GREEN "HOUMO_PATH is $HOUMO_PATH"
PRINT_GREEN "HMCC_PATH is $HMCC_PATH"
PRINT_GREEN "TCIM_RUNTIME_PATH is $TCIM_RUNTIME_PATH"
PRINT_GREEN "MODELZOO_PATH is $MODELZOO_PATH"
PRINT_GREEN "DATASETS_PATH is $DATASETS_PATH"
PRINT_GREEN "MODEL_PATH is $MODEL_PATH"
PRINT_GREEN "PYTHONPATH is $PYTHONPATH"
PRINT_GREEN "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
PRINT_GREEN "PATH is $PATH"
PRINT_GREEN "HDPL_PLATFORM is $HDPL_PLATFORM"
