#!/usr/bin/env bash

# main path
__dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HOUMO_MODELZOO_PATH=${__dir}
export HMASSIST_PATH=$HOUMO_MODELZOO_PATH/hmassist

# install requirements
pip3 install -r $HOUMO_MODELZOO_PATH/requirements.txt

# common define
PRINT_GREEN() { echo -e "\033[1;32m$@\033[0m"; }
PRINT_YELLOW() { echo -e "\033[1;33m$@\033[0m"; }

if [[ -z $HOUMO_TARGET ]]; then
  export HOUMO_TARGET=xh1
fi

if [[ -z $HOUMO_PATH ]]; then
  PRINT_YELLOW "[warn] HOUMO_PATH not set. use default."
  export HOUMO_PATH=/usr/local/houmo
fi

if [[ -z $TCIM_RUNTIME_PATH ]]; then
  PRINT_YELLOW "[warn] TCIM_RUNTIME_PATH not set. use default."
  export TCIM_RUNTIME_PATH=$HOUMO_PATH
fi

if [[ -z $HOUMO_MODELZOO_URL ]]; then
  export HOUMO_MODELZOO_URL=http://139.224.0.199:8082/artifactory/houmo/release
fi

# paths for build
export PATH=$HOUMO_PATH/bin:$PATH

# paths for runtime
export PYTHONPATH=$TCIM_RUNTIME_PATH/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TCIM_RUNTIME_PATH/lib:$HOUMO_PATH/lib:$LD_LIBRARY_PATH

# paths for hmassist
export PYTHONPATH=$HMASSIST_PATH:$PYTHONPATH
export PATH=$HMASSIST_PATH:$PATH

# data and model path
if [[ -z $HOUMO_DATASETS_PATH ]]; then
  export HOUMO_DATASETS_PATH=$HOUMO_MODELZOO_PATH/data/datasets
fi

if [[ -z $HOUMO_MODEL_PATH ]]; then
  CI_MODEL_PATH=/data02/modelzoo_ci/models
  if test -d $CI_MODEL_PATH; then
    export HOUMO_MODEL_PATH=$CI_MODEL_PATH
  else
    export HOUMO_MODEL_PATH=$HOUMO_MODELZOO_PATH/models
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
PRINT_GREEN "HOUMO_SDK_PATH is $HOUMO_SDK_PATH"
PRINT_GREEN "TCIM_RUNTIME_PATH is $TCIM_RUNTIME_PATH"
PRINT_GREEN "HOUMO_MODELZOO_PATH is $HOUMO_MODELZOO_PATH"
PRINT_GREEN "HOUMO_DATASETS_PATH is $HOUMO_DATASETS_PATH"
PRINT_GREEN "HOUMO_MODEL_PATH is $HOUMO_MODEL_PATH"
PRINT_GREEN "PYTHONPATH is $PYTHONPATH"
PRINT_GREEN "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
PRINT_GREEN "PATH is $PATH"
PRINT_GREEN "HDPL_PLATFORM is $HDPL_PLATFORM"
