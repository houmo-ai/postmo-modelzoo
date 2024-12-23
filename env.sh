#!/usr/bin/env bash
pip3 install -r requirements.txt

# main path
export MODELZOO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd)
export HMASSIST_PATH=$MODELZOO_PATH/hmassist
export CMAKE_CONFIG_PATH=$MODELZOO_PATH/release.cmake

if [[ -z $HOUMO_TARGET ]]; then
  export HOUMO_TARGET=houmo
fi

if [[ -z $HOUMO_PATH ]]; then
  echo "[warn] HOUMO_PATH not set. use default."
  export HOUMO_PATH=/usr/local/houmo
fi

if [[ -z $TCIM_RUNTIME_PATH ]]; then
echo "[warn] TCIM_RUNTIME_PATH not set. use default."
  export TCIM_RUNTIME_PATH=$HOUMO_PATH
fi

if [[ -z $MODELZOO_URL ]]; then
  export MODELZOO_URL=http://139.224.0.199:8082/artifactory/houmo/release
fi

export TCIM_RUNTIME_PATH=$HOUMO_PATH

# paths for build
export PATH=$HOUMO_PATH/bin:$PATH

# paths for c/c++ compiling
export TCIM_INC_PATH=$TCIM_RUNTIME_PATH/include
export TCIM_LIB_PATH=$TCIM_RUNTIME_PATH/lib

# paths for runtime
export PYTHONPATH=$TCIM_RUNTIME_PATH/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOUMO_PATH/lib:$LD_LIBRARY_PATH

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

echo "[Please check the following path. Unset the environment variable if you want to use the default path!]"
echo "HOUMO_TARGET is $HOUMO_TARGET"
echo "HOUMO_PATH is $HOUMO_PATH"
echo "HOUMO_SDK_PATH is $HOUMO_SDK_PATH"
echo "TCIM_RUNTIME_PATH is $TCIM_RUNTIME_PATH"
echo "MODELZOO_PATH is $MODELZOO_PATH"
echo "DATASETS_PATH is $DATASETS_PATH"
echo "MODEL_PATH is $MODEL_PATH"
echo "PYTHONPATH is $PYTHONPATH"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
echo "PATH is $PATH"
echo "HDPL_PLATFORM is $HDPL_PLATFORM"
