#!/usr/bin/env bash

# main path
__dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HOUMO_EXAMPLES_PATH=${__dir}

# common define
PRINT_GREEN() { echo -e "\033[1;32m$@\033[0m"; }
PRINT_YELLOW() { echo -e "\033[1;33m$@\033[0m"; }

if [[ -z $HOUMO_SDK_PATH ]]; then
  PRINT_YELLOW "[warn] HOUMO_SDK_PATH not set. use default /usr/local/houmo-sdk."
  export HOUMO_PATH=/usr/local/houmo-sdk
fi

if [[ -z $HOUMO_PATH ]]; then
  PRINT_YELLOW "[warn] HOUMO_PATH not set. use default /usr/local/houmo."
  export HOUMO_PATH=/usr/local/houmo
fi

if [[ -z $TCIM_RUNTIME_PATH ]]; then
  PRINT_YELLOW "[warn] TCIM_RUNTIME_PATH not set. use default $HOUMO_PATH."
  export TCIM_RUNTIME_PATH=$HOUMO_PATH
fi

if [[ -z $HOUMO_MODELZOO_URL ]]; then
  export HOUMO_MODELZOO_URL=http://artifactory.houmo.ai/artifactory/toolchain/release
fi

# paths for build
export PATH=$HOUMO_EXAMPLES_PATH/tools/bin:$HOUMO_PATH/bin:$PATH

# paths for runtime
export LD_LIBRARY_PATH=$TCIM_RUNTIME_PATH/lib:$HOUMO_PATH/lib:$HOUMO_SDK_PATH/hal/lib:$LD_LIBRARY_PATH

# data and model path
if [[ -z $HOUMO_DATASETS_PATH ]]; then
  export HOUMO_DATASETS_PATH=$HOUMO_EXAMPLES_PATH/data/datasets
fi

if [[ -z $HOUMO_MODEL_PATH ]]; then
  CI_MODEL_PATH=/data02/modelzoo_ci/models
  if test -d $CI_MODEL_PATH; then
    export HOUMO_MODEL_PATH=$CI_MODEL_PATH
  fi
fi

# paths for xh2 modelzoo
export PYTHONPATH=$HOUMO_EXAMPLES_PATH/apis/common/python:$HOUMO_EXAMPLES_PATH/hmodel/xh2:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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

# use asic if detected
if [[ -z $HDPL_PLATFORM ]]; then
  if { [[ "$HOUMO_TARGET" == "xh2" ]] && ls /dev/ | grep -q 'xh2a'; } then
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
PRINT_GREEN "HOUMO_EXAMPLES_PATH is $HOUMO_EXAMPLES_PATH"
PRINT_GREEN "HOUMO_DATASETS_PATH is $HOUMO_DATASETS_PATH"
PRINT_GREEN "HOUMO_MODEL_PATH is $HOUMO_MODEL_PATH"
PRINT_GREEN "PYTHONPATH is $PYTHONPATH"
PRINT_GREEN "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
PRINT_GREEN "PATH is $PATH"
PRINT_GREEN "HDPL_PLATFORM is $HDPL_PLATFORM"
