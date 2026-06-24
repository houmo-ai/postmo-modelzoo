#!/usr/bin/env bash

# main path
__dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HOUMO_EXAMPLES_PATH=${__dir}

# common define
PRINT_GREEN() { echo -e "\033[1;32m$@\033[0m"; }
PRINT_YELLOW() { echo -e "\033[1;33m$@\033[0m"; }

if [[ -z $HOUMO_SDK_PATH ]]; then
  PRINT_YELLOW "[warn] HOUMO_SDK_PATH not set. use default /usr/local/houmo-sdk."
  export HOUMO_SDK_PATH=/usr/local/houmo-sdk
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
  export HOUMO_MODELZOO_URL=http://artifactory.houmo.ai/artifactory/Dadao
fi

# paths for build
export PATH=$HOUMO_EXAMPLES_PATH/tools/bin:$HOUMO_PATH/bin:$PATH

# paths for runtime
export LD_LIBRARY_PATH=$HOUMO_EXAMPLES_PATH/tools/common/lib:$TCIM_RUNTIME_PATH/lib:$HOUMO_PATH/lib:$HOUMO_SDK_PATH/hal/lib:$LD_LIBRARY_PATH

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
export PYTHONPATH=$HOUMO_EXAMPLES_PATH/apis/common/python:$HOUMO_EXAMPLES_PATH/hmodel/xh2:$HOUMO_EXAMPLES_PATH/hmodel/gptqmodel:$PYTHONPATH


# 清理环境变量中的重复路径
_remove_duplicate_paths() {
  local var_value="$1"
  local delimiter=":"
  local -A seen
  local result=""
  local IFS=":"

  for path in $var_value; do
    if [[ -n "$path" ]] && [[ -z "${seen[$path]}" ]]; then
      if [[ -z "$result" ]]; then
        result="$path"
      else
        result="$result:$path"
      fi
      seen[$path]=1
    fi
  done

  echo "$result"
}

export PATH=$(_remove_duplicate_paths "$PATH")
export LD_LIBRARY_PATH=$(_remove_duplicate_paths "$LD_LIBRARY_PATH")
export PYTHONPATH=$(_remove_duplicate_paths "$PYTHONPATH")

PRINT_YELLOW "[Please check the following path. Unset the env and source again if you want to use the default path!]"
PRINT_GREEN "HOUMO_TARGET=$HOUMO_TARGET"
PRINT_GREEN "HOUMO_VERSION=$HOUMO_VERSION"
PRINT_GREEN "HOUMO_PATH=$HOUMO_PATH"
PRINT_GREEN "HOUMO_SDK_PATH=$HOUMO_SDK_PATH"
PRINT_GREEN "TCIM_RUNTIME_PATH=$TCIM_RUNTIME_PATH"
PRINT_GREEN "HOUMO_EXAMPLES_PATH=$HOUMO_EXAMPLES_PATH"
PRINT_GREEN "HOUMO_DATASETS_PATH=$HOUMO_DATASETS_PATH"
PRINT_GREEN "HOUMO_MODEL_PATH=$HOUMO_MODEL_PATH"
PRINT_GREEN "PYTHONPATH=$PYTHONPATH"
PRINT_GREEN "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
PRINT_GREEN "PATH=$PATH"
