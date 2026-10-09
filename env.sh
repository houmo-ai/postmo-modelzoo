#!/usr/bin/env bash

# main path
__dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export POSTMO_EXAMPLES_PATH=${__dir}

# common define
PRINT_GREEN() { echo -e "\033[1;32m$@\033[0m"; }
PRINT_YELLOW() { echo -e "\033[1;33m$@\033[0m"; }

if [[ -z $POSTMO_SDK_PATH ]]; then
  PRINT_YELLOW "[warn] POSTMO_SDK_PATH not set. use default /usr/local/houmo-sdk."
  export POSTMO_SDK_PATH=/usr/local/houmo-sdk
fi

if [[ -z $POSTMO_PATH ]]; then
  PRINT_YELLOW "[warn] POSTMO_PATH not set. use default /usr/local/houmo."
  export POSTMO_PATH=/usr/local/houmo
fi

if [[ -z $TCIM_RUNTIME_PATH ]]; then
  PRINT_YELLOW "[warn] TCIM_RUNTIME_PATH not set. use default $POSTMO_PATH."
  export TCIM_RUNTIME_PATH=$POSTMO_PATH
fi

if [[ -z $POSTMO_MODELZOO_URL ]]; then
  export POSTMO_MODELZOO_URL=http://artifactory.houmo.ai/artifactory/Dadao
fi

# paths for build
export PATH=$POSTMO_EXAMPLES_PATH/tools/bin:$POSTMO_PATH/bin:$PATH

# paths for runtime
export LD_LIBRARY_PATH=$POSTMO_EXAMPLES_PATH/utils/lib:$TCIM_RUNTIME_PATH/lib:$POSTMO_PATH/lib:$POSTMO_SDK_PATH/hal/lib:$LD_LIBRARY_PATH

# data and model path
if [[ -z $POSTMO_DATASETS_PATH ]]; then
  export POSTMO_DATASETS_PATH=$POSTMO_EXAMPLES_PATH/data/datasets
fi

if [[ -z $POSTMO_MODEL_PATH ]]; then
  CI_MODEL_PATH=/data02/modelzoo_ci/models
  if test -d $CI_MODEL_PATH; then
    export POSTMO_MODEL_PATH=$CI_MODEL_PATH
  fi
fi

# paths for xh2 modelzoo
export PYTHONPATH=$POSTMO_EXAMPLES_PATH/hmatc:$POSTMO_EXAMPLES_PATH/utils/python:$POSTMO_EXAMPLES_PATH/hmodel/xh2:$POSTMO_EXAMPLES_PATH/hmodel/gptqmodel:$POSTMO_EXAMPLES_PATH/hmodel/gptqmodel/third_party/auto-round:$PYTHONPATH


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
PRINT_GREEN "POSTMO_TARGET=$POSTMO_TARGET"
PRINT_GREEN "POSTMO_VERSION=$POSTMO_VERSION"
PRINT_GREEN "POSTMO_PATH=$POSTMO_PATH"
PRINT_GREEN "POSTMO_SDK_PATH=$POSTMO_SDK_PATH"
PRINT_GREEN "TCIM_RUNTIME_PATH=$TCIM_RUNTIME_PATH"
PRINT_GREEN "POSTMO_EXAMPLES_PATH=$POSTMO_EXAMPLES_PATH"
PRINT_GREEN "POSTMO_DATASETS_PATH=$POSTMO_DATASETS_PATH"
PRINT_GREEN "POSTMO_MODEL_PATH=$POSTMO_MODEL_PATH"
PRINT_GREEN "PYTHONPATH=$PYTHONPATH"
PRINT_GREEN "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
PRINT_GREEN "PATH=$PATH"

if [[ "$1" == "--download_datasets" ]]; then
  (
      cd "$POSTMO_EXAMPLES_PATH/data/datasets" || exit 1
      python3 get_datasets.py --remove-archive
  )
fi
