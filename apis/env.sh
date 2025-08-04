#!/usr/bin/env bash

# main path
__dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HOUMO_EXAMPLES_PATH=${__dir}

# common define
PRINT_GREEN() { echo -e "\033[1;32m$@\033[0m"; }
PRINT_YELLOW() { echo -e "\033[1;33m$@\033[0m"; }

# paths for runtime
export PYTHONPATH=$TCIM_RUNTIME_PATH/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TCIM_RUNTIME_PATH/lib:$LD_LIBRARY_PATH

# use asic if detected
if [[ -z $HDPL_PLATFORM ]]; then
  if ls /dev/ | grep -q 'hmcl_client_mgr'; then
    export HDPL_PLATFORM=ASIC
  else
    export HDPL_PLATFORM=ISIM
  fi
fi

PRINT_GREEN "HOUMO_EXAMPLES_PATH is $HOUMO_EXAMPLES_PATH"
PRINT_GREEN "HOUMO_SDK_PATH is $HOUMO_SDK_PATH"
PRINT_GREEN "TCIM_RUNTIME_PATH is $TCIM_RUNTIME_PATH"
PRINT_GREEN "PYTHONPATH is $PYTHONPATH"
PRINT_GREEN "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
PRINT_GREEN "HDPL_PLATFORM is $HDPL_PLATFORM"
