#!/usr/bin/env bash

# install requirements
# sudo pip3 install -r requirements.txt
# sudo pip3 uninstall -y houmo-tcim2

source env.sh

# main path
unset HOUMO_PATH
export HMCC_PATH=/develop02/yan.cao/hmcc
export TCIM_RUNTIME_PATH=$HMCC_PATH/builds/tcim_lite

# paths for build
export HMCC_SOURCE_PATH=$HMCC_PATH
export HMCC_BUILD_PATH=$HMCC_PATH/builds/debugo3_clang
export HDPLCC_PATH=$HMCC_PATH/builds/hdpl_cc
export HDPL_LIB_PATH=$HMCC_PATH/builds/hdpl_lib
export ISIM_PATH=$HMCC_PATH/builds/isim
export HAL_DRIVER_PATH=$HMCC_PATH/builds/hal_driver
export PYTHONPATH=$HMCC_BUILD_PATH/tools/hmcc/python_packages/hmcc:$HMCC_PATH/compiler/python:$PYTHONPATH
# export PATH=$HMCC_BUILD_PATH/bin:$HMCC_PATH/compiler/tools/hdpl-compile:$HDPLCC_PATH/bin:$PATH

# paths for runtime
export PYTHONPATH=$TCIM_RUNTIME_PATH/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HMCC_BUILD_PATH/lib:$TCIM_RUNTIME_PATH/lib:$HDPL_LIB_PATH/lib:$ISIM_PATH/lib:$LD_LIBRARY_PATH

export PATH=$(_remove_duplicate_paths "$PATH")
export LD_LIBRARY_PATH=$(_remove_duplicate_paths "$LD_LIBRARY_PATH")
export PYTHONPATH=$(_remove_duplicate_paths "$PYTHONPATH")

PRINT_YELLOW "[env modified for develop, please check the following paths]"
PRINT_GREEN "HOUMO_PATH=$HOUMO_PATH"
PRINT_GREEN "HMCC_PATH=$HMCC_PATH"
PRINT_GREEN "TCIM_RUNTIME_PATH=$TCIM_RUNTIME_PATH"
PRINT_GREEN "PYTHONPATH=$PYTHONPATH"
PRINT_GREEN "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
PRINT_GREEN "PATH=$PATH"
