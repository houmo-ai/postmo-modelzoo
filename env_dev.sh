#!/usr/bin/env bash
pip3 install onnx_graphsurgeon -i https://pypi.ngc.nvidia.com

__dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODELZOO_PATH=${__dir}
export MODELZOO_PATH
HMASSIST_PATH=$MODELZOO_PATH/hmassist
export HMASSIST_PATH

export HMASSIST_TARGET=H30
export CMAKE_CONFIG_PATH=$MODELZOO_PATH/develop.cmake
if [[ -z $MODELZOO_URL ]]; then
  export MODELZOO_URL=http://139.224.0.199:8082/artifactory/houmo/release
fi

# set hal library log level
export HM800_HAL_CONSOLE_LEVEL=0
export HDPL_API_TIMEOUT=30000

# main path
export QUANTOOL_PATH=$MODELZOO_PATH/../../quantool
if [[ -z $HDPL_TOOLCHAIN_ITVM_INSTALL ]] && [[ -d ${TVM_ROOT} ]]; then
  HDPL_TOOLCHAIN_ITVM_INSTALL=$TVM_ROOT/build/install
fi

# paths for c/c++ compiling
export TCIM_INC_PATH=$HDPL_TOOLCHAIN_ITVM_INSTALL/include
export TCIM_LIB_PATH=$HDPL_TOOLCHAIN_ITVM_INSTALL/lib
export IDNNL_INC_PATH=$IDNNL_PATH/include
export IDNNL_LIB_PATH=$IDNNL_PATH/lib
export HDPL_INC_PATH=$HDPL_PATH/include
export HDPL_LIB_PATH=$HDPL_PATH/lib
export CLANG_LIB_PATH=$CLANG_PATH/lib

# paths for runtime
if [[ -d ${QUANTOOL_PATH} ]]; then
  export PYTHONPATH=$HMASSIST_PATH:$QUANTOOL_PATH:$PYTHONPATH
else
  export PYTHONPATH=$HMASSIST_PATH:$PYTHONPATH
fi
export PATH=$HMASSIST_PATH:$PATH

# data and model path
if [[ -z $DATASETS_PATH ]]; then
  export DATASETS_PATH=$MODELZOO_PATH/data/datasets
fi
if [[ -z $MODEL_PATH ]]; then
  export MODEL_PATH=$MODELZOO_PATH/data/models
fi

# use asic if detected
if [[ -z $HDPL_PLATFORM ]]; then
  if [ -c /dev/hm_host_pcie* ]; then
    export HDPL_PLATFORM=ASIC
  else
    export HDPL_PLATFORM=ISIM
  fi
fi

echo "[Please check the following path. Unset the environment variable if you want to use the default path!]"
echo "HMASSIST_TARGET is $HMASSIST_TARGET"
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
