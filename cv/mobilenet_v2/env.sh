#!/usr/bin/env bash
PROJ_ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit ; pwd)

if [ -n "${TVM_ROOT}" ]; then
  # whether is development environment
  export IS_DEV=1
fi

if [ -z ${IS_DEV} ]; then
  export HOUMO_PATH=/usr/local/houmo
  export LD_LIBRARY_PATH="${HOUMO_PATH}/lib":${LD_LIBRARY_PATH}
fi

if [ -n "${IS_DEV}" ]; then
  HDPL_TOOLCHAIN_ITVM_INSTALL="$TVM_ROOT/build/install/"
fi

if [ -z "$HDPL_TOOLCHAIN_ITVM_INSTALL" ]; then
  HDPL_TOOLCHAIN_ITVM_INSTALL="$(pip3 show houmo-tvm | grep Location: | awk '{print($2)}')/tvm"
fi
if [ -z "$HDPL_TOOLCHAIN_ITVM_INSTALL" ]; then
  HDPL_TOOLCHAIN_ITVM_INSTALL="$(python3 -c 'import site;print(site.getsitepackages()[0])')/tvm"
fi
if [ ! -f "${HDPL_TOOLCHAIN_ITVM_INSTALL}/libtvm.so" ]; then
  echo "Could not found houmo tvm installed dir"
fi
export HDPL_TOOLCHAIN_ITVM_INSTALL

#数据集路径
export DATASETS_PATH=$PROJ_ROOT_PATH/data/datasets
export MODEL_PATH=$PROJ_ROOT_PATH/data/models


echo "HOUMO_PATH is ${HOUMO_PATH}"
echo "HDPL_TOOLCHAIN_ITVM_INSTALL is ${HDPL_TOOLCHAIN_ITVM_INSTALL}"
echo "PROJ_ROOT_PATH is ${PROJ_ROOT_PATH}"
echo "DATASETS_PATH is $DATASETS_PATH, please replace it to path where you want to save datasets"
echo "MODEL_PATH ${MODEL_PATH}"
