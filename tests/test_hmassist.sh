#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"
echo "cd $SCRIPT_PATH"

pushd hmassist/resnet50
# hmquant.sh
hmbuild.sh
popd
