#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"


bash ../models/autodrive/yolop/perf.sh

# bash ../models/autodrive/yolop/eval.sh

bash ../models/autodrive/yolop/test.sh
bash ../models/asr/wenet/test.sh
