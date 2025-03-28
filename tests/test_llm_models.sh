#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"


bash ../models/llm/qwen2/test.sh 2>&1 | tee qwen2_test.log
bash ../models/llm/qwen2.5/test.sh 2>&1 | tee qwen2.5_test.log
bash ../models/llm/deepseek/test.sh 2>&1 | tee deepseek_test.log
bash ../models/diffusion/sdxl/test.sh 2>&1 | tee sdxl_test.log
