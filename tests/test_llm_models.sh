#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"


bash ../models/llm/qwen2/test.sh
bash ../models/llm/qwen2.5/test.sh
bash ../models/diffusion/sdxl/test.sh
