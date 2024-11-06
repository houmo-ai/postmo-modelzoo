#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"


bash ../models/llm/qwen/perf.sh
bash ../models/llm/qwen_mix/perf.sh
bash ../models/llm/qwen_mix_multibatches/perf.sh
bash ../models/llm/qwen2/perf.sh
