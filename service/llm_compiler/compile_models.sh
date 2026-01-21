#!/bin/bash
# Copyright 2025 HOUMO AI
#
# File: compile_models.sh
# Description:
#   Script to compile LLM models with various configurations for different targets.
#   This script automates the compilation process for multiple LLM models with 
#   configurable parameters like context length, batch size, and core numbers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

set -e

RELEASE="OFF"
PERF="OFF"
VERSION="1.0.0"
##### XH1 #####
# models=(
#     "qwen3;model_size:8b;target:xh1;context_length:8192;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh1_qwen3_8b_8k_20250603.zip"
#     "qwen3;model_size:8b;target:xh1;context_length:8192;prefill_length:256;batch:1;device_num:1;core_num:4;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models/qwen3/hmquant_xh1_qwen3_8b_256_8k_20250815.zip"
#     "deepseek-r1-qwen;model_size:7b;target:xh1;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:4;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models/deepseek/hmquant_deepseek_256_8192_20250922.zip"
#     "deepseek-r1-qwen;model_size:7b;target:xh1;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models/deepseek/hmquant_deepseek_256_8192_20250922.zip"
#     "qwen2.5-vl;model_size:7b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:4;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_7b_256_2k_20250903.zip"
#     "qwen2.5-vl;model_size:7b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_7b_256_2k_20250903.zip"
#     "qwen2.5-vl;model_size:3b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_3b_256_2k_448x448_20251219.zip"
#     "qwen2.5-vl;model_size:3b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:4;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_3b_256_2k_448x448_20251219.zip"
#     "qwen3-vl;model_size:4b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:4;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh1_qwen3-vl_4b_256_2k_448x448_20251111.zip"
#     "qwen3-vl;model_size:4b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh1_qwen3-vl_4b_256_2k_448x448_20251111.zip"
# )
##### XH2 #####
models=(
    "qwen3;model_size:4b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_4b_2507_256_2k_20260116.zip"
    "qwen3;model_size:8b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20251127.zip"
    "qwen3;model_size:8b;target:xh2;context_length:16384;prefill_length:256;batch:4;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20251127.zip"
    "qwen3;model_size:14b;target:xh2;context_length:16384;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20251127.zip"
    "qwen3;model_size:14b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:2;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20251127.zip"
    "qwen3;model_size:14b;target:xh2;context_length:256;prefill_length:256;batch:2;device_num:2;core_num:2;flash_attention:0;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20251127.zip"
    "qwen3;model_size:32b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:4;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_32b_256_2k_20251211.zip"
    "qwen3-vl;model_size:4b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh2_qwen3-vl_4b_256_2k_448x448_20251217.zip"
    "qwen3-vl;model_size:8b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh2_qwen3-vl_8b_256_2k_448x448_20251216.zip"
    "qwen2.5-vl;model_size:7b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;flash_attention:0 1;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen2.5-vl/hmquant_xh2_qwen2.5-vl_7b_256_2k_448x448_20251128.zip"
    "qwen3;model_size:30b_a3b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_30b_a3b_256_2k_20251210.zip"
    "qwen3-vl;model_size:30b_a3b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh2_qwen3-vl_30b_a3b_256_2k_448x448_20260120.zip"
    "qwen2.5;model_size:7b;target:xh2;context_length:256;prefill_length:256;batch:1;device_num:0;core_num:2;flash_attention:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20251216.zip"
    "qwen2.5;model_size:7b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;flash_attention:0;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20250904.zip"
    "qwen2.5;model_size:7b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;flash_attention:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20251216.zip"
    "deepseek-r1-qwen3;model_size:8b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/deepseek/hmquant_xh2_deepseek_qwen3_8b_2k_20250903.zip"
    "bge;model_size:0.5b;target:xh2;context_length:512;prefill_length:0;batch:10;device_num:0;core_num:2;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/bge/hmquant_xh2_bge_0.5b_0.5k_b10_20251022.zip"
    "gte;model_size:1.5b;target:xh2;context_length:0;prefill_length:256;batch:0;device_num:-1;core_num:2;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/gte/hmquant_xh2_gte_1.5b_256_20251104.zip"
    "qwen3_coder;model_size:30b_a3b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3_coder/hmquant_xh2_qwen3_coder_30b_a3b_256_16k_20260116.zip"
    "gpt;model_size:20b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/gpt/hmquant_xh2_gpt_oss_20b_256_2k_20251127.zip"
    "minicpmo;model_size:7b;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/minicpmo/hmquant_xh2_minicpmo_7b_256_4k_20251216.zip"
    "whisper;model_size:medium;target:xh2;context_length:0;prefill_length:0;batch:0;device_num:-1;core_num:2;strip:copy;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/whisper/hmquant_xh2_whisper_medium_20251104.zip"
    "qwen3;model_size:8b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20251127.zip"
    "qwen3;model_size:14b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20251127.zip"
    "qwen3-vl;model_size:8b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh2_qwen3-vl_8b_256_2k_448x448_20251107.zip"
    "deepseek-r1-qwen3;model_size:8b;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/deepseek/hmquant_xh2_deepseek_qwen3_8b_2k_20250903.zip"
    "deepseek-r1-qwen3;model_size:8b;target:xh2;context_length:4096;prefill_length:256;batch:2;device_num:0;core_num:2;strip:overwrite;quant_model_path:http://artifactory.houmo.ai/artifactory/toolchain/release/models_outdated/deepseek/hmquant_xh2_deepseek_qwen3_8b_2k_20250903.zip"
)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR" || {
    echo "Cannot enter the directory of the script: $SCRIPT_DIR" >&2
    exit 1
}

current_time=$(date +"%Y%m%d_%H%M%S")
perf_cfg_file="${SCRIPT_DIR}/perf_cfg_v${VERSION}_${current_time}.json"
# Iterate through the list of models
for item in "${models[@]}"; do
    # Split model name and parameters (model name is before the first semicolon)
    IFS=';' read -r model_name params <<< "$item"
    echo "Model name: $model_name"

    # Parse parameters
    declare -A model_params
    # Split parameter pairs (skip the model name before the first semicolon)
    IFS=';' read -ra param_pairs <<< "$params"
    for pair in "${param_pairs[@]}"; do
        IFS=':' read -r key value <<< "$pair"
        model_params["$key"]="$value"
        echo "  $key: $value"
    done

    cmd="python3 compile_llms.py"
    cmd+=" --model_name $model_name"
    for key in "${!model_params[@]}"; do
        value="${model_params[$key]}"
        # Skip parameters with value of none
        if [ "$value" != "none" ]; then
            cmd+=" --$key $value"
        else
            echo "  Skipping parameter $key (value is none)"
        fi
    done

    if [[ $cmd != *"--version"* ]]; then
        cmd+=" --version $VERSION"
    fi
    if [[ $cmd != *"--result_dir"* ]]; then
        model_size="${model_params['model_size']}"
        result_dir="/data02/services/model_results/${model_name}_${model_size}"
        cmd+=" --result_dir $result_dir"
    fi
    cmd+=" -up -compile"
    if [ "$RELEASE" = "ON" ]; then
        cmd+=" -release"
    fi
    if [ "$PERF" = "ON" ]; then
        cmd+=" -perf ${perf_cfg_file}"
    fi

    # Execute the command
    echo -e "Executing command: $cmd"
    eval "$cmd" || true

    # Check execution result
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Model ${model_name} compiled successfully"
    else
        echo "Error: Model ${model_name} compilation failed, Exit code: ${EXIT_CODE}"
    fi
    RET=$EXIT_CODE

    # Clean up parameter array
    unset model_params
done
echo "All model compilation completed $(date)"

if [ "$PERF" = "ON" ]; then
    if [ -f "$perf_cfg_file" ]; then
        # Read credentials from environment variables
        XH2_SERVER_USER="${XH2_SERVER_USER:-}"
        XH2_SERVER_IP="${XH2_SERVER_IP:-}"
        # Validate required environment variables
        if [ -z "$XH2_SERVER_USER" ] || [ -z "$XH2_SERVER_IP" ]; then
            echo "ERROR: PERF Failed! XH2_SERVER_USER and XH2_SERVER_IP environment variables must be set!" >&2
            exit 1
        fi
        
        echo "Starting XH2 Perf on ${XH2_SERVER_IP} $(date)"
        ssh "${XH2_SERVER_USER}@${XH2_SERVER_IP}" "python3 ${SCRIPT_DIR}/perf_llms.py -v ${VERSION} -perf ${perf_cfg_file} -log ${SCRIPT_DIR}/compiler_perf_${current_time}.log"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "XH2 remote Perf execution failed (Exit code: $EXIT_CODE)" >&2
            RET=$EXIT_CODE
        fi
    fi
fi

exit $RET