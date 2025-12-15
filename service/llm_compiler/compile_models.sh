#!/bin/bash
set -e

RELEASE="OFF"
PERF="OFF"
VERSION="0.6.0"
# models=(
#     # "qwen3;model_size:8b;target:xh1;context_length:8192;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh1_qwen3_8b_8k_20250603.zip"
#     # "qwen3;model_size:8b;target:xh1;context_length:8192;prefill_length:256;batch:1;device_num:1;core_num:4;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models/qwen3/hmquant_xh1_qwen3_8b_256_8k_20250815.zip"
#     "deepseek-r1-qwen;model_size:7b;target:xh1;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:4;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models/deepseek/hmquant_deepseek_256_8192_20250922.zip"
#     # "deepseek-r1-qwen;model_size:7b;target:xh1;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models/deepseek/hmquant_deepseek_256_8192_20250922.zip"
#     # "qwen2.5-vl;model_size:7b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:4;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_7b_256_2k_20250903.zip"
#     # "qwen2.5-vl;model_size:7b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_7b_256_2k_20250903.zip"
#     # "qwen2.5-vl;model_size:3b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_3b_256_2k_448x448_20251219.zip"
#     # "qwen2.5-vl;model_size:3b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:4;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5-vl/hmquant_xh1_qwen2.5-vl_3b_256_2k_448x448_20251219.zip"
#     # "qwen3-vl;model_size:4b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:4;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh1_qwen3-vl_4b_256_2k_448x448_20251111.zip"
#     # "qwen3-vl;model_size:4b;target:xh1;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh1_qwen3-vl_4b_256_2k_448x448_20251111.zip"
# )
models=(
    "qwen3;model_size:8b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20250812.zip"
    "qwen3;model_size:8b;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20250812.zip"
    "qwen3;model_size:8b;target:xh2;context_length:8192;prefill_length:256;batch:4;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20250812.zip"
    "qwen3;model_size:14b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20250902.zip"
    "qwen3;model_size:14b;target:xh2;context_length:16384;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20250902.zip"
    "qwen3;model_size:14b;target:xh2;context_length:16384;prefill_length:256;batch:1;device_num:2;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20250902.zip"
    "qwen3;model_size:14b;target:xh2;context_length:16384;prefill_length:256;batch:2;device_num:2;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20250902.zip"
    "qwen3;model_size:14b;target:xh2;context_length:32768;prefill_length:256;batch:1;device_num:2;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20250902.zip"
    "qwen3;model_size:32b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:4;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_32b_256_2k_20250625.zip"
    "qwen3;model_size:30b_a3b;target:xh2;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_30b_a3b_256_8k_20251027.zip"
    "qwen3;model_size:30b_a3b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_30b_a3b_256_8k_20251027.zip"
    "qwen3_coder;model_size:30b_a3b;target:xh2;context_length:16384;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3_coder/hmquant_xh2_qwen3_coder_30b_a3b_256_2k_20251127.zip"
    "gpt;model_size:20b;target:xh2;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/gpt/hmquant_xh2_gpt_oss_20b_256_2k_20251127.zip"
    "qwen2.5-vl;model_size:7b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5-vl/hmquant_xh2_qwen2.5-vl_7b_256_2k_448x448_20251128.zip"
    "qwen3-vl;model_size:8b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh2_qwen3-vl_8b_256_2k_448x448_20251107.zip"
    "qwen3-vl;model_size:4b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh2_qwen3-vl_4b_256_2k_448x448_20251107.zip"
    # "qwen3-vl;model_size:2b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3-vl/hmquant_xh2_qwen3-vl_2b_256_2k_448x448_20251107.zip"
    "deepseek-r1-qwen3;model_size:8b;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/deepseek/hmquant_xh2_deepseek_qwen3_8b_2k_20250903.zip"
    "qwen2.5;model_size:7b;target:xh2;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20250904.zip"
    "qwen2.5;model_size:7b;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20250904.zip"
    "qwen2.5;model_size:7b;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20250904.zip"
    "whisper;model_size:medium;target:xh2;context_length:0;prefill_length:0;batch:0;device_num:-1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/whisper/hmquant_xh2_whisper_medium_20251104.zip"
    "minicpmo;model_size:7b;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models/minicpmo/hmquant_xh2_minicpmo_7b_256_4k_20251120.zip"
    "bge;model_size:0.5b;target:xh2;context_length:512;prefill_length:0;batch:10;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/bge/hmquant_xh2_bge_0.5b_0.5k_b10_20251022.zip"
    "gte;model_size:1.5b;target:xh2;context_length:0;prefill_length:256;batch:0;device_num:-1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/gte/hmquant_xh2_gte_1.5b_256_20251104.zip"
    "qwen3;model_size:8b;target:xh2;context_length:2048;prefill_length:256;batch:1;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20250812.zip"
    "deepseek-r1-qwen3;model_size:8b;target:xh2;context_length:4096;prefill_length:256;batch:2;device_num:0;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/deepseek/hmquant_xh2_deepseek_qwen3_8b_2k_20250903.zip"
)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR" || {
    echo "无法进入测试脚本所在目录: $SCRIPT_DIR" >&2
    exit 1
}

current_time=$(date +"%Y%m%d_%H%M%S")
perf_cfg_file="${SCRIPT_DIR}/perf_cfg_v${VERSION}_${current_time}.json"
# 遍历模型列表
for item in "${models[@]}"; do
    # 分割模型名称和参数部分（第一个分号前是模型名）
    IFS=';' read -r model_name params <<< "$item"
    echo "模型名称: $model_name"

    # 解析参数
    declare -A model_params
    # 分割参数对（跳过第一个分号前的模型名）
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
        # 忽略值为none的参数
        if [ "$value" != "none" ]; then
            cmd+=" --$key $value"
        else
            echo "  跳过参数 $key（值为none）"
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

    # 执行命令
    echo -e "\n执行命令: $cmd"
    eval "$cmd" || true

    # 检查执行结果
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "模型${model_name}编译成功"
    else
        echo "错误: 模型${model_name}编译失败, 错误码:${EXIT_CODE}"
    fi
    RET=$EXIT_CODE

    # 清理参数数组
    unset model_params
done
echo "所有模型编译完成 $(date)"

if [ "$PERF" = "ON" ]; then
    if [ -f "$perf_cfg_file" ]; then
        echo "开始在 10.64.34.58 执行XH2 Perf $(date)"
        ssh wanyu.li@10.64.34.58 "python3 ${SCRIPT_DIR}/perf_llms.py -v ${VERSION} -perf ${perf_cfg_file} -log ${SCRIPT_DIR}/compiler_perf_${current_time}.log"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "XH2远端Perf执行失败(退出码: $EXIT_CODE)" >&2
            RET=$EXIT_CODE
        fi
    fi
fi

exit $RET