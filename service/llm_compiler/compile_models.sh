#!/bin/bash
set -e

models=(
    "qwen3;model_size:8b;version:0.3.0;target:xh2;context_length:2048;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20250812.zip"
    "qwen3;model_size:8b;version:0.3.0;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20250812.zip"
    "qwen3;model_size:8b;version:0.3.0;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_8b_2k_20250812.zip"
    "qwen3;model_size:14b;version:0.3.0;target:xh2;context_length:2048;prefill_length:256;batch:1;device_num:2;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20250902.zip"
    "qwen3;model_size:14b;version:0.3.0;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:2;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20250902.zip"
    "qwen3;model_size:14b;version:0.3.0;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:2;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen3/hmquant_xh2_qwen3_14b_2k_20250902.zip"
    "qwen2.5;model_size:7b;version:0.3.0;target:xh2;context_length:2048;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20250904.zip"
    "qwen2.5;model_size:7b;version:0.3.0;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20250904.zip"
    "qwen2.5;model_size:7b;version:0.3.0;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/qwen2.5/hmquant_xh2_qwen2.5_7b_2k_20250904.zip"
    "deepseek-r1-qwen3;model_size:8b;version:0.3.0;target:xh2;context_length:2048;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/deepseek/hmquant_xh2_deepseek_qwen3_8b_2k_20250903.zip"
    "deepseek-r1-qwen3;model_size:8b;version:0.3.0;target:xh2;context_length:4096;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/deepseek/hmquant_xh2_deepseek_qwen3_8b_2k_20250903.zip"
    "deepseek-r1-qwen3;model_size:8b;version:0.3.0;target:xh2;context_length:8192;prefill_length:256;batch:1;device_num:1;core_num:2;quant_model_path:http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/deepseek/hmquant_xh2_deepseek_qwen3_8b_2k_20250903.zip"
)

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

    if [[ $cmd != *"--result_dir"* ]]; then
        model_size="${model_params['model_size']}"
        result_dir="/data/services/model_results/${model_name}_${model_size}"
        cmd+=" --result_dir $result_dir"
    fi
    cmd+=" -up -compile"

    # 执行命令
    echo -e "\n执行命令: $cmd"
    eval "$cmd"

    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "模型${model_name}编译成功"
    else
        echo "错误: 模型${model_name}编译失败" >&2
        # exit 1
    fi

    # 清理参数数组
    unset model_params
done

echo "所有模型编译完成"