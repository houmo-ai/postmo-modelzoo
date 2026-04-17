<!-- SPDX-License-Identifier: Apache-2.0 -->


# Qwen3.5 MoE模型量化导出示例。请参考以下内容按要求实现或修改../ptq.py和../quant_pipeline.py

## step1：对Qwen3.5 MoE做旋转
python src/example_qwen35_moe_vl_rotate_fp.py \
  --model /data02/datasets/Qwen3.5-35B-A3B \
  --out xxx/Qwen3.5-35B-A3B_rotated_fp \
  --llm-rotation hadamard \
  --vision-rotation last \
  --device cuda:0 \
  --validate

## step2：使用GPTQModel量化Qwen3.5 MoE模型
python src/example_qwen35moe.py \
  --model /data02/datasets/Qwen3.5-35B-A3B \
  --out xxx/Qwen3.5-35B-A3B_gptq_4bit
  --rotation hadamard --hessian-mse --moe-routing bypass --nsamples=256 \
  --shared-expert-bits 4  \
  --self-attn-bits 4 \
  --expert-bits 4

## step3：导出Vision部分的hmonnx，--hf_model_dir依赖于step1的输出
python src/qwen3_5_moe_vision_xh2a_export_hmonnx.py \
  --config src/configs/qwen3_5_moe/qwen3_5_moe_vision_config.py \
  --hf_model_dir xxx/Qwen3.5-35B-A3B_rotated_fp \
  --model_name xxx \
  --output_dir xxx


## step4：导出LLM部分的hmonnx，--quant-weight依赖于step2的输出，--model就是原始模型
python src/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model /data02/datasets/Qwen3.5-35B-A3B \
  --quant-weight xxx/Qwen3.5-35B-A3B_gptq_4bit \
  --quant-type w4a8h0_sefp \
  --context-length 2048 --input-sequence-length 256 \
  --output-dir xxx


# 补充说明：
1. step1，step2是模型量化，step3，step4是模型导出
2. step1~2的中间结果存放位置由ptq.py中--work-dir参数指定。step3~4的结果存放位置由ptq.py中--out-dir指定。原始模型由ptq.py中--model参数指定
3. export_llm，quant_llm函数调用step1~4的功能时，除了输入输出等必要参数外
    step1功能参数均采用示例中设置，示例中未出现的参数就不使用。
    step2功能参数均采用示例中设置。
    step3 --model_name 参数使用ptq.py中--model-name参数的值，其余默认。
    step4功能参数均采用示例中设置。