<!-- SPDX-License-Identifier: Apache-2.0 -->

# 任务说明
1. 参考当前目录下的脚本，实现../quant_pipeline.py和ptq.py，quant_pipeline.py编写规范可以参考~/code/imodelzoo/models/vlm/qwen3-vl


# Qwen3.5 9B Dense模型量化导出示例，其余尺寸的模型（例如Qwen3.5 27B）过程完全相同，依赖的config文件也是相同的

## step1：对Qwen3.5做旋转
export CUDA_VISIBLE_DEVICES=0
python example_qwen35_vl_rotate_fp.py \
  --model /data01/datasets/Qwen3.5-9B \
  --out ../output/Qwen3.5-9B-rotated-fp \
  --llm-rotation hadamard \
  --vision-rotation last \
  --device cuda \
  --validate

## step2：使用GPTQModel量化Qwen3.5 Dense模型
export CUDA_VISIBLE_DEVICES=0
python example_qwen35dense.py \
  --model /data01/datasets/Qwen3.5-9B \
  --out ../output/Qwen3.5-9B-quarot-gptq-4bit-mse24-hessian \
  --bits 4 \
  --group-size 64 \
  --rotation hadamard \
  --nsamples 256 \
  --seqlen 1024 \
  --mse 2.4 \
  --hessian-mse \
  --device cuda

## step3：导出Vision部分的hmonnx，依赖于step1的输出
export CUDA_VISIBLE_DEVICES=0
python qwen3_5_vision_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_instruct_vision_config.py \
  --hf_model_dir ../output/Qwen3.5-9B-rotated-fp/ \
  --model_name Qwen3_5 \
  --output_root ../output


## step4：导出LLM部分的hmonnx，依赖于step2的输出
export CUDA_VISIBLE_DEVICES=0
python qwen3_5_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_xh2a.py \
  --hf_model_dir ../output/Qwen3.5-9B-quarot-gptq-4bit-mse24-hessian \
  --dtype fp16 \
  --valid \
  --golden \
  --work_dir ../output/qwen3_5_llm_gptq_export


# 该任务的补充说明：
1. step1，step2是模型量化，对应../quant_pipeline.py中的quant_llm函数。step3，step4是模型导出，对应../quant_pipeline.py中的export_llm函数。
2. step1~2的中间结果存放位置由ptq.py中--work-dir参数指定。step3~4的结果存放位置由ptq.py中--out-dir指定。
3. export_llm，quant_llm函数调用step1~4的功能时，除了输入输出等必要参数外
    step1功能参数均采用示例中默认设置。
    step2功能参数均采用示例中默认设置。
    step3 --model_name 参数使用ptq.py中--model-name参数的值，其余默认。
    step4功能参数均采用示例中默认设置。
4. quant_pipeline.py中尽可能保持代码简洁，不要把/src文件夹脚本中的复杂代码复制到quant_pipeline.py中，建议采用调用接口的方式。你可以对/src中的脚本做一些必要的改动