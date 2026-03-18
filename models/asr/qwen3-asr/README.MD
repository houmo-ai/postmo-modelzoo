# Qwen3-ASR

本示例展示如何把Qwen3-ASR模型部署到后摩鸿途芯片的设备上。


### 1.准备模型

自行下载编译好的模型，并解压到当前目录下。


### 2.模型部署

#### 2.1 创建虚拟环境

```bash
bash test.sh
source qwen3-asr/bin/activate
```
#### 2.1.1 ASR语音识别演示

```bash
python3 demo_asr.py --processor_dir Qwen3-ASR-0.6B \
    --audio ../../../data/audio/audio.mp3 \
    --encode_path models/qwen3_asr_0.6b/qwen3_asr_encode.hmm \
    --decode_path models/qwen3_asr_0.6b/qwen3_asr_decode.hmm \
    --prefill_path models/qwen3_asr_0.6b/qwen3_asr_prefill.hmm \
    --embedding_path models/qwen3_asr_0.6b/hmquant/quant_embedding.pt

python3 demo_asr.py --processor_dir Qwen3-ASR-1.7B \
    --audio ../../../data/audio/audio.mp3 \
    --encode_path models/qwen3_asr_1.7b/qwen3_asr_encode.hmm \
    --decode_path models/qwen3_asr_1.7b/qwen3_asr_decode.hmm \
    --prefill_path models/qwen3_asr_1.7b/qwen3_asr_prefill.hmm \
    --embedding_path models/qwen3_asr_1.7b/hmquant/quant_embedding.pt
```

#### 2.1.2 ForcedAligner强制对齐演示

```bash
python3 demo_forcealigner.py --processor_dir Qwen3-ForcedAligner-0.6B \
    --audio ../../../data/audio/61-70968-0000.wav \
    --text "HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT" \
    --encode_path models/qwen3_forcealigner_0.6b/qwen3_asr_encode.hmm \
    --prefill_path models/qwen3_forcealigner_0.6b/qwen3_asr_prefill.hmm \
    --embedding_path models/qwen3_forcealigner_0.6b/hmquant/quant_embedding.pt
```

## 3 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。