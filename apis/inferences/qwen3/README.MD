# Qwen3(通义千问3)

本示例展示如何把编译后的Qwen3模型部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于[https://modelscope.cn/models/Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)。

目前芯片模型只能在linux环境下量化和编译，请参考houmo-modelzoo/models/llm/qwen3中的量化和编译。


## 2 快速开始

目前Qwen3模型仅提供python脚本编译方式。

1. 将python目录加入PYTHONPATH环境变量

2. 设置运行库路径
    - linux环境将lib目录加入LD_LIBRARY_PATH环境变量

3. 如系统为linux aarch64，需设置预加载动态库路径
    - export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

4. 获取模型文件，提供4 core模型和2 core模型，可通过参数--ncore指定模型core数量，执行命令：

```bash
# (默认)下载4core模型
python3 get_model.py
# 下载2core模型
python3 get_model.py --ncore 2
```

如果需要其他规格，请参考houmo-modelzoo自行编译或通过其他途径获取。

5. Qwen3模型使用python API进行演示，执行演示程序：

```bash
python3 demo.py
```

## 3 模型c++推理
目前c++推理Qwen3-8b模型，仅支持运行在x86_64平台的xh2设备上。 
### 3.1 依赖三方库环境
    tokenizer-cpp和eigen，目前在xh2设备上Ubuntu24.04平台x86_64版本的的三方库已经编译好，已经集成到get_model.py和install.sh中，并且在CMakeLists.txt进行了调用，执行cmake ..时会自动进行三方库的安装。如需手动安装，可以参考以下bash脚本示例中的详细步骤进行手动安装。
```
#!/bin/bash
if [ $(uname -m) = "x86_64" ] && [ "$HOUMO_TARGET" = "xh2" ]; then
  # 安装expect工具（处理交互所需）
  sudo apt-get update
  sudo apt-get install -y expect libeigen3-dev
  if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
        echo "环境变量已配置"
        echo "Rust版本: $(rustc --version)"
        cd $PWD/..
        if [ -e 3rdparty ];then
          cd 3rdparty
        else
          mkdir 3rdparty && cd 3rdparty
        fi

        git clone https://github.com/mlc-ai/tokenizers-cpp.git
        cd tokenizers-cpp
        git submodule update --init --recursive
        exit 0
  fi
# 创建临时的expect脚本
cat << 'EOF' > rust_install.exp
#!/usr/bin/expect -f
set timeout 1800 ;# 设置30分钟超时，适应慢网络

puts "开始自动安装Rust..."

# 启动安装过程
spawn sh -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"

# 处理所有可能的交互提示
expect {
    # 处理需要回车的提示
    ">" {
        send "\r"
        exp_continue
    }
    # 处理安装完成的提示
    eof
}

puts "Rust安装完成！"
EOF

  # 赋予执行权限并运行expect脚本
  chmod +x rust_install.exp
  ./rust_install.exp

  # 清理临时文件
  rm rust_install.exp

  # 配置环境变量
  if [ -f "$HOME/.cargo/env" ]; then
      source "$HOME/.cargo/env"
      echo "环境变量已配置"
      echo "Rust版本: $(rustc --version)"
      cd $PWD/..
      if [ -e 3rdparty ];then
        cd 3rdparty
      else
        mkdir 3rdparty && cd 3rdparty
      fi

      git clone https://github.com/mlc-ai/tokenizers-cpp.git
      cd tokenizers-cpp
      git submodule update --init --recursive
      exit 0
  else
      echo "请手动运行以下命令配置环境变量:"
      echo "source $HOME/.cargo/env"
  fi
else
  if [ $(uname -m) != "x86_64" ]; then
    echo "UnSupport Platform" $(uname -m)
  fi

  if [ "$HOUMO_TARGET" = "xh2" ]; then
    echo "No xh2Backhend found!"
  fi
fi
```

### 3.2 模型获取与模型准备
    由于目前c++的demo仅支持xh2的Ubuntu24.04系统的x86_64设备，因此在xh2平台可以直接使用get_model.py获取模型和三方库，为了让c++方便读取embedding权重，需要对embedding权重的pt文件进行转换，转换示例代码如下：
```python
embedding_path = "hmquant/quant_embedding.pt"
#权重转换
if os.path.exists(embedding_path) and HOUMO_TARGET == "xh2":
    import torch
    import numpy as np

    embedding_weight = torch.load(embedding_path, map_location="cpu", weights_only=True)['weight']
    embedding_data = embedding_weight.cpu().numpy() 
    embedding_data.tofile(embedding_path.replace(".pt", ".bin"))
#三方库下载
import platform
if platform.system() == 'Linux' and platform.machine() == 'x86_64':
    thirdparty_path = "models/qwen3/3rdparty.zip"
    get_file_from_jfrog(thirdparty_path, model_dir, "./")
```
    目前这部分转换代码已经集成到get_model.py中，在xh2的Ubuntu24.04系统的x86_64设备下载好模型后，会自动进行转换，并下载三方库。

### 3.3 程序编译和运行
    提供run_cpp_linux.sh脚本进行一键编译和运行，如需手动编译和运行参考指令如下：
```bash
if [ $(uname -m) = "x86_64" ] && [ "$HOUMO_TARGET" = "xh2" ]; then
  set -e

  WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  cd "${WORK_PATH}" || exit 1

  # get test model
  python3 get_model.py
  mkdir -p build
  cd build || exit 1

  cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release ..
  make
  make install

  cd $WORK_PATH
  ./example_cxx_qwen3
  #如需手动指定对应模型文件路径使用如下方式启动
  # ./example_cxx_qwen3 ${prefillModelPath} ${decodeModelPath} ${tokenizerJsonPath} ${embeddingWeightPath}
else
  echo "UnSupport Environment"
fi
```

### 3.4 C++样例相关文件说明
```
.
├── CMakeLists.txt              CMake配置文件
├── README.MD                   
├── run_cpp_qwen3.sh            编译和运行qwen3 c++样例脚本         
├── get_model.py                获取运行所需要的模型文件
├── include                     相关头文件
├── HmQwenInfer.cc              qwen3 c++推理类封装
├── Hmtokenizer.cc              qwen3 tokenizer功能实现封装
├── qwen3.cc                    qwen3 样例主函数实现
└── install.sh                  三方依赖库自动编译安装脚本
```

### 3.5 C++演示结果
```
Question : 请介绍一下存算一体技术的优势
Response : 存算一体（Compute-in-Memory, CIM）技术是一种将存储与计算功能集成在同一芯片或模块中的新型计算架构，它突破了传统冯·诺依曼架构中存储与计算分离的限制，具有显著的优势。以下是存算一体技术的主要优势：

---

### 1. **降低能耗**
- **减少数据移动**：传统计算架构中，数据需要在存储器和处理器之间频繁搬运，这会消耗大量能量。存算一体技术将计算直接在存储单元中完成，大大减少了数据移动，从而显著降低能耗。
- **提升能效比**：由于减少了数据传输的能耗，存算一体技术在低功耗场景（如物联网、边缘计算）中表现尤为突出。

---

### 2. **提高计算效率**
- **并行计算能力**：存算一体技术可以同时进行存储和计算操作，支持并行处理，提升计算效率。
- **加速矩阵运算**：在人工智能、深度学习等需要大量矩阵运算的领域，存算一体技术可以显著加快计算速度，提高模型训练和推理效率。

---

### 3. **提升计算密度**
- **高集成度**：将存储器与计算单元集成在同一芯片上，可以实现更高的计算密度，适用于需要高并发计算的场景。
- **节省空间**：由于不需要额外的计算模块，存算一体技术可以减少硬件体积，适用于嵌入式系统、可穿戴设备等对空间要求较高的设备。

---

### 4. **适用于新兴计算场景**
- **人工智能与机器学习**：存算一体技术特别适合神经网络中的矩阵运算，可以显著提升AI模型的推理速度和能效。
- **边缘计算**：在边缘端进行本地计算，减少对云端的依赖，提高实时性和隐私性。
- **物联网（IoT）**：适用于低功耗、高效率的传感器节点和边缘设备。

---

### 5. **支持新型存储技术**
- 存算一体技术可以与新型非易失性存储器（如忆阻器、相变存储器、铁电存储器等）结合，实现高性能、低功耗的存储与计算一体化系统。

---

### 6. **降低系统复杂度**
- 由于存储与计算功能集成，系统架构更加简洁，减少了硬件复杂度，降低了系统设计和维护的难度。

---

### 总结

存算一体技术通过将存储与计算功能融合，显著提升了计算效率、降低了能耗，并适用于人工智能、边缘计算、物联网等新兴领域。它被认为是未来高性能计算、低功耗计算的重要发展方向之一。

如果你对某一具体应用场景（如AI、物联网、边缘计算）感兴趣，我可以进一步展开说明。
```

### 3.6 性能结果
```bash
#linux x86_64
total: 572 tokens, cost 54.163 s
prefill time: 1655.214 ms, 18.12 tokens/s
decode average time: 91.957 ms, 10.87 tokens/s
end2end average time: 94.691 ms, 10.56 tokens/s
```

## 4 参考结果

### 4.1 演示结果

```bash
question:
请介绍一下存算一体技术的优势
response:
<think>
嗯，用户让我介绍一下存算一体技术的优势。首先，我需要明确什么是存算一体技术。存算一体，也就是存储和计算一体化，可能是指将存储和计算功能集成在一个系统中，这样可以减少数据在存储和计算单元之间的传输，从而提高效率。不过，可能用户指的是更具体的，比如在神经网络或AI芯片中的应用，比如存内计算（In-Memory Computing）或者存算一体架构。

接下来，我得考虑用户可能的背景。他们可能是学生、研究人员，或者企业技术人员，想了解这项技术在哪些方面有优势。可能他们正在做项目，或者想了解技术趋势，比如在AI、物联网或边缘计算中的应用。

用户的需求可能不仅仅是列举优势，还可能希望了解这些优势如何应用在实际场景中，或者与其他技术的对比。比如，传统架构中数据需要在存储和计算单元之间传输，导致延迟和能耗，而存算一体可以解决这些问题。所以，我需要分点说明优势，比如能效、速度、可扩展性、灵活性、降低延迟、支持复杂计算等。

另外，可能需要提到具体的应用场景，比如在AI芯片、物联网设备、边缘计算中的应用，以及如何提升性能。同时，可能还要提到技术挑战，比如设计复杂性、制造工艺等，但用户可能更关注优势，所以这部分可能需要简要提及。

还要注意术语的准确性，比如存算一体可能有不同的实现方式，比如存内计算、存内逻辑、存内模拟等，需要明确这些概念。不过用户可能不需要太深入的技术细节，所以保持简洁。

最后，确保回答结构清晰，分点列出优势，并给出实际例子，帮助用户理解。可能还需要比较传统架构和存算一体的差异，突出其优势。比如，传统架构的数据搬运瓶颈，存算一体如何解决，从而提升性能和能效。
</think>

存算一体技术（Compute-in-Memory, CIM）是一种将存储和计算功能集成在同一物理结构中的技术，旨在通过减少数据在存储和计算单元之间的传输，提升系统效率、降低能耗并优化性能。以下是其主要优势：

---

### **1. 提升计算效率**
- **减少数据搬运瓶颈**：传统架构中，数据需要在存储单元（如内存）和计算单元之间频繁传输，导致延迟和能耗。存算一体技术将计算直接嵌入存储单元，避免数据搬运，显著提升计算效率。
- **并行计算能力**：在存内计算中，数据可以直接在存储器中进行运算，支持并行处理，尤其适合处理大规模数据（如AI模型训练和推理）。

---

### **2. 降低能耗**
- **减少功耗**：数据搬运是计算过程中的主要能耗来源。存算一体技术通过减少数据移动，降低整体功耗，尤其适用于低功耗设备（如物联网终端、可穿戴设备）。
- **支持绿色计算**：在数据中心和边缘计算场景中，存算一体技术有助于降低碳排放，符合可持续发展需求。

---

### **3. 支持复杂计算任务**
- **加速AI与机器学习**：存算一体技术特别适合处理神经网络等复杂计算任务。例如，在存内模拟（In-Memory Computing）中，权重和数据可以直接在存储器中进行矩阵运算，显著提升AI推理速度。
- **高密度计算**：通过将计算逻辑嵌入存储器，存算一体技术可支持更高密度的计算任务，满足高性能计算需求。

---

### **4. 可扩展性与灵活性**
- **适应不同应用场景**：存算一体技术可根据需求调整存储和计算单元的配置，适用于从边缘设备到云端的多场景（如边缘AI、自动驾驶、智能传感器）。
- **兼容现有架构**：可与传统冯·诺依曼架构结合，通过混合架构实现性能与成本的平衡。

---

### **5. 降低系统复杂性**
- **简化硬件设计**：通过减少数据传输需求，存算一体技术降低了系统复杂性，简化了硬件设计和软件编程。
- **支持新型存储技术**：如忆阻器（ReRAM）、相变存储器（PCM）等，这些技术在存算一体架构中能更好地发挥其潜力。

---

### **6. 降低延迟**
- **实时响应需求**：在需要低延迟的场景（如自动驾驶、工业控制）中，存算一体技术能快速处理数据，满足实时性要求。
- **优化通信延迟**：在分布式系统中，减少数据传输延迟可提升整体系统性能。

---

### **应用场景**
- **AI芯片**：如TPU、GPU等加速器采用存算一体设计，提升AI模型训练和推理效率。
- **物联网设备**：在边缘端实现本地数据处理，减少对云端依赖。
- **数据中心**：降低能耗和延迟，提升大规模数据处理能力。
- **自动驾驶**：实时处理传感器数据，提升决策速度。

---

### **挑战与局限**
尽管存算一体技术具有显著优势，但其设计和实现仍面临挑战，如：
- **技术复杂性**：需要新型存储器和计算单元的协同设计。
- **制造工艺限制**：现有半导体工艺可能难以完全实现存算一体的集成。
- **算法适配**：需针对特定应用场景优化算法以充分利用存算一体特性。

---

### **总结**
存算一体技术通过整合存储与计算，解决了传统架构中的数据搬运瓶颈，显著提升了能效、速度和灵活性，是未来高性能计算、AI和边缘计算的重要方向。其优势在AI、物联网、自动驾驶等领域尤为突出，正在推动新一代计算架构的变革。
```

### 4.2 性能结果

```bash
# linux aarch64
total: 1199 tokens, cost 216.051 s
prefill time: 1444.001 ms, 0.69 tokens/s
decode average time: 179.130 ms, 5.58 tokens/s
end2end average time: 180.193 ms, 5.55 tokens/s

---

# linux x86_64
total: 1199 tokens, cost 114.017 s
prefill time: 741.065 ms, 1.35 tokens/s
decode average time: 94.553 ms, 10.58 tokens/s
end2end average time: 95.093 ms, 10.52 tokens/s
```

## 5 一键执行

可以使用run脚本一键执行查看结果

脚本在linux x86_64环境中默认下载4core模型，在linux aarch64环境中默认下载2core模型。

linux执行脚本命令：

```bash
bash run.sh
```

## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
