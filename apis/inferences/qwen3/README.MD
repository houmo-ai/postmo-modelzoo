# Qwen3(通义千问 3)

本示例展示如何把编译后的 Qwen3 模型部署到后摩鸿途芯片的设备上。

[TOC]

## 1 模型说明

本例使用的模型实现来源于：

-   模型名称: Qwen3-8B
-   来源: [https://modelscope.cn/models/Qwen/Qwen3-8B](https://modelscope.cn/models/Qwen/Qwen3-8B)
-   许可: [Apache License 2.0](https://modelscope.cn/models/Qwen/Qwen3-8B/file/view/master/LICENSE)

预训练模型在运行时下载，工程发布**不包含**该模型。
目前芯片模型只能在 linux 环境下量化和编译，请参考 houmo-modelzoo/models/llm/qwen3 中的量化和编译。

## 2 快速开始

目前 Qwen3 模型仅提供 python 脚本编译方式。

1. 将 python 目录加入 PYTHONPATH 环境变量

2. 设置运行库路径

    - linux 环境将 lib 目录加入 LD_LIBRARY_PATH 环境变量

3. 如系统为 linux aarch64，需设置预加载动态库路径

    - export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

4. 获取模型文件，提供 2 core 模型，执行命令：

```bash
# 下载 2 core 模型
python3 get_model.py
```

如果需要其他规格，请参考 houmo-modelzoo 自行编译或通过其他途径获取。

5. Qwen3 模型使用 python API 进行演示，提供两种方式的 demo：

-   `Normal`: 执行单轮对话
-   `Prefix Caching`: 基于 Prefix 缓存，执行多轮对话

```bash
# Normal demo 执行方式
python3 demo.py
# Prefix Caching demo 执行方式
python3 demo_prefix_caching.py
```

## 3 参考结果

### 3.1 演示结果

#### 3.1.1 Normal demo 参考结果

```bash
qwen_base:__init__:78 - prefill model loaded
qwen_base:__init__:92 - decode model loaded
__main__:chat:183 - question:
请介绍一下存算一体技术的优势
__main__:chat:261 - response:
存算一体（Compute-in-Memory, CIM）技术是一种将存储与计算功能集成在同一芯片或模块中的技术，它通过在存储器中直接执行计算操作，显著提升了计算效率并降低了能耗。以下是存算一体技术的主要优势：

---

### 1. **提升计算效率**
- **减少数据移动**：传统计算架构中，数据需要在存储器和处理器之间频繁搬运，这会带来显著的延迟和能耗。存算一体技术将计算直接在存储器中完成，避免了数据搬运，从而大幅提升计算效率。
- **并行计算能力**：存算一体架构支持大规模并行计算，特别适合处理矩阵运算、卷积等计算密集型任务，如人工智能和机器学习中的神经网络计算。

---

### 2. **降低能耗**
- **减少功耗**：由于数据不需要在存储器和计算单元之间频繁传输，存算一体技术显著降低了能耗。特别是在大规模数据处理场景中，这种节能效果尤为明显。
- **适用于低功耗设备**：这种技术非常适合用于移动设备、物联网（IoT）设备、边缘计算节点等对功耗敏感的应用场景。

---

### 3. **提高计算密度**
- **集成度高**：存算一体技术将存储器与计算单元集成在同一芯片上，提高了芯片的计算密度，使得单位面积上可以实现更高的计算能力。
- **支持高并发计算**：由于计算和存储在同一物理位置，可以实现更高并发度的计算任务，适用于需要实时处理的场景。

---

### 4. **适用于AI与大数据处理**
- **加速AI推理**：存算一体技术特别适合用于人工智能中的推理任务，如卷积神经网络（CNN）、循环神经网络（RNN）等，能够显著加快模型推理速度。
- **支持大数据处理**：在大数据分析、图像识别、自然语言处理等领域，存算一体技术可以有效提升数据处理效率，降低系统复杂度。

---

### 5. **支持新型存储技术**
- **兼容非易失性存储器（NVM）**：存算一体技术可以与新型存储器（如忆阻器、相变存储器、存储器阵列等）结合，实现高性能、低功耗的计算能力。
- **支持存储器计算（Memristor Computing）**：利用忆阻器等新型存储器件的非线性特性，存算一体技术可以实现更高效的计算方式。

---

### 6. **降低系统复杂度**
- **简化系统架构**：由于计算和存储集成，系统架构更加简洁，减少了对高速互连和缓存的需求，降低了系统设计的复杂性。
- **提高系统可靠性**：减少数据传输路径，降低了系统故障率，提高了整体可靠性。

---

### 总结

存算一体技术通过将存储与计算功能融合，显著提升了计算效率、降低了能耗，并适用于人工智能、大数据处理、边缘计算等高性能计算场景。随着新型存储器和计算架构的发展，存算一体技术正成为未来计算系统的重要发展方向之一。

qwen_demo_utils:show_statistics:81 - Total Input: 30 tokens, Output 647 tokens, Prefill Cost 134.156 ms, Decode Cost 34682.281 ms
qwen_demo_utils:show_statistics:88 - Prefill Speed: 223.62 tokens/s; Decode Speed: 18.63 tokens/s
qwen_demo_utils:show_statistics:91 - TTFT (Time to First Token): 152.335 ms
qwen_demo_utils:show_statistics:92 - TPOT (Time Per Output Token): 53.688 ms/token
qwen_demo_utils:show_statistics:95 - E2E Latency (End-to-End Latency): 39.292 seconds
qwen_demo_utils:show_statistics:96 - E2E TPS (End-to-End Tokens Per Second): 16.47 tokens/s
```

#### 3.1.2 Prefix Caching demo 参考结果

```bash
qwen_base:__init__:78 - prefill model loaded
qwen_base:__init__:92 - decode model loaded
__main__:chat:284 - question:
你是谁？
__main__:find_longest_matched:186 - ❌ No matching cached prefix found
__main__:chat:376 - response:
我是电商平台的客服助手，负责解答订单处理相关的问题。
__main__:insert:156 - ✅ Successfully inserted prefix cache, token length: 648
qwen_demo_utils:show_statistics:81 - Total Input: 635 tokens, Output 14 tokens, Prefill Cost 425.193 ms, Decode Cost 700.975 ms
qwen_demo_utils:show_statistics:88 - Prefill Speed: 1493.44 tokens/s; Decode Speed: 18.55 tokens/s
qwen_demo_utils:show_statistics:91 - TTFT (Time to First Token): 462.604 ms
qwen_demo_utils:show_statistics:92 - TPOT (Time Per Output Token): 53.921 ms/token
qwen_demo_utils:show_statistics:95 - E2E Latency (End-to-End Latency): 1.261 seconds
qwen_demo_utils:show_statistics:96 - E2E TPS (End-to-End Tokens Per Second): 11.10 tokens/s
qwen_base:__init__:78 - prefill model loaded
qwen_base:__init__:92 - decode model loaded
__main__:insert:156 - ✅ Successfully inserted prefix cache, token length: 648
__main__:chat:284 - question:
回答用户A最后的两个问题，需严格遵循背景中的所有规则。
__main__:find_longest_matched:182 - 🔍 Found longest matching prefix, match length: 623
__main__:chat:376 - response:
订单3如果审核通过，由于其金额超过5万元，属于异常订单，且收货地址为上海浦东新区（非偏远地区），因此发货时间将根据正常规则处理。根据订单3的付款时间（昨天15点），商家需在付款后48小时内发货，因此预计最晚今天23点前发货。

如果之后申请退款，流程将根据订单状态进行判断：

- **若订单3已发货但未确认收货**：需用户退回商品，商家验收后72小时内退款；
- **若订单3已完成**：仅在存在质量问题、错发漏发等售后问题，并提供质检报告的情况下，支持退款，审核时间3个工作日。

因此，退款流程会根据订单实际状态而变化，建议在订单状态变更后及时查看系统通知或联系客服确认具体流程。
__main__:insert:156 - ✅ Successfully inserted prefix cache, token length: 823
qwen_demo_utils:show_statistics:81 - Total Input: 648 tokens, Output 176 tokens, Prefill Cost 139.806 ms, Decode Cost 9421.571 ms
qwen_demo_utils:show_statistics:88 - Prefill Speed: 4634.99 tokens/s; Decode Speed: 18.57 tokens/s
qwen_demo_utils:show_statistics:91 - TTFT (Time to First Token): 157.681 ms
qwen_demo_utils:show_statistics:92 - TPOT (Time Per Output Token): 53.838 ms/token
qwen_demo_utils:show_statistics:95 - E2E Latency (End-to-End Latency): 10.807 seconds
qwen_demo_utils:show_statistics:96 - E2E TPS (End-to-End Tokens Per Second): 16.29 tokens/s
__main__:chat:284 - question:
订单3若审核不通过，会进入什么状态？原因是什么？
__main__:find_longest_matched:182 - 🔍 Found longest matching prefix, match length: 623
__main__:chat:376 - response:
订单3若审核不通过，会进入“已取消”状态。原因可能是：
1. **订单金额超过5万元**，触发异常订单规则，需人工审核；
2. **审核未通过**，可能涉及以下原因：
   - 订单信息不完整或不符合平台政策（如商品类别、支付方式、收货地址等）；
   - 商家未通过审核（如商家资质、订单合规性等）；
   - 系统判定为高风险订单（如频繁下单、大额订单等）。

审核不通过后，订单将自动取消，用户可联系客服了解具体原因并申请复核。
__main__:insert:156 - ✅ Successfully inserted prefix cache, token length: 788
qwen_demo_utils:show_statistics:81 - Total Input: 647 tokens, Output 142 tokens, Prefill Cost 139.362 ms, Decode Cost 7575.316 ms
qwen_demo_utils:show_statistics:88 - Prefill Speed: 4642.59 tokens/s; Decode Speed: 18.61 tokens/s
qwen_demo_utils:show_statistics:91 - TTFT (Time to First Token): 149.219 ms
qwen_demo_utils:show_statistics:92 - TPOT (Time Per Output Token): 53.726 ms/token
qwen_demo_utils:show_statistics:95 - E2E Latency (End-to-End Latency): 8.716 seconds
qwen_demo_utils:show_statistics:96 - E2E TPS (End-to-End Tokens Per Second): 16.29 tokens/s
```

### 3.2 性能结果

```bash
# Normal Demo, Linux x86_64
Total Input: 30 tokens, Output 647 tokens, Prefill Cost 134.156 ms, Decode Cost 34682.281 ms
Prefill Speed: 223.62 tokens/s; Decode Speed: 18.63 tokens/s
TTFT (Time to First Token): 152.335 ms
TPOT (Time Per Output Token): 53.688 ms/token
E2E Latency (End-to-End Latency): 39.292 seconds
E2E TPS (End-to-End Tokens Per Second): 16.47 tokens/s
```

```bash
# Prefix Caching Demo, Linux x86_64
# No Cache
Total Input: 635 tokens, Output 14 tokens, Prefill Cost 425.193 ms, Decode Cost 700.975 ms
Prefill Speed: 1493.44 tokens/s; Decode Speed: 18.55 tokens/s
TTFT (Time to First Token): 462.604 ms
TPOT (Time Per Output Token): 53.921 ms/token
E2E Latency (End-to-End Latency): 1.261 seconds
E2E TPS (End-to-End Tokens Per Second): 11.10 tokens/s
# Use Cache
Total Input: 648 tokens, Output 176 tokens, Prefill Cost 139.806 ms, Decode Cost 9421.571 ms
Prefill Speed: 4634.99 tokens/s; Decode Speed: 18.57 tokens/s
TTFT (Time to First Token): 157.681 ms
TPOT (Time Per Output Token): 53.838 ms/token
E2E Latency (End-to-End Latency): 10.807 seconds
E2E TPS (End-to-End Tokens Per Second): 16.29 tokens/s
```

## 4 Qwen3 c++推理

模型库提供基于 C++的 Qwen3-8b 模型的样例，目前验证支持的平台信息如下：

-   Linux xh2 x86_64 (cmake 3.28.3 gcc/g++ 13.3.0)
-   Android xh2 arm64-v8a (cmake 3.28.3 clang/clang++ 19.0.1)
-   Windows11 xh2 x64 (cmake 4.1.1 Visual Studio 17 2022)

### 4.1 依赖三方库环境

Qwen3 的 C++推理样例依赖以下三方库：

1. tokenizer-cpp tokenizer 实现
2. eigen3 科学计算库

目前在上述平台的三方库已经编译好，已经集成到 get_model.py 中，并且在 CMakeLists.txt 进行了调用，构建时会自动进行三方库的链接。如需手动编译安装，可以参考 get_model.py 下载的 3rdparty.zip 压缩包中 README.MD 的详细步骤进行手动安装。

### 4.2 模型获取与模型准备

    目前验证支持平台，可以直接使用get_model.py获取模型和三方库，为了让c++方便读取embedding权重，需要对embedding权重的pt文件进行转换，转换示例代码如下：

```python
    # 权重转换
    embedding_path = "hmquant/quant_embedding.pt"
    if os.path.exists(embedding_path):
        # Load embedding weights
        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )['weight']

        # Convert bfloat16 to float16 if necessary
        if embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.float().half()

        # Save as binary file
        embedding_data = embedding_weight.cpu().numpy()
        embedding_data.tofile(embedding_path.replace(".pt", ".bin"))
```

    目前这部分转换代码已经集成到get_model.py中，会自动进行转换，并下载编译好的三方库。

### 4.3 程序编译和运行

编译和运行脚本支持的平台及说明如下：

-   run.sh 支持一键编译和运行如下平台的 C++样例：
    1. Linux_x86_64_xh2
-   run.bat 支持一键编译如下平台的 C++样例：
    1. win11_x64_xh2
    -   run.bat 会同时运行 win11 的 python 和 c++样例，运行 C++样例依赖 libhal_xh2a.dll 和 tcim_runtime_lite.dll。
-   build_ndk.sh 支持一键编译如下平台的 C++样例:
    1. Android_arm64-v8a_xh2
        ```bash
        # 1. 需要下载Android NDK，并设置NDK_PATH。
        # 2. 需要下载Android版本runtime-sdk，设置TCIM_RUNTIME_PATH为当前Android版本路径。
        # 示例如下：
        # export NDK_PATH={the_path_of_android}/android-ndk-r28c
        # export TCIM_RUNTIME_PATH={the_path_of_android}/houmo-tcim-runtime-xh2
        ./build_ndk.sh
        ```

### 4.4 C++样例相关文件说明

```
.
├── CMakeLists.txt              CMake配置文件
├── README.MD
├── run.sh                      编译和运行脚本，支持平台参考4.3节
├── run.bat                     win11编译和运行C++脚本,需要提前使用run.bat下载好模型
├── get_model.py                获取运行所需要的模型文件和三方库
├── include                     相关头文件
├── HmQwenInfer.cc              qwen3 c++推理类封装
├── Hmtokenizer.cc              qwen3 tokenizer功能实现封装
├── qwen3.cc                    qwen3 样例主函数实现
├── run.bat                     win11编译和运行脚本
└── build_ndk.sh                安卓编译脚本
```

### 4.5 C++演示结果

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

### 4.6 性能结果

```bash
# Linux x86_64
Total Input: 26 tokens, Output 1261 tokens, Prefill Cost 340.324 ms, Decode Cost 63976.289 ms
Prefill Speed: 76.40 tokens/s; Decode Speed: 19.69 tokens/s
TTFT (Time to First Token): 340.324 ms
TPOT (Time Per Output Token): 50.775 ms/token
E2E Latency (End-to-End Latency): 64.317 seconds
E2E TPS (End-to-End Tokens Per Second): 19.61 tokens/s
All Embedding takes 0.22 ms.
```

## 5 一键执行

可以使用 run 脚本一键执行查看结果

linux 执行脚本命令：

```bash
# 执行 Normal Demo
bash run.sh
# 执行 Prefix Caching Demo
bash run.sh -t prefix_caching
```

## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
