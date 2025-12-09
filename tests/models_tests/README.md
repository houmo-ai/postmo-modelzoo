# 模型测试

当前目录下的所有 python 脚本，均用于测试对外发布模型。

## 1. 环境依赖

测试依赖 pytest python 库，可使用 `pip3 install pytest` 安装。

当前仅支持 linux 系统。

## 2. 测试说明

### 2.1 测试类型

提供下述七种测试类型用于测试模型用例的不同功能，测试时可自定义测试模型用例的哪些功能。

#### 2.1.1 get_model

测试模型用例中的下载模型及相关资源功能，即测试`get_model.py`脚本。详细测试步骤：

1. 根据模型 json 配置文件中的`get_model_params`配置，生成`get_model.py`脚本的所有测试参数。
2. 依次使用测试参数执行`get_model.py`脚本。
3. 校验`get_model.py`脚本是否成功结束 (不校验下载结果的正确性和完整性)。

#### 2.1.2 quant

主要测试模型量化功能，详细测试步骤：

1. 获取原始模型
2. 根据模型 json 配置文件中的`hmquant_params`配置或`quant_params`配置，生成量化测试的所有测试参数。其中，`hmquant_params`配置用于 hmatc，`quant_params`配置用于`ptq.py`脚本。
3. 对原始模型进行量化得到量化模型 (优先使用`hmatc quant`，如模型不支持 hmatc 则使用`ptq.py`)
4. 校验量化过程中是否存在`fail`字样，不存在且量化正常结束则认为量化测试通过。

#### 2.1.3 compile

主要测试模型编译功能，详细测试步骤：

1. 获取量化模型
2. 根据模型 json 配置文件中的`hmbuild_params`配置或`compile_params`配置，生成编译测试的所有测试参数。其中，`hmbuild_params`配置用于 hmatc，`compile_params`配置用于`build.py`脚本。
3. 将量化模型编译为可在后摩设备执行的编译模型 (优先使用`hmatc build`，如模型不支持 hmatc 则使用`build.py`)
4. 对编译模型进行 golden 结果校验，编译正常执行且 golden 结果校验通过，认为编译测试通过。

#### 2.1.4 demo

主要测试模型推理功能，详细测试步骤：

1. 获取编译后模型，如不支持则获取量化后模型进行编译
2. 根据模型 json 配置文件中的`hmdemo_params`配置或`demo_params`配置，生成模型推理测试的所有测试参数。其中，`hmdemo_params`配置用于 hmatc，`demo_params`配置用于`demo.py`脚本。
3. 执行模型推理 (优先使用`hmatc demo`，如模型不支持 hmatc 则使用`demo.py`)
4. 校验模型推理过程是否正常结束，正常结束则认为测试通过。

#### 2.1.5 compare

(模型需支持 hmatc) 主要测试模型推理结果正确性，需提供输入数据用于比较模型结果是否正确，详细测试步骤：

1. 获取量化后模型，通过 hmatc 进行编译
2. 根据模型 json 配置文件中的`hmcompare_params`配置，生成模型结果正确性测试的所有测试参数。其中，`hmcompare_params`配置用于 hmatc。
3. 通过 hmatc 的`hmatc compare`命令，执行模型推理并验证结果正确性。
4. 校验步骤 3 是否正常结束，正常结束则认为测试通过。

#### 2.1.6 perf

主要测试模型性能并校验是否有显著下降(较 benchmark 下降超过 5%)，详细测试步骤：

1. 获取编译后模型，如不支持则获取量化后模型进行编译
2. 根据模型 json 配置文件中的`hmperf_params`配置或`perf_params`配置，生成模型性能测试的所有测试参数。其中，`hmperf_params`配置用于 hmatc，`perf_params`配置用于`demo.py`脚本 (当前 llm 模型用例中均通过`demo.py`脚本执行并统计性能数据)。
3. 执行模型性能测试 (优先使用`hmatc perf`，如模型不支持 hmatc 则使用`demo.py`)
4. 获取模型性能测试的性能结果，读取模型配置文件中的`perf_metrics`参数获取 benchmark 性能数据，比较性能测试结果是否存在显著下降，若性能无显著下降且性能测试正式结束则认为性能测试通过。

#### 2.1.7 eval

(模型需支持 hmatc) 主要测试模型精度并校验是否有显著下降(基于相同数据集，比较 hm 模型和 onnx 模型的 map), 需提供数据集用于精度测试，详细测试步骤：

1. 获取编译后模型，如不支持则获取量化后模型进行编译
2. 根据模型 json 配置文件中的`hmeval_params`配置，生成模型精度测试的所有测试参数。其中，`hmeval_params`配置用于 hmatc。
3. 通过 hmatc 的`hmatc eval`命令，执行模型精度测试。
4. 获取模型精度测试的 map 结果，读取模型配置文件中的`eval_threshold`参数获取精度阈值，比较`hm map >= (onnx map * threshold)`，若精度无显著下降且精度测试正式结束则认为精度测试通过。

### 2.2 测试文件说明

- `model_configs`: 文件夹中为所有模型测试配置文件。文件命名规则：`"model_cfg_" + 模型名 + ".json"`。
- `update_test_py.py`: 根据 model*configs 文件夹下的模型配置文件，自动更新 python 测试用例脚本(仅增加用例)。脚本中将自动转换模型名称中的"-"和"."，"-"转为下划线"*"，"."转为"dot"。
- `test_<test_type>_models.py`: python 测试用例脚本，用于 pytest 执行测例。根据章节 2.1 的 7 种测试类型，共计有 7 个 python 测试脚本。
  - <test_type>: get, quant, compile, demo, compare, perf, eval。
- `test_models_utils.py`: 包含了 7 种测试类型的测试逻辑代码，如现有测试逻辑无法覆盖新增模型或场景，则需修改此文件。
- `conftest.py`: 在 pytest 框架中，是一个用于存放共享测试配置和 fixture 函数的特殊文件。如需详细了解可参考：https://pytest.cn/en/stable/getting-started.html
  - 本文件中定义了支持的 pytest markers, 当前已支持的 markers: get_model, quant, compile, demo, compare, eval, perf。

### 2.3 测试配置文件说明

测试配置文件为 tests/models_tests/model_configs 文件夹中的 json 文件，其中每个文件包含了：模型的测试所需的基础信息，模型支持的测试类型，模型支持的平台，模型支持的后摩设备，以及用于不同测试类型的测试参数。

提供了 CV 模型和 LLM 模型模板配置文件，便于新增模型的时候进行修改。其中，测试仓库包含两个默认路径：

- `cached_models`: `/develop02/wanyu.li/modelzoo`，主要用于缓存原始模型，避免重复下载。
- `cached_results`: `/data02/services/imodelzoo/tests/model_results_{HOUMO_TARGET}`，用于缓存单次测试模型量化和编译结果，避免测试用例之间重复量化编译，同时用于支持分阶段测试（如: 在 10.64.35.39 执行量化编译，在 10.64.35.71 执行 xh2 推理）。

注意：

- LLM 模型的原始模型下载需要手工提前下载到`cached_models`路径下，不要在测试用例中再下载，容易断开连接无法下载成功导致相关测例无法执行成功，且严重拉长整体测试时间。
- 当前 Device 相关的配置，仅支持配置为 1，不支持多 Device 场景。

模型模板配置文件说明：

- CV 模板配置文件：`tests/models_tests/model_configs/model_cfg_template_cv.json`，配置项说明。

```JSON
// (必需)标识模型是否废弃, true表示已废弃，false表示未废弃
"obsolete": false,
// (必需)模型用例路径(相对路径, 根目录为: imodelzoo)
"model_dir": "models/<model_type>/<model_name>",
// (必需)模型用例支持的系统平台: x86_64, aarch64
"support_platform": ["x86_64"],
// (必需)模型用例支持的后摩Backend: xh1, xh2
"support_backend": ["xh1", "xh2"],
// (必需)get_model.py脚本中默认值支持的hmm模型core数量: 1, 2, 4 ...
"support_core_num": {
    "xh1": [1],
    "xh2": [1]
},
// (必需)模型用例支持的测试类型:
// "get_model": (必需)模型用例中包含get_model.py脚本
// "quant": 符合下述任一情况即认为支持：
//     1) 模型用例中包含ptq.py且可基于后摩设备量化。
//     2) 模型用例支持hmatc的quant命令行: hmatc quant。
// "compile": 符合下述任一情况即认为支持：
//     1) 模型用例中包含build.py可在linux x86_64成功编译量化后模型。
//     2) 模型用例支持hmatc的build命令行: hmatc build。
// "demo": 符合下述任一情况即认为支持：
//     1) 模型用例中包含demo.py可成功执行推理。
//     2) 模型用例支持hmatc的demo命令行: hmatc demo。
// "compare": 模型用例支持hmatc的copmare命令行。
// "perf": 符合下述任一情况即认为支持：
//     1) 模型用例中包含demo.py且计算并打印了性能数据(prefill, decode, end2end)。
//     2) 模型用例支持hmatc的perf命令行: hmatc perf。
// "eval": 模型用例支持hmatc的eval命令行。
"support_flow": {
    "xh1": ["get_model", "quant", "compile", "demo", "compare", "perf", "eval"],
    "xh2": ["get_model", "quant", "compile", "demo", "compare", "perf", "eval"]
},
// (必需)模型用例支持的hmatc功能:
// "hmquant": 模型用例支持hmatc量化: hmatc quant
// "hmbuild": 模型用例支持hmatc编译: hmatc build
// "hmdemo": 模型用例支持hmatc推理: hmatc demo
// "hmcompare": 模型用例支持hmatc比较推理结果: hmatc compare
// "hmeval": 模型用例支持hmatc评估精度: hmatc eval
// "hmperf": 模型用例支持hmatc评估性能: hmatc perf
"support_hmatc": {
    "xh1": ["hmquant", "hmbuild", "hmdemo", "hmeval", "hmperf", "hmcompare"],
    "xh2": ["hmquant", "hmbuild", "hmdemo", "hmeval", "hmperf", "hmcompare"]
},
// (可选)模型性能benchmark，用于性能测试。如果support_flow中支持perf，则此配置为必需。
"perf_metrics": {
    "xh1":{
        "x86_64": 1733.207
    },
    "xh2":{
        "x86_64": 484.383
    }
},
// (可选)模型精度阈值，用于精度测试。如果support_flow中支持eval，则此配置为必需。其中key值为精度指标的名称。
"eval_threshold": {
    "top1_acc": 0.95
},
// (必需)get_model测试中，get_model.py脚本支持的入参及对应待测的参数，待测参数需要一一对应，使用默认值则配置为null。
"get_model_params": {
    "xh1": {
        "type": ["raw", "quant", "hmm"],
        "model_dir": ["cached_models", "cached_models", "cached_models"]
    },
    "xh2": {
        "type": ["raw", "quant", "hmm"],
        "model_dir": ["cached_models", "cached_models", "cached_models"]
    }
},
// (可选)quant测试中，如通过hmatc quant命令行量化模型，则此配置为必需。
// hmatc quant支持的入参及对应待测的参数，待测参数需要一一对应，使用默认值则配置为null。
"hmquant_params": {
    "params": {
        "required": {
            "config": ["./config.yml"]
        },
        "optional": {
            "target": [null],
            "result_path": [null]
        }
    }
},
// (可选)compile测试中，如通过hmatc build命令行编译模型，则此配置为必需。
// hmatc build支持的入参及对应待测的参数，待测参数需要一一对应，使用默认值则配置为null。
"hmbuild_params": {
    "xh1": {
        "required": {
            "config": ["./config.yml", "./config.yml", "./config.yml", "./config.yml"]
        },
        "optional": {
            "result_path": [null, null, null, null],
            "batch": [null, "1", "2", "4"],
            "ncore": [null, "1", "2", "4"],
            "opt_level": [null, "0", "1", "2"],
            "roi_num": [null, null, null, null]
        }
    },
    "xh2": {
        "required": {
            "config": ["./config.yml", "./config.yml", "./config.yml"]
        },
        "optional": {
            "result_path": [null, null, null],
            "batch": [null, "1", "1"],
            "ncore": [null, "1", "2"],
            "opt_level": ["0", "1", "2"],
            "roi_num": [null, null, null]
        }
    }
},
// (可选)demo测试中，如通过hmatc demo命令行执行推理，则此配置为必需。
// hmatc demo支持的入参及对应待测的参数，待测参数需要一一对应，使用默认值则配置为null。
"hmdemo_params": {
    "params": {
        "required": {
            "config": ["./config.yml"]
        },
        "optional": {
            "target": [null],
            "result_path": [null],
            "onnx": [null]
        }
    }
},
// (可选)perf测试中，如通过hmatc perf命令行评估模型性能，则此配置为必需。
// hmatc perf支持的入参及对应待测的参数，待测参数需要一一对应，使用默认值则配置为null。
"hmperf_params": {
    "params": {
        "required": {
            "config": ["./config.yml", "./config.yml", "./config.yml"],
            "warmup": ["1", "10", "10"],
            "sample": ["1", "1000", "1000"]
        },
        "optional": {
            "target": [null, null, null],
            "result_path": [null, null, null],
            "loop_num": [null, null, null],
            "thread": ["1", "4", "8"],
            "device": [null, null, null]
        }
    }
},
// (可选)如支持eval测试类型，则此配置为必需。
// 用于eval测试中, hmatc eval支持的入参及对应待测的参数，待测参数需要一一对应，使用默认值则配置为null。
"hmeval_params": {
    "params": {
        "required": {
            "config": ["./config.yml"]
        },
        "optional": {
            "target": [null],
            "result_path": [null],
            "onnx": [null]
        }
    }
},
// (可选)如支持compare测试类型，则此配置为必需。
// 用于compare测试中, hmatc compare支持的入参及对应待测的参数，待测参数需要一一对应，使用默认值则配置为null。
"hmcompare_params": {
    "params": {
        "required": {
            "config": ["./config.yml"],
            "data_path": ["./imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG"]
        },
        "optional": {
            "target": [null],
            "result_path": [null]
        }
    }
}
```

- LLM 模板配置文件：`tests/models_tests/model_configs/model_cfg_template_llm.json`，配置项说明。

```JSON
// (必需)标识模型是否废弃, true表示已废弃，false表示未废弃
"obsolete": false,
// (必需)模型用例路径(相对路径, 根目录为: imodelzoo)
"model_dir": "models/<model_type>/<model_name>",
// (必需)模型用例支持的系统平台: x86_64, aarch64
"support_platform": ["x86_64"],
// (必需)模型用例支持的后摩Backend: xh1, xh2
"support_backend": ["xh1", "xh2"],
// (必需)get_model.py脚本中默认值支持的hmm模型core数量: 1, 2, 4 ...
"support_core_num": {
    "xh1": [4],
    "xh2": [2]
},
// (必需)模型用例支持的测试类型:
// "get_model": (必需)模型用例中包含get_model.py脚本
// "quant": 符合下述任一情况即认为支持：
//     1) 模型用例中包含ptq.py且可基于后摩设备量化。
// "compile": 符合下述任一情况即认为支持：
//     1) 模型用例中包含build.py可在linux x86_64成功编译量化后模型。
// "demo": 符合下述任一情况即认为支持：
//     1) 模型用例中包含demo.py可成功执行推理。
// "perf": 符合下述任一情况即认为支持：
//     1) 模型用例中包含demo.py且计算并打印了性能数据(prefill, decode, end2end)。
"support_flow": {
    "xh1": ["get_model", "compile", "demo", "perf"],
    "xh2": ["get_model", "quant", "compile", "demo", "perf"]
},
// hmatc工具当前均不支持LLM模型，配置为空。
"support_hmatc": null,
// (可选)模型性能benchmark，用于性能测试。如果support_flow中支持perf，则此配置为必需
"perf_metrics": {
    "xh1": {
        "x86_64": {
            "prefill": 1.05,
            "decode": 7.92,
            "end2end": 7.88
        }
    },
    "xh2": {
        "x86_64": {
            "prefill": 40.0,
            "decode": 15.93,
            "end2end": 15.69
        }
    }
},
// LLM模型当前暂不支持精度评测，配置为空。
"eval_threshold": null,
// (必需)get_model测试中，get_model.py脚本支持的入参及对应待测的参数，待测参数需要一一对应，使用默认值则配置为null。
"get_model_params": {
    "xh1": {
        "type": ["raw", "quant", "hmm"],
        "quant_model_dir": [null, "cached_models/hmquant_xh1", null],
        "build_model_dir": [null, null, "cached_models/hmm_xh1"],
        "model_dir": ["cached_models", "cached_models", "cached_models"]
    },
    "xh2": {
        "type": ["raw", "quant", "hmm"],
        "quant_model_dir": [null, "cached_models/hmquant_xh2", null],
        "build_model_dir": [null, null, "cached_models/hmm_xh2"],
        "model_dir": ["cached_models", "cached_models", "cached_models"]
    }
},
// (可选)quant测试中，如通过ptq.py脚本量化模型，则此配置为必需。
// ptq.py脚本支持的入参及对应待测的参数，其中需要预先下载原始模型至cached_models下，再将cached_models/<raw_model_folder> 路径配置到测试文件中。
"quant_params": {
    "xh1": null,
    "xh2": {
        "model": ["cached_models/<raw_model_folder>", "cached_models/<raw_model_folder>", "cached_models/<raw_model_folder>"],
        "out-dir": ["cached_results/hmquant_xh2_2k", "cached_results/hmquant_xh2_4k", "cached_results/hmquant_xh2_8k"],
        "context-length": ["2048", "4096", "8192"],
        "input-sequence-length": [null, null, null],
        "calibration-dataset": [null, null, null]
    }
},
// (可选)compile测试中，如通过build.py脚本编译模型，则此配置为必需。
// build.py脚本支持的入参及对应待测的参数。如果参数中配置了与量化结果相同的文件夹，则会直接使用该文件夹的结果，不会重新量化。
"compile_params": {
    "xh1": {
        "model_dir": ["cached_results/hmquant_xh1"],
        "model_name": [null],
        "ncore": [null],
        "stage": ["build"],
        "output_dir": ["cached_results/hmm_xh1"]
    },
    "xh2": {
        "model_dir": ["cached_results/hmquant_xh2_2k", "cached_results/hmquant_xh2_2k"],
        "batch": [null, null],
        "ncore": [null, null],
        "ndevice": [null, null],
        "stage": ["build", "build"],
        "context_length": ["2048", "8192"],
        "output_dir": ["cached_results/hmm_xh2_2k", "cached_results/hmm_xh2_8k"]
    }
},
// (可选)demo测试中，如通过demo.py脚本执行推理，则此配置为必需。
// demo.py脚本支持的入参及对应待测的参数。如果参数中配置了与量化/编译结果相同的文件夹，则会直接使用该文件夹的结果，不会重新量化/编译。
"demo_params": {
    "xh1": {
        "tokenizer_dir": ["cached_models/<raw_model_folder>"],
        "embedding_path": ["cached_results/hmm_xh1/hmquant/quant_embedding.pt"],
        "prefill_path": ["cached_results/hmm_xh1/<model_name>_prefill.hmm"],
        "decode_path": ["cached_results/hmm_xh1/<model_name>_decode.hmm"],
        "ndevice": ["1"]
    },
    "xh2": {
        "tokenizer_dir": ["cached_models/<raw_model_folder>", "cached_models/<raw_model_folder>"],
        "embedding_path": ["cached_results/hmm_xh2_2k/hmquant/quant_embedding.pt", "cached_results/hmm_xh2_8k/hmquant/quant_embedding.pt"],
        "prefill_path": ["cached_results/hmm_xh2_2k/<model_name>_prefill.hmm", "cached_results/hmm_xh2_8k/<model_name>_prefill.hmm"],
        "decode_path": ["cached_results/hmm_xh2_2k/<model_name>_decode.hmm", "cached_results/hmm_xh2_8k/<model_name>_decode.hmm"],
        "ndevice": ["1", "1"]
    }
},
// (可选)perf测试中，如通过demo.py脚本评估模型性能，则此配置为必需，无需修改配置值。
"perf_params": "demo"，
```

### 2.4 测试方法

```bash
cd imodelzoo/tests

# 执行当前文件夹tests下所有测试用例
pytest -s -v
# 执行models_tests文件夹下所有测试用例
pytest -s -v models_tests/
# 执行当前文件夹下yolov5s模型所有测试用例
pytest -s -v models_tests/ -m "yolov5s"
# 执行当前文件夹下qwen2.5-vl模型所有测试用例
pytest -s -v models_tests/ -m "qwen2dot5_vl"
# 执行当前文件夹下llm模型所有测试用例,llm模型指模型用例在models/llm文件夹下的模型
pytest -s -v models_tests/ -k "_llm_"
# 执行当前文件夹下所有性能测试用例(不支持性能测试的模型会自动跳过)
pytest -s -v models_tests/ -m "perf"
# 执行当前文件夹下所有精度测试用例(不支持精度测试的模型会自动跳过)
pytest -s -v models_tests/ -m "eval"
# 执行当前文件夹下resnet50模型的性能测试和精度测试用例
pytest -s -v models_tests/ -m resnet50 -m "perf or eval"
```

上述示例的测试方法命令中，`pytest -s -v`可作为固定命令前缀，`-k`和`-m`均为 pytest 框架中提供过滤执行测例的关键词，详细说明可参考：

- https://pytest.cn/en/stable/example/markers.html
- https://zhuanlan.zhihu.com/p/629592323

## 3. 新增模型

本章节介绍了新增模型加入测试的方法。

下文示例均假设新增模型的名称为: template-v1.0。

### 3.1 创建模型测试配置文件

1. 在文件夹`tests/models_tests/model_configs/`中新增模型配置文件`model_cfg_template-v1.0.json`。
2. 从模板配置文件`tests/models_tests/model_configs/model_cfg_template_<cv/llm>.json`复制其内容至新增模型的配置文件中，按模型信息进行修改内容。配置文件各配置项说明可参考章节 2.3。

### 3.2 应用模型测试配置文件

```bash
cd imodelzoo/tests/models_tests
python3 update_test_py.py
```

执行结果如下：

```bash
Detect new model template-v1.0-->template_v1dot0, support flow ['get_model', 'perf'].
Add test_xxx_template_v1dot0_get_model into get_model python file
Detect new model template-v1.0-->template_v1dot0, support flow ['get_model', 'perf'].
Add test_xxx_template_v1dot0_perf into perf python file
```

说明：脚本打印的 log 中包含了模型原始名称`template-v1.0`和脚本转换后的模型名称`template_v1dot0`，其中转换后的模型名称同时会添加到 `./model_names.txt` 文件中。


### 3.3 执行模型测试

```bash
cd imodelzoo/tests
# 假设新增模型的名称为: template-v1.0，转换后模型名称为: template_v1dot0
pytest -s -v models_tests/ -m "template_v1dot0"
```
