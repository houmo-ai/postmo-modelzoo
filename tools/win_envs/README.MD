# win11环境变量自动设置工具

本代码用于在使用后摩xh2芯片的win11设备上一键设置支持example样例运行必要的环境变量。

## 目录

- [win11快速开始](#1win11快速开始)
- [配置说明](#2配置说明)
- [免责声明](#3免责声明)


## 1.win11快速开始
1. 使用准备：
    - win11从后摩智能开发者社区[**https://developer.houmoai.com/**](https://developer.houmoai.com/)下载相关软件，参照社区文档中心完成固件烧录、驱动安装。
    - 下载example样例，安装必要的python运行环境。推荐使用miniconda3[**https://repo.anaconda.com/miniconda/**](https://repo.anaconda.com/miniconda/)进行安装，如果使用conda，可以通过conda create -n hmenv python == {version>= 3.9}创建运行环境。创建成功后，通过conda activate hmenv进入环境，安装必要的python依赖。
    - 参照当前目录下environment.yml中的参考版本自行安装。
    - 从后摩开发者社区下载Runtime SDK，解压后进入houmo_tcim_runtime，执行python setup.py install安装，安装好之后通过pip list查看houmo-tcim-runtime-xh2是否存在，确定安装是否完成。 

2. 自动设置环境变量。
    - 特别注意，设置完成后，需要新的终端方可生效环境变量。可以新开终端cd到对应样例目录，执行run.bat。也可以直接双击对应目录的run.bat运行。
    - 使用管理员权限运行cmd终端, cd到examples样例目录后，激活安装好houmo-tcim-runtime-xh2的python环境，执行env.bat进入环境变量自动设置流程。
    1. 支持在examples目录的使用方式如下： 
        | 指令        | 说明                                     |
        | -------------- | ---------------------------------------- |
        | env.bat --help/-h     | 帮助信息 |
        | env.bat --set         | 设置环境变量，已经设置的会自动跳过 |
        | env.bat --clear       | 清除脚本设置的环境变量 |
        | env.bat --reset       | 清除环境变量，并重新进入设置环境变量流程 |
    2. 执行env.bat --set进入自动设置环境变量流程，运行结果示例：
        ``` bat
        rem 首先会打印examples支持的示例情况及当前环境支持运行的样例信息
        --------------------------------------------------
          imodelzoo win11 support demos
        --------------------------------------------------
        √ qwen3           : python, c++
        √ resnet50        : python, c++
        √ yolov5s         : python, c++
        --------------------------------------------------
        --------------------------------------------------
                current env support python demos
        --------------------------------------------------
        qwen3           ×
        resnet50        ×
        yolov5s         ×
        --------------------------------------------------
        --------------------------------------------------
                current env support cpp demos
        --------------------------------------------------
        qwen3           ×
        resnet50        ×
        yolov5s         ×
        --------------------------------------------------
        rem 然后会自动搜索环境变量，部分环境变量可能需要用户手动设置，0.6.0之前的版本HOUMO_SDK_PATH需要用户自行设置，0.6.0及之后的版本如果正确安装之后会自动搜索，搜索失败会提示用户手动设置。[Required]的环境变量是必要的，搜索不到用户必须设置。[Optional]的环境变量如果不设置可以回车跳过，但是会影响部分样例的运行。
        ✅ Found tcim_package_path at path:
        D:\miniconda\envs\imodelzoo\Lib\site-packages\tcim_lite
        Manual Set CMAKE_PATH [Optional]  abspath:
        Manual Set OPENCV_PATH [Optional]  abspath:
        rem 设置完成后，会再次打印当前环境变量支持的样例情况，用户可以根据设置后的支持情况运行支持的样例。
        --------------------------------------------------
                imodelzoo win11 support demos
        --------------------------------------------------
        √ qwen3           : python, c++
        √ resnet50        : python, c++
        √ yolov5s         : python, c++
        --------------------------------------------------
        --------------------------------------------------
                current env support python demos
        --------------------------------------------------
        qwen3           √
        resnet50        √
        yolov5s         √
        --------------------------------------------------
        --------------------------------------------------
                current env support cpp demos
        --------------------------------------------------
        qwen3           ×
        resnet50        ×
        yolov5s         ×
        --------------------------------------------------
        rem 如果用户还需要设置环境变量支持其他样例，可以通过env.bat --reset进行重新设置，此步骤对于设置过的变量也会重设，功能等价于执行env.bat --clear + env.bat。也可以通过env.bat --clear + env.bat清楚后再设置。
        ```
    3. 执行env.bat --clear可以清除设置的环境变量。
    4. 执行env.bat --reset会清除环境变量后重新进入设置流程，功能等价于env.bat --clear + env.bat。
    5. 执行env.bat --help/-h打印帮助信息。

3. 初始环境变量会备份为initial_env_backup.json请不要修改和删除此文件，否则应用env.bat前的环境变量可能无法恢复。

4. 首次运行env.bat需要确保终端中python指令可用，且已经按照使用准备完成了需要的python库文件。设置完环境变量一定要重新启动终端才会生效，env.bat需要管理员权限运行。

## 2.配置说明

### 2.1 env.json字段说明
可以参考tools/win_envs目录下的template.json，comment字段代表注释信息。详细信息如下：

    ``` json
    {
        "__comment__": "支持的样例目录",
        "example_dir": "apis/inferences",
        "__comment__1": "支持的平台",
        "support_target": "xh2",
        "__comment__2": "支持的样例文件夹",
        "support_demos": {
            "qwen3": [
                "python",
                "c++"
            ],
            "resnet50": [
                "python",
                "c++"
            ],
            "yolov5s": [
                "python",
                "c++"
            ]
        },
        "__comment__3": "env.bat将会设置的所有环境变量",
        "all_environments": {
            "HOUMO_MODELZOO_URL": "",
            "HDPL_PLATFORM": "",
            "HOUMO_TARGET": "",
            "TCIM_BACKEND": "",
            "HOUMO_EXAMPLES_PATH": "",
            "PYTHON_DIR": "",
            "TCIM_RUNTIME_PATH": "",
            "CMAKE_PATH": "",
            "HOUMO_PATH": "",
            "HOUMO_SDK_PATH": "",
            "OPENCV_PATH": "",
            "HOUMO_VERSION": "",
            "PATH": ""
        },
        "__comment__4": "get_model脚本需要的环境变量",
        "get_model_environments": [
            "PYTHON_DIR",
            "HOUMO_MODELZOO_URL",
            "HOUMO_TARGET",
            "HOUMO_VERSION",
            "HDPL_PLATFORM",
            "HOUMO_EXAMPLES_PATH"
        ],
        "__comment__5": "python样例需要的环境变量 = get_model脚本需要的环境变量 + 当前的设置",
        "py_example_environments": {
            "qwen3": [
                "TCIM_BACKEND",
                "TCIM_RUNTIME_PATH",
                "HOUMO_SDK_PATH"
            ],
            "resnet50": [
                "TCIM_BACKEND",
                "TCIM_RUNTIME_PATH",
                "HOUMO_SDK_PATH"
            ],
            "yolov5s": [
                "TCIM_BACKEND",
                "TCIM_RUNTIME_PATH",
                "HOUMO_SDK_PATH"
            ]
        },
        "__comment__6": "cpp样例需要的环境变量 = python样例需要的环境变量 + 当前的设置",
        "cpp_example_environments": {
            "qwen3": [
                "CMAKE_PATH",
                "HOUMO_PATH"
            ],
            "resnet50": [
                "CMAKE_PATH",
                "OPENCV_PATH"
            ],
            "yolov5s": [
                "CMAKE_PATH",
                "OPENCV_PATH"
            ]
        }
    }
    ```

### 2.2 添加新的支持
当前支持添加新的样例支持及环境依赖，
1. 添加样例：
    - 需要修改env.json中的support_demos、py_example_environments、 cpp_example_environments字段
    - get_model_environments字段的环境变量已经满足了必要的下载条件，请不要在get_model.py中引入额外的环境变量，否则需要自行修改env.json和set_environs.py添加支持。
- 添加环境依赖：
    - 目前已经设置了必要的环境变量，可以在json中的all_environments新增新的环境变量，默认是[Optional]，不会进行强制设置要求。
    - 如果需要添加的环境变量是[Required]，则需要在set_environs.py中WinEnvironsGenerater类的autoSetEnvirons中添加支持和校验，可以参考self.nullEnvManualSet("XXX_PATH", need=True)。

## 3.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
