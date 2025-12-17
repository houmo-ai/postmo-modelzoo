# win11环境变量自动设置工具

本代码用于在使用后摩xh2芯片的win11设备上一键设置支持example样例运行必要的环境变量。

## 目录

- [win11快速开始](#1win11快速开始)
- [配置说明](#2配置说明)
- [models目录样例支持说明(仅供参考)](#3models目录样例支持说明(仅供参考))
- [免责声明](#4免责声明)


## 1.win11快速开始
**此README仅支持python≥3.9，其他版本未作验证，建议使用python≥3.9。**
1. 使用准备与使用前提：
    - win11从后摩智能开发者社区[**https://developer.houmoai.com/**](https://developer.houmoai.com/)下载相关软件，参照社区文档中心完成固件烧录、驱动安装。
    - **驱动安装后需要确认windows的用户环境变量是否有HOUMO_SDK_PATH，该工具依赖此环境变量否则无法运行。**
    1. 直接使用exe安装包安装驱动，HOUMO_SDK_PATH会自动添加到用户环境变量，只需要用户确认是否正确即可。
    2. 使用zip包安装时，**解压时解压到zip包同名目录，否则工具可能无法获取到驱动版本号**。此外，用户需要自行添加HOUMO_SDK_PATH环境变量，路径示例D:\houmo_drv_xh2_v0.6.0_windows11_x86_64\houmo_drv_xh2。
    - 下载example样例，安装必要的python运行环境，**建议使用python≥3.9版本**。
    1. 安装后需要确认cmd终端python的版本号是否符合要求，有些用户存在多个版本的python需要确认默认的版本是否符合要求，如需查看windows的python位置，可以终端输入where python查看，建议从环境变量删除不需要的python版本。
    2. **部分用户终端python指令无效，但是py指令有效，具体处理方式为**：在cmd终端输入where py，将输出的路径从环境变量PATH删除。或者手动到对应目录删除py.exe文件。处理完成后重新开新的终端，确保py指令失效，python指令有效。

2. win11 python环境的安装：
    1. 方式一：
        打开windows自带的Microsoft Store，搜索python后，点击获取会自动下载安装。
    2. 方式二：
        - 在python官网[**https://www.python.org/downloads/windows/**](https://www.python.org/downloads/windows/) 下载python≥3.9安装包。
        - 下载完成后，双击下载的安装文件启动安装程序。
        - 在安装界面中，勾选 Add python.exe to PATH（非常重要，避免后续手动配置环境变量）。
        - 点击 Customize installation（可选），根据需要自定义安装路径或组件，建议默认即可。
        - 如果需要更改安装路径，点击 Browse 并选择新路径（如 D:\Python）。
        - 确认所有选项后，点击 Install 开始安装。
        - 安装完成后，点击 Disable path length limit（建议禁用路径长度限制以避免潜在问题），然后点击 Close 关闭安装程序。
        - 安装后，win+R打开终端，输入python --version确认python≥3.9。

3. examples环境安装详细步骤：
    - 开始菜单搜索cmd，选择命令提示符，右键以管理员身份运行，执行cd /d {path/of/examples}
    - 执行 pip install -r tools/win_envs/requirements.txt安装相关依赖。
    - 在[**https://visualstudio.microsoft.com/visual-cpp-build-tools/**](https://visualstudio.microsoft.com/visual-cpp-build-tools/)下载vs生成工具，后续C++样例编译需要使用。
    - 运行vs生成工具，勾选使用c++的桌面开发后，点击安装。
    - 执行cd /d {path/of/Runtime SDK}/切换到Runtime SDK压缩包下载的目录。
    - 执行pip install {Runtime SDK.tar.gz}进行houmo-tcim-runtime-xh2的安装。 
    - 安装结束后，执行pip list查看houmo-tcim-runtime-xh2是否安装成功以及版本是否正确。
    - cd /d {path/of/examples}切换回examples的路径。

4. 以下步骤为可选，主要是为了支持C/C++样例。
    - 在[**https://cmake.org/**](https://cmake.org/)下载cmake并运行安装。
    - 在opencv官网[**https://opencv.org/**](https://opencv.org/)选择Library->Release下载编译好的opencv并运行解压安装。

5. 自动设置环境变量。
    1. 支持在examples目录的使用方式如下： 
        | 指令        | 说明                                     |
        | -------------- | ---------------------------------------- |
        | env.bat --help/-h     | 帮助信息 |
        | env.bat --set         | 设置环境变量，已经设置的会自动跳过 |
        | env.bat --clear       | 清除工具设置的环境变量 |
        | env.bat --reset       | 对于安装过其他版本的环境，使用reset清除环境变量后重新设置 |
    2. 执行env.bat --set进入自动设置环境变量流程，[Required]的环境变量是运行样例必须的，[Optional]用户可以自行选择是否输入，不输入会有部分样例无法运行。目前部分样例需要给到的路径示例如下：
    ``` bat
    OPENCV_PATH [Optional]: D:\Program Files\Opencv\opencv\build
    CMAKE_PATH: D:\\Program Files\\CMake\\bin
    ```
    运行结果示例：
    ``` bat
    rem 首先会打印examples支持的示例情况及当前环境支持运行的样例信息
    ===============Set Win Envs===============
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
    C:\Users\liuzhenyao\AppData\Local\Programs\Python\Python311\Lib\site-packages\tcim_lite
    Manual Set HOUMO_SDK_PATH [Required]  abspath:D:\0.5.0\houmo_drv_xh2_v0.5.0_windows11_x86_64\houmo_drv_xh2
    Manual Set OPENCV_PATH [Optional]  abspath:D:\Program Files\Opencv\opencv\build
    Set HOUMO_MODELZOO_URL to http://10.10.1.53:8082/artifactory/toolchain/release ('user-level')
    Set HDPL_PLATFORM to ASIC ('user-level')
    Set HOUMO_TARGET to xh2 ('user-level')
    Set TCIM_BACKEND to Xh2HalBackend ('user-level')
    Set HOUMO_EXAMPLES_PATH to D:\win11\imodelzoo ('user-level')
    Set PYTHON_DIR to C:\Users\liuzhenyao\AppData\Local\Programs\Python\Python311 ('user-level')
    Set TCIM_RUNTIME_PATH to C:\Users\liuzhenyao\AppData\Local\Programs\Python\Python311\Lib\site-packages\tcim_lite ('user-level')
    Set CMAKE_PATH to D:\Program Files\CMake\bin ('user-level')
    Set HOUMO_PATH to C:\Users\liuzhenyao\AppData\Local\Programs\Python\Python311\Lib\site-packages\tcim_lite ('user-level')
    Set HOUMO_SDK_PATH to D:\0.5.0\houmo_drv_xh2_v0.5.0_windows11_x86_64\houmo_drv_xh2 ('user-level')
    Set OPENCV_PATH to D:\Program Files\Opencv\opencv\build ('user-level')
    Set HOUMO_VERSION to v0.5.0 ('user-level')
    Added to 'user-level' PATH: C:\Users\liuzhenyao\AppData\Local\Programs\Python\Python311\Lib\site-packages\tcim_lite\bin;D:\0.5.0\houmo_drv_xh2_v0.5.0_windows11_x86_64\houmo_drv_xh2\hal/lib;
    Note: New PATH takes effect in new processes. Restart window to apply.
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
    qwen3           √
    resnet50        √
    yolov5s         √
    --------------------------------------------------
    ========Set all envs Finished, please reopen cmd window=========
    rem 如果用户还需要设置环境变量支持其他样例，可以通过env.bat --reset进行重新设置，此步骤对于设置过的变量也会重设，功能等价于执行env.bat --clear + env.bat --set。也可以通过env.bat --clear + env.bat --set清楚后再设置。
    ```
    3. 执行env.bat --clear可以清除设置的环境变量。
    4. 执行env.bat --reset会清除旧版本的环境变量。
    5. 执行env.bat --help/-h打印帮助信息。

6. 初始环境变量会备份为initial_env_backup.json请不要修改和删除此文件，否则应用env.bat前的环境变量可能无法恢复。

7. 首次运行env.bat需要确保终端中python指令可用，且已经按照使用准备完成了需要的python库文件。设置完环境变量一定要重新启动终端才会生效，env.bat需要管理员权限运行。

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
            "resnet50_multistreams": [
                "python",
                "c++"
            ],
            "resnet50_pipeline": [
                "python",
                "c++"
            ],
            "yolov5s": [
                "python",
                "c++"
            ]
        },
        "__comment__3": "是否使用内网下载链接",
        "develop_url": false,
        "__comment__4": "env.bat将会设置的所有环境变量",
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
        "__comment__5": "get_model脚本需要的环境变量",
        "get_model_environments": [
            "PYTHON_DIR",
            "HOUMO_MODELZOO_URL",
            "HOUMO_TARGET",
            "HOUMO_VERSION",
            "HDPL_PLATFORM",
            "HOUMO_EXAMPLES_PATH"
        ],
        "__comment__6": "python样例需要的环境变量 = get_model脚本需要的环境变量 + 当前的设置",
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
            "resnet50_multistreams": [
                "TCIM_BACKEND",
                "TCIM_RUNTIME_PATH",
                "HOUMO_SDK_PATH"
            ],
            "resnet50_pipeline": [
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
        "__comment__7": "cpp样例需要的环境变量 = python样例需要的环境变量 + 当前的设置",
        "cpp_example_environments": {
            "qwen3": [
                "CMAKE_PATH",
                "HOUMO_PATH"
            ],
            "resnet50": [
                "CMAKE_PATH",
                "OPENCV_PATH"
            ],
            "resnet50_multistreams": [
                "CMAKE_PATH",
                "OPENCV_PATH"
            ],
            "resnet50_pipeline": [
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
2. 添加环境依赖：
    - 目前已经设置了必要的环境变量，可以在json中的all_environments新增新的环境变量，默认是[Optional]，不会进行强制设置要求。
    - 如果需要添加的环境变量是[Required]，则需要在set_environs.py中WinEnvironsGenerater类的autoSetEnvirons中添加支持和校验，可以参考self.nullEnvManualSet("XXX_PATH", need=True)。

## 3. models目录样例支持说明(仅供参考)
### 3.1 hmatc的编译和安装
1. 确保已经安装上述的步骤完成了env.bat的自动环境变量设置。
2. cd /d {path/of/examples}/hmatc 
3. python setup.py install或者python setup.py bdist_wheel后执行pip install dist/hmatc-xxx.whl

### 3.2 样例的运行
1. 进入models对应样例目录。
2. 执行python get_model.py --type raw下载原始模型。
3. 执行python get_model.py --type hmm下载编译好的支持后摩设备的模型。
4. 运行时需要执行 set HOUMO_DATASETS_PATH=%HOUMO_EXAMPLES_PATH%/data/datasets 设置临时环境变量（每次新开一个终端都需要）。
5. 执行以下指令测试：
    - perf性能测试：hmatc perf -c config.yml -wn 10 -sn 1000 -tn 1
    - demo运行测试：hmatc demo -c config.yml
    - onnx运行测试：hmatc demo -c config.yml --onnx
    - hmm模型评估：hmatc eval -c config.yml
    - onnx模型评估：hmatc eval -c config.yml --onnx

备注：models下的模型在windows平台仅支持运行，不支持量化和编译，如需量化和编译请使用linux平台进行。env.bat打印支持的apis目录下样例确保可以运行，目前models下的windows样例运行仅供用户参考，样例可能有需要做适当修改才可以运行的情况。

## 4.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
