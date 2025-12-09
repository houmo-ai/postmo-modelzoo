#! /bin/bash

# default params
VERSION=""
BACKEND=""
IMAGE_NAME=""
TYPE="all"
RELEASE="off"
NO_APIS="off"
NO_HMATC="off"
NO_MODELS="off"
KEY_STR=""
MODEL_STR=""

show_help() {
    echo "Usage: $0 [options]"
    echo "  -v, --version     Houmo Dadao software version, example: 0.3.0, 2.5.0"
    echo "  -b, --backend     Houmo backend, support: xh1, xh2"
    echo "  -t, --test_type   CD test type, support: all, infer, no-infer, default is all."
    echo "  --release         Use released models for testing."
    echo "  --no_apis         Don't execute apis testing."
    echo "  --no_hmatc        Don't execute hmatc testing."
    echo "  --no_models       Don't execute models testing."
    echo "  -k, --key_str     Filter test cases using the -k parameter."
    echo "  -m, --model_str   Filter test cases using the -m parameter."
    echo "  -h, --help        help information"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
        ;;
        -b|--backend)
            BACKEND="$2"
            shift 2
        ;;
        -t|--test_type)
            TYPE="$2"
            shift 2
        ;;
        --release)
            RELEASE="on"
            shift
        ;;
        --no_apis)
            NO_APIS="on"
            shift
        ;;
        --no_hmatc)
            NO_HMATC="on"
            shift
        ;;
        --no_models)
            NO_MODELS="on"
            shift
        ;;
        -k|--key_str)
            KEY_STR="$2"
            shift 2
        ;;
        -m|--model_str)
            MODEL_STR="$2"
            shift 2
        ;;
        -h|--help)
            show_help
        ;;
        *)
            echo "Error: Unknown parameter '$1'" >&2
            show_help
        ;;
    esac
done

valid_version_pattern='^[0-9]+\.[0-9]+\.[0-9]+$'
if [ -z "$VERSION" ] || { [ -n "$VERSION" ] && ! [[ "$VERSION" =~ $valid_version_pattern ]]; }; then
    echo "请提供合法的待测试的后摩大道软件版本号,例如: 0.3.0, 2.5.0。" >&2
    exit -1
fi

if [[ -z "$BACKEND" || ( -n "$BACKEND" && "$BACKEND" != "xh1" && "$BACKEND" != "xh2" ) ]]; then
    echo "请提供合法的待测试的后摩Backend, 支持的Backend: xh1, xh2。" >&2
    exit -2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR" || {
    echo "无法进入测试脚本所在目录: $SCRIPT_DIR" >&2
    exit 1
}
rm -rf "./../../tests/test_logs/"
rm -rf ./*.xml
echo "===> 开始CD测试, 软件版本: ${BACKEND}-${VERSION}, 当前服务器：$(hostname), 当前时间: $(date), 当前目录: $(pwd)"

if [ -n "$KEY_STR" ]; then
    echo "-k ${KEY_STR}"
fi
if [ -n "$MODEL_STR" ]; then
    echo "-m ${MODEL_STR}"
fi

RET=0
if [ "$BACKEND" = "xh1" ]; then
    IMAGE_NAME="Dadao-xh1-v${VERSION}-ubuntu20.04-x86.64.latest"
    python3 inference_tests.py -v $VERSION -t $BACKEND --release $RELEASE --no_apis $NO_APIS --no_hmatc $NO_HMATC --no_models $NO_MODELS -k "${KEY_STR}" -m "${MODEL_STR}"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "XH1 CD测试执行失败(退出码: $EXIT_CODE)" >&2
        RET=$EXIT_CODE
        # exit $EXIT_CODE
    fi
else
    IMAGE_NAME="Dadao-xh2-v${VERSION}-ubuntu24.04-x86.64.latest"
    if { [ "$TYPE" = "all" ] || [ "$TYPE" = "no-infer" ]; } && [ "$NO_MODELS" = "off" ]; then
        echo "开始执行量化及编译测试 $(date)"
        python3 quant_compile_tests.py -v $VERSION -t $BACKEND --release $RELEASE -k "${KEY_STR}" -m "${MODEL_STR}"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "XH2量化及编译测试执行失败(退出码: $EXIT_CODE)" >&2
            RET=$EXIT_CODE
            # exit $EXIT_CODE
        fi
    else
        echo "跳过量化及编译测试 $(date)"
    fi

    if [ "$TYPE" = "all" ] || [ "$TYPE" = "infer" ]; then
        echo "开始在 10.64.34.58 执行XH2推理 $(date)"
        ssh wanyu.li@10.64.34.58 "python3 ${SCRIPT_DIR}/inference_tests.py -v ${VERSION} -t ${BACKEND} --release ${RELEASE} --no_apis ${NO_APIS} --no_hmatc ${NO_HMATC} --no_models ${NO_MODELS} -k '${KEY_STR}' -m '${MODEL_STR}' > ${SCRIPT_DIR}/test_cd.log 2>&1"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "XH2远端推理执行失败(退出码: $EXIT_CODE)" >&2
            RET=$EXIT_CODE
            # exit $EXIT_CODE
        fi
    else
        echo "跳过推理测试 $(date)"
    fi
fi

IMAGE_ID=$(docker images | grep ${IMAGE_NAME} | awk '{print $3}')
echo "开始生成测试结果报告 $(date), docker image id: $IMAGE_ID"
python3 generate_test_report.py -v ${VERSION} -t ${BACKEND} -id ${IMAGE_ID} --release ${RELEASE}

exit $RET

echo "<=== CD测试结束, 当前时间: $(date)"