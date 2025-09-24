#! /bin/bash

# default params
VERSION=""
BACKEND=""
TYPE="all"

show_help() {
    echo "Usage: $0 [options]"
    echo "  -v, --version     Houmo Dadao software version, example: 0.3.0, 2.5.0"
    echo "  -b, --backend     Houmo backend, support: xh1, xh2"
    echo "  -t, --test_type   CD test type, support: all, infer, no-infer, default is all."
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

echo "===> 开始CD测试, 软件版本: ${BACKEND}-${VERSION}, 当前服务器：$(hostname), 当前时间: $(date)"

RET=0
if [ "$BACKEND" = "xh1" ]; then
    python3 inference_tests.py -v ${VERSION} -t ${BACKEND} > ./cd_tester_xh1.log 2>&1
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "XH1 CD测试执行失败(退出码: $EXIT_CODE)" >&2
        RET=$EXIT_CODE
        # exit $EXIT_CODE
    fi
else
    if [ "$TYPE" = "all" ] || [ "$TYPE" = "no-infer" ]; then
        echo "开始执行量化及编译测试 $(date)"
        python3 quant_compile_tests.py -v $VERSION -t $BACKEND
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
        echo "开始在 10.64.35.71 执行XH2推理 $(date)"
        ssh wanyu.li@10.64.35.71 "python3 /develop02/wanyu.li/imodelzoo_develop/service/cd_tester/inference_tests.py -v ${VERSION} -t ${BACKEND} > /develop02/wanyu.li/imodelzoo_develop/service/cd_tester/test_cd.log 2>&1"
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

exit $RET

echo "<=== CD测试结束, 当前时间: $(date)"