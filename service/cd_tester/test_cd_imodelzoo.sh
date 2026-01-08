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
    echo "Please provide a valid Houmo Dadao software version number, for example: 0.3.0, 2.5.0." >&2
    exit -1
fi

if [[ -z "$BACKEND" || ( -n "$BACKEND" && "$BACKEND" != "xh1" && "$BACKEND" != "xh2" ) ]]; then
    echo "Please provide a valid Houmo Backend for testing, supported backends: xh1, xh2." >&2
    exit -2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR" || {
    echo "Unable to enter test script directory: $SCRIPT_DIR" >&2
    exit 1
}
rm -rf "./../../tests/test_logs/"
rm -rf ./*.xml
echo "===> Starting CD tests, software version: ${BACKEND}-${VERSION}, current server: $(hostname), current time: $(date), current directory: $(pwd)"

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
        echo "XH1 CD test execution failed (exit code: $EXIT_CODE)" >&2
        RET=$EXIT_CODE
        # exit $EXIT_CODE
    fi
else
    IMAGE_NAME="Dadao-xh2-v${VERSION}-ubuntu24.04-x86.64.latest"
    if { [ "$TYPE" = "all" ] || [ "$TYPE" = "no-infer" ]; } && [ "$NO_MODELS" = "off" ]; then
        echo "Starting quantization and compilation tests $(date)"
        python3 quant_compile_tests.py -v $VERSION -t $BACKEND --release $RELEASE -k "${KEY_STR}" -m "${MODEL_STR}"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "XH2 quantization and compilation test execution failed (exit code: $EXIT_CODE)" >&2
            RET=$EXIT_CODE
            # exit $EXIT_CODE
        fi
    else
        echo "Skipping quantization and compilation tests $(date)"
    fi

    if [ "$TYPE" = "all" ] || [ "$TYPE" = "infer" ]; then
        # Read credentials from environment variables
        XH2_SERVER_USER="${XH2_SERVER_USER:-}"
        XH2_SERVER_IP="${XH2_SERVER_IP:-}"
        # Validate required environment variables
        if [ -z "$XH2_SERVER_USER" ] || [ -z "$XH2_SERVER_IP" ]; then
            echo "ERROR: Inference Failed! XH2_SERVER_USER and XH2_SERVER_IP environment variables must be set!" >&2
            exit 1
        fi
        echo "Starting XH2 inference on ${XH2_SERVER_IP}, time: $(date)"
        ssh "${XH2_SERVER_USER}@${XH2_SERVER_IP}" "python3 ${SCRIPT_DIR}/inference_tests.py -v ${VERSION} -t ${BACKEND} --release ${RELEASE} --no_apis ${NO_APIS} --no_hmatc ${NO_HMATC} --no_models ${NO_MODELS} -k '${KEY_STR}' -m '${MODEL_STR}' > ${SCRIPT_DIR}/test_cd.log 2>&1"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR: XH2 remote inference execution failed (exit code: $EXIT_CODE)" >&2
            RET=$EXIT_CODE
            # exit $EXIT_CODE
        fi
    else
        echo "Skipping inference tests $(date)"
    fi
fi

IMAGE_ID=$(docker images | grep ${IMAGE_NAME} | awk '{print $3}')
echo "Generating test results report $(date), docker image id: $IMAGE_ID"
python3 generate_test_report.py -v ${VERSION} -t ${BACKEND} -id ${IMAGE_ID} --release ${RELEASE}

exit $RET

echo "<=== CD tests finished, current time: $(date)"