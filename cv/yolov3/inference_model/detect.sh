#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

python3 preprocess.py --output-path ./preprocessed \
        --img-path "${SCRIPT_DIR}/data/dog.jpg"

if [ -c /dev/hmcl_feature_in ]; then
  export HDPL_PLATFORM=ASIC
fi

mkdir -p build
cd build
cmake ..
make
cd ../
#./build/hdpl_yolov3 "${SCRIPT_DIR}/data/dog.json" ./preprocessed 1

python3 draw_box.py --img-path "${SCRIPT_DIR}/data/dog.jpg" \
	            --output-path "${SCRIPT_DIR}/boxed_dog.jpg" \
	            --coco-names "${DATASETS_PATH}/coco.names" \
		    --detect-json ./detections.json
