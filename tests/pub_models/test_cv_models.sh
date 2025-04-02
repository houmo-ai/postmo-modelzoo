#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"


bash ../models/backbone/resnet50/perf.sh
bash ../models/backbone/mobilenetv2/perf.sh
bash ../models/backbone/efficientnet/perf.sh
# bash ../models/backbone/vit/perf.sh
bash ../models/detection/yolov5s/perf.sh
bash ../models/detection/yolov3/perf.sh
bash ../models/detection/yolov8m/perf.sh

bash ../models/backbone/resnet50/eval.sh
bash ../models/backbone/mobilenetv2/eval.sh
bash ../models/backbone/efficientnet/eval.sh
# bash ./models/backbone/vit/eval.sh
bash ../models/detection/yolov5s/eval.sh
bash ../models/detection/yolov3/eval.sh
bash ../models/detection/yolov8m/eval.sh

bash ../models/backbone/resnet50/test.sh
bash ../models/backbone/mobilenetv2/test.sh
bash ../models/backbone/efficientnet/test.sh
# bash ./models/backbone/vit/test.sh
bash ../models/detection/yolov5s/test.sh
bash ../models/detection/yolov3/test.sh
bash ../models/detection/yolov8m/test.sh

