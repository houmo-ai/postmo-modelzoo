#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh1" ]; then
    echo "Only supports HOUMO_TARGET as xh1."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
  python3 get_model.py
  python3 quant_compile.py --model_name "ocr_rec" \
                           --model_path "./paddleocr_rec-sim.onnx" \
                           --output_path "./output" \
                           --compile

  python3 compare.py --model_name "ocr_rec" \
                     --onnx_model_path "./paddleocr_rec-sim.onnx" \
                     --output_path "./output" \
                     --data_path "CCPD2020/quant_data/rec/2_0_0_3_33_30_27_33_27.jpg"

  python3 perf.py ./output/${HOUMO_TARGET}/ocr_rec.hmm

  python3 run_model.py --model_path ./output/${HOUMO_TARGET}/ocr_rec.hmm \
                       --data_path CCPD2020/PPOCR/val/crop_imgs \
                       --infer_mode demo \
                       --num 10
  python3 run_model.py --model_path ./paddleocr_rec-sim.onnx \
                       --data_path CCPD2020/PPOCR/val/crop_imgs \
                       --infer_mode demo \
                       --num 10

  python3 run_model.py --model_path ./output/${HOUMO_TARGET}/ocr_rec.hmm \
                       --data_path CCPD2020 \
                       --infer_mode eval 
  python3 run_model.py --model_path ./paddleocr_rec-sim.onnx \
                       --data_path CCPD2020 \
                       --infer_mode eval 
fi
