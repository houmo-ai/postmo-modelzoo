import os
import sys
import argparse
import numpy as np
from prettytable import PrettyTable
from hmatc.infer import onnx_infer
from hmatc.utils.preprocess import convert_bgr_to_yuv

SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIP_DIR)
from base_utils import *

def compare(args):
    model_name = args.model_name
    onnx_model_path = args.onnx_model_path
    inputs_info_list, _ = get_net_input_output_infos(onnx_model_path)

    onnxInfer = onnx_infer.OnnxInfer()
    onnxInfer.load(onnx_model_path)

    onnx_in_datas = dict()
    hmquant_in_datas = dict()
    xh_in_datas = dict()

    data_path = args.data_path
    if not os.path.exists(data_path):
        data_path = os.path.join(HOUMO_DATASETS_PATH, data_path)
        if not os.path.exists(data_path):
            logger.error(f"{data_path} or {args.data_path} not exists!")
            assert(0)        
    cv_image = cv2.imread(data_path)
    if cv_image is None:
        logger.error("Failed to decode image")
        assert(0)
    onnx_in_datas[inputs_info_list[0]['name']] = onnx_preprocess(cv_image, inputs_info_list[0]['input_shape'][2:])
    
    if HOUMO_TARGET == "xh1":
        from hmatc.infer import hmquant_infer, xh1_infer
        
        sequencer_path =os.path.join(args.output_path, f"{HOUMO_TARGET}/{SUB_QUANT_PATH[HOUMO_TARGET]}/{model_name}.pkl")
        hmquantInfer = hmquant_infer.HmQuantInfer()
        hmquantInfer.load(sequencer_path)

        hmm_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/{model_name}_{HOUMO_TARGET}_b1_1core_O2.hmm")
        xhInfer = xh1_infer.Xh1Infer()
        xhInfer.load(hmm_path)
        data_fmt = xhInfer.inputs_info[inputs_info_list[0]['name']].format.name

        input_tensor, dyn_info = xh_preprocess(cv_image, inputs_info_list[0]['input_shape'][2:])
        yuv_pad_hwc = convert_bgr_to_yuv(input_tensor, fmt=data_fmt)
        h, w, c = yuv_pad_hwc.shape
        yuv_pad = yuv_pad_hwc.view(1, c, h, w)

        hmquant_in_datas[inputs_info_list[0]['name']] = yuv_pad.contiguous()
        hmquant_in_datas[f"resizer_crop_{inputs_info_list[0]['name']}"] = dyn_info

        yuv_pad = yuv_pad_hwc.detach().cpu().numpy().flatten()
        if data_fmt == "YUV420SP":
            valid_len = yuv_pad.size // 2
        elif data_fmt == "YUV422SP":
            valid_len = yuv_pad.size * 2 // 3
        elif data_fmt in ["YUV444SP", "YUV400"]:
            valid_len = yuv_pad.size
        else:
            logger.error(f"Unsupported format: {data_fmt}!")
            assert(0)
        yuv = yuv_pad[:valid_len]
        yuv = yuv.reshape(1, -1)

        xh_in_datas[inputs_info_list[0]['name']] = np.ascontiguousarray(yuv)  # np.ndarray
        xh_in_datas[f"resizer_crop_{inputs_info_list[0]['name']}"] = dyn_info.detach().cpu().numpy()

    elif HOUMO_TARGET == "xh2":
        from hmatc.infer import xhquant_infer, xh2_infer

        sequencer_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/{SUB_QUANT_PATH[HOUMO_TARGET]}/{model_name}.onnx")
        hmquantInfer = xhquant_infer.Xh2HmQuantInfer()
        hmquantInfer.load(sequencer_path)

        hmm_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/{model_name}_{HOUMO_TARGET}_b1_1core_O2.hmm")
        xhInfer = xh2_infer.Xh2Infer()
        xhInfer.load(hmm_path)

        onnx_data_fp16 = onnx_in_datas[inputs_info_list[0]['name']].copy().astype(np.float16)
        hmquant_in_datas[inputs_info_list[0]['name']] = torch.from_numpy(onnx_data_fp16).cpu()

        xh_in_datas[inputs_info_list[0]['name']] = onnx_data_fp16
    else:
        logger.error(f"Unsupported CHIP TARGET: {HOUMO_TARGET}")
        assert(0)
    
    onnx_outputs = onnxInfer.run(onnx_in_datas)
    hmquant_outputs = hmquantInfer.run(hmquant_in_datas)
    _, xh_outputs = xhInfer.run(xh_in_datas)

    header = [
            "name",
            "onnx vs hmquant",
            "onnx vs xh1",
            "hmquant vs xh1",
        ]
    table = PrettyTable(header)
    table.title = "Cosine Distance"

    for output_name in onnx_outputs:
        # onnx
        onnx_output = onnx_outputs[output_name]
        
        # hmquant
        hmquant_output = hmquant_outputs[output_name]
        if HOUMO_TARGET == "xh1":
            hmquant_output = xhInfer.dequantize(output_name, hmquant_output)
        
        # xh
        xh_output = xh_outputs[output_name]

        # compare
        onnx_vs_hmquant = cosine_distance(onnx_output, hmquant_output)
        onnx_vs_xh = cosine_distance(onnx_output, xh_output)
        hmquant_vs_xh = cosine_distance(hmquant_output, xh_output)

        table.add_row(
                [
                    output_name,
                    f"{onnx_vs_hmquant:.6f}",
                    f"{onnx_vs_xh:.6f}",
                    f"{hmquant_vs_xh:.6f}",
                ]
            )
    logger.info(f"Compare...\n{table}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ppocrv3_rec')
    parser.add_argument('--onnx_model_path', type=str, default='./paddleocr_rec-sim.onnx')
    parser.add_argument("--output_path", default="./output", type=str)
    parser.add_argument("--data_path", default="CCPD2020/quant_data/rec/2_0_0_3_33_30_27_33_27.jpg", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    
    compare(args)

if __name__ == "__main__":
    main()