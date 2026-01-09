#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: quant_compile.py
# Description:
#   Post-training quantization and compilation of the ppocrv3 recognition model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import time
import glob
from datetime import datetime
import json
from tcim_lite.runtime import Tensor

SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIP_DIR)
from base_utils import *

def get_xh1_quant_cfg(precision, input_name, net_input_size):
    qcfg = dict(
        inputs_cfg = {
            input_name: dict(
                data_format="BGR",
                first_layer_weight_denorm_mean=[0.5, 0.5, 0.5],
                first_layer_weight_denorm_std=[0.5, 0.5, 0.5],
                #resizer_crop={"top": 0, "left": 0, "height": 0, "width": 0},
                resizer_resize={
                    "height": net_input_size[0],
                    "width": net_input_size[1],
                    "align_corners": False,
                    "method": "bilinear"},
                toYUV_format = YUV_FORMAT,
                resizer_pad = {"value": [-1, 0, 0]},
                dynamic_crop = True,
                ####  dynamic resizer mode 0(len(dyn_info)=4) set fold=True, dynamic resizer mode 1(len(dyn_info)=10) set fold=False
                fold = False,
            )
        },
        quant_cfg = dict(
            mix_search = dict(
                activation = dict(
                    method = dict(name="all" if precision == "int16" else "auto"),
                    target_cos = 0.99,
                    calib_samples = 1
                )
            )
        )
    )
    if precision == "int8":
        del qcfg['quant_cfg']['mix_search']
    return qcfg       

if HOUMO_TARGET == 'xh1':
    CUSTOM_MSG = dict()
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='ppocrv3_rec')
        parser.add_argument('--model_path', type=str, default='./paddleocr_rec-sim.onnx')
        parser.add_argument("--output_path", default="./output/", type=str)
        parser.add_argument("--calibset_path", type=str, default="CCPD2020/quant_data/rec/", help="quant calib data path")
        parser.add_argument("--precision", type=str, default="int16", help="quant precision, xh1 support int16, auto or int8")
        parser.add_argument("--calib_num", type=int, default=200, help="calibset use number")
        parser.add_argument("--compile", action="store_true", help="compile quanted model or no")
        parser.add_argument("--enable_upload", action="store_true", help="Compress hmm model and upload or no")
        return parser.parse_args()
    
    def quantize(args):
        from hmquant.api import generate_golden, quant_single_onnx_network, convert_profiling, quantize_profiling
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_infos_list, _ = get_net_input_output_infos(args.model_path)
        calibset_path = os.path.join(HOUMO_DATASETS_PATH, args.calibset_path) \
            if not os.path.isabs(args.calibset_path) and "./" != args.calibset_path[:2] else args.calibset_path
        if not os.path.exists(calibset_path):
            logger.error(f"{calibset_path} is not exist!")
        calibdata_list = glob.glob(os.path.join(calibset_path, "*.jpg"))
        calibdata_list = calibdata_list[:args.calib_num]
        input_data = list()
        for idx, file_dir in enumerate(calibdata_list):
            image = cv2.imread(file_dir)
            sequence_data, crop_info = xh_preprocess(image, input_infos_list[0]['input_shape'][2:])
            input_data.append([sequence_data, crop_info])

        onnx_data = onnx_preprocess(cv2.imread(calibdata_list[0]), input_infos_list[0]['input_shape'][2:])

        qcfg = get_xh1_quant_cfg(args.precision, input_infos_list[0]['name'], input_infos_list[0]['input_shape'][2:])

        single_batch_sequence_input = {input_infos_list[0]['name']: input_data[0][0],
                                      f"resizer_crop_{input_infos_list[0]['name']}": input_data[0][1]}
        single_batch_onnx_input = {input_infos_list[0]['name']: onnx_data}
        convert_profiling(args.model_path, 
                          onnx_input=[single_batch_onnx_input],
                          cfg=qcfg,
                          sequencer_input=[single_batch_sequence_input],
                          device=device)
        
        sequencer = quant_single_onnx_network(cfg=qcfg,
                                              calibration_data=input_data,
                                              onnx_model_or_path=args.model_path,
                                              device=device)
        quantize_profiling(sequencer, [single_batch_sequence_input], 
                           device=device,
                           mode=0,
                           quant_mode="quant_forward")
        save_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/{SUB_QUANT_PATH[HOUMO_TARGET]}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        generate_golden(sequencer, single_batch_sequence_input, save_path, args.model_name)
        sequencer.save_pkl(save_path, args.model_name)

        msg_mean = qcfg['inputs_cfg'][input_infos_list[0]['name']]['first_layer_weight_denorm_mean']
        msg_mean = [msg_mean[idx] * 255 for idx in range(len(msg_mean))]
        msg_std = qcfg['inputs_cfg'][input_infos_list[0]['name']]['first_layer_weight_denorm_std']
        msg_std = [msg_std[idx] * 255 for idx in range(len(msg_std))]
        CUSTOM_MSG[input_infos_list[0]['name']] = {
            'shape': input_infos_list[0]['input_shape'],
            'resizer_mode': 1,
            'input_cfg': {
                'shape': input_infos_list[0]['input_shape'],
                'data_format': qcfg['inputs_cfg'][input_infos_list[0]['name']]['data_format'],
                'mean': msg_mean,
                'std': msg_std,
                'resize_type': 1,
                'padding_mode': 1,
                'padding_values': [127.5, 127.5, 127.5],
                'resizer': {
                    'toYUV_format': f"{YUV_FORMAT}SP",
                    'max_input_size': MAX_INPUT_SIZE,
                }
            }
        }

    def calculate_golden_infer_output(infer_output: Tensor, golden_path: str, model_name: str, output_name: str):
        peat_name = output_name.replace("/", "_")
        golden_fix_output_dir = os.path.join(golden_path, f"hmquant_{model_name}_{peat_name}_output.npy")
        golden_fix_output_data = np.load(golden_fix_output_dir)
        infer_fix_output_data = infer_output.numpy()
        re_flag = np.array_equal(infer_fix_output_data, golden_fix_output_data)
        if re_flag:
            return re_flag, 1.0
        else:
            golden_fp32_output_dir = os.path.join(golden_path, f"hmquant_{model_name}_{output_name}_dequant_output.npy")
            golden_fp32_output_data = np.load(golden_fp32_output_dir)
            infer_fp32_output_data = infer_output.astype(np.float32).numpy()
            cos_dist = cosine_distance(infer_fp32_output_data, golden_fp32_output_data)
            return re_flag, cos_dist
    
    def compile_model(args):
        import tcim
        base_path = os.path.join(args.output_path, f"{HOUMO_TARGET}")
        quanted_model_dir = os.path.join(base_path, f"{SUB_QUANT_PATH[HOUMO_TARGET]}/hmquant_{args.model_name}_with_act.onnx")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tcim.build_from_hmonnx(
            onnx_model=quanted_model_dir,
            output_name=f"{args.model_name}_xh1_b1_1core_O2",
            ncore=1,
            opt_level='O2',
            target=HOUMO_TARGET,
            batch=1,
            output_dir=base_path,
            work_dir=os.path.join(base_path, "tcim"),
            enable_dynamic_image_resize=DYNAMIC_RESIZE[HOUMO_TARGET],
            custom_msg=json.dumps(CUSTOM_MSG, ensure_ascii=False)
        )
        hmm_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/{args.model_name}_xh1_b1_1core_O2.hmm")
        if args.enable_upload:
            logger.info("Compressing hmmodel...")
            hmcc_version = get_package_version(f"houmo-tcim-xh1")
            runtime_version = get_package_version(f"houmo_tcim_runtime_xh1")
            with open(os.path.join(args.output_path, "xh1", "VERSION.txt"), "w") as f:
                f.write(f"hmquant_version: {get_hmquant_xh1_version()}\n")
                f.write(f"tcim_version: {hmcc_version}\n")
                f.write(f"tcim_runtime_version: {runtime_version}\n")
                f.write(f"build_time: {now}\n")
            filename = f"{args.model_name}_xh1_b1_1core_O2_{get_houmo_version()}.tar.xz"
            compress_hmm_path = os.path.join(
                args.output_path,
                "xh1",
                filename,
            )
            compress_files_to_tar_xz_with_progress(
                [hmm_path, os.path.join(args.output_path, "xh1", "VERSION.txt")],
                compress_hmm_path,
            )
            logger.info(
                f"MD5: {get_file_md5(compress_hmm_path)}, save path: {compress_hmm_path}"
            )
            upload_file_to_artifactory(
                compress_hmm_path,
                f"models/{HOUMO_TARGET}-{get_houmo_version()}/{args.model_name}/{filename}",
                max_retries=3,
            )
            logger.info(f"Compressing hmmodel done.")
    
elif HOUMO_TARGET == 'xh2':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='ppocrv3_rec')
        parser.add_argument('--model_path', type=str, default='./paddleocr_rec-sim.onnx')
        parser.add_argument("--output_path", default="./output", type=str)
        parser.add_argument("--precision", type=str, default="w8a8_sefp", help="quant precision, xh2 support w8a8_sefp, w4a8_ssfp or w8a16_sefp")
        parser.add_argument("--calibset_path", type=str, default="CCPD2020/quant_data/rec/", help="quant calib data path")
        parser.add_argument("--calib_num", type=int, default=1, help="calibset use number")
        parser.add_argument("--mix_search_cfg", type=str, default="./xh2_mix_search_cfg.yaml", help="quant mix search config path")
        parser.add_argument("--compile", action="store_true", help="compile quanted model or no")
        parser.add_argument("--enable_upload", action="store_true", help="Compress hmm model and upload or no")
        return parser.parse_args()
    
    def quantize(args):
        from xhquant.api import (
            DeviceType,
            HMONNXGoldenInference,
            HMONNXInference,
            QuantScheme,
            convert_onnx_to_hmonnx,
            create_quant_config,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.precision)
        quant_config = create_quant_config(quant_scheme)

        input_infos_list, output_infos_list = get_net_input_output_infos(args.model_path)

        try:
            calibset_path = os.path.join(HOUMO_DATASETS_PATH, args.calibset_path) \
                if not os.path.isabs(args.calibset_path) and "./" != args.calibset_path[:2] else args.calibset_path
            if not os.path.exists(calibset_path):
                logger.error(f"{calibset_path} is not exist!")
            calibdata_list = glob.glob(os.path.join(calibset_path, "*.jpg"))
            calibdata_list = calibdata_list[:args.calib_num]
            in_datas = list()
            for idx, file_dir in enumerate(calibdata_list):
                image = cv2.imread(file_dir)
                sequence_data = onnx_preprocess(image, input_infos_list[0]['input_shape'][2:])
                dtype = input_infos_list[0]['dtype']
                data = sequence_data.astype(dtype)
                in_datas.append(torch.from_numpy(data).to(device=device))
        except FileNotFoundError:
            logger.warning("Calibset not found, use random data!")
            in_datas = list()
            for input_info in input_infos_list:
                input_shape = input_info['input_shape']
                dtype = input_info['dtype']
                rng = np.random.default_rng()
                data = rng.standard_normal(input_shape, dtype=dtype)
                in_datas.append(torch.from_numpy(data))

        save_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/xhquant")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_model_path = os.path.join(save_path, f"{args.model_name}.onnx")

        mix_search_cfg = None
        if os.path.exists(args.mix_search_cfg):
            mix_search_cfg = args.mix_search_cfg
        
        convert_onnx_to_hmonnx(args.model_path,
                               in_datas,
                               device_type=DeviceType.XH2a,
                               out_hmonnx_file=output_model_path,
                               quant_config=quant_config,
                               input_names=[input_info['name'] for input_info in input_infos_list],
                               output_names=[output_info['name'] for output_info in output_infos_list],
                               mix_search=mix_search_cfg)
        
        debug_path = os.path.join(save_path, f"hmquant_{args.model_name}_with_act")
        if os.path.exists(debug_path):
            import shutil
            shutil.rmtree(debug_path, ignore_errors=True)
        
        session = HMONNXGoldenInference(output_model_path)
        session.to(device)
        session.save_golden = True
        session.golden_dir = save_path
        for idx, data in enumerate(in_datas):
            in_datas[idx] = data.half().to(device)
        session(*in_datas)
    
    def calculate_golden_infer_output(infer_output: Tensor, golden_path: str, model_name: str, output_name: str):
        peat_name = output_name.replace("/", "_")
        golden_output_dir = os.path.join(golden_path, f"hmquant_{model_name}_{peat_name}_output.npy")
        golden_output_data = np.load(golden_output_dir)
        infer_output_data = infer_output.numpy()
        re_flag = np.array_equal(infer_output_data, golden_output_data)
        if re_flag:
            return re_flag, 1.0
        else:
            cos_dist = cosine_distance(golden_output_data, infer_output_data)
            return re_flag, cos_dist

    def compile_model(args):
        import tcim
        base_path = os.path.join(args.output_path, f"{HOUMO_TARGET}")
        quanted_model_dir = os.path.join(base_path, f"{SUB_QUANT_PATH[HOUMO_TARGET]}/hmquant_{args.model_name}_with_act.onnx")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tcim.build_from_hmonnx(
            onnx_model=quanted_model_dir,
            output_name=f"{args.model_name}_xh2_b1_1core_O2",
            ncore=1,
            opt_level='O2',
            target=HOUMO_TARGET,
            batch=1,
            output_dir=base_path,
            work_dir=os.path.join(base_path, "tcim"),
            enable_dynamic_image_resize=DYNAMIC_RESIZE[HOUMO_TARGET],
        )
        hmm_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/{args.model_name}_{HOUMO_TARGET}_b1_1core_O2.hmm")
        if args.enable_upload:
            logger.info("Compressing hmmodel...")
            hmcc_version = get_package_version(f"houmo-tcim-xh2")
            runtime_version = get_package_version(f"houmo_tcim_runtime_xh2")
            with open(os.path.join(args.output_path, "xh2", "VERSION.txt"), "w") as f:
                f.write(f"hmquant_version: {get_hmquant_xh2_version()}\n")
                f.write(f"tcim_version: {hmcc_version}\n")
                f.write(f"tcim_runtime_version: {runtime_version}\n")
                f.write(f"build_time: {now}\n")
            filename = f"{args.model_name}_{HOUMO_TARGET}_b1_1core_O2_{get_houmo_version()}.tar.xz"
            compress_hmm_path = os.path.join(
                args.output_path,
                "xh2",
                filename,
            )
            compress_files_to_tar_xz_with_progress(
                [hmm_path, os.path.join(args.output_path, "xh2", "VERSION.txt")],
                compress_hmm_path,
            )
            logger.info(
                f"MD5: {get_file_md5(compress_hmm_path)}, save path: {compress_hmm_path}"
            )
            upload_file_to_artifactory(
                compress_hmm_path,
                f"models/{HOUMO_TARGET}-{get_houmo_version()}/{args.model_name}/{filename}",
                max_retries=3,
            )
            logger.info(f"Compressing hmmodel done.")

def compare_hmm_golden(args):
    import tcim_lite
    quanted_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/{SUB_QUANT_PATH[HOUMO_TARGET]}")
    hmm_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/{args.model_name}_{HOUMO_TARGET}_b1_1core_O2.hmm")

    module = tcim_lite.runtime.load(hmm_path)

    input_num = module.get_num_inputs()
    input_names = [module.get_input_name(i) for i in range(input_num)]
    input_infos = [module.get_input_info(input_name) for input_name in input_names]

    out_num = module.get_num_outputs()
    out_names = [module.get_output_name(i) for i in range(out_num)]

    golden_input_data_list = []
    for idx, in_name in enumerate(input_names):
        golden_input_path = os.path.join(quanted_path, f"hmquant_{args.model_name}_{in_name}_input.npy")
        golden_input = np.load(golden_input_path).astype(input_infos[idx].dtype)
        golden_input_data_list.append(golden_input)
    
    total_time = 0
    for i in range(1):
        for idx, in_name in enumerate(input_names):
            module.set_input(in_name, golden_input_data_list[idx])
        t_start = time.time()
        module.run()
        module.sync()
        t_end = time.time()
        cost_time = t_end - t_start
        total_time += cost_time
        logger.info("current run iter cost: %.4f, global_avg cost: %.4f."%(cost_time, total_time / (i + 1))) 
    
    for idx, out_name in enumerate(out_names):
        infer_output = module.get_output(out_name)
        normal, cos_dist = calculate_golden_infer_output(infer_output, quanted_path, args.model_name, out_name)
        if not normal and cos_dist < NORAM_DIST[HOUMO_TARGET]:
            logger.warning(f"Output '{out_name}' golden compare is failed! cosine_similarity = %.6f"%cos_dist)
        else:
            logger.info(f"Output '{out_name}' golden compare is successful! cosine_similarity = %.6f"%cos_dist)

def main():
    args = parse_args()
    
    quantize(args)
    if args.compile:
        compile_model(args)
        compare_hmm_golden(args)

if __name__ == "__main__":
    main()