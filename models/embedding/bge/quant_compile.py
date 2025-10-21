import os
import sys
import argparse
import time
import onnx
import logging
import numpy as np
import torch
import tcim
import tcim_lite
from tcim_lite.runtime import Tensor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2."
HOUMO_CORE_NUM = int(os.getenv('HOUMO_CORE_NUM', 2))
SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIP_DIR) 

def cosine_distance(data1, data2):
    if data1.shape != data2.shape:
        logging.error(f"[error] shape not equal {data1.shape} vs {data2.shape}")
        return -1
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    cosine_dist = np.dot(v1_norm, v2_norm)
    if np.isnan(cosine_dist):
        return -1
    return cosine_dist

def get_net_input_output_infos(model_path):
    if not os.path.exists(model_path):
        logging.error(f"{model_path} is not found!")
        assert 0
    onnx_model = onnx.load_model(model_path)
    input_infos_dict = {}
    for idx, net_input in enumerate(onnx_model.graph.input):
        input_name = net_input.name
        #net_input_size = []
        input_shape = [d.dim_value if d.dim_value > 0 else 1 for d in net_input.type.tensor_type.shape.dim]
        input_info = {
            'input_shape': input_shape,
            'dtype': onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(net_input.type.tensor_type.elem_type).name
        }
        input_infos_dict[input_name] = input_info
    output_infos_dict = {}
    for idx, net_output in enumerate(onnx_model.graph.output):
        output_name = net_output.name
        output_info = {
            'dtype': onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(net_output.type.tensor_type.elem_type).name
        }
        output_infos_dict[output_name] = output_info
    return input_infos_dict, output_infos_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./onnx', help='path of onnx model')
    parser.add_argument("--output_path", default="./output", type=str)
    parser.add_argument("--precision", type=str, default="w8a8_sefp", help="quant precision, xh2 support w8a8_sefp, w4a8_ssfp or w8a16_sefp")
    parser.add_argument("--compile", action="store_true", help="compile quanted model or no")
    return parser.parse_args()

def quantize(args, model_path, model_name):
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

    input_infos, output_infos = get_net_input_output_infos(model_path)

    in_datas = list()
    for input_name in input_infos.keys():
        input_shape = input_infos[input_name]['input_shape']
        dtype = input_infos[input_name]['dtype']
        if input_name == "input_ids":
            data = np.random.randint(0, 151645, size=input_shape, dtype=dtype)
        elif input_name == "attention_mask":
            data = np.ones(input_shape, dtype=dtype)
        elif input_name == "token_type_ids":
            data = np.zeros(input_shape, dtype=dtype)
        else:
            rng = np.random.default_rng()
            if np.issubdtype(dtype, np.integer):
                data = rng.integers(low=0, high=127, size=input_shape, dtype=dtype)
            else:
                data = rng.standard_normal(input_shape, dtype=dtype)
        in_datas.append(torch.from_numpy(data))

    save_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/xhquant")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_model_path = os.path.join(save_path, f"{model_name}.onnx")
    
    convert_onnx_to_hmonnx(model_path,
                            in_datas,
                            device_type=DeviceType.XH2a,
                            quant_config=quant_config,
                            out_hmonnx_file=output_model_path,
                            input_names=list(input_infos.keys()),
                            output_names=list(output_infos.keys()),
                            )
    debug_path = os.path.join(save_path, f"hmquant_{model_name}_with_act")
    if os.path.exists(debug_path):
        import shutil
        shutil.rmtree(debug_path, ignore_errors=True)
    
    quant_input_infos, _ = get_net_input_output_infos(output_model_path)

    session = HMONNXGoldenInference(output_model_path)
    session.to(device)
    session.save_golden = True
    session.golden_dir = save_path
    for idx, data in enumerate(in_datas):
        data = data.detach().cpu().numpy().astype(quant_input_infos[list(quant_input_infos.keys())[idx]]['dtype'])
        data = torch.from_numpy(data)
        in_datas[idx] = data.to(device)
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

def compile_model(output_path, model_name):
    base_path = os.path.join(output_path, f"{HOUMO_TARGET}")
    quanted_model_dir = os.path.join(base_path, f"xhquant/hmquant_{model_name}_with_act.onnx")

    tcim.build_from_hmonnx(
        onnx_model=quanted_model_dir,
        output_name=model_name,
        ncore=HOUMO_CORE_NUM,
        opt_level='O2',
        target=HOUMO_TARGET,
        output_dir=base_path,
        work_dir=os.path.join(base_path, "tcim"),
    )

def compare_hmm_golden(output_path, model_name):
    quanted_path = os.path.join(output_path, f"{HOUMO_TARGET}/xhquant")
    hmm_path = os.path.join(output_path, f"{HOUMO_TARGET}/{model_name}.hmm")

    # wt_manager = tcim_lite.runtime.WeightManager(1)
    # option = tcim_lite.runtime.Option(wt_manager)
    # module = tcim_lite.runtime.load(hmm_path, option=option)
    module = tcim_lite.runtime.load(hmm_path)

    input_num = module.get_num_inputs()
    input_names = [module.get_input_name(i) for i in range(input_num)]
    input_infos = [module.get_input_info(input_name) for input_name in input_names]

    out_num = module.get_num_outputs()
    out_names = [module.get_output_name(i) for i in range(out_num)]

    golden_input_data_list = []
    for idx, in_name in enumerate(input_names):
        golden_input_path = os.path.join(quanted_path, f"hmquant_{model_name}_{in_name}_input.npy")
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
        logging.info("current run iter cost: %.4f, global_avg cost: %.4f."%(cost_time, total_time / (i + 1))) 
    
    for idx, out_name in enumerate(out_names):
        infer_output = module.get_output(out_name)
        normal, cos_dist = calculate_golden_infer_output(infer_output, quanted_path, model_name, out_name)
        if not normal and cos_dist < 0.99:
            logging.warning(f"Output '{out_name}' golden comparison has low similarity, so please pay attention! cosine_similarity = %.6f"%cos_dist)
        else:
            logging.info(f"Output '{out_name}' golden compare is successful! cosine_similarity = %.6f"%cos_dist)

def main():
    args = parse_args()
    
    bge_m3_model_name = "bge-m3"
    onnx_path = os.path.join(args.model_path, f"{bge_m3_model_name}/{bge_m3_model_name}.onnx")
    if not os.path.exists(onnx_path):
        logging.error(f"{onnx_path} is not found!")
        return
    quantize(args, onnx_path, bge_m3_model_name)
    bge_reranker_model_name = "bge-reranker-v2-m3"
    onnx_path = os.path.join(args.model_path, f"{bge_reranker_model_name}/{bge_reranker_model_name}.onnx")
    if not os.path.exists(onnx_path):
        logging.error(f"{onnx_path} is not found!")
        return
    quantize(args, onnx_path, bge_reranker_model_name)
    args.compile = True
    if args.compile:
        compile_model(args.output_path, bge_m3_model_name)
        compile_model(args.output_path, bge_reranker_model_name)
        compare_hmm_golden(args.output_path, bge_m3_model_name)
        compare_hmm_golden(args.output_path, bge_reranker_model_name)

if __name__ == "__main__":
    main()