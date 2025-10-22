import os
import argparse
import logging
import onnx
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh2", "Only support HOUMO_TARGET: xh2."

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

    save_path = os.path.join(args.output_path, f"{HOUMO_TARGET}/hmquant")
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

if __name__ == "__main__":
    main()