import os
import numpy as np
import time
import argparse
import tcim
import tcim_lite

import logging
logging.basicConfig(level="INFO")

HOUMO_TARGET = os.getenv('HOUMO_TARGET')
HOUMO_CORE_NUM = os.getenv('HOUMO_CORE_NUM', 2)
GOLDEN_THRESH = 0.99

def cosine_distance(data1, data2):
    if data1.shape != data2.shape:
        print(f"[error] shape not equal {data1.shape} vs {data2.shape}")
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

def build(model_dir, model_name, ncore, output_dir):

    model = os.path.join(model_dir, f"hmquant_{model_name}_with_act.onnx")
    if not os.path.exists(model):
        logging.error(f"{model} is not exists!")
        exit(0)
    tcim.build_from_hmonnx(
        model,
        output_name=model_name,
        ncore=ncore,
        target=HOUMO_TARGET,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim"),
        llm_opt=True,
    )

    hmm_path = os.path.join(output_dir, f"{model_name}.hmm")
    module = tcim_lite.runtime.load(hmm_path)
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        input_data_path = os.path.join(model_dir, f"hmquant_{model_name}_{input_name}_input.npy")
        input_data = np.load(input_data_path).astype(input_info.dtype)
        module.set_input(input_name, input_data)
    module.run()
    module.sync()
    output_num = module.get_num_outputs()
    result_check = True
    for id in range(output_num):
        output_name = module.get_output_name(id)
        output_data = module.get_output(output_name).numpy()
        golden_data_path = os.path.join(model_dir, f"hmquant_{model_name}_{output_name}_output.npy")
        golden_output = np.load(golden_data_path)
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)
            is_match = (golden_output == output_data).all()
            logging.info(f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist:.6f}")
            if is_match:
                continue
            if cosine_dist < GOLDEN_THRESH:
                result_check = False
        else:
            result_check = False
            logging.warning(f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape}") 
    if not result_check:
        logging.warning("The difference in golden inspection is large and needs attention")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./output", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    
    base_path_1st_list = ["llm_decoder", "llm_prefill", "vision"]
    model_names_list = ["minicpmo_llm_7b_xh2a_4k_decode",
                        "minicpmo_llm_7b_xh2a_4k_prefill",
                        "minicpmo_vision_7b_xh2a_2k"]
    for idx, base_path_1st in enumerate(base_path_1st_list):
        model_dir = os.path.join(args.output_path, f"{HOUMO_TARGET}/hmquant/{base_path_1st}")
        model_name = model_names_list[idx]
        output_dir = os.path.join(args.output_path, f"{HOUMO_TARGET}")
        build(model_dir, model_name, HOUMO_CORE_NUM, output_dir)
        logging.info(f"{base_path_1st} compile successful!")

if __name__ == "__main__":
    main()