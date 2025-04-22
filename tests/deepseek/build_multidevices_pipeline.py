import os
import numpy as np
import time
import argparse

import logging
logging.basicConfig(level="INFO")

HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')


def sanitize_name(name: str):
    return name.replace(":", "_").replace("/", "_")


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


def save_submodel_golden(model_dir, model_name, output_names):
    for name in output_names:
        file_path = os.path.join(model_dir, f"hmquant_{model_name}_with_act/{name}.npy")
        print(file_path)
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True).item().get("output_tensor")
            save_path1 = os.path.join(model_dir, f"hmquant_{model_name}_{name}_input.npy")
            save_path2 = os.path.join(model_dir, f"hmquant_{model_name}_{name}_output.npy")
            np.save(save_path1, data)
            np.save(save_path2, data)
            print(f"{os.path.basename(save_path1)} saved in {os.path.dirname(save_path1)}")
            print(f"{os.path.basename(save_path2)} saved in {os.path.dirname(save_path2)}")
            
            
def extract_model(src, dest, input_names, output_names):
    import onnx
    import onnx_graphsurgeon
    model = onnx.load(src)
    graph = onnx_graphsurgeon.import_onnx(model)
    tensors = graph.tensors()
    print(f"inputs: {input_names}")
    print(f"outputs: {output_names}")
    graph.inputs = [tensors[in_t.strip()] for in_t in input_names]
    graph.outputs = [tensors[out_t.strip()] for out_t in output_names]
    graph.cleanup()
    onnx.save(onnx_graphsurgeon.export_onnx(graph), dest)
    print(f"extracted model saved in", dest)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, 'hmquant'),
        help='path to the model dir',
    )
    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='deepseek',
        help='output houmo model name',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=1,
        help='batch size',
    )
    parser.add_argument(
        '--ncore',
        dest='ncore',
        type=int,
        default=4,
        help='core number',
    )
    parser.add_argument(
        '--nblocks',
        dest='nblocks',
        type=int,
        default=48,
        help='block number',
    )
    parser.add_argument(
        '--stage',
        dest='stage',
        type=str,
        default="all",
        help='build stage choise=["build", "test", "all"]',
    )
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET),
        help='build output dir',
    )
    args = parser.parse_args()
    return args


def clip(raw_path, part1_path, part2_path, nblocks):
    dir_name = os.path.dirname(raw_path)
    mid_layer_id = 25
    mid_layer_name = f'model_layers_{mid_layer_id}_resadd2'
    save_submodel_golden(dir_name, 'deepseek', [mid_layer_name])
    input_names = ['input_1', 'valid_length', 'current_length']
    for i in range(mid_layer_id+1):
        input_names.append(f'model_layers_{i}_self_attn_kcache_input')
        input_names.append(f'model_layers_{i}_self_attn_vcache_input')
        input_names.append(f'model_layers_{i}_self_attn_kcache_history_sum')
    # onnx.utils.extract_model(raw_path, part1_path, input_names=input_names, 
    #                          output_names=[mid_layer_name], check_model=True)
    extract_model(raw_path, part1_path, input_names=input_names, 
                    output_names=[mid_layer_name])
    input_names = [mid_layer_name, 'valid_length', 'current_length']
    for i in range(mid_layer_id+1, nblocks):
        input_names.append(f'model_layers_{i}_self_attn_kcache_input')
        input_names.append(f'model_layers_{i}_self_attn_vcache_input')
        input_names.append(f'model_layers_{i}_self_attn_kcache_history_sum')
    # onnx.utils.extract_model(raw_path, part2_path, input_names=input_names, 
    #                          output_names=['Output_lm_head_add_list_1'], check_model=True)
    extract_model(raw_path, part2_path, input_names=input_names, 
                    output_names=['Output_lm_head_add_list_1'])


def build(model_name, model_dir, model_path, output_dir, profile, ncore=1):
    import tcim
    start = time.time()
    print(f"\n===> {model_name} build start...")
    decode_model = os.path.join(model_dir, model_path)
    tcim.build_from_hmonnx(
        decode_model,
        weights=os.path.join(model_dir, "weight.npy"),
        output_name=model_name,
        ncore=ncore,
        llm_opt=True,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim"),
    )
    profile["build"] = time.time() - start
    print(f'{model_name} build completed in {profile["build"]:.3f} s.', flush=True)


def test(model_name, model_dir, output_dir, profile, batch=1, prefix=None):
    import tcim_lite
    print(f"\n===> {model_name} test start...")
    # load model
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    start = time.time()
    option = tcim_lite.runtime.Option(0)
    module = tcim_lite.runtime.load(model_path, option)
    profile["load"] = time.time() - start
    print(f'{model_name} load completed in {profile["load"]:.3f} s.', flush=True)

    # set input
    current_length = 0
    profile["set_input"] = 0
    if prefix is None:
        prefix = model_name
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print(f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}")
        input_data_path = os.path.join(model_dir, f"hmquant_{prefix}_{sanitize_name(input_name)}_input.npy")
        input_data = np.load(input_data_path).astype(input_info.dtype)
        if input_name == 'current_length':
            current_length = input_data[0]
            print("current_length is", current_length)
        input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        print(f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}")
        start = time.time()
        module.set_input(input_name, input_data)
        profile["set_input"] += time.time() - start
    print(f'{model_name} set {input_num} inputs completed in {profile["set_input"]*1000:.3f} ms.')

    # infer model
    start = time.time()
    module.run()
    module.sync()
    profile["infer"] = time.time() - start
    print(f'{model_name} infer completed in {profile["infer"]*1000:.3f} ms.')

    # get output and compare with golden
    profile["get_output"] = 0
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name)
        print(f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}")
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        if len(output_data.shape) == 3:
            output_data = output_data[:1, :current_length, :]
        profile["get_output"] += time.time() - start
        print(f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}")
        output_data_path = os.path.join(model_dir, f'hmquant_{prefix}_{sanitize_name(output_name)}_output.npy')
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            if len(golden_output.shape) == 3:
                golden_output = golden_output[:1, :current_length, :]
            golden_output = np.concatenate([golden_output for i in range(batch)], axis=0)
        else:
            result_check = False
            print(f"[warning] compare canceled while golden data not found -> {output_data_path}")
            continue
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)
            is_match = (golden_output == output_data).all()
            print(f"[compare] golden output [{output_name}] match={is_match}, similarity={cosine_dist:.6f}")
            if is_match:
                continue
            if cosine_dist < 0.999:
                result_check = False
        else:
            result_check = False
            print(f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape}")
    print(f'{model_name} get {output_num} ouputs completed in {profile["get_output"]*1000:.3f} ms.')
    if not result_check:
        print("[error] result check failed.")
        exit(-1)
    print(f"<=== {model_name} test success.")


if __name__ == '__main__':
    args = get_args()
    curdir = os.getcwd()
    model_dir = args.model_dir
    model_name = args.model_name
    nblocks = args.nblocks
    output_dir = args.output_dir
    ncore = args.ncore
    batch = args.batch
    profile = {}
 
    # clip model to 2 parts
    raw_path = os.path.join(model_dir, "prefill/hmquant_deepseek_with_act.onnx")
    part1_path = os.path.join(model_dir, "prefill/hmquant_deepseek_part1_with_act.onnx")
    part2_path = os.path.join(model_dir, "prefill/hmquant_deepseek_part2_with_act.onnx")
    clip(raw_path, part1_path, part2_path, nblocks)
    raw_path = os.path.join(model_dir, "decoder/hmquant_deepseek_with_act.onnx")
    part1_path = os.path.join(model_dir, "decoder/hmquant_deepseek_part1_with_act.onnx")
    part2_path = os.path.join(model_dir, "decoder/hmquant_deepseek_part2_with_act.onnx")
    clip(raw_path, part1_path, part2_path, nblocks)

    # build model
    if args.stage == "build" or args.stage == "all":
        model_path = f"prefill/hmquant_{model_name}_part1_with_act.onnx"
        build("deepseek_prefill_part1", model_dir, model_path, output_dir, profile, ncore)
        model_path = f"prefill/hmquant_{model_name}_part2_with_act.onnx"
        build("deepseek_prefill_part2", model_dir, model_path, output_dir, profile, ncore)
        model_path = f"decoder/hmquant_{model_name}_part1_with_act.onnx"
        build("deepseek_decode_part1", model_dir, model_path, output_dir, profile, ncore)
        model_path = f"decoder/hmquant_{model_name}_part2_with_act.onnx"
        build("deepseek_decode_part2", model_dir, model_path, output_dir, profile, ncore)

    # test model
    if args.stage == 'test' or args.stage == 'all':
        part_dir = os.path.join(model_dir, "prefill")
        test("deepseek_prefill_part1", part_dir, output_dir, profile, prefix=model_name)
        test("deepseek_prefill_part2", part_dir, output_dir, profile, prefix=model_name)
        part_dir = os.path.join(model_dir, "decoder")
        test("deepseek_decode_part1", part_dir, output_dir, profile, prefix=model_name)
        test("deepseek_decode_part2", part_dir, output_dir, profile, prefix=model_name)
