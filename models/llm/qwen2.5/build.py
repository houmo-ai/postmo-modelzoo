import os
import numpy as np
import time
import argparse

import logging
logging.basicConfig(level="ERROR")

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
        default='qwen2.5',
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
        default=28,
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


def build(model_name, model_dir, model_path, output_dir, profile, ncore=1):
    import tcim
    start = time.time()
    print(f"\n===> {model_name} build start...")
    decode_model = os.path.join(model_dir, model_path)
    tcim.build_from_hmonnx(
        decode_model,
        weights=os.path.join(model_dir, "weight.npy"),
        model_name=model_name,
        ncore=ncore,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim"),
        op_version={'Gather':1}
    )
    profile["build"] = time.time() - start
    print(f'{model_name} build completed in {profile["build"]:.3f} s.', flush=True)


def test(model_name, model_dir, output_dir, profile, batch=1, prefix=None):
    import tcim_lite
    print(f"\n===> {model_name} test start...")
    # load model
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    start = time.time()
    module = tcim_lite.runtime.load(model_path)
    profile["load"] = time.time() - start
    print(f'{model_name} load completed in {profile["load"]:.3f} s.', flush=True)

    # set input
    profile["set_input"] = 0
    if prefix is None:
        prefix = model_name
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print("input_info[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape,
                                                                          input_info.dtype, input_info.format.name))
        input_data_path = os.path.join(model_dir, f"hmquant_{model_name}_{sanitize_name(input_name)}_input.npy")
        input_data = np.load(input_data_path).astype(input_info.dtype)
        input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        print("golden input[{}] shape = {}, dtype = {}".format(input_name, input_data.shape, input_data.dtype))
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
        print("output_info[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape,
                                                                            output_info.dtype, output_info.format.name))
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        profile["get_output"] += time.time() - start
        print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
        output_data_path = os.path.join(model_dir, 'hmquant_' + prefix + '_' + output_name + '_output.npy')
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            golden_output = np.concatenate([golden_output for i in range(batch)], axis=0)
        else:
            result_check = False
            print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))
            continue
        if golden_output.shape == output_data.shape:
            cosine_dist = cosine_distance(golden_output, output_data)
            is_match = (golden_output == output_data).all()
            print("[compare] golden output [{}] match={}, similarity={:.6f}"
                    .format(output_name, is_match, cosine_dist))
            if is_match:
                continue
            if cosine_dist < 0.999:
                result_check = False
        else:
            result_check = False
            print("[compare] golden output [{}] shape not match {} vs {}"
                    .format(output_name, golden_output.shape, output_data.shape))
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

    # build model
    if args.stage == "build" or args.stage == "all":
        model_path = f"prefill/hmquant_{model_name}_with_act.onnx"
        build("qwen2.5_prefill", model_dir, model_path, output_dir, profile, ncore)
        model_path = f"decoder/hmquant_{model_name}_with_act.onnx"
        build("qwen2.5_decode", model_dir, model_path, output_dir, profile, ncore)

    # test model
    if args.stage == 'test' or args.stage == 'all':
        part_dir = os.path.join(model_dir, "prefill")
        test("qwen2.5_prefill", part_dir, output_dir, profile, prefix=model_name)
        part_dir = os.path.join(model_dir, "decoder")
        test("qwen2.5_decode", part_dir, output_dir, profile, prefix=model_name)
