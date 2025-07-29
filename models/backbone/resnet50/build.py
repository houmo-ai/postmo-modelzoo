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
        default='resnet50',
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
        default=1,
        help='core number',
    )
    parser.add_argument(
        '--input_shape',
        dest='input_shape',
        type=lambda s:[int(item) for item in s.split(',')],
        default=None,
        help='new input shape if want change',
    )
    parser.add_argument(
        '--dynamic_resize',
        dest='dynamic_resize',
        action='store_true',
        help='set dynamic crop/resize/pad',
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
    parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
        help='print details',
    )
    args = parser.parse_args()
    return args


def build(args=None):
    """build and test houmo model."""
    model_dir = args.model_dir
    model_name = args.model_name
    batch = args.batch
    ncore = args.ncore
    input_shape = args.input_shape
    enable_dynamic_image_resize = args.dynamic_resize
    stage = args.stage
    output_dir= args.output_dir
    verbose = args.verbose
    quant_name = "hmquant_" + model_name + "_with_act"
    onnx_name = quant_name + ".onnx"
    onnx_path = os.path.join(model_dir, onnx_name)
    hmm_path = os.path.join(output_dir, f"{model_name}.hmm")
    profile = {}
    
    # arg check
    if enable_dynamic_image_resize:
        if input_shape is None:
            raise RuntimeError("input_shape should be set when dynamic_resize is set.")

    # 1. build model
    if stage == 'build' or stage == 'all':
        import tcim
        print(f"\n===> {model_name} build start...")
        start = time.time()
        tcim.build_from_hmonnx(
            onnx_path,
            output_name=model_name,
            ncore=ncore,
            output_dir=output_dir,
            work_dir=os.path.join(output_dir, "tcim"),
            enable_dynamic_image_resize=enable_dynamic_image_resize,
        )
        profile["build"] = time.time() - start
        print(f'{model_name} build completed in {profile["build"]:.3f} s.')

    # 2. test model
    if stage == 'test' or stage == 'all':
        import tcim_lite
        print(f"\n===> {model_name} test start...")
        # 2.1 load model
        start = time.time()
        module = tcim_lite.runtime.load(hmm_path)
        profile["load"] = time.time() - start
        print(f'{model_name} load completed in {profile["load"]*1000:.3f} ms.')

        # 2.2 set input with golden
        profile["set_input"] = 0
        input_num = module.get_num_inputs()
        print("input_num:", input_num)
        for id in range(input_num):
            input_name = module.get_input_name(id)
            input_info = module.get_input_info(input_name)
            print(f"input_info[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}")
            if "resizer_crop_" in input_name:
                crop = [0, 0, input_shape[2], input_shape[3]]  # y1, x1, h, w
                resize = [224, 224]  # h, w
                pad = [0, 0, 0, 0]  # top, left, bottom, right
                input_data = np.concatenate((crop, resize, pad))
            else:
                input_data_path = os.path.join(model_dir, f"hmquant_{model_name}_{sanitize_name(input_name)}_input.npy")
                input_data = np.load(input_data_path).astype(input_info.dtype)
            input_data = np.concatenate([input_data for i in range(batch)], axis=0)
            print(f"golden input[{input_name}] shape = {input_data.shape}, dtype = {input_data.dtype}")
            start = time.time()
            module.set_input(input_name, input_data)
            profile["set_input"] += time.time() - start
        print(f'{model_name} set {input_num} inputs completed in {profile["set_input"]*1000:.3f} ms.')

        # 2.3 infer model
        start = time.time()
        module.run()
        module.sync()
        profile["infer"] = time.time() - start
        print(f'{model_name} infer completed in {profile["infer"]*1000:.3f} ms.')

        # 2.4. get output and compare with golden
        result_check = True
        profile["get_output"] = 0
        profile["dequant"] = 0
        output_num = module.get_num_outputs()
        print("output_num:", output_num)
        for id in range(output_num):
            output_name = module.get_output_name(id)
            output_info = module.get_output_info(output_name)
            print(f"output_info[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}")
            start = time.time()
            output_data = module.get_output(output_name)
            profile["get_output"] += time.time() - start
            start = time.time()
            dequanted_data = output_data.cast(np.float32)
            profile["dequant"] += time.time() - start
            output_data = output_data.numpy()
            dequanted_data = dequanted_data.numpy()
            print(f"output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}")
            print(f"dequanted output[{output_name}] shape = {dequanted_data.shape}, dtype = {dequanted_data.dtype}")
            output_data_path = os.path.join(model_dir, f'hmquant_{model_name}_{sanitize_name(output_name)}_output.npy')
            dequanted_data_path = os.path.join(model_dir, f'hmquant_{model_name}_{sanitize_name(output_name)}_dequant_output.npy')
            if os.path.exists(output_data_path) and os.path.exists(dequanted_data_path):
                golden_output = np.load(output_data_path)
                golden_dequanted = np.load(dequanted_data_path)
                golden_output = np.concatenate([golden_output for i in range(batch)], axis=0)
                golden_dequanted = np.concatenate([golden_dequanted for i in range(batch)], axis=0)
            elif not os.path.exists(output_data_path):
                print(f"[warning] compare canceled while golden data not found -> {output_data_path}")
                result_check &= False
                continue
            elif not os.path.exists(dequanted_data_path):
                print(f"[warning] compare canceled while golden data not found -> {dequanted_data_path}")
                result_check &= False
                continue
            if golden_output.shape == output_data.shape and golden_dequanted.shape == dequanted_data.shape:
                cosine_dist1 = cosine_distance(golden_output, output_data)
                is_match1 = (golden_output == output_data).all()
                print(f"[compare] golden output [{output_name}] match={is_match1}, similarity={cosine_dist1:.6f}")
                cosine_dist2 = cosine_distance(golden_dequanted, dequanted_data)
                is_match2 = (golden_dequanted == dequanted_data).all()
                print(f"[compare] dequanted golden output [{output_name}] match={is_match2}, similarity={cosine_dist2:.6f}")
                if is_match1 and is_match2:
                    continue
                if cosine_dist1 < 0.999:
                    result_check &= False
                    if verbose:
                        print("output_data:\n", output_data)
                        print("golden_output:\n", golden_output)
                if cosine_dist2 < 0.999:
                    result_check &= False
                    if verbose:
                        print("dequanted_data:\n", dequanted_data)
                        print("golden_dequanted:\n", golden_dequanted)
            else:
                result_check &= False
                print(f"[compare] golden output [{output_name}] shape not match {golden_output.shape} vs {output_data.shape},",
                      f"{golden_dequanted.shape} vs {dequanted_data.shape}")
        print(f'{model_name} get {output_num} ouputs completed in {profile["get_output"]*1000:.3f} ms.')
        print(f'{model_name} {output_num} dequants completed in {profile["dequant"]*1000:.3f} ms.')
        if not result_check:
            raise RuntimeError("[error] result check failed.")
        print(f"<=== {model_name} test success.")


if __name__ == '__main__':
    import platform
    arch = platform.machine()
    if arch != "x86_64":
        print(f"[error] tcim not support platform: {arch}")
        exit(0)
    args = get_args()
    print(args)
    build(args)
