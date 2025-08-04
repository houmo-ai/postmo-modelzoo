import argparse
import json
import math
import os
import sys
import time
import platform
import numpy as np
import onnx
import multiprocessing
import tcim_lite

# pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-arguments


class Xh1ConvUtil:
    """utils for xh1 related shapes or constraints"""

    @staticmethod
    def get_kp_shape(kernel_size, input_channel, output_channel):
        def get_part(align_kernel_size):
            if align_kernel_size <= 3:
                return 1
            if 6 <= align_kernel_size:
                return 8
            return 2 if (align_kernel_size == 4 and input_channel <= 64 and output_channel <= 64) else 4

        align_kernel_size = max(kernel_size[0], kernel_size[1])
        co_align_val = 64
        ci_align_val = 512 if align_kernel_size == 1 else 64
        ci1 = math.ceil(input_channel / ci_align_val)
        co1 = math.ceil(output_channel / co_align_val)
        return (co1, ci1, get_part(align_kernel_size), co_align_val)

    @staticmethod
    def get_padding_for_same_fmap_size(input_feature_size, kernel_size, stride):
        total_pad_lower_bound = tuple(
            math.ceil(input_feature_size[i] / stride[i] - 1) * stride[i] + kernel_size[i] - input_feature_size[i]
            for i in range(2)
        )
        total_pad_upper_bound = tuple(total_pad_lower_bound[i] + stride[i] - 1 for i in range(2))
        total_pad_h = total_pad_lower_bound[0] if total_pad_lower_bound[0] >= 0 else total_pad_upper_bound[0]
        total_pad_w = total_pad_lower_bound[1] if total_pad_lower_bound[1] >= 0 else total_pad_upper_bound[1]
        padding_low = (max(0, math.floor(total_pad_h / 2)), max(0, math.floor(total_pad_w / 2)))
        padding_high = (total_pad_h - padding_low[0], total_pad_w - padding_low[1])
        assert len(set(padding_low)) == len(set(padding_high)) == 1
        return padding_low[0], padding_high[0]


def run_model_wrapper(tid, did, model_path, wm, input_datas, round_num, barrier, verbose):
    # load model, create a stream and set to the model
    option = tcim_lite.runtime.Option(wm)
    model = tcim_lite.runtime.load(model_path, option=option)
    if verbose:
        print(f"thread {tid} on device {did} load model done.")
    stream = tcim_lite.runtime.Stream(True)
    model.set_stream(stream)
    # set input to the model
    for input_name, input_data in zip(input_names, input_datas):
        input_info = model.get_input_info(input_name)
        model.set_input(input_name, tcim_lite.runtime.Tensor(input_info, input_data))

    # wait until all threads ready
    barrier.wait()
    # infer loop
    for i in range(round_num):
        model.run()
    model.sync()


def run_multi_streams(
    hmm_path: str, input_shapes, process_num: int, round_num: int, device_id: int, verbose: bool = False
):
    # pylint: disable=unused-argument,unused-variable,too-many-locals,redefined-outer-name

    # 1. prepare input data
    input_datas = [np.random.randint(-128, 128, input_shape, dtype=np.int8) for input_shape in input_shapes]

    # 2. define barrier
    barrier = multiprocessing.Barrier(process_num)

    # 3. create processes
    processes = []
    tid = 0
    for _ in range(process_num):
        wm = tcim_lite.runtime.WeightManager(device_id)
        p = multiprocessing.Process(
            target=run_model_wrapper, args=(tid, device_id, hmm_path, wm, input_datas, round_num, barrier, verbose)
        )
        processes.append(p)
        tid += 1

    # 4. start processes
    start_time = time.time()
    for p in processes:
        p.start()

    # 5. wait for processes to finish
    for p in processes:
        p.join()
    end_time = time.time()

    return end_time - start_time


def gen_conv_model_and_tops():
    """model with graph: conv2d"""
    dtype = onnx.TensorProto.INT8
    batch_size = 1
    channel = 256
    feature_size = (256, 256)
    kernel_size = (3, 3)
    stride = (1, 1)
    num_layers = 64
    padding = Xh1ConvUtil.get_padding_for_same_fmap_size(feature_size, kernel_size, stride)
    input_name = "input"

    input_tensor = make_tensor(input_name, (batch_size, *feature_size, channel), dtype)
    current_tensor = input_tensor
    initializers = []
    nodes = []
    mid_tensors = []

    weight_shape = [*kernel_size, channel, channel]
    bias_shape = (weight_shape[-1],)
    kp_shape = Xh1ConvUtil.get_kp_shape(kernel_size, channel, channel)
    ko_shape = (weight_shape[-1],)
    weight_data = np.random.randint(low=-128, high=128, size=weight_shape, dtype=np.int8)
    bias_data = np.random.randint(low=-128, high=128, size=bias_shape, dtype=np.int16)
    kp_data = np.random.randint(low=0, high=5, size=kp_shape, dtype=np.int8)
    ko_data = np.random.randint(low=0, high=5, size=ko_shape, dtype=np.int8)
    for i in range(num_layers):
        current_layer_name = f"conv{i}"
        current_tensor = make_base_cimd_conv2d_node(
            name=current_layer_name,
            input_names=[current_tensor.name],
            output_names=[current_layer_name],
            weight_shape=weight_shape,
            weight_data=weight_data,
            bias_data=bias_data,
            kp_shape=kp_shape,
            kp_data=kp_data,
            ko_data=ko_data,
            kernel_size=kernel_size,
            padding=padding,
            strides=stride,
            feature_map_layout="NHWC",
            initializers=initializers,
            nodes=nodes,
        )
        mid_tensors.append(current_tensor)

    model = make_model(
        nodes,
        "test",
        inputs=[input_tensor],
        outputs=[current_tensor],
        value_info=mid_tensors,
        initializer=initializers,
        opset_imports=[
            onnx.helper.make_opsetid(GLOBAL_XH1_ONNX_DOMAIN, GLOBAL_XH1_ONNX_VERSION),
        ],
    )

    num_tops = (
        num_layers
        * batch_size
        * channel
        * channel
        * feature_size[0]
        * feature_size[1]
        * kernel_size[0]
        * kernel_size[1]
        * 2
        / (1000**4)
    )

    return model, num_tops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default="./xh1_perf_test", help="work dir.")
    parser.add_argument("--device-id", type=int, default=0, help="the device id to run the test on.")
    parser.add_argument("--skip-build", action="store_true", help="skip model building.")
    parser.add_argument("--skip-run", action="store_true", help="skip model running.")
    args = parser.parse_args()

    MODEL_NAME = "xh1_conv"

    print("#########################################")
    print("##  AI core compute performance test   ##")
    print("#########################################")
    print()

    platform_name = platform.machine()

    output_dir = os.path.join(args.work_dir, "output")
    build_tmp_dir = os.path.join(args.work_dir, "tcim_temp")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(build_tmp_dir, exist_ok=True)

    hmm_path = os.path.join(output_dir, f"{MODEL_NAME}.hmm")

    # build model
    print("=========================================")
    if os.path.exists(hmm_path):
        print(f"Skipping model generation since model already exists in {hmm_path}.")
        args.skip_build = True
    elif platform_name != "x86_64":
        print(f"Skipping model generation since the current platform is {platform_name}.")
        args.skip_build = True

        # download the compiled model for inference
        HOUMO_EXAMPLES_PATH = os.environ.get('HOUMO_EXAMPLES_PATH', '../..')
        sys.path.append(f'{HOUMO_EXAMPLES_PATH}/common/python')
        from utils import get_file_from_jfrog

        if "HOUMO_MODELZOO_URL" not in os.environ:
            os.environ["HOUMO_MODELZOO_URL"] = "http://139.224.0.199:8082/artifactory/houmo/release"
        HOUMO_TARGET = os.environ.get('HOUMO_TARGET', 'houmo')
        model_dir = os.path.join(HOUMO_EXAMPLES_PATH, "models")
        zipped_hmm_path = "models/computing_perf/hmm_xh1_conv_4cores_20250613.zip"
        get_file_from_jfrog(zipped_hmm_path, model_dir, "./")

    if not args.skip_build:
        import tcim
        from tcim.test_utils.onnx_builder.onnx_builder import (
            GLOBAL_XH1_ONNX_DOMAIN,
            GLOBAL_XH1_ONNX_VERSION,
            make_model,
            make_tensor,
        )
        from tcim.test_utils.onnx_builder.xh1_op_builder import make_base_cimd_conv2d_node

        print("Generating onnx model...")
        hmonnx_model, tops_per_sample = gen_conv_model_and_tops()
        model_path = os.path.join(args.work_dir, f"{MODEL_NAME}.onnx")
        print(f"Model generated successfully, saving to {model_path}.")
        onnx.save(hmonnx_model, model_path)
        model_tops_path = os.path.join(build_tmp_dir, f"{MODEL_NAME}_tops.txt")
        with open(model_tops_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(tops_per_sample, indent=4))

        print("=========================================")
        print(f"Building model with tcim in output dir: {output_dir}")
        tcim.build_from_hmonnx(
            hmonnx_model,
            output_name=MODEL_NAME,
            output_dir=output_dir,
            work_dir=build_tmp_dir,
            target="xh1",
            ncore=1,
            opt_level="O2",
            io_layout="any",
        )
        print("Model built successfully.")

    if not args.skip_run:
        print("=========================================")
        print(f"Running model {hmm_path}")
        if not os.path.exists(hmm_path):
            raise FileNotFoundError(f"Model file {hmm_path} not found, please re-run the build step.")
        json_path = os.path.join(build_tmp_dir, "model.json")
        with open(json_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            input_shapes = [input_info["shape"] for input_info in model_info["Model"]["inputs"]]
            input_names = [input_info["name"] for input_info in model_info["Model"]["inputs"]]
        PROCESS_NUM = 4
        # warm up
        _ = run_multi_streams(
            hmm_path=hmm_path,
            input_shapes=input_shapes,
            process_num=PROCESS_NUM,
            round_num=1,
            device_id=args.device_id,
            verbose=False,
        )
        SAMPLE_NUM_PER_THREAD = 64  # about 5 second
        elapsed_time = run_multi_streams(
            hmm_path=hmm_path,
            input_shapes=input_shapes,
            process_num=PROCESS_NUM,
            round_num=SAMPLE_NUM_PER_THREAD,
            device_id=args.device_id,
            verbose=True,
        )
        print(f"Model run successfully on device {args.device_id}, elapsed time: {elapsed_time:.2f} seconds.")
        model_tops_path = os.path.join(build_tmp_dir, f"{MODEL_NAME}_tops.txt")
        with open(model_tops_path, "r", encoding="utf-8") as f:
            tops_per_sample = json.load(f)

        print("=========================================")
        print(f"Performance is {tops_per_sample * PROCESS_NUM * SAMPLE_NUM_PER_THREAD / elapsed_time:.2f} TOPS.\n")
