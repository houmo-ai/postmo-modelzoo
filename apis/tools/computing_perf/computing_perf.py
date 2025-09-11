import os
import sys
import time
import math
import json
import onnx
import logging
import argparse
import platform
import tcim_lite
import multiprocessing
import numpy as np
import tcim
from tcim.test_utils.onnx_builder.onnx_builder import make_model, make_tensor
from tcim.test_utils.onnx_builder.common import make_initializer
from tcim.hmcc_converter.utils import convert_to_onnx_dtype

logging.basicConfig(level=logging.INFO)


class Xh1ConvUtil:
    """utils for xh1 related shapes or constraints"""

    @staticmethod
    def get_kp_shape(kernel_size, input_channel, output_channel):
        def get_part(align_kernel_size):
            if align_kernel_size <= 3:
                return 1
            if 6 <= align_kernel_size:
                return 8
            return (
                2
                if (
                    align_kernel_size == 4
                    and input_channel <= 64
                    and output_channel <= 64
                )
                else 4
            )

        align_kernel_size = max(kernel_size[0], kernel_size[1])
        co_align_val = 64
        ci_align_val = 512 if align_kernel_size == 1 else 64
        ci1 = math.ceil(input_channel / ci_align_val)
        co1 = math.ceil(output_channel / co_align_val)
        return (co1, ci1, get_part(align_kernel_size), co_align_val)

    @staticmethod
    def get_padding_for_same_fmap_size(input_feature_size, kernel_size, stride):
        total_pad_lower_bound = tuple(
            math.ceil(input_feature_size[i] / stride[i] - 1) * stride[i]
            + kernel_size[i]
            - input_feature_size[i]
            for i in range(2)
        )
        total_pad_upper_bound = tuple(
            total_pad_lower_bound[i] + stride[i] - 1 for i in range(2)
        )
        total_pad_h = (
            total_pad_lower_bound[0]
            if total_pad_lower_bound[0] >= 0
            else total_pad_upper_bound[0]
        )
        total_pad_w = (
            total_pad_lower_bound[1]
            if total_pad_lower_bound[1] >= 0
            else total_pad_upper_bound[1]
        )
        padding_low = (
            max(0, math.floor(total_pad_h / 2)),
            max(0, math.floor(total_pad_w / 2)),
        )
        padding_high = (total_pad_h - padding_low[0], total_pad_w - padding_low[1])
        assert len(set(padding_low)) == len(set(padding_high)) == 1
        return padding_low[0], padding_high[0]


class Xh2ConvUtil:
    """utils for xh2 related shapes or constraints"""

    @staticmethod
    def get_padding_for_same_fmap_size(input_feature_size, kernel_size, stride):
        total_pad_lower_bound = tuple(
            math.ceil(input_feature_size[i] / stride[i] - 1) * stride[i]
            + kernel_size[i]
            - input_feature_size[i]
            for i in range(2)
        )
        total_pad_upper_bound = tuple(
            total_pad_lower_bound[i] + stride[i] - 1 for i in range(2)
        )
        total_pad_h = (
            total_pad_lower_bound[0]
            if total_pad_lower_bound[0] >= 0
            else total_pad_upper_bound[0]
        )
        total_pad_w = (
            total_pad_lower_bound[1]
            if total_pad_lower_bound[1] >= 0
            else total_pad_upper_bound[1]
        )
        padding_low = (
            max(0, math.floor(total_pad_h / 2)),
            max(0, math.floor(total_pad_w / 2)),
        )
        padding_high = (total_pad_h - padding_low[0], total_pad_w - padding_low[1])
        assert len(set(padding_low)) == len(set(padding_high)) == 1
        return padding_low[0], padding_high[0]


def make_conv2d_node(
    name,
    input_names,
    output_names,
    kernel_shape,
    out_channels,
    pads,
    strides,
    weight_data,
    bias_data,
    batch_size,
    out_feature_shape,
    initializers,
    nodes,
    output_dtype="float32",
):

    w_name = name + ".quant_weight"
    b_name = name + ".quant_bias"
    initializers.extend(
        [
            make_initializer(w_name, weight_data),  # weight
            make_initializer(b_name, bias_data),  # bias
        ]
    )
    output_shape = [batch_size, out_channels, *out_feature_shape]

    conv_node = onnx.helper.make_node(
        "Conv",
        inputs=[input_names[0], w_name, b_name],
        outputs=output_names,
        name=name,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )
    nodes.append(conv_node)

    out_tensor = onnx.helper.make_tensor_value_info(
        output_names[0], convert_to_onnx_dtype(output_dtype), output_shape
    )
    return out_tensor


def make_transpose_node(
    name, input_names, output_names, axes, output_dtype, nodes, **kwargs
):
    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=input_names,
        outputs=output_names,
        name=name,
        axes=axes,
    )
    nodes.append(transpose_node)
    out_tensor = onnx.helper.make_tensor_value_info(
        output_names[0],
        convert_to_onnx_dtype(output_dtype),
        kwargs.get("out_shape", []),
    )
    return out_tensor


def gen_conv_model_and_tops_xh1():
    """model with graph: conv2d"""
    dtype = onnx.TensorProto.INT8
    batch_size = 1
    channel = 256
    feature_size = (256, 256)
    kernel_size = (3, 3)
    stride = (1, 1)
    num_layers = 64
    padding = Xh1ConvUtil.get_padding_for_same_fmap_size(
        feature_size, kernel_size, stride
    )
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
    weight_data = np.random.randint(
        low=-128, high=128, size=weight_shape, dtype=np.int8
    )
    bias_data = np.random.randint(low=-128, high=128, size=bias_shape, dtype=np.int16)
    kp_data = np.random.randint(low=0, high=5, size=kp_shape, dtype=np.int8)
    ko_data = np.random.randint(low=0, high=5, size=ko_shape, dtype=np.int8)
    from tcim.test_utils.onnx_builder.xh1_op_builder import (
        make_base_cimd_conv2d_node,
    )

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

    from tcim.test_utils.onnx_builder.onnx_builder import (
        GLOBAL_XH1_ONNX_DOMAIN,
        GLOBAL_XH1_ONNX_VERSION,
    )

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
    model_tops_info = {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "in_channel": channel,
        "out_channel": channel,
        "feature_size": feature_size,
        "kernel_shape": kernel_size,
    }
    model_tops_info["num_tops"] = num_tops
    return model, model_tops_info


def gen_conv_model_and_tops_xh2():
    """model with graph: conv2d"""
    dtype = onnx.TensorProto.FLOAT16
    batch_size = 1
    channel = 256
    feature_size = (64, 64)
    kernel_shape = (3, 3)
    strides = (1, 1)
    num_layers = 64
    padding = Xh2ConvUtil.get_padding_for_same_fmap_size(
        feature_size, kernel_shape, strides
    )
    input_name = "input"

    input_tensor = make_tensor(input_name, (batch_size, *feature_size, channel), dtype)
    current_tensor = input_tensor
    initializers = []
    nodes = []
    mid_tensors = []

    weight_shape = [*kernel_shape, channel, channel]
    bias_shape = (weight_shape[-1],)
    weight_data = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=[
            weight_shape[3],
            weight_shape[2],
            weight_shape[0],
            weight_shape[1],
        ],  # co, ci, kh, kw
    ).astype(np.float16)
    bias_data = np.random.uniform(low=-10, high=10, size=bias_shape).astype(np.float16)
    # permute from NHWC to NCHW
    current_tensor = make_transpose_node(
        "nhwc_to_nchw",
        input_names=[current_tensor.name],
        output_names=["nhwc_to_nchw"],
        axes=[0, 3, 1, 2],
        out_shape=(batch_size, channel, *feature_size),
        nodes=nodes,
        output_dtype="float16",
    )
    mid_tensors.append(current_tensor)
    for i in range(num_layers):
        current_layer_name = f"conv{i}"
        current_tensor = make_conv2d_node(
            name=current_layer_name,
            input_names=[current_tensor.name],
            output_names=[current_layer_name],
            kernel_shape=kernel_shape,
            out_channels=channel,
            pads=[padding[0], padding[0], padding[1], padding[1]],
            strides=strides,
            weight_data=weight_data,
            bias_data=bias_data,
            batch_size=batch_size,
            out_feature_shape=feature_size,
            initializers=initializers,
            nodes=nodes,
            output_dtype="float16",
        )
        mid_tensors.append(current_tensor)

    # permute back to NHWC
    current_tensor = make_transpose_node(
        "nchw_to_nhwc",
        input_names=[current_tensor.name],
        output_names=["nchw_to_nhwc"],
        axes=[0, 2, 3, 1],
        out_shape=(batch_size, *feature_size, channel),
        nodes=nodes,
        output_dtype="float16",
    )
    mid_tensors.append(current_tensor)

    from tcim.hmcc_converter.base_hmonnx import HMOnnxModelVersion

    opset_import = onnx.OperatorSetIdProto()
    opset_import.version = HMOnnxModelVersion.ONNX.get_opset_version()
    opset_import.domain = "ai.onnx"

    model = make_model(
        nodes,
        "test",
        inputs=[input_tensor],
        outputs=[current_tensor],
        value_info=mid_tensors,
        initializer=initializers,
        opset_imports=[HMOnnxModelVersion.ONNX.get_opset_id()],
    )

    model_tops_info = {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "in_channel": channel,
        "out_channel": channel,
        "feature_size": feature_size,
        "kernel_shape": kernel_shape,
    }
    model_tops_info["num_tops"] = (
        num_layers
        * batch_size
        * channel
        * channel
        * feature_size[0]
        * feature_size[1]
        * kernel_shape[0]
        * kernel_shape[1]
        * 2
        / (1000**4)
    )

    return model, model_tops_info


def run_model_wrapper(
    tid, did, model_path, wm, input_names, input_datas, round_num, barrier, verbose
):
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
    hmm_path: str,
    input_names,
    input_shapes,
    process_num: int,
    round_num: int,
    device_id: int,
    verbose: bool = False,
):
    # pylint: disable=unused-argument,unused-variable,too-many-locals,redefined-outer-name

    # 1. prepare input data
    input_datas = [
        np.random.randint(-128, 128, input_shape, dtype=np.int8)
        for input_shape in input_shapes
    ]

    # 2. define barrier
    barrier = multiprocessing.Barrier(process_num)

    # 3. create processes
    processes = []
    tid = 0
    for _ in range(process_num):
        wm = tcim_lite.runtime.WeightManager(device_id)
        p = multiprocessing.Process(
            target=run_model_wrapper,
            args=(
                tid,
                device_id,
                hmm_path,
                wm,
                input_names,
                input_datas,
                round_num,
                barrier,
                verbose,
            ),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default="./output", help="work dir.")
    parser.add_argument(
        "--device_id", type=int, default=0, help="the device id to run the test on."
    )
    parser.add_argument(
        "--skip_build", action="store_true", help="skip model building."
    )
    parser.add_argument("--skip_run", action="store_true", help="skip model running.")
    parser.add_argument(
        "--sample_num",
        type=int,
        default=512,
        help="total number of samples to run, the default value runs about 5 seconds.",
    )
    args = parser.parse_args()

    target = os.getenv("HOUMO_TARGET")
    if target not in ["xh1", "xh2"]:
        raise ValueError("HOUMO_TARGET must be xh1 or xh2")

    if os.getenv("HDPL_PLATFORM", "ASIC") == "ISIM":
        print("ISIM does not support this tool.")
        exit(0)

    MODEL_NAME = f"{target}_conv"
    print("#########################################")
    print("##  AI core compute performance test   ##")
    print("#########################################")
    print()

    platform_name = platform.machine()
    output_dir = os.path.join(args.work_dir, target)
    build_tmp_dir = os.path.join(output_dir, "tcim_temp")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(build_tmp_dir, exist_ok=True)

    hmm_path = os.path.join(output_dir, f"{MODEL_NAME}.hmm")

    # build model
    print("=========================================")
    if os.path.exists(hmm_path):
        print(f"Skipping model generation since model already exists in {hmm_path}.")
        args.skip_build = True
    elif platform_name != "x86_64":
        print(
            f"Skipping model generation since the current platform is {platform_name}."
        )
        args.skip_build = True
        # download the compiled model for inference
        HOUMO_EXAMPLES_PATH = os.environ.get("HOUMO_EXAMPLES_PATH", "../..")
        from hmatc.utils.utils import get_file_from_jfrog

        if "HOUMO_MODELZOO_URL" not in os.environ:
            os.environ["HOUMO_MODELZOO_URL"] = (
                "http://139.224.0.199:8082/artifactory/houmo/release"
            )
        if target == "xh1":
            zipped_hmm_path = "models/computing_perf/hmm_xh1_conv_1core_20250908.zip"
        elif target == "xh2":
            zipped_hmm_path = "models/computing_perf/hmm_xh2_conv_1core_20250908.zip"
        get_file_from_jfrog(zipped_hmm_path, "./", "./")

    if not args.skip_build:
        if target == "xh1":
            gen_conv_model_and_tops = gen_conv_model_and_tops_xh1
        elif target == "xh2":
            gen_conv_model_and_tops = gen_conv_model_and_tops_xh2
        print("Generating onnx model...")
        hmonnx_model, model_tops_info = gen_conv_model_and_tops()
        model_path = os.path.join(output_dir, f"{MODEL_NAME}.onnx")
        print(f"Model generated successfully, saving to {model_path}.")
        onnx.save(hmonnx_model, model_path)
        model_tops_path = os.path.join(build_tmp_dir, f"{MODEL_NAME}_tops.json")
        with open(model_tops_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_tops_info, indent=4))
        print("=========================================")
        print(f"Building model with tcim in output dir: {output_dir}")
        tcim.build_from_hmonnx(
            hmonnx_model,
            output_name=MODEL_NAME,
            output_dir=output_dir,
            work_dir=build_tmp_dir,
            target=target,
            ncore=1,
            opt_level="O2",
            io_layout="any",
        )
        print("Model built successfully.")

    if not args.skip_run:
        print("=========================================")
        print(f"Running model {hmm_path}")
        if not os.path.exists(hmm_path):
            raise FileNotFoundError(
                f"Model file {hmm_path} not found, please re-run the build step."
            )
        json_path = os.path.join(build_tmp_dir, "model.json")
        with open(json_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            input_shapes = [
                input_info["shape"] for input_info in model_info["Model"]["inputs"]
            ]
            input_names = [
                input_info["name"] for input_info in model_info["Model"]["inputs"]
            ]
        PROCESS_NUM = 2 * int(os.getenv("HOUMO_CORE_NUM", "2"))
        # warm up
        _ = run_multi_streams(
            hmm_path=hmm_path,
            input_names=input_names,
            input_shapes=input_shapes,
            process_num=PROCESS_NUM,
            round_num=1,
            device_id=args.device_id,
            verbose=False,
        )
        assert args.sample_num % PROCESS_NUM == 0
        SAMPLE_NUM_PER_THREAD = args.sample_num // PROCESS_NUM
        print(
            f"Running model using {PROCESS_NUM} threads for {SAMPLE_NUM_PER_THREAD} samples per thread"
        )
        elapsed_time = run_multi_streams(
            hmm_path=hmm_path,
            input_names=input_names,
            input_shapes=input_shapes,
            process_num=PROCESS_NUM,
            round_num=SAMPLE_NUM_PER_THREAD,
            device_id=args.device_id,
            verbose=True,
        )
        print(
            f"Model run successfully on device {args.device_id}, elapsed time: {elapsed_time:.2f} seconds."
        )
        model_tops_path = os.path.join(build_tmp_dir, f"{MODEL_NAME}_tops.json")
        with open(model_tops_path, "r", encoding="utf-8") as f:
            model_tops_info = json.load(f)
        TOPS = (
            model_tops_info["num_tops"]
            * PROCESS_NUM
            * SAMPLE_NUM_PER_THREAD
            / elapsed_time
        )
        print("\n" + "=" * 60)
        print("           Performance Test Summary")
        print("=" * 60)

        print(f"{'Model Parameter':<25} | {'Value':>15}")
        print("-" * 60)
        print(f"{'Number of Conv2d Layers':<25} | {model_tops_info['num_layers']:>15}")
        print(f"{'Batch Size':<25} | {model_tops_info['batch_size']:>15}")
        print(f"{'Input Channels':<25} | {model_tops_info['in_channel']:>15}")
        print(f"{'Output Channels':<25} | {model_tops_info['out_channel']:>15}")
        print(f"{'Feature Map Size':<25} | {str(model_tops_info['feature_size']):>15}")
        print(f"{'Kernel Shape':<25} | {str(model_tops_info['kernel_shape']):>15}")
        print("-" * 60)
        print(
            f"{'Computing amount':<25} | {model_tops_info['num_tops']:>15.4f} TOPs/sample"
        )
        print(f"{'Test time':<25} | {elapsed_time:>15.4f} seconds")
        print(f"{'PERFORMANCE':<25} | {TOPS:>15.2f} TOPS")
        print("=" * 60 + "\n")
