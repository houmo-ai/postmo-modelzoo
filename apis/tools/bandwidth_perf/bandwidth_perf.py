import argparse
import json
import math
import os
import shutil
import platform
import numpy as np
import onnx
import time
import multiprocessing
from itertools import zip_longest

try:
    import tcim_lite
except ImportError:
    print("Please install tcim_lite")
    exit(-1)


def gen_copy_model_and_data_size():
    """model with graph: transpose"""
    from tcim.test_utils.onnx_builder import make_model, make_transpose_node
    from tcim.test_utils.onnx_builder.common import make_initializer
    from tcim.hmcc_converter.base_hmonnx import HMOnnxModelVersion

    dtype = onnx.TensorProto.INT8
    input_shape = (4, 2, 1048512)
    input_name = "input"

    input_tensor = make_initializer(
        input_name, np.random.randint(-128, 127, input_shape, np.int8), dtype
    )
    current_tensor = input_tensor
    initializers = []
    nodes = []
    mid_tensors = []

    current_tensor = make_transpose_node(
        "transpose1",
        [current_tensor.name],
        ["transpose1_out"],
        perm=[0, 1, 2],
        output_dtype="int8",
        nodes=nodes,
        out_shape=input_shape,
    )
    mid_tensors.append(current_tensor)
    initializers.append(input_tensor)

    model = make_model(
        nodes,
        "test",
        inputs=[],
        outputs=[current_tensor],
        value_info=mid_tensors,
        initializer=initializers,
        opset_imports=[HMOnnxModelVersion.ONNX.get_opset_id()],
    )

    data_size = {"data_size": math.prod(input_shape)}

    return model, data_size


def parse_dtype(dtype: dict):
    bits = dtype.get("bits", 0)
    if dtype.get("code", "") == "int":
        return {8: "int8", 16: "int16", 32: "int32"}[bits]
    if dtype.get("code", "") == "uint":
        return {8: "uint8", 16: "uint16", 32: "uint32"}[bits]
    if dtype.get("code", "") == "float":
        return {16: "float16", 32: "float32"}[bits]
    raise RuntimeError(f"unknown dtype:{dtype}")


def gen_random_data(input_shape, input_dtype):
    if np.issubdtype(input_dtype, np.integer):
        data_size_in_bytes = np.prod(input_shape) * np.dtype(input_dtype).itemsize
        random_int8_data = np.random.randint(
            -128, 128, data_size_in_bytes, dtype=np.int8
        )
        return random_int8_data.view(input_dtype).reshape(input_shape)
    if np.issubdtype(input_dtype, np.floating):
        return np.random.random(input_shape).astype(input_dtype)
    raise ValueError(f"Unsupported input dtype: {input_dtype}")


# pylint: disable=too-many-arguments,too-many-locals
def run_model_wrapper(
    tid,
    did,
    model_path,
    wm,
    input_names,
    input_datas,
    inner_round_num,
    outer_round_num,
    barrier,
    queue_for_start_time,
    verbose,
):

    time.sleep(
        0.1 * tid
    )  # stagger thread start time a bit to avoid loading model at the same time

    # load model, create a stream and set to the model
    option = tcim_lite.runtime.Option(wm)
    run_option = tcim_lite.runtime.RunOption(inner_round_num)
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
    if verbose:
        print(
            f"thread {tid} on device {did} will start run when all threads are ready..."
        )
    barrier.wait()
    queue_for_start_time.put(time.time())
    # infer loop
    for _ in range(outer_round_num):
        model.run(False, run_option)
    model.sync()


# pylint: disable=too-many-arguments
def run_multi_streams(
    hmm_path: str,
    json_path: str,
    process_num: int,
    inner_round_num: int,
    outer_round_num: int,
    device_id: int,
    verbose: bool = False,
):
    # pylint: disable=unused-argument,unused-variable,too-many-locals,redefined-outer-name

    with open(json_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)
        input_shapes = [
            input_info["shape"] for input_info in model_info["Model"]["inputs"]
        ]
        input_names = [
            input_info["name"] for input_info in model_info["Model"]["inputs"]
        ]
        input_dtypes = [
            parse_dtype(input_info["dtype"])
            for input_info in model_info["Model"]["inputs"]
        ]

    queue_for_start_time = multiprocessing.Queue()

    # 1. prepare input data
    input_datas = [
        gen_random_data(input_shape, input_dtype)
        for input_shape, input_dtype in zip_longest(input_shapes, input_dtypes)
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
                inner_round_num,
                outer_round_num,
                barrier,
                queue_for_start_time,
                verbose,
            ),
        )
        processes.append(p)
        tid += 1

    # 4. start processes
    for p in processes:
        p.start()

    # 5. wait for processes to finish
    for p in processes:
        p.join()
    end_time = time.time()

    # 6. get start time
    start_time_list = []
    while not queue_for_start_time.empty():
        start_time_list.append(queue_for_start_time.get())
    start_time = min(start_time_list)  # use the earliest start time

    return end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and run a model to get the bandwidth performance of AI core.",
        usage="python bandwidth_test.py --work-dir ./output",
    )
    parser.add_argument("--work-dir", type=str, default="./output", help="work dir.")
    parser.add_argument(
        "--type", type=str, choices=("r", "w"), default="r", help="bandwidth type."
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="the device id to run the test on."
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="skip model building."
    )
    parser.add_argument("--skip-run", action="store_true", help="skip model running.")
    parser.add_argument(
        "--sample-num",
        type=int,
        default=8,
        help="total number of samples to run, the default value runs about 5 seconds.",
    )
    args = parser.parse_args()

    target = os.getenv("HOUMO_TARGET")
    if target not in ["xh1", "xh2"]:
        raise ValueError("HOUMO_TARGET must be xh1 or xh2")
    MODEL_NAME = f"{target}_transpose"

    bandwidth_type = "Read" if args.type == "r" else "Write"

    print("#########################################")
    print("##        AI core bandwidth test       ##")
    print("#########################################")
    print()

    platform_name = platform.machine()

    output_dir = os.path.join(args.work_dir, target, bandwidth_type.lower())
    build_tmp_dir = os.path.join(output_dir, "tcim_temp")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(build_tmp_dir, exist_ok=True)

    hmm_path = os.path.join(output_dir, f"{MODEL_NAME}.hmm")

    not_found_tcim = False
    try:
        import tcim
    except ImportError:
        not_found_tcim = True

    if platform_name != "x86_64" or not_found_tcim:
        print(
            f"Skipping model generation since the current platform is {platform_name} or not found tcim."
        )
        args.skip_build = True

    # build model
    print("=========================================")
    if not os.path.exists(hmm_path) and args.skip_build:
        print(f"Skipping model generation since download hmm from network.")
        args.skip_build = True
        from hmatc.utils.utils import get_file_from_jfrog

        if "HOUMO_MODELZOO_URL" not in os.environ:
            os.environ["HOUMO_MODELZOO_URL"] = (
                "http://139.224.0.199:8082/artifactory/houmo/release"
            )
        zipped_hmm_path = f"models/bandwidth_perf/hmm_{target}_transpose_{bandwidth_type.lower()}_1core_20251028.tar.xz"
        get_file_from_jfrog(zipped_hmm_path, "./", "./")

    if not args.skip_build:
        print("Generating onnx model...")
        hmonnx_model, model_data_size_info = gen_copy_model_and_data_size()
        model_path = os.path.join(output_dir, f"{MODEL_NAME}.onnx")
        print(f"Model generated successfully, saving to {model_path}.")
        onnx.save(hmonnx_model, model_path)
        model_data_size_path = os.path.join(output_dir, f"{MODEL_NAME}_data_size.json")
        with open(model_data_size_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_data_size_info, indent=4))

        print("=========================================")
        print(f"Building model with tcim in output dir: {output_dir}")
        os.environ["SPLIT_XH2_LOAD"] = "1"
        import tcim

        tcim.build_from_hmonnx(
            hmonnx_model,
            output_name=MODEL_NAME,
            output_dir=output_dir,
            enable_model_connect=False,
            work_dir=build_tmp_dir,
            target=target,
            ncore=1,
            opt_level="O2",
            io_layout="any",
            emit_cpp_extra_args=f"only-emit-op-list=hmint.{'load' if args.type == 'r' else 'store'}",
        )
        json_path = os.path.join(output_dir, "model.json")
        shutil.copyfile(os.path.join(build_tmp_dir, "model.json"), json_path)
        print("Model built successfully.")

    if not args.skip_run:
        INNER_ROUND = 10000
        print("=========================================")
        print(f"Running model {hmm_path}")
        if not os.path.exists(hmm_path):
            raise FileNotFoundError(
                f"Model file {hmm_path} not found, please re-run the build step."
            )
        json_path = os.path.join(output_dir, "model.json")
        PROCESS_NUM = 4
        # warm up
        _ = run_multi_streams(
            hmm_path=hmm_path,
            json_path=json_path,
            process_num=PROCESS_NUM,
            inner_round_num=1,
            outer_round_num=1,
            device_id=args.device_id,
            verbose=False,
        )
        assert args.sample_num % PROCESS_NUM == 0
        SAMPLE_NUM_PER_PROCESS = args.sample_num // PROCESS_NUM
        print(
            f"Running model using {PROCESS_NUM} processes for {SAMPLE_NUM_PER_PROCESS} samples per thread"
        )
        elapsed_time = run_multi_streams(
            hmm_path=hmm_path,
            json_path=json_path,
            process_num=PROCESS_NUM,
            inner_round_num=INNER_ROUND,
            outer_round_num=SAMPLE_NUM_PER_PROCESS,
            device_id=args.device_id,
            verbose=True,
        )
        print(
            f"Model run successfully on device {args.device_id}, elapsed time: {elapsed_time:.2f} seconds."
        )
        model_data_size_path = os.path.join(output_dir, f"{MODEL_NAME}_data_size.json")
        with open(model_data_size_path, "r", encoding="utf-8") as f:
            model_data_size_info = json.load(f)

        data_size_in_MB = model_data_size_info["data_size"] / (1024**2)
        total_round_num = args.sample_num * INNER_ROUND
        bandwidth = (
            model_data_size_info["data_size"] * total_round_num / elapsed_time
        )  # Bytes per second
        bandwidth_GBps = bandwidth / (1024**3)  # GB per second

        print("\n" + "=" * 60)
        print(f"      {bandwidth_type} Bandwidth Test Summary")
        print("=" * 60)

        print(f"{'Model Parameter':<25} | {'Value':>15}")
        print("-" * 60)
        print(f"{'Data size':<25} | {data_size_in_MB:>15.2f} MiB")
        print(f"{'Round number':<25} | {total_round_num:>15d}")
        print(f"{'Test time':<25} | {elapsed_time:>15.4f} seconds")
        print(f"{bandwidth_type + ' Bandwidth':<25} | {bandwidth_GBps:>15.2f} GiB/s")
        print("=" * 60 + "\n")
