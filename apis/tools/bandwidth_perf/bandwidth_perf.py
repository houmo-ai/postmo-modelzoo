import argparse
import json
import math
import os
import shutil
import platform
import numpy as np
import onnx

try:
    import tcim
except ImportError:
    print("Please install tcim")
    exit(0)
from tcim.hmcc_converter.base_hmonnx import HMOnnxModelVersion
from tcim.runner.multi_stream_runner import run_multi_streams
from tcim.test_utils.onnx_builder.common import make_initializer
from tcim.test_utils.utils import DeviceLock

# pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-arguments


def gen_copy_model_and_data_size():
    """model with graph: transpose"""
    dtype = onnx.TensorProto.INT8
    input_shape = (4, 2, 1048576)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and run a model to get the bandwidth performance of AI core.",
        usage="python bandwidth_test.py --work-dir ./output",
    )
    parser.add_argument("--work-dir", type=str, default="./output", help="work dir.")
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

    print("#########################################")
    print("##        AI core bandwidth test       ##")
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
        from hmatc.utils.utils import get_file_from_jfrog

        if "HOUMO_MODELZOO_URL" not in os.environ:
            os.environ["HOUMO_MODELZOO_URL"] = (
                "http://139.224.0.199:8082/artifactory/houmo/release"
            )
        if target == "xh1":
            zipped_hmm_path = (
                "models/bandwidth_perf/hmm_xh1_transpose_1core_20250916.tar.xz"
            )
        elif target == "xh2":
            zipped_hmm_path = (
                "models/bandwidth_perf/hmm_xh2_transpose_1core_20250916.tar.xz"
            )
        get_file_from_jfrog(zipped_hmm_path, "./", "./")

    if not args.skip_build:
        import tcim
        from tcim.test_utils.onnx_builder.onnx_builder import (
            make_model,
        )
        from tcim.hmcc_converter.base_hmonnx import HMOnnxModelVersion
        from tcim.test_utils.onnx_builder import make_transpose_node

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
        tcim.build_from_hmonnx(
            hmonnx_model,
            output_name=MODEL_NAME,
            output_dir=output_dir,
            work_dir=build_tmp_dir,
            target=target,
            ncore=1,
            opt_level="O2",
            io_layout="any",
            emit_cpp_extra_args="only-emit-op-list=hmint.load",
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
        with DeviceLock("xh2", args.device_id, "bandwidth pref", False):
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
        print("           Bandwidth Test Summary")
        print("=" * 60)

        print(f"{'Model Parameter':<25} | {'Value':>15}")
        print("-" * 60)
        print(f"{'Data size':<25} | {data_size_in_MB:>15.2f} MiB")
        print(f"{'Round number':<25} | {total_round_num:>15d}")
        print(f"{'Test time':<25} | {elapsed_time:>15.4f} seconds")
        print(f"{'BANDWIDTH':<25} | {bandwidth_GBps:>15.2f} GiB/s")
        print("=" * 60 + "\n")
