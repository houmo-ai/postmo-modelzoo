import os
import numpy as np
import onnx
import argparse


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default=os.path.join('output', os.getenv('HOUMO_TARGET', ''), 'result'),
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
        '--core',
        dest='core',
        type=int,
        default=1,
        help='core number',
    )
    parser.add_argument(
        '--stage',
        dest='stage',
        type=str,
        default="all",
        help='build stage choise=["build", "test", "all"]',
    )
    args = parser.parse_args()
    return args


def build(args=None):
    """build and test houmo model."""
    model_name = args.model_name
    batch = args.batch
    stage = args.stage
    model_dir = args.model_dir
    quant_name = "hmquant_" + model_name + "_with_act"
    onnx_name = quant_name + ".onnx"
    model_path = os.path.join(model_dir, onnx_name)
    model_dir = os.path.dirname(model_path)

    # 1. build model
    if stage == 'build' or stage == 'all':
        import tcim
        print(f"\n===> {model_name} build start...")
        onnx_model = onnx.load(model_path)
        compile_config = {}
        if batch % 4 == 0:
            compile_config["tcim.core_num"] = 4
        input_cfg = {}
        inputs = onnx_model.graph.input
        for input in inputs:
            dims = input.type.tensor_type.shape.dim
            input_shape = [dim.dim_value for dim in dims]
            input_shape[0] *= batch
            input_cfg[input.name] = tcim.HMInput(shape=input_shape)
        tcim.build.build_from_hmonnx(onnx_model, model_name=model_name, inputs=input_cfg, compiler_cfg=compile_config, lite=True)
        print(f"<=== {model_name} build success.")

    # 2. test model
    if stage == 'test' or stage == 'all':
        import tcim_lite
        print(f"\n===> {model_name} test start...")
        # 2.1 load model
        module = tcim_lite.runtime.load(model_name + ".hmm")

        # 2.2 set input with golden
        input_num = module.get_num_inputs()
        for id in range(input_num):
            input_name = module.get_input_name(id)
            input_info = module.get_input_info(input_name)
            print("input_info[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape,
                                                                              input_info.dtype, input_info.format.name))
            input_file_name = 'hmquant_' + model_name + '_' + input_name + '_input.npy'
            input_data_path = os.path.join(model_dir, input_file_name)
            input_data = np.load(input_data_path).astype(input_info.dtype)
            input_data = np.concatenate([input_data for i in range(batch)], axis=0)
            print("golden input[{}] shape = {}, dtype = {}".format(input_name, input_data.shape, input_data.dtype))
            module.set_input(input_name, input_data)

        # 2.3 infer model
        module.run()
        module.sync()

        # 2.4. get output and compare with golden
        result_check = True
        output_num = module.get_num_outputs()
        for id in range(output_num):
            output_name = module.get_output_name(id)
            output_info = module.get_output_info(output_name)
            print("output_info[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape,
                                                                               output_info.dtype, output_info.format.name))
            output_data = module.get_output(output_name).numpy()
            print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
            output_data_path = os.path.join(model_dir, 'hmquant_' + model_name + '_' + output_name + '_output.npy')
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path)
                golden_output = np.concatenate([golden_output for i in range(batch)], axis=0)
            else:
                result_check = False
                print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))
                continue
            if golden_output.shape == output_data.shape:
                from hmassist.utils.dist_metrics import cosine_distance
                cosine_dist = cosine_distance(golden_output, output_data)
                is_match = (golden_output == output_data).all()
                print("[compare] golden output [{}] match={}, similarity={:.6f}"
                            .format(output_name, is_match, cosine_dist))
                if cosine_dist < 0.999:
                    result_check = False
            else:
                result_check = False
                print("[compare] golden output [{}] shape not match {} vs {}"
                            .format(output_name, golden_output.shape, output_data.shape))
        if not result_check:
            print("[error] result check failed.")
            exit(-1)
        print(f"<=== {model_name} test success.")


if __name__ == '__main__':
    args = get_args()
    build(args)
