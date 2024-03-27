import os
import numpy as np
import onnx
import argparse
import tcim
from hmassist.utils.dist_metrics import cosine_distance


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='output/H30/result',
        help='path to the model dir',
    )
    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='mobilenetv2',
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
        '--stage',
        dest='stage',
        type=str,
        default="all",
        help='build stage choise=["build", "test", "all"]',
    )
    args = parser.parse_args()
    return args


def build(args=None):
    # build model
    format = 'YUV422SP'
    model_name = args.model_name
    batch = args.batch
    stage = args.stage
    model_dir = args.model_dir
    quant_name = "hmquant_" + model_name + "_with_act"
    onnx_name = quant_name + ".onnx"
    model_path = os.path.join(model_dir, onnx_name)
    model_dir = os.path.dirname(model_path)

    # 1. build model
    onnx_model = onnx.load(model_path)
    inputs = onnx_model.graph.input
    if stage == 'build' or stage == 'all':
        compile_config={}
        if batch > 1:
            compile_config["tcim.spec_batch_num"] = batch
        tcim.build.build_from_hmonnx(onnx_model, model_name=model_name, compiler_cfg=compile_config)
        print(model_name + ' build completed.')

    # 2. test model
    if stage == 'test' or stage == 'all':
        # 2.1 load model
        module = tcim.runtime.load(model_name + ".hmm.so")

        # 2.2 set input with golden
        input_num = module.get_num_inputs()
        assert(len(inputs) == input_num)
        for input in inputs:
            input_name = input.name
            # id = module.get_input_index(input_name)
            # name = module.get_input_name_by_index(id)
            input_info = module.get_input(0).numpy()
            print("input[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape, input_info.dtype, format))
            input_file_name = 'hmquant_' + model_name + '_' + input_name + '_input.npy'
            input_data_path = os.path.join(model_dir, input_file_name)
            input_data = np.load(input_data_path).astype("int8")
            input_data = np.concatenate([input_data for i in range(batch)], axis=0)
            print("golden input[{}] shape = {}, dtype = {}".format(input_name, input_data.shape, input_data.dtype))
            module.set_input(input_name, input_data, format)

        # 2.3 infer model
        module.run()

        # 2.4. get output and compare with golden
        result_check = True
        output_num = module.get_num_outputs()
        for id in range(0, output_num):
            output_name = module.get_output_name_by_index(id)
            output_data = module.get_output_by_name(output_name).numpy()
            print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
            output_data_path = os.path.join(model_dir, 'hmquant_' + model_name + '_with_act', output_name + '.npy')
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path, allow_pickle=True).item().get("output_tensor")
                golden_output = np.concatenate([golden_output for i in range(batch)], axis=0)
            else:
                result_check = False
                print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))
            if golden_output.shape == output_data.shape:
                cosine_dist = cosine_distance(golden_output, output_data)
                is_match = (golden_output == output_data).all()
                print("[compare] golden output [{}] match={}, similarity={:.6f}"
                            .format(output_name, is_match, cosine_dist))
                if cosine_dist < 0.99:
                    result_check = False
            else:
                result_check = False
                print("[compare] golden output [{}] shape not match {} vs {}"
                            .format(output_name, golden_output.shape, output_data.shape))
        if not result_check:
            exit(-1)


if __name__ == '__main__':
    args = get_args()
    build(args)
