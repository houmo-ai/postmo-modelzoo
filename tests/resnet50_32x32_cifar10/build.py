import os
import numpy as np
import onnx
import argparse
import tvm
import tvm.relay as relay
import tvm.tcim as tcim
from tvm.relay.backend import Executor
from hmassist.utils.dist_metrics import cosine_distance


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        default='output/H30/result/hmquant_resnet50_32x32_cifar10_with_act.onnx',
        help='path to the model path',
    )
    parser.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        default='resnet50_32x32_cifar10',
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


def build(args):
    # build model
    image_format = 'YUV422SP'
    model_name = args.model_name
    batch = args.batch
    model_path = args.model_path
    stage = args.stage
    model_dir = os.path.dirname(model_path)
    shape_dict = {}
    layout_dict = {}
    convert_config = {}
    
    onnx_model = onnx.load(model_path)
    inputs = onnx_model.graph.input

    if stage == 'build' or stage == 'all':
        for input in inputs:
            dims = input.type.tensor_type.shape.dim
            input_shape = (
                batch * dims[0].dim_value, dims[1].dim_value,
                dims[2].dim_value, dims[3].dim_value,
            )
            print('input name:', input.name)
            print('input shape:', input_shape)
            shape_dict[input.name] = input_shape

        mod = relay.frontend.from_hmonnx(
            onnx_model, shape_dict, layout=layout_dict,
            resizer_attr=None, convert_config=convert_config,
        )
        executor = Executor('aot')
        compile_config = {
            'tcim.fuse_strategy': 1,
            'tcim.codegen_pic': True,
            "tcim.for_benchmark": True
        }
        target = tvm.target.Target('hdpl', host='c')
        with tvm.transform.PassContext(opt_level=3, config=compile_config):
            graph, lib, params = relay.build(
                mod, target, executor=executor, mod_name=model_name,
            )
        tcim.store_so(model_name, lib)
        print(model_name, ' saved as a aot model.')

    # compare with golden
    if stage == 'test' or stage == 'all':
        module = tcim.load_so(model_name)
        for input in inputs:
            input_file_name = 'hmquant_' + model_name + '_' + input.name + '_input.npy'
            input_data_path = os.path.join(model_dir, input_file_name)
            input_data = np.load(input_data_path).astype("int8")
            print("input[{}] shape = {}, dtype = {}".format(input.name, input_data.shape, input_data.dtype))
            module.set_input(input.name, input_data, image_format)

        module.run()

        output_num = module.get_num_outputs()
        for id in range(0, output_num):
            output_name = module.get_output_name_by_index(id)
            output_data = module.get_output_by_name(output_name).numpy()
            print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
            output_data_path = os.path.join(model_dir, 'hmquant_' + model_name + '_with_act', output_name + '.npy')
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path, allow_pickle=True).item().get("output_tensor")
            else:
                print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))
            if golden_output.shape == output_data.shape:
                cosine_dist = cosine_distance(golden_output, output_data)
                is_match = (golden_output == output_data).all()
                print("[compare] golden output [{}] match={}, similarity={:.6f}"
                            .format(output_name, is_match, cosine_dist))
            else:
                print("[compare] golden output [{}] shape not match {} vs {}"
                             .format(output_name, golden_output.shape, output_data.shape))


if __name__ == '__main__':
    args = get_args()
    build(args)
