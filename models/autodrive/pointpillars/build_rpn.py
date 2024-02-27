import os
import numpy as np
import onnx
import argparse
import tvm
import tvm.relay as relay
import tvm.tcim as tcim
from tvm.relay.backend import Executor
from hmassist.utils.dist_metrics import cosine_distance

import logging
logger = logging.getLogger("__FILE__")
logger.setLevel(logging.INFO)


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
        default='rpn',
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
    image_format = "ND"
    model_name = args.model_name
    batch = args.batch
    stage = args.stage
    model_dir = args.model_dir
    quant_name = "hmquant_" + model_name + "_with_act"
    onnx_name = quant_name + ".onnx"
    shape_dict = {}
    layout_dict = {}
    convert_config = {}

    onnx_model = onnx.load(os.path.join(model_dir, onnx_name))
    inputs = onnx_model.graph.input
    if stage == 'build' or stage == 'all':
        for input in inputs:
            dims = input.type.tensor_type.shape.dim
            input_shape = [dim.dim_value for dim in dims]
            input_shape[0] *= batch
            print('input name:', input.name)
            print('input shape:', input_shape)
            shape_dict[input.name] = input_shape
            # layout_dict[input.name] = 'NHWC'

        mod = relay.frontend.from_hmonnx(
            onnx_model, shape_dict, layout=layout_dict,
            resizer_attr=None, convert_config=convert_config,
        )
        executor = Executor('aot')
        compile_config = {
            'tcim.fuse_strategy': 1,
            'tcim.gen_intrinsic': False,
            'tcim.sync_strategy': 0,
            'tcim.codegen_pic': True,
            'tcim.for_benchmark': False,
            'tcim.core_num': 1,
        }
        target = tvm.target.Target('hdpl', host='c')
        with tvm.transform.PassContext(opt_level=3, config=compile_config):
            graph, lib, params = relay.build(
                mod, target, executor=executor, mod_name=model_name,
            )
        tcim.store_so(model_name, lib)
        print(model_name, 'saved as a aot model.')

    # compare with golden
    if stage == 'test' or stage == 'all':
        module = tcim.load_so(model_name)
        for input in inputs:
            input_file_name = 'bev_input.npy'
            input_data_path = os.path.join(model_dir, "hmquant_pfe_1_with_act", input_file_name)
            input_data = np.load(input_data_path, allow_pickle=True).item().get("output_tensor").astype("int8")
            logger.info("input[{}] shape = {}, dtype = {}".format(input.name, input_data.shape, input_data.dtype))
            input_data = np.transpose(input_data, (0, 2, 3, 1))
            module.set_input(input.name, input_data, image_format)

        module.run()

        output_num = module.get_num_outputs()
        for id in range(0, output_num):
            output_name = module.get_output_name_by_index(id)
            output_data = module.get_output_by_name(output_name).numpy()
            logger.info("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
            output_data_path = os.path.join(model_dir, quant_name, output_name + '.npy')
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path, allow_pickle=True).item().get("output_tensor")
                if golden_output.shape == output_data.shape:
                    cosine_dist = cosine_distance(golden_output, output_data)
                    is_match = (golden_output == output_data).all()
                    logger.info("[compare] golden output [{}] match={}, similarity={:.6f}"
                                .format(output_name, is_match, cosine_dist))
                else:
                    logger.error("[compare] golden output [{}] shape not match {} vs {}"
                                 .format(output_name, golden_output.shape, output_data.shape))
            else:
                logger.warning("compare canceled while golden data not found -> {}".format(output_data_path))


if __name__ == '__main__':
    args = get_args()
    build(args)
