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
        '--model_name',
        dest='model_name',
        type=str,
        default='resnet50_32x32_cifar10',
        help='output houmo model name',
    )
    parser.add_argument(
        '--driver_path',
        dest='driver_path',
        type=str,
        default='/home/debug.sw/0.9.8.20231206/',
        help='the path of hmipu driver',
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

local_path = "resnet50_32x32_cifar10"
def get_onnx_module_and_golden():
    if not os.path.exists(local_path):
        os.system('curl -upublic:Password@123 \
                http://10.10.1.53:8082/artifactory/hdpl_test_data/resnet50_32x32_cifar10.zip  -o resnet50_32x32_cifar10.zip')
        if not os.path.exists(local_path):
            os.makedirs(local_path)
            os.system('unzip -o resnet50_32x32_cifar10.zip -d %s' % (local_path))

    return os.path.join(local_path, "hmquant_resnet50_32x32_cifar10_with_act.onnx")

def get_so_path(schedue_strategy, gen_intrinsic_mode, fuse_conv2d_add_relu):
    so_path = "resnet50_32x32";
    so_path += "_schedule_" + str(schedue_strategy);
    so_path += "_intrinsic_" + str(gen_intrinsic_mode);
    if fuse_conv2d_add_relu:
        so_path += "_" + "fuse_convaddrelu";
    return so_path

def reset_asic(driver_path):
    print("dirver path: ", driver_path)
    cur_path = os.getcwd()
    os.chdir(driver_path)
    reset_aicore_cmd = "./reset_aicore.sh"
    os.system(reset_aicore_cmd)
    os.chdir(cur_path)

def build(args):
    # build model
    image_format = 'YUV422SP'
    model_name = args.model_name
    batch = args.batch
    model_path = get_onnx_module_and_golden()
    driver_path = args.driver_path
    stage = args.stage
    model_dir = os.path.dirname(model_path)
    shape_dict = {}
    layout_dict = {}
    convert_config={"layout": "NHWC"}


    onnx_model = onnx.load(model_path)
    inputs = onnx_model.graph.input
    core_num = 1
    schedue_strategy = 0
    fuse_conv2d_add_relu = True
    gen_intrinsic_mode = 2
    if batch > 4:
        core_num = 4
        schedue_strategy = 2
    platform_env_var = "HDPL_PLATFORM"
    platform_env_val = os.getenv(platform_env_var)
    need_recover_platform_env = False
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
            'tcim.gen_intrinsic': gen_intrinsic_mode,
            "tcim.schedule_strategy": schedue_strategy,
            "tcim.use_convaddrelu": fuse_conv2d_add_relu,
            "tcim.sync_strategy": 1,
            'tcim.codegen_pic': False,
            "tcim.for_benchmark": True,
            "tcim.core_num": core_num,
            "tcim.opt_layout": 0b101,
        }
        # workaround for build for asic and run will by sync time out
        if (not isinstance(platform_env_val, type(None))) and platform_env_val == "ASIC":
            os.environ[platform_env_var] = "ISIM"
            os.environ["HMIPU_HEAP_BASE_ADDR"] = '0x2e0000000'
            need_recover_platform_env = True
        target = tvm.target.Target('hdpl', host='c')
        with tvm.transform.PassContext(opt_level=3, config=compile_config):
            graph, lib, params = relay.build(
                mod, target, executor=executor, mod_name=model_name,
            )

        so_path = get_so_path(schedue_strategy, gen_intrinsic_mode, fuse_conv2d_add_relu)
        tcim.store_so(model_name, lib, so_path, hdplcc_options=["-O2"])
        if need_recover_platform_env:
            os.environ[platform_env_var] = platform_env_val
        print(model_name, ' saved as a aot model.')

    # compare with golden
    if stage == 'test' or stage == 'all':
        module = tcim.load_so(model_name)
        for input in inputs:
            input_file_name = 'hmquant_' + model_name + '_' + input.name + '_input.npy'
            input_data_path = os.path.join(model_dir, input_file_name)
            input_data = np.load(input_data_path).astype("int8")
            batch_input_data = input_data
            for idx in range(1, batch):
                batch_input_data = np.vstack((batch_input_data, input_data))
            print("input[{}] shape = {}, dtype = {}".format(input.name, batch_input_data.shape, input_data.dtype))
            module.set_input(input.name, batch_input_data, image_format)
        if os.environ[platform_env_var] == "ASIC":
            reset_asic(driver_path)
        module.prepare()
        module.run()

        output_num = module.get_num_outputs()
        for id in range(0, output_num):
            output_name = module.get_output_name_by_index(id)
            output_data = module.get_output_by_name(output_name).numpy()
            print("Output_data shape::", output_data.shape)
            output_data_path = os.path.join(model_dir, 'hmquant_' + model_name + '_with_act', output_name + '.npy')
            if os.path.exists(output_data_path):
                golden_output = np.load(output_data_path, allow_pickle=True).item().get("output_tensor")
            else:
                print("[warning] compare canceled while golden data not found -> {}".format(output_data_path))

            print("golden output shape = {}, dtype = {}".format(golden_output.shape, golden_output.dtype))
            for batch_idx in range(batch):
                cur_output_data = output_data[batch_idx].reshape((1, -1))
                print("batch {} output[{}] shape = {}, dtype = {}".format(batch_idx, output_name, cur_output_data.shape, output_data[batch_idx].dtype))
                cosine_dist = cosine_distance(golden_output, cur_output_data)
                is_match = (golden_output == cur_output_data).all()
                print("[compare] golden output [{}] match={}, similarity={:.6f}"
                                .format(output_name, is_match, cosine_dist))
                if golden_output.shape == cur_output_data.shape:
                    cosine_dist = cosine_distance(golden_output, cur_output_data)
                    is_match = (golden_output == cur_output_data).all()
                    print("[compare] golden output [{}] match={}, similarity={:.5f}"
                               .format(output_name, is_match, cosine_dist))
                else:
                    print("[compare] golden output [{}] shape not match {} vs {}"
                                .format(output_name, golden_output.shape, cur_output_data.shape))


if __name__ == '__main__':
    args = get_args()
    build(args)
