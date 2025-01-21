import os
import numpy as np
import time
import onnx
import argparse

HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')


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
        default='qwen2',
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
        default=4,
        help='core number',
    )
    parser.add_argument(
        '--nblocks',
        dest='nblocks',
        type=int,
        default=28,
        help='block number',
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
    args = parser.parse_args()
    return args


def extract_model(src, dest, input_names, output_names):
    import onnx_graphsurgeon
    model = onnx.load(src)
    graph = onnx_graphsurgeon.import_onnx(model)
    tensors = graph.tensors()
    print(f"inputs: {input_names}")
    print(f"outputs: {output_names}")
    graph.inputs = [tensors[in_t.strip()] for in_t in input_names]
    graph.outputs = [tensors[out_t.strip()] for out_t in output_names]
    graph.cleanup()
    onnx.save(onnx_graphsurgeon.export_onnx(graph), dest)
    print(f"extracted model saved in", dest)


def save_submodel_golden(model_dir, model_name, submodel_name, output_names):
    for name in output_names:
        file_path = os.path.join(model_dir, f"{submodel_name}/hmquant_{model_name}_with_act/{name}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True).item().get("output_tensor")
            save_path1 = os.path.join(model_dir, f"{submodel_name}/hmquant_{model_name}_{name}_input.npy")
            save_path2 = os.path.join(model_dir, f"{submodel_name}/hmquant_{model_name}_{name}_output.npy")
            np.save(save_path1, data)
            np.save(save_path2, data)
            print(f"{os.path.basename(save_path1)} saved in {os.path.dirname(save_path1)}")
            print(f"{os.path.basename(save_path2)} saved in {os.path.dirname(save_path2)}")


def build(model_name, model_dir, model_path, output_dir, profile, ncore=1):
    import tcim
    start = time.time()
    print(f"\n===> {model_name} build start...")
    decode_model = os.path.join(model_dir, model_path)
    tcim.build_from_hmonnx(
        decode_model,
        weights=os.path.join(model_dir, "weight.npy"),
        model_name=model_name,
        ncore=ncore,
        output_dir=output_dir,
        work_dir=os.path.join(output_dir, "tcim"),
        op_version={'Gather':1}
    )
    profile["build"] = time.time() - start
    print(f'{model_name} build completed in {profile["build"]:.3f} s.', flush=True)


def test(model_name, model_dir, output_dir, profile, batch=1, prefix=None):
    import tcim_lite
    print(f"\n===> {model_name} test start...")
    # load model
    model_path = os.path.join(output_dir, f"{model_name}.hmm")
    start = time.time()
    module = tcim_lite.runtime.load(model_path)
    profile["load"] = time.time() - start
    print(f'{model_name} load completed in {profile["load"]:.3f} s.', flush=True)

    # set input
    current_length = 0
    profile["set_input"] = 0
    if prefix is None:
        prefix = model_name
    input_num = module.get_num_inputs()
    for id in range(input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name)
        print("input_info[{}] shape = {}, dtype = {}, format = {}".format(input_name, input_info.shape,
                                                                          input_info.dtype, input_info.format.name))
        input_file_name = 'hmquant_' + prefix + '_' + input_name + '_input.npy'
        input_data_path = os.path.join(model_dir, input_file_name)
        input_data = np.load(input_data_path).astype(input_info.dtype)
        if input_name == 'current_length':
            current_length = input_data[0]
        input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        print("golden input[{}] shape = {}, dtype = {}".format(input_name, input_data.shape, input_data.dtype))
        start = time.time()
        module.set_input(input_name, input_data)
        profile["set_input"] += time.time() - start
    print(f'{model_name} set {input_num} inputs completed in {profile["set_input"]*1000:.3f} ms.')

    # infer model
    start = time.time()
    module.run()
    module.sync()
    profile["infer"] = time.time() - start
    print(f'{model_name} infer completed in {profile["infer"]*1000:.3f} ms.')

    # get output and compare with golden
    profile["get_output"] = 0
    result_check = True
    output_num = module.get_num_outputs()
    for id in range(output_num):
        output_name = module.get_output_name(id)
        output_info = module.get_output_info(output_name)
        print("output_info[{}] shape = {}, dtype = {}, format = {}".format(output_name, output_info.shape,
                                                                            output_info.dtype, output_info.format.name))
        start = time.time()
        output_data = module.get_output(output_name).numpy()
        profile["get_output"] += time.time() - start
        print("output[{}] shape = {}, dtype = {}".format(output_name, output_data.shape, output_data.dtype))
        # only compare [1,current_length,4096]
        if model_name == "qwen2_prefill_body":
            output_data = output_data[:1, :current_length, :]
        output_data_path = os.path.join(model_dir, 'hmquant_' + prefix + '_' + output_name + '_output.npy')
        if os.path.exists(output_data_path):
            golden_output = np.load(output_data_path)
            if model_name == "qwen2_prefill_body":
                golden_output = golden_output[:1, :current_length, :]
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
    print(f'{model_name} {model_part} get {output_num} ouputs completed in {profile["get_output"]*1000:.3f} ms.')
    if not result_check:
        print("[error] result check failed.")
        exit(-1)
    print(f"<=== {model_name} {model_part} test success.")


if __name__ == '__main__':
    args = get_args()
    curdir = os.getcwd()
    model_dir = args.model_dir
    model_name = args.model_name
    nblocks = args.nblocks
    output_dir = args.output_dir
    ncore = args.ncore
    batch = args.batch
    profile = {}

    # split prefill into 2 parts: body and head
    prefill_model = os.path.join(model_dir, f"prefill/hmquant_{model_name}_with_act.onnx")
    prefill_model_body = os.path.join(model_dir, f"prefill/hmquant_{model_name}_body_with_act.onnx")
    prefill_model_head = os.path.join(model_dir, f"prefill/hmquant_{model_name}_head_with_act.onnx")
    body_input_names = ['input_1', 'valid_length', 'current_length']
    body_output_names = [f'model_layers_{nblocks-1}_resadd2']
    head_input_names = [f'model_layers_{nblocks-1}_resadd2', 'current_length']
    head_output_names = ['lm_head_add_list_0']
    if not os.path.exists(prefill_model_body) or not os.path.exists(prefill_model_head):
        if os.path.exists(prefill_model):
            extract_model(prefill_model, prefill_model_body, input_names=body_input_names,
                output_names=body_output_names)
            extract_model(prefill_model, prefill_model_head, input_names=head_input_names,
                output_names=head_output_names)
            save_submodel_golden(model_dir, model_name, "prefill", body_output_names)
        else:
            print(f"[error] {prefill_model} not exist.")
            exit(-1)

    # build model
    if args.stage == "build" or args.stage == "all":
        model_part = "qwen2_prefill_body"
        model_path = f"prefill/hmquant_{model_name}_body_with_act.onnx"
        build(model_part, model_dir, model_path, output_dir, profile, ncore)
        model_part = "qwen2_prefill_head"
        model_path = f"prefill/hmquant_{model_name}_head_with_act.onnx"
        build(model_part, model_dir, model_path, output_dir, profile, ncore)
        model_part = "qwen2_decode"
        model_path = f"decoder/hmquant_{model_name}_with_act.onnx"
        build(model_part, model_dir, model_path, output_dir, profile, ncore)

    # test model
    if args.stage == 'test' or args.stage == 'all':
        model_part = "qwen2_prefill_body"
        part_dir = os.path.join(model_dir, "prefill")
        test(model_part, part_dir, output_dir, profile, prefix=model_name)
        model_part = "qwen2_prefill_head"
        part_dir = os.path.join(model_dir, "prefill")
        test(model_part, part_dir, output_dir, profile, prefix=model_name)
        model_part = "qwen2_decode"
        part_dir = os.path.join(model_dir, "decoder")
        test(model_part, part_dir, output_dir, profile, prefix=model_name)