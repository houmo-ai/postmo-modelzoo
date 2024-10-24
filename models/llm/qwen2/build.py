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
        '--core',
        dest='core',
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


if __name__ == '__main__':
    args = get_args()
    curdir = os.getcwd()
    model_name = args.model_name
    nblocks = args.nblocks

    if args.stage == "build" or args.stage == "all":
        # model split
        prefill_model = os.path.join(args.model_dir, f"prefill/hmquant_{model_name}_with_act.onnx")
        prefill_model_part1 = os.path.join(args.model_dir, f"prefill/hmquant_{model_name}_part1_with_act.onnx")
        prefill_model_part2 = os.path.join(args.model_dir, f"prefill/hmquant_{model_name}_part2_with_act.onnx")
        prefill_model_head = os.path.join(args.model_dir, f"prefill/hmquant_{model_name}_head_with_act.onnx")
        decode_model = os.path.join(args.model_dir, f"decoder/hmquant_{model_name}_with_act.onnx")
        decode_model_part1 = os.path.join(args.model_dir, f"decoder/hmquant_{model_name}_part1_with_act.onnx")
        decode_model_part2 = os.path.join(args.model_dir, f"decoder/hmquant_{model_name}_part2_with_act.onnx")
        decode_model_head = os.path.join(args.model_dir, f"decoder/hmquant_{model_name}_head_with_act.onnx")
        part1_input_names = ['input_1', 'valid_length', 'current_length']
        part1_output_names = [f'model_layers_{nblocks//2-1}_resadd2']
        part2_input_names = [f'model_layers_{nblocks//2-1}_resadd2', 'valid_length', 'current_length']
        part2_output_names = [f'model_layers_{nblocks-1}_resadd2']
        head_input_names = [f'model_layers_{nblocks-1}_resadd2', 'current_length']
        head_output_names = ['lm_head_add_list_0']
        if args.batch == 1:
            for i in range(nblocks):
                part1_input_names.append(f'model_layers_{i}_self_attn_kcache_input')
                part1_input_names.append(f'model_layers_{i}_self_attn_vcache_input')
                part2_input_names.append(f'model_layers_{i}_self_attn_kcache_input')
                part2_input_names.append(f'model_layers_{i}_self_attn_vcache_input')
        elif args.batch == 4:
            for i in range(nblocks):
                for j in range(args.batch):
                    part1_input_names.append(f'model_layers_{i}_self_attn_kcache_input_batch{j}')
                    part1_input_names.append(f'model_layers_{i}_self_attn_vcache_input_batch{j}')
                    part2_input_names.append(f'model_layers_{i}_self_attn_kcache_input_batch{j}')
                    part2_input_names.append(f'model_layers_{i}_self_attn_vcache_input_batch{j}')
        else:
            print(f"[error] batch = {args.batch} not supported.")
            exit(-1)
        if os.path.exists(prefill_model) and not os.path.exists(prefill_model_part1):
            extract_model(prefill_model, prefill_model_part1, input_names=part1_input_names,
                output_names=part1_output_names)
            extract_model(prefill_model, prefill_model_part2, input_names=part2_input_names,
                output_names=part2_output_names)
            extract_model(prefill_model, prefill_model_head, input_names=head_input_names,
                output_names=head_output_names)
            save_submodel_golden(args.model_dir, model_name, "prefill", part1_output_names+part2_output_names)
        if os.path.exists(decode_model) and not os.path.exists(decode_model_part1):
            extract_model(decode_model, decode_model_part1, input_names=part1_input_names,
                output_names=part1_output_names)
            extract_model(decode_model, decode_model_part2, input_names=part2_input_names,
                output_names=part2_output_names)
            extract_model(decode_model, decode_model_head, input_names=head_input_names,
                output_names=head_output_names)
            save_submodel_golden(args.model_dir, model_name, "decoder", part1_output_names+part2_output_names)
        print(f"{model_name} model split commpleted.")

    if os.system("python3 build_prefill_part1.py --stage {} --model_dir {} --model_name {} --core {}"
                 .format(args.stage, args.model_dir, model_name, args.core)):
        exit(-1)
    if os.system("python3 build_prefill_part2.py --stage {} --model_dir {} --model_name {} --core {}"
                 .format(args.stage, args.model_dir, model_name, args.core)):
        exit(-1)
    if os.system("python3 build_decode_part1.py --stage {} --model_dir {} --model_name {} --core {}"
                 .format(args.stage, args.model_dir, model_name, args.core)):
        exit(-1)
    if os.system("python3 build_decode_part2.py --stage {} --model_dir {} --model_name {} --core {}"
                 .format(args.stage, args.model_dir, model_name, args.core)):
        exit(-1)
    if os.system("python3 build_prefill_head.py --stage {} --model_dir {} --model_name {} --core {}"
                 .format(args.stage, args.model_dir, model_name, args.core)):
        exit(-1)
    if os.system("python3 build_decode_head.py --stage {} --model_dir {} --model_name {} --core {}"
                 .format(args.stage, args.model_dir, model_name, args.core)):
        exit(-1)
