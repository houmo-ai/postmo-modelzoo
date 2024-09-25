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
        '--stage',
        dest='stage',
        type=str,
        default="build",
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


if __name__ == '__main__':
    args = get_args()
    curdir = os.getcwd()

    if args.stage == "build" or args.stage == "all":
        prefill_model = os.path.join(args.model_dir, "prefill/hmquant_qwen_with_act.onnx")
        prefill_model_part1 = os.path.join(args.model_dir, "prefill/hmquant_qwen_part1_with_act.onnx")
        prefill_model_part2 = os.path.join(args.model_dir, "prefill/hmquant_qwen_part2_with_act.onnx")
        decode_model = os.path.join(args.model_dir, "decoder/hmquant_qwen_with_act.onnx")
        decode_model_part1 = os.path.join(args.model_dir, "decoder/hmquant_qwen_part1_with_act.onnx")
        decode_model_part2 = os.path.join(args.model_dir, "decoder/hmquant_qwen_part2_with_act.onnx")
        if os.path.exists(prefill_model):
            extract_model(prefill_model, prefill_model_part1, input_names=['input_1', 'valid_length', 'current_length'],
                    output_names=['model_layers_15_resadd2'])
            extract_model(prefill_model, prefill_model_part2, input_names=['model_layers_15_resadd2', 'valid_length', 'current_length'],
                    output_names=['model_layers_31_resadd2'])
        if os.path.exists(decode_model):
            extract_model(decode_model, decode_model_part1, input_names=['input_1', 'valid_length', 'current_length'],
                    output_names=['model_layers_15_resadd2'])
            extract_model(decode_model, decode_model_part2, input_names=['model_layers_15_resadd2', 'valid_length', 'current_length'],
                    output_names=['model_layers_31_resadd2'])

    if os.system("python3 build_prefill_part1.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
    if os.system("python3 build_prefill_part2.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
    if os.system("python3 build_decode_part1.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
    if os.system("python3 build_decode_part2.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
    if os.system("python3 build_prefill_head.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
    if os.system("python3 build_decode_head.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
