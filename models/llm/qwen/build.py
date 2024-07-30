import os
import numpy as np
import argparse


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
        default="all",
        help='build stage choise=["build", "test", "all"]',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    curdir = os.getcwd()
    if os.system("python3 build_prefill.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
    if os.system("python3 build_decode.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
    if os.system("python3 build_prefill_head.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
    if os.system("python3 build_decode_head.py --stage {} --model_dir {} --core {}"
                 .format(args.stage, args.model_dir, args.core)):
        exit(-1)
