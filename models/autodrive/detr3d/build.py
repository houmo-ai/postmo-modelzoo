import os
import numpy as np
import argparse


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage',
        dest='stage',
        type=str,
        default="all",
        help='build stage choise=["build", "test", "all"]',
    )
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=4,
        help='batch size',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    os.system("python3 build_part1.py --stage {} --batch {}".format(args.stage, args.batch))
    os.system("python3 build_part2.py --stage {} --batch {}".format(args.stage, args.batch))
