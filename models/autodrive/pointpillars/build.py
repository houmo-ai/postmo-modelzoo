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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    os.system("python3 build_pfe_1.py --stage {}".format(args.stage))
    os.system("python3 build_rpn.py --stage {}".format(args.stage))
