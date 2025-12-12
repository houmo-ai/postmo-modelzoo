import os
import sys
import argparse
from hmatc.utils.utils import get_file_from_jfrog, get_houmo_version

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '.')


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest="model_type",
        type=str,
        default="hmm",
        choices=["raw", "hmm"],
        help="which model type to get, choise in [raw, hmm]",
    )
    parser.add_argument(
        "--build_model_dir",
        dest="build_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
        help="where to save build_model",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=".",
        help="where to save downloaded model",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir

    model_name = "ppocrv3_rec"
    ncore = 1
    batch = 1
    opt_level = "O2"
    version = get_houmo_version()
    target = HOUMO_TARGET
    raw_path = f"models/PPOCRv3/paddleocr_rec-sim.onnx"
    data_path = f"models/PPOCRv3/CCPD2020_PPOCRv3_eval.tar.gz"
    build_path = f"models/{target.lower()}-{version}/{model_name}/{model_name}_{target}_b{batch}_{ncore}core_{opt_level}_{version}.tar.xz"

    if model_type in ["raw"]:
        file_path = get_file_from_jfrog(raw_path, model_dir)

    if not os.path.exists(
        os.path.join(HOUMO_DATASETS_PATH, "CCPD2020_PPOCRv3_eval.tar.gz")
    ):
        get_file_from_jfrog(data_path, HOUMO_DATASETS_PATH, HOUMO_DATASETS_PATH)

    if model_type in ["hmm"] and not get_file_from_jfrog(
        build_path, model_dir, build_model_dir
    ):
        sys.exit(1)
