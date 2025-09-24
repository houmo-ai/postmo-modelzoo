import os
import argparse

from hmatc.utils.utils import get_file_from_jfrog


HOUMO_TARGET = os.getenv("HOUMO_TARGET", "houmo")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest="model_type",
        type=str,
        default="quant",
        help="which resource to get, choise in [raw, quant, hmm, all]",
    )
    parser.add_argument(
        "--quant_model_dir",
        dest="quant_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="where to save quant_model",
    )
    parser.add_argument(
        "--build_model_dir",
        dest="build_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET),
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default="",
        help="where to save downloaded model",
    )
    parser.add_argument(
        "--ndevice",
        dest="ndevice",
        type=int,
        default=1,
        choices=[1, 2],
        help="device number",
    )
    parser.add_argument(
        "--context_length",
        dest="context_length",
        type=str,
        default="2k",
        help="context_length",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    quant_model_dir = args.quant_model_dir
    build_model_dir = args.build_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", ".")
    HOUMO_MODEL_PATH = os.getenv("HOUMO_MODEL_PATH", ".")
    wiki_path = "models/datasets/wikitext-2-raw-v1.zip"
    if HOUMO_TARGET == "xh1":
        print("[error] not support xh1.")
    elif HOUMO_TARGET == "xh2":
        quant_path = "models/qwen3/hmquant_xh2_qwen3_14b_256_2k_20250625.zip"
        hmm_map = {
            (1, "2k"): "models/qwen3/hmm_xh2_qwen3_14b_256_2k_2cores_20250701.zip",
            (1, "8k"): "models/qwen3/hmm_xh2_qwen3_14b_256_8k_2cores_20250701.zip",
            (
                2,
                "2k",
            ): "models/qwen3/hmm_xh2_qwen3_14b_256_2k_2cores_2devices_202508013.zip",
            (
                2,
                "16k",
            ): "models/qwen3/hmm_xh2_qwen3_14b_256_16k_2cores_2devices_20250813.zip",
        }
        key = (args.ndevice, args.context_length)
        hmm_path = hmm_map.get(key)

    if model_type in ["raw", "all"]:
        ignore_patterns = []
        get_file_from_jfrog(wiki_path, model_dir, HOUMO_DATASETS_PATH)
    else:
        ignore_patterns = ["*.safetensors"]

    from modelscope import snapshot_download

    snapshot_download(
        "qwen/qwen3-14b",
        local_dir=f"{model_dir}/qwen3-14b",
        ignore_patterns=ignore_patterns,
    )

    if model_type in ["quant", "all"]:
        get_file_from_jfrog(quant_path, model_dir, quant_model_dir)

    if model_type in ["hmm", "all"]:
        get_file_from_jfrog(hmm_path, model_dir, build_model_dir)
