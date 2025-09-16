import os
import argparse
import logging
from compiler_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Check quant folder")
    parser.add_argument(
        "-n",
        "--model_name",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "-qm",
        "--quant_model",
        required=True,
        type=str,
        help="(required) model path, example: /models/llm/hmquant_qwen3",
    )
    parser.add_argument(
        "-path",
        "--quant_model_path",
        required=True,
        type=str,
        help="(required) Quantized model path (Jfrog url)",
    )

    args = parser.parse_args()
    return args


def _check_folder(folder_path):
    if not os.path.isdir(folder_path):
        logger.error(f"Missing folder {folder_path}.")
        return False
    return True


def _check_file(file_path):
    if not os.path.isfile(file_path):
        logger.error(f"Missing file {file_path}")
        return False
    return True


def _check_model_source(quant_model_path):
    # 如果需要额外处理Jfrog上的量化压缩包，在此处增加类型判断
    if "http" in quant_model_path:
        return "jfrog"
    return "local"


def check_quant_model(quant_model_path: str, quant_model: str, model_name: str) -> bool:
    import glob

    target = os.getenv("HOUMO_TARGET", None)
    if target is None:
        return False

    quant_model_src = _check_model_source(quant_model_path)
    if quant_model_src in ["jfrog"]:
        # quant model path is Jfrog url
        import sys

        sys.path.append(f"{script_dir}/../../apis/common/python")
        from utils import get_file_from_jfrog

        get_file_from_jfrog(quant_model_path, quant_model, quant_model)

        if os.path.exists(f"{quant_model}/hmquant"):
            os.system(f"mv -f {quant_model}/hmquant/* {quant_model}/")
            os.system(f"rm -rf {quant_model}/hmquant")

    # folders
    decoder_dir = os.path.join(quant_model, "decoder")
    prefill_dir = os.path.join(quant_model, "prefill")
    folder_list = [decoder_dir, prefill_dir]
    # files
    embedding_file = os.path.join(quant_model, "quant_embedding.pt")
    decoder_file = os.path.join(decoder_dir, f"hmquant_{model_name}_with_act.onnx")
    prefill_file = os.path.join(prefill_dir, f"hmquant_{model_name}_with_act.onnx")
    file_list = [embedding_file, decoder_file, prefill_file]
    if target == "xh1":
        weight_file = os.path.join(quant_model, "weight.npy")
        file_list.append(weight_file)
    if all(_check_folder(ele) for ele in folder_list) is False:
        return False
    if all(_check_file(ele) for ele in file_list) is False:
        return False
    if target == "xh2":
        decoder_external = list(glob.glob(decoder_dir + "/*_decode_external_data"))
        prefill_external = list(glob.glob(prefill_dir + "/*_prefill_external_data"))
        if len(decoder_external) == 0 or len(prefill_external) == 0:
            logger.error("Missing external data.")
            return False

    return True


if __name__ == "__main__":
    args = parse_args()

    quant_model_path = args.quant_model_path
    quant_model = args.quant_model
    model_name = args.model_name

    if not check_quant_model(quant_model_path, quant_model, model_name):
        exit(-1)
