import os
import onnx
import argparse
from hmatc.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "houmo")
HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH", ".")


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest="model_type",
        type=str,
        default="raw",
        help="which model type to get, choise in [raw, quant, all]",
    )
    parser.add_argument(
        "--quant_model_dir",
        dest="quant_model_dir",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="where to save quant_model",
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


if __name__ == "__main__":
    args = get_args()
    quant_model_dir = args.quant_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    model_name = "yolov5l-face.onnx"
    raw_path = (
        "http://10.10.1.53:8082/artifactory/toolchain/support/custom/saimo/"
        + model_name
    )
    data_path = "http://10.10.1.53:8082/artifactory/customer_service_models/%E8%B5%9B%E6%91%A9/CelebA.tar.gz"

    if model_type == "raw" or model_type == "all":
        get_file_from_jfrog(raw_path, model_dir)
        extract_path = os.path.join(model_dir, "yolov5l-face_clip.onnx")
        print("extract_path:", model_dir, extract_path)
        onnx.utils.extract_model(
            model_dir + "/" + model_name,
            extract_path,
            input_names=["input"],
            output_names=[
                "/model.23/Concat_output_0",
                "/model.23/Concat_2_output_0",
                "/model.23/Concat_4_output_0",
            ],
            check_model=True,
        )

    # if not os.path.exists(os.path.join(HOUMO_DATASETS_PATH, "CelebA")):
    #     get_file_from_jfrog(data_path, HOUMO_DATASETS_PATH, HOUMO_DATASETS_PATH)
