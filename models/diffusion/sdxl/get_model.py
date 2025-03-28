import os
import onnx
import argparse
from hmassist.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='quant',
        help='which model type to get, choise in [raw, quant, all]',
    )
    parser.add_argument(
        '--quant_model_dir',
        dest='quant_model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, 'hmquant'),
        help='where to save quant_model',
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='',
        help='where to save downloaded model',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    quant_model_dir = args.quant_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    quant_path = "models/sdxl/hmquant_sd_unet_20241231.zip"
    hmm_path = "models/sdxl/hmm_sdxl_4cores_20250314.zip"

    if model_type in ["raw", "all"]:
        from modelscope import snapshot_download
        if not os.path.exists('stable-diffusion-xl-base-1.0'):
            snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', local_dir='stable-diffusion-xl-base-1.0')
        if not os.path.exists('TCD-SDXL-LoRA'):
            snapshot_download('AI-ModelScope/TCD-SDXL-LoRA', local_dir='TCD-SDXL-LoRA')

    if model_type in ["quant", "all"]:
        get_file_from_jfrog(quant_path, model_dir, quant_model_dir)

    if model_type in ["hmm", "all"]:
        get_file_from_jfrog(hmm_path, model_dir, os.path.join('output', HOUMO_TARGET))
