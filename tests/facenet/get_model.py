import os
import onnx
import argparse

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='all',
        help='model_type to get',
    )
    parser.add_argument(
        '--quant_model_dir',
        dest='quant_model_dir',
        type=str,
        default='output/H30/result',
        help='model_path to get',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    quant_model_dir = argsquant_model_dirmodel_dir
    if args.model_type == "raw" or args.model_type == "all":
        print("no raw model was provided.")

    if args.model_type == "quant" or args.model_type == "all":
        if not os.path.exists(os.path.join(quant_model_dir, "hmquant_facenet_with_act.onnx")):
            if not os.path.exists("hmquant_facenet_with_act_20240108.zip"):
                os.system('wget http://10.10.1.53:8082/artifactory/toolchain/release/models/facenet/hmquant_facenet_with_act_20240108.zip')
            os.system('mkdir -p ' + quant_model_dir)
            os.system('unzip -d ' + quant_model_dir + ' hmquant_facenet_with_act_20240108.zip')
