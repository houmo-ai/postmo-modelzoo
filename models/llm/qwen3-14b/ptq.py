import argparse, os

HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '')
HOUMO_TARGET = os.getenv('HOUMO_TARGET', '')


def check_gpu():
    import subprocess

    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=count --format=csv,noheader,nounits | wc -l",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        if result.returncode == 0 and int(result.stdout.strip()) > 0:
            return True
        return False
    except Exception as e:
        print(f"Not install GPU driver, error msg: {e}")
        return False

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1',""):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if HOUMO_TARGET == 'xh2':
    from quant_pipline import quant_llm, export_llm, move_llm

    def parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--model", type=str, default="qwen3-14b")
        parser.add_argument("--model-name", type=str, default="qwen3", help="output hmonnx model name")
        parser.add_argument("--work-dir", type=str, default="work_dirs/")
        parser.add_argument("--out-dir", type=str, default="output/{}".format(HOUMO_TARGET))
        parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
        parser.add_argument("--skip-gptq", action="store_true", help="skip_gptq")
        parser.add_argument("--w-bits", type=int, default=4)
        parser.add_argument("--seed", type=int, default=1024)
        parser.add_argument("--resume", action="store_true", help="resume from the cache")
        parser.add_argument("--debug", action="store_true", help="debug mode")
        parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
        parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
        parser.add_argument("--quant-type", default="w4a8_ssfp", help="quant type, default is w4a8_ssfp")
        parser.add_argument("--calibration-dataset", type=str, default=None, help="customized calibrate dataset, should be a json file")
        args = parser.parse_args()
        return args

    def main():
        args = parse_args()
        if args.calibration_dataset:
            args.gptqmodel = True
        else:
            args.gptqmodel = False
        quant_llm(args)
        export_llm(args)
        move_llm(args)

if __name__ == "__main__":
    if not check_gpu():
        print("Error: Not found GPU device.")
        exit(-1)
    main()