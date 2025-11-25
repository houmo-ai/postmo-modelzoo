import os
import sys
import argparse
SCRIP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIP_DIR)
from model import *
from dataset import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./output/xh1/ocr_rec.hmm')
    parser.add_argument("--data_path", default="CCPD2020", type=str)
    parser.add_argument('--infer_mode', type=str, default='eval')
    parser.add_argument('--num', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    
    ppocr_v3 = OCRRec(args.model_path)
    infer_mode = args.infer_mode
    test_num = args.num
    data_path = args.data_path
    if not os.path.exists(data_path):
        data_path = os.path.join(HOUMO_DATASETS_PATH, data_path)
        if not os.path.exists(data_path):
            logger.error(f"{data_path} or {args.data_path} not exists!")
            assert(0) 
    if infer_mode == "eval":
        data_module = CCPD2020DataSet(data_path)
        res = ppocr_v3.evaluate(data_module, test_num)
        logger.info(res)
    elif infer_mode == "demo":
        import glob
        demo_data_list = glob.glob(os.path.join(data_path, "*.jpg"))
        total_num = len(demo_data_list)
        test_num = total_num if test_num <= 0 else min(total_num, test_num)
        ppocr_v3.demo(demo_data_list[0:test_num])

if __name__ == "__main__":
    main()