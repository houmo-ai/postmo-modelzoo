import argparse, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import threading

HOUMO_DATASETS_PATH = os.getenv('HOUMO_DATASETS_PATH', '')
HOUMO_TARGET = os.getenv('HOUMO_TARGET', '')

class ProcessMemoryMonitor:
    """
    Monitors the memory usage of the current Python process in real-time using psutil.
    """
    def __init__(self, interval=2, log_file=None):
        """
        Initializes the monitor.
        Args:
            interval (int): Time between measurements in seconds.
            log_file (str, optional): Path to a file to log results. If None, prints to console.
        """
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.log_file = log_file
        self.is_monitoring = False
        self.peak_memory_mb = 0

    def get_memory_info(self):
        """
        Gets current memory usage information.
        Returns:
            dict: A dictionary containing memory usage data.
        """
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size in MB
        percent = self.process.memory_percent()   # Percentage of system memory
        return {'rss_mb': rss_mb, 'percent': percent}

    def start(self):
        """Starts the monitoring loop in a separate daemon thread."""
        self.is_monitoring = True
        self.peak_memory_mb = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True  # Thread will exit when main program does
        self.monitor_thread.start()
        print(f"Memory monitoring started (interval: {self.interval}s)")

    def _monitor_loop(self):
        """The internal loop that runs in the thread."""
        while self.is_monitoring:
            mem_info = self.get_memory_info()
            self.peak_memory_mb = max(self.peak_memory_mb, mem_info['rss_mb'])

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - RSS: {mem_info['rss_mb']:.2f} MB, System%: {mem_info['percent']:.2f}%"

            # Output to console or file
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(log_message + '\n')

            time.sleep(self.interval)

    def stop(self):
        """Stops the monitoring loop and prints peak usage."""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1) # Wait a moment for the thread to finish
        print(f"[Monitoring stopped. Peak RSS: {self.peak_memory_mb:.2f} MB]")

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

if HOUMO_TARGET == 'xh1':
    import hmquant.llm.llm_utils  as utils
    from hmquant.llm.llm_api import QwenQuantPipline

    def parse_args():
        parser = argparse.ArgumentParser(description="Quant Qwen3")
        parser.add_argument("--model", type=str, default="qwen3-8b", help="path to model")
        # 1. quant
        parser.add_argument("--quant_config", type=str, default="quant_config.py")
        parser.add_argument("--n_calib", type=int, default=16)
        # 2. export
        parser.add_argument("--model_name", type=str, default="qwen3")
        parser.add_argument("--save_path",type=str,default=f"output/{HOUMO_TARGET}/hmquant")
        parser.add_argument("--prefill_shape", type=int, nargs='+', default=[4, 64], help="List of integers for prefill shape")
        parser.add_argument("--cache_len", type=int, default=8192)
        parser.add_argument("--multi_batch",action="store_true",default=False,help="weather use multi batch for export")
        # 3. others
        parser.add_argument("--wikitext_local",type=str,default=os.path.join(HOUMO_DATASETS_PATH,"wikitext-2-raw-v1"),help="if has local wikitext, set it here")
        parser.add_argument("--eval_ppl",action="store_true",default=False)

        """  args below are for debug, please not used """
        parser.add_argument("--blocks", default=36, type=int)
        parser.add_argument("--decoder_shape", type=int, nargs='+', default=[1, 1], help="List of integers for decoder shape")
        parser.add_argument("--gptq",type=str2bool,default=False,help="weather use gptq to quant weight") # boost precision
        parser.add_argument("--cache_2_input",type=str2bool,default=True)
        parser.add_argument("--rotate_ov",type=str2bool,default=True,help="weather rotate o_proj and v_proj")
        parser.add_argument("--rotate_pre_rope",type=str2bool,default=False,help="weather rotate acts before rope")
        parser.add_argument("--rotate_post_rope",type=str2bool,default=False,help="weather rotate acts after rope")
        parser.add_argument("--use_klt",type=str2bool,default=True,help="weather use klt for rotation")
        parser.add_argument("--compile_mode",type=str2bool,default=False,help="weather show convert err")
        """  args above are for debug, please not used """
        args = parser.parse_args()
        if args.multi_batch:
            args.decoder_shape = [4,1]
        else:
            args.decoder_shape = [1,1]
        return args

    def main():
        args = parse_args()
        model, tokenizer = AutoModelForCausalLM.from_pretrained(args.model), AutoTokenizer.from_pretrained(args.model)
        quant_pipline = QwenQuantPipline()
        # 1. quant model
        qmodel = quant_pipline.quant_llm(model, tokenizer, args=args)
        if args.eval_ppl:
            utils.eval_ppl(qmodel, tokenizer, disk_file=args.wikitext_local)
            ques_res = qmodel.stream_chat(tokenizer,"hello")
        # 2. export model
        quant_pipline.export_llm(qmodel, tokenizer, args)

        # 3. chat
        if not args.cache_2_input:
            while True:
                prompt = input("\n你的问题：")
                prompt = prompt.replace("\\n", "\n")
                quant_pipline.chat(prompt,args)

        # 4. generate golden
        quant_pipline.generate_golden(args, save_path=args.save_path, model_name=args.model_name)

elif HOUMO_TARGET == 'xh2':
    from quant_pipline import quant_llm, export_llm, move_llm

    def parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--model", type=str, default="qwen3-8b")
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
    memory_monitor = ProcessMemoryMonitor(interval=2)
    memory_monitor.start()
    main()
    memory_monitor.stop()
