import os
import json
import argparse
import logging
from compiler_utils import setup_logging, execute_cmd, update_perf_values
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Perf LLMs")
    parser.add_argument(
        "--perf_cfg",
        type=str,
        help="the path of perf_file.",
    )
    parser.add_argument(
        "-log",
        "--log_file",
        type=str,
        default="",
        help="the path of log.",
    )

    args = parser.parse_args()
    return args


def find_first_matched_keyword(line: str, keyword_list: list) -> str | None:
    if not keyword_list or not line:
        return None
    for keyword in keyword_list:
        if keyword in line:
            return keyword  # 找到第一个匹配项立即返回
    return None


def _generate_perf_table(cfg_path, outputs):
    perf_flag = False
    perf_metric = {
        "model_name": list(),
        "input_token": list(),
        "output_token": list(),
        "loop": list(),
        "prefill_time": list(),
        "decode_time": list(),
        "vision_time": list(),
        "prefill_speed": list(),
        "decode_speed": list(),
        "TTFT": list(),
        "TPOT": list(),
        "e2e_latency": list(),
        "e2e_tps": list(),
        "embedding_time": list(),
    }

    keywords_dict = {
        "Prefill Time": "prefill_time",
        "Decode Time": "decode_time",
        "Vision Time": "vision_time",
        "Prefill Speed": "prefill_speed",
        "Decode Speed": "decode_speed",
        "TTFT": "TTFT",
        "TPOT": "TPOT",
        "E2E Latency": "e2e_latency",
        "E2E TPS": "e2e_tps",
        "Embedding Time": "embedding_time",
    }
    for line in outputs:
        if "Start of Task" in line:
            for key in perf_metric.keys():
                perf_metric[key].append("NA")
            model_name = line.strip().split("ModelName:", 1)[-1].rsplit(".", 1)[0][1:-1]
            perf_metric["model_name"][-1] = model_name
        elif "input token len" in line:
            perf_metric["input_token"][-1] = line.strip().rsplit(":", 1)[-1].strip()
        elif "stop token len" in line:
            perf_metric["output_token"][-1] = line.strip().rsplit(":", 1)[-1].strip()
        elif "loop :" in line:
            perf_metric["loop"][-1] = line.strip().rsplit(":", 1)[-1].strip()
        elif "End of Task" in line:
            perf_flag = False
            continue
        elif "LLM Perf Avarage Information" in line:
            perf_flag = True
            continue
        elif perf_flag is True:
            keyword = find_first_matched_keyword(line, keywords_dict.keys())
            if keyword is not None:
                perf_metric[keywords_dict[keyword]][-1] = (
                    line.strip().rsplit(" ", 2)[-2].strip()
                )

    for key, value in perf_metric.items():
        print(f"{key}, length: {len(value)}")

    perf_df = pd.DataFrame(perf_metric)
    perf_df.columns = [
        "model_name",
        "input(token)",
        "output(token)",
        "loop",
        "prefill_time(ms)",
        "decode_time(ms)",
        "vision_time(ms)",
        "prefill_speed(token/s)",
        "decode_speed(token/s)",
        "TTFT(ms)",
        "TPOT(ms/token)",
        "e2e_latency(s)",
        "e2e_tps(tokens/s)",
        "embedding_time(ms)",
    ]
    perf_csv_path = cfg_path.replace(".json", ".csv")
    perf_df.to_csv(perf_csv_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()

    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Perf cfg path: %s", args.perf_cfg)

    cfg_path = args.perf_cfg
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
    else:
        logger.error(f"Invalid perf config path: {cfg_path}")
        exit(-1)

    llm_perf_dir = f"{script_dir}/../../tools/llm_perf"
    os.chdir(llm_perf_dir)
    logger.info("Current dir: %s", os.getcwd())
    os.environ["HDPL_PLATFORM"] = "ASIC"

    for perf_md in cfg_data["Streams"]:
        embed_bin = perf_md["embedding"]
        embed_pt = embed_bin.replace(".bin", ".pt")
        if os.path.exists(embed_bin):
            continue
        model_type = "llm" if "visual" not in perf_md else "vllm"
        execute_cmd(
            ["python3", "convert_embed.py", "--path", embed_pt, "--type", model_type],
            args.log_file,
        )

    cmds = ["./llm_perf", "--config", cfg_path]

    ret, outputs = execute_cmd(cmds, args.log_file, get_outputs=True)
    if not ret:
        exit(-1)

    _generate_perf_table(cfg_path, outputs)


# if __name__ == "__main__":
#     args = parse_args()

#     file_path = "./compiler_perf_20251204_1054.log"
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     cfg_path = "./perf_cfg_v0.6.0_251204.json"
#     _generate_perf_table(cfg_path, lines)
