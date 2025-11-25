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
        execute_cmd(["python3", "convert_embed.py", "--path", embed_pt], args.log_file)

    cmds = ["./llm_perf", "--config", cfg_path]

    ret, outputs = execute_cmd(cmds, args.log_file, get_outputs=True)
    if not ret:
        exit(-1)

    perf_flag = False
    perf_metric = {
        "model_name": list(),
        "prefill_time": list(),
        "decode_time": list(),
        "prefill_speed": list(),
        "decode_speed": list(),
        "TTFT": list(),
        "TPOT": list(),
        "e2e_latency": list(),
        "e2e_tps": list(),
        "embedding_time": list(),
    }
    for line in outputs:
        if "Start of Task" in line:
            model_name = line.strip().split("ModelName:", 1)[-1].rsplit(".", 1)[0][1:-1]
            perf_metric["model_name"].append(model_name)
        elif "End of Task" in line:
            perf_flag = False
            continue
        elif "LLM Perf Avarage Information" in line:
            perf_flag = True
            continue
        elif perf_flag is True and "Prefill Time" in line:
            perf_metric["prefill_time"].append(line.strip().rsplit(" ", 2)[-2].strip())
        elif perf_flag is True and "Decode Time" in line:
            perf_metric["decode_time"].append(line.strip().rsplit(" ", 2)[-2].strip())
        elif perf_flag is True and "Prefill Speed" in line:
            perf_metric["prefill_speed"].append(line.strip().rsplit(" ", 2)[-2].strip())
        elif perf_flag is True and "Decode Speed" in line:
            perf_metric["decode_speed"].append(line.strip().rsplit(" ", 2)[-2].strip())
        elif perf_flag is True and "TTFT" in line:
            perf_metric["TTFT"].append(line.strip().rsplit(" ", 2)[-2].strip())
        elif perf_flag is True and "TPOT" in line:
            perf_metric["TPOT"].append(line.strip().rsplit(" ", 2)[-2].strip())
        elif perf_flag is True and "E2E Latency" in line:
            perf_metric["e2e_latency"].append(line.strip().rsplit(" ", 2)[-2].strip())
        elif perf_flag is True and "E2E TPS" in line:
            perf_metric["e2e_tps"].append(line.strip().rsplit(" ", 2)[-2].strip())
        elif perf_flag is True and "Embedding Time" in line:
            perf_metric["embedding_time"].append(
                line.strip().rsplit(" ", 2)[-2].strip()
            )

    perf_df = pd.DataFrame(perf_metric)
    perf_df.columns = [
        "model_name",
        "prefill_time(ms)",
        "decode_time(ms)",
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
