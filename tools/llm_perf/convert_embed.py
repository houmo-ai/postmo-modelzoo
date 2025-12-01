import os
import argparse
import torch
import numpy as np

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh1", "xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

def parse_args():
    parser = argparse.ArgumentParser(description="Convert embedding format")
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Embedding pt file path",
    )
    
    parser.add_argument(
        "--type",
        required=True,
        type=str,
        default="llm",
        help="Embedding pt file path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    embedding_path = args.path
    type = args.type
    if os.path.exists(embedding_path) and embedding_path.endswith(".pt"):
        if type == "llm":
            embedding_weight = torch.load(
                embedding_path, map_location="cpu", weights_only=True
            )
            embedding_weight = embedding_weight['weight']
        if type == "vllm":
            embedding_weight = torch.load(embedding_path, map_location="cpu", weights_only=False)
            if HOUMO_TARGET == "xh2":
                embedding_weight = embedding_weight.weight
        if embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.float().half()

        embedding_data = embedding_weight.detach().cpu().numpy()
        embedding_data.tofile(embedding_path.replace(".pt", ".bin"))
