import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Convert embedding format")
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Embedding pt file path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    embedding_path = args.path

    if os.path.exists(embedding_path) and embedding_path.endswith(".pt"):
        import torch

        embedding_weight = torch.load(
            embedding_path, map_location="cpu", weights_only=True
        )
        embedding_weight = embedding_weight['weight']
        if embedding_weight.dtype == torch.bfloat16:
            embedding_weight = embedding_weight.float().half()

        embedding_data = embedding_weight.cpu().numpy()
        embedding_data.tofile(embedding_path.replace(".pt", ".bin"))
