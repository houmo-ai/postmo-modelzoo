import os 
embedding_path = "hmquant/quant_embedding.pt"
if os.path.exists(embedding_path):
    import torch
    import numpy as np

    embedding_weight = torch.load(embedding_path, map_location="cpu", weights_only=True)
    embedding_weight = embedding_weight['weight']
    if embedding_weight.dtype == torch.bfloat16:
        embedding_weight = embedding_weight.float().half()
    embedding_data = embedding_weight.cpu().numpy() 
    embedding_data.tofile(embedding_path.replace(".pt", ".bin"))