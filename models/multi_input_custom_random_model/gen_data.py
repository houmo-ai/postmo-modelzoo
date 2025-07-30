import os
import numpy as np


os.makedirs('data', exist_ok=True)

for idx in range(100):
    x = np.random.uniform(low=0, high=128, size=(1, 4, 4)).astype(np.float32)
    y = np.random.uniform(low=0, high=128, size=(1, 4, 4)).astype(np.float32)
    z = {"x": x, "y": y}
    np.savez_compressed(f'data/{idx}.npz', **z)
