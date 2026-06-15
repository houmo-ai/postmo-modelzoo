from pathlib import Path

import torch
from torch import nn


def load_auto_model(model_dir: Path, model_revision: str):
    from funasr import AutoModel

    auto_model = AutoModel(
        model=str(model_dir),
        model_revision=model_revision,
        disable_pbar=True,
        disable_update=True,
    )
    return auto_model


class StaticSinusoidalPositionEncoder(nn.Module):
    def __init__(self, max_length: int, input_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        positions = torch.arange(1, max_length + 1, dtype=dtype).unsqueeze(0)
        log_timescale_increment = torch.log(torch.tensor([10000.0], dtype=dtype)) / (input_dim / 2 - 1)
        inv_timescales = torch.exp(torch.arange(input_dim / 2, dtype=dtype) * (-log_timescale_increment))
        inv_timescales = inv_timescales.reshape(1, -1)
        scaled_time = positions.reshape(1, max_length, 1) * inv_timescales.reshape(1, 1, -1)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        self.register_buffer("position_encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_encoding[:, : x.size(1), :].to(dtype=x.dtype, device=x.device)


def freeze_encoder_position_encoder(model: nn.Module, fixed_length: int) -> nn.Module:
    input_dim = model.encoder.output_size()
    model.encoder.embed = StaticSinusoidalPositionEncoder(fixed_length, input_dim)
    return model