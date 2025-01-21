
import typing
import torch
from torch import nn

from wenet.utils.hadamard_utils import (
    random_hadamard_matrix,
    apply_exact_had_to_linear,
    is_pow2,
)

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_attention_inputs(layer, Q, is_self) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    if is_self:
        layer_list = [layer.self_attn.linear_q, 
                    layer.self_attn.linear_k, 
                    layer.self_attn.linear_v]
    else:
        layer_list = [layer.src_attn.linear_q]

    for W in layer_list:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)


def rotate_attention_output(layer, Q, is_self) -> None:
    # Rotate output matrix of the self-attention layer.
    if is_self:
        W = layer.self_attn.linear_out
    else:
        W = layer.src_attn.linear_out
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)


def rotate_mlp_input(layer, Q) -> None:
    # Rotate the MLP input weights.\
    mlp_inputs = [layer.feed_forward.w_1]

    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

def rotate_mlp_output(layer, Q):
    out_layer = layer.feed_forward.w_2
    # out_layer = layer.mlp.c_proj if hasattr(layer.mlp, "c_proj") else layer.mlp.fc2
    # Rotate the MLP output weights and bias.
    dtype = out_layer.weight.data.dtype
    W_ = out_layer.weight.data.to(dtype=torch.float64)
    out_layer.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype)
    
    if out_layer.bias is not None:
        b = out_layer.bias.data.to(dtype=torch.float64)
        out_layer.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)

def rotate_output_layer(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    kv_proj = model.output_layer
    dtype = kv_proj.weight.dtype
    W_ = kv_proj.weight.to(dtype=torch.float64)
    kv_proj.weight.data = torch.matmul(W_, Q).to(dtype=dtype)

def decoder_rotate(decoder_model: nn.Module):
    embed_dim = decoder_model.embed[0].weight.shape[-1]
    device = decoder_model.embed[0].weight.device
    Q_v = get_orthogonal_matrix(embed_dim, "hadamard", device)

    dtype = decoder_model.embed[0].weight.data.dtype
    decoder_model.embed[0].weight.data = torch.matmul(
        decoder_model.embed[0].weight.data.double(), Q_v
    ).to(dtype)

    dtype = decoder_model.embed[1].pe.dtype
    decoder_model.embed[1].pe.data = torch.matmul(
        decoder_model.embed[1].pe.data.double(), Q_v
    ).to(dtype)
    
    for layer in decoder_model.decoders:
        rotate_attention_inputs(layer, Q_v, is_self=True)
        rotate_attention_output(layer, Q_v, is_self=True)

        rotate_attention_inputs(layer, Q_v, is_self=False)
        rotate_attention_output(layer, Q_v, is_self=False)
        
        rotate_mlp_input(layer, Q_v)
        rotate_mlp_output(layer, Q_v)

    rotate_output_layer(decoder_model, Q_v)
    


        
        