import typing
import torch
from torch import nn

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if (
                replace_layers
            ):  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True) 
        x = x * torch.rsqrt(variance + self.eps)
        return x


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64).to(W_)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)
        
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def decoder_ln2rms(decoder_model: nn.Module):
    dtype = decoder_model.embed[0].weight.dtype
    decoder_model.embed[0].weight.data = (
        decoder_model.embed[0].weight.data
        - decoder_model.embed[0].weight.data.double().mean(
            dim=-1, keepdim=True
        )
    ).to(dtype)
    
    dtype = decoder_model.embed[1].pe.dtype
    decoder_model.embed[1].pe.data = (
        decoder_model.embed[1].pe.data
        - decoder_model.embed[1].pe.data.double().mean(
            dim=-1, keepdim=True
        )
    ).to(dtype)
   
    for layer in decoder_model.decoders:
        fuse_ln_linear(
            layer.norm1,
            [
                layer.self_attn.linear_q,
                layer.self_attn.linear_k,
                layer.self_attn.linear_v,
            ],
        )
        fuse_ln_linear(
            layer.norm2, 
            [            
                layer.src_attn.linear_q,
            ],
        )
        fuse_ln_linear(
            layer.norm3, 
            [            
                layer.feed_forward.w_1,
            ],
        )

        bake_mean_into_linear(layer.self_attn.linear_out)
        bake_mean_into_linear(layer.src_attn.linear_out)
        bake_mean_into_linear(layer.feed_forward.w_2)
    
    fuse_ln_linear(decoder_model.after_norm, [decoder_model.output_layer])

    replace_modules(
        decoder_model,
        torch.nn.LayerNorm,
        lambda _: RMSN(decoder_model.embed[0].weight.shape[-1], eps=1e-5),
        replace_layers=False,
    )
    

    
