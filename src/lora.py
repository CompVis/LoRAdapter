import torch
from torch import nn
from typing import Union, Tuple
from src.utils import DataProvider
from jaxtyping import Float
from einops import rearrange


class SimpleLoraLinear(torch.nn.Module):
    def __init__(
        self,
        out_features: int,
        in_features: int,
        c_dim: int,
        rank: int | float,
        data_provider: DataProvider,
        alpha: float = 1.0,
        lora_scale: float = 1.0,
        broadcast_tokens: bool = True,
        depth: int | None = None,
        use_depth: bool = False,
        n_transformations: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.data_provider = data_provider
        self.lora_scale = lora_scale
        self.broadcast_tokens = broadcast_tokens
        self.depth = depth
        self.use_depth = use_depth
        self.n_transformations = n_transformations
        self.rank = rank

        # original weight of the matrix
        self.W = nn.Linear(in_features, out_features, bias=False)
        for p in self.W.parameters():
            p.requires_grad_(False)

        if type(rank) == float:
            self.rank = int(in_features * self.rank)

        self.A = nn.Linear(in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, out_features, bias=False)

        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_normal_(self.A.weight, a=1)

        self.emb_gamma = nn.Linear(c_dim, self.rank * n_transformations, bias=False)
        self.emb_beta = nn.Linear(c_dim, self.rank * n_transformations, bias=False)

    def forward(self, x, *args, **kwargs):
        w_out = self.W(x)

        if self.lora_scale == 0.0:
            return w_out

        c = self.data_provider.get_batch()
        if self.use_depth:
            assert self.depth is not None, "block depth has to be provided"
            c = c[self.depth]

        scale = self.emb_gamma(c) + 1.0
        shift = self.emb_beta(c)

        # we need to do that when we only get a single embedding vector
        # e.g pooled clip img embedding
        # out is [B, 1, rank]
        if self.broadcast_tokens:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        if self.n_transformations > 1:
            # out is [B, 1, trans, rank]
            scale = scale.reshape(-1, 1, self.n_transformations, self.rank)
            shift = shift.reshape(-1, 1, self.n_transformations, self.rank)

        a_out = self.A(x)  # [B, N, D]
        if self.n_transformations > 1:
            a_out = a_out.unsqueeze(-2).expand(-1, -1, self.n_transformations, -1)  # [B, N, trans, rank]
        a_cond = scale * a_out

        a_cond = a_cond + shift

        if self.n_transformations > 1:
            a_cond = a_cond.mean(dim=-2)

        b_out = self.B(a_cond)

        return w_out + b_out * self.lora_scale


# FiLM style LoRA conditioning
class LoRAConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        c_dim: int,
        rank: int,
        depth: int,
        data_provider: DataProvider,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.lora_scale = lora_scale
        self.data_provider = data_provider

        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_normal_(self.A.weight, a=1)

        self.emb_gamma = nn.Linear(c_dim, rank, bias=False)
        self.emb_beta = nn.Linear(c_dim, rank, bias=False)

    def forward(self, x: Float[torch.Tensor, "B C H W"], *args, **kwargs) -> Float[torch.Tensor, "B C H W"]:
        w_out = self.W(x)

        c = self.data_provider.get_batch()
        scale = self.emb_gamma(c) + 1.0
        shift = self.emb_beta(c)

        a_out = self.A(x)
        a_cond = scale[..., None, None] * a_out + shift[..., None, None]

        b_out = self.B(a_cond)

        return w_out + b_out * self.lora_scale


class NewStructLoRAConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        c_dim: int,
        rank: int,
        depth: int,
        data_provider: DataProvider,
        lora_scale: float = 1.0,
    ):
        super().__init__()

        self.lora_scale = lora_scale
        self.depth = depth

        self.data_provider = data_provider

        self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_normal_(self.A.weight, a=1)

        self.beta = nn.Conv2d(c_dim, rank, 1, bias=False)
        self.gamma = nn.Conv2d(c_dim, rank, 1, bias=False)

    def forward(self, x: Float[torch.Tensor, "B C H W"], *args, **kwargs) -> Float[torch.Tensor, "B C H W"]:
        w_out = self.W(x)

        if self.lora_scale == 0.0:
            return w_out

        cs = self.data_provider.get_batch()  # tuple
        c = cs[self.depth]

        element_shift = self.beta(c)
        element_scale = self.gamma(c) + 1.0

        # check if norm is actually needed
        # if doesn't work add norm on a_out
        a_out = self.A(x)

        a_cond = a_out * element_scale + element_shift

        b_out = self.B(a_cond)

        return w_out + b_out * self.lora_scale
