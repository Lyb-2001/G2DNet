from typing import Optional, Dict, Any, Literal, Callable
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torchdiffeq import odeint
from models.dinov3.module.mona_with_select import MonaOpClipDowm,MonaOpMultiClass,MonaUncertaintyPrompt
from models.dinov3.module.IRBranch import EdgeFeatureGuidance
from toolbox.losses.lovasz_losses import lovasz_softmax
import numpy as np
from module.DySample import DySample
from typing import Optional, List, Tuple
import math
from bb.convnextv2 import convnextv2_femto

activation = nn.SiLU()
normalization = lambda x: nn.GroupNorm(max(1, x // 16), x)


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        emb_channels: Optional[int] = None,
        dropout: float = 0.1,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            activation,
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        self.h_upd = self.x_upd = (
            Upsample() if up else Downsample() if down else nn.Identity()
        )

        if self.emb_channels:
            self.emb_layers = nn.Sequential(
                activation,
                nn.Linear(self.emb_channels, 2 * self.out_channels),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        self.skip_connection = (
            nn.Identity()
            if self.out_channels == in_channels
            else nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]):
        emb = None
        if isinstance(x, tuple):
            x, emb = x

        if self.updown:
            h = self.in_layers[:-1](x)
            h, x = self.h_upd(h), self.x_upd(x)  # up/down sampling
            h = self.in_layers[-1](h)
        else:
            h = self.in_layers(x)

        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            if emb_out.ndim == 4:
                emb_out = emb_out.permute(0, 3, 1, 2)  # to channels first
            else:
                while len(emb_out.shape) < len(h.shape):
                    emb_out = emb_out[..., None]
            scale, shift = emb_out.chunk(2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(
        self, in_channels: int, num_heads: int = 1, num_head_channels: int = -1
    ):
        super().__init__()
        self.in_channels = in_channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                in_channels % num_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = in_channels // num_head_channels
        self.norm = normalization(in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(x, tuple):
            x, _ = x
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    def __init__(self, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor):
        bs, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct, bcs -> bts",
            (q * scale).view(bs * self.num_heads, ch, length),
            (k * scale).view(bs * self.num_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts, bcs -> bct", weight, v.reshape(bs * self.num_heads, ch, length)
        )
        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        model_channels: int,
        out_channels: int = 6,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [8],
        dropout: float = 0.1,
        channel_mult: List[int] = [1, 2,4,8],
        num_heads: int = 4,
        num_head_channels: int = 8,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.image_size = input_shape[-1]
        self.in_channels = input_shape[0]
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        emb_channels = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_channels),
            activation,
            nn.Linear(emb_channels, emb_channels),
        )
        # image-based conditioning
        ch = int(channel_mult[0] * model_channels)
        self.label_embed = nn.Sequential(
            nn.Conv2d(3, ch, 1),  # NOTE: change me, hard coded input channels
            normalization(ch),
            activation,
            nn.Conv2d(ch, ch, 1),
        )
        ch = input_ch = int(channel_mult[0] * model_channels)
        # self.stem = nn.Conv2d(self.in_channels, ch, 3, padding=1)
        self.stem = nn.Conv2d(self.in_channels, ch, 4, stride=4)
        self.encoder_blocks = nn.ModuleList([])
        encoder_ch = [ch]
        res = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):

                layers = []
                out_ch = int(mult * model_channels)
                layers.append(ResBlock(ch, out_ch, emb_channels, dropout))
                ch = out_ch

                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels))

                self.encoder_blocks.append(nn.Sequential(*layers))
                encoder_ch.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.encoder_blocks.append(
                    ResBlock(ch, out_ch, emb_channels, dropout, down=True)
                )
                ch = out_ch
                encoder_ch.append(ch)
                res *= 2  # downsampling factor

        if attention_resolutions[0] == -1:
            self.middle_block = ResBlock(ch, emb_channels=emb_channels, dropout=dropout)
        else:
            self.middle_block = nn.Sequential(
                ResBlock(ch, emb_channels=emb_channels, dropout=dropout),
                AttentionBlock(ch, num_heads, num_head_channels),
                ResBlock(ch, emb_channels=emb_channels, dropout=dropout),
            )

        self.decoder_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):

                layers = []
                in_ch = encoder_ch.pop()
                out_ch = int(mult * model_channels)
                layers.append(ResBlock(ch + in_ch, out_ch, emb_channels, dropout))
                ch = out_ch

                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels))

                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, out_ch, emb_channels, dropout, up=True))
                    res //= 2
                self.decoder_blocks.append(nn.Sequential(*layers))

        self.head = nn.Sequential(
            normalization(ch),
            activation,
            zero_module(nn.Conv2d(input_ch, self.out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        emb = None
        if t is not None:
            # (b, emb_dim)
            emb = self.time_embed(timestep_embedding(t, self.model_channels))

        y_emb = None
        if y is not None:
            y = y.repeat(1, 3, 1, 1) if y.shape[1] == 1 else y
            drop = None
            if self.training:
                drop = torch.rand(y.shape[0], 1, 1, 1, device=y.device) > 0.1
            # print(y.shape)
            y_emb = self.label_embed(y if drop is None else y * drop)  # v1

        hs = []
        h = self.stem(x)
        if y_emb is not None:
            h = h + y_emb
        hs.append(h)

        res = h.shape[-1]
        for block in self.encoder_blocks:
            # checks if the next block downsamples the input
            if isinstance(block, ResBlock) and block.updown:
                res //= 2
            h = block((h, emb))
            hs.append(h)

        h = self.middle_block((h, emb))

        for block in self.decoder_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block((h, emb))
            res = h.shape[-1]
        return F.interpolate(self.head(h),size=(480,640),mode="bilinear",align_corners=False)

def zero_module(module: nn.Module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    t = t * 1000  # [0, 1] -> [0, 1000]
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding  # (b, dim)
# import math
# from functools import partial
# from inspect import isfunction
#
# import torch
# import torch.nn.functional as F
# from einops import rearrange, reduce
# from einops.layers.torch import Rearrange
# from torch import einsum, nn
# from torch.utils.checkpoint import checkpoint
#
#
# # -------------------- Helpers --------------------
# def exists(x):
#     return x is not None
#
#
# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d
#
#
# def num_to_groups(num, divisor):
#     groups = num // divisor
#     remainder = num % divisor
#     arr = [divisor] * groups
#     if remainder > 0:
#         arr.append(remainder)
#     return arr
#
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x
#
#
# def Upsample(dim, dim_out=None):
#     return nn.Sequential(
#         nn.Upsample(scale_factor=2, mode="nearest"),
#         nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
#     )
#
#
# def Downsample(dim, dim_out=None):
#     # No More Strided Convolutions or Pooling
#     return nn.Sequential(
#         Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
#         nn.Conv2d(dim * 4, default(dim_out, dim), 1),
#     )
#
#
# # -------------------- Positional embeddings --------------------
# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#
#         return embeddings
#
#
# # -------------------- Resnet block --------------------
# class WeightStandardizedConv2d(nn.Conv2d):
#     """
#     https://arxiv.org/abs/1903.10520
#     weight standardization purportedly works synergistically with group normalization
#     """
#
#     def forward(self, x):
#         eps = 1e-5 if x.dtype == torch.float32 else 1e-3
#
#         weight = self.weight
#         mean = reduce(weight, "o ... -> o 1 1 1", "mean")
#         var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
#         normalized_weight = (weight - mean) / (var + eps).rsqrt()
#
#         return F.conv2d(
#             x,
#             normalized_weight,
#             self.bias,
#             self.stride,
#             self.padding,
#             self.dilation,
#             self.groups,
#         )
#
#
# class Block(nn.Module):
#     def __init__(self, dim, dim_out, groups=8):
#         super().__init__()
#         self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
#         self.norm = nn.GroupNorm(groups, dim_out)
#         self.act = nn.SiLU()
#
#     def forward(self, x, scale_shift=None):
#         x = self.proj(x)
#         x = self.norm(x)
#
#         if exists(scale_shift):
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift
#
#         x = self.act(x)
#         return x
#
#
# class ResnetBlock(nn.Module):
#     """https://arxiv.org/abs/1512.03385"""
#
#     def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
#         super().__init__()
#         self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
#
#         self.block1 = Block(dim, dim_out, groups=groups)
#         self.block2 = Block(dim_out, dim_out, groups=groups)
#         self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
#
#     def forward(self, x, time_emb=None):
#         scale_shift = None
#         if exists(self.mlp) and exists(time_emb):
#             time_emb = self.mlp(time_emb)
#             time_emb = rearrange(time_emb, "b c -> b c 1 1")
#             scale_shift = time_emb.chunk(2, dim=1)
#
#         h = self.block1(x, scale_shift=scale_shift)
#         h = self.block2(h)
#         return h + self.res_conv(x)
#
#
# # -------------------- Attention module --------------------
# class Attention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
#         q = q * self.scale
#
#         sim = einsum("b h d i, b h d j -> b h i j", q, k)
#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attn = sim.softmax(dim=-1)
#
#         out = einsum("b h i j, b h d j -> b h i d", attn, v)
#         out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
#         return self.to_out(out)
#
#
# class LinearAttention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
#
#         self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
#
#         q = q.softmax(dim=-2)
#         k = k.softmax(dim=-1)
#
#         q = q * self.scale
#         context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
#
#         out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
#         out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
#         return self.to_out(out)
#
#
# # -------------------- Group normalization --------------------
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.GroupNorm(1, dim)
#
#     def forward(self, x):
#         x = self.norm(x)
#         return self.fn(x)
#
#
# # -------------------- Unet --------------------
# class Unet(nn.Module):
#     def __init__(
#         self,
#         dim,
#         init_dim=None,
#         out_dim=6,
#         dim_mults=(1, 2, 4, 8),
#         channels=6,
#         self_condition=True,
#         resnet_block_groups=4,
#     ):
#         super().__init__()
#
#         # determine dimensions
#         self.channels = channels
#         self.self_condition = self_condition
#         input_channels = channels + (3 if self_condition else 0)
#
#         init_dim = default(init_dim, dim)
#         self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=4, stride=4)  # changed to 1 and 0 from 7,3
#
#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))
#
#         block_klass = partial(ResnetBlock, groups=resnet_block_groups)
#
#         # time embeddings
#         time_dim = dim * 4
#
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(dim),
#             nn.Linear(dim, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim),
#         )
#
#         # layers
#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)
#
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)
#
#             self.downs.append(
#                 nn.ModuleList(
#                     [
#                         block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#                         block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#                         Residual(PreNorm(dim_in, LinearAttention(dim_in))),
#                         Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
#                     ]
#                 )
#             )
#
#         mid_dim = dims[-1]
#         self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
#         self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
#         self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
#
#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
#             is_last = ind == (len(in_out) - 1)
#
#             self.ups.append(
#                 nn.ModuleList(
#                     [
#                         block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
#                         block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
#                         Residual(PreNorm(dim_out, LinearAttention(dim_out))),
#                         Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
#                     ]
#                 )
#             )
#
#         self.out_dim = default(out_dim, channels)
#
#         self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
#         self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
#
#     def forward(self, x, time, x_self_cond):
#         if self.self_condition:
#             x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
#             x = torch.cat((x_self_cond, x), dim=1)
#
#         x = self.init_conv(x)
#         r = x.clone()
#
#         t = self.time_mlp(time)
#
#         h = []
#
#         for block1, block2, attn, downsample in self.downs:
#             x = block1(x, t)
#             h.append(x)
#
#             x = block2(x, t)
#             x = attn(x)
#             h.append(x)
#
#             x = downsample(x)
#
#         x = self.mid_block1(x, t)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x, t)
#
#         for block1, block2, attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim=1)
#             x = block1(x, t)
#
#             x = torch.cat((x, h.pop()), dim=1)
#             x = block2(x, t)
#             x = attn(x)
#
#             x = upsample(x)
#
#         x = torch.cat((x, r), dim=1)
#
#         x = self.final_res_block(x, t)
#         #x = checkpoint(self.final_res_block, x, t)  if cuda-out-of-memory
#
#         return F.interpolate(self.final_conv(x),size=(480,640),mode="bilinear",align_corners=False)
class ContinuousFlowSSN(nn.Module):
    def __init__(
        self,
        flow_net: nn.Module,
        base_net: Optional[nn.Module] = None,
        num_classes: int = 2,
        cond_base: bool = False,
        cond_flow: bool = False,
        base_std: float = 0.0,
    ):
        super().__init__()
        # assert flow_net.out_channels == num_classes
        self.flow_net = flow_net
        self.base_net = base_net
        self.num_classes = num_classes
        self.cond_base = cond_base
        self.cond_flow = cond_flow
        self.base_std = base_std  # learns diag cov of base dist if 0, else fixed
        self.criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(
            [2.0268, 4.2508, 23.6082, 23.0149, 11.6264, 25.8710])).float())
        class_weight = torch.from_numpy(np.array([2.0268, 4.2508, 23.6082, 23.0149, 11.6264, 25.8710])).float()
        self.class_weight = class_weight/class_weight.mean()
        self.criterion_bound = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.4584, 18.7187])).float())

        if self.cond_base:
            assert base_net is not None
            assert base_net.out_channels == num_classes * 2  # pred (loc, scale)
        else:
            assert flow_net.input_shape[0] == num_classes
            base_shape = flow_net.input_shape  # (num_classes, h, w)
            self.register_buffer("base_loc", torch.zeros(*base_shape))
            self.register_buffer("base_scale", torch.ones(*base_shape))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mc_samples: int = 1,
        ode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        batch_size, _, h, w = batch["x"].shape
        # h = h//4
        # w = w//4
        # print(context.shape)
        loss, logits, probs,train_out,rgb_features,loc, std = *(None,) * 6, 0.0
        base_dist,loc,rgb_features = self.get_base_dist(batch["x"],batch["ir"])
        # scale = base_dist.scale
        sample_shape = torch.Size(
            [mc_samples] if self.cond_base else [mc_samples, batch_size]
        )
        # eps = torch.randn(mc_samples*batch_size,self.num_classes,h,w,device=scale.device)
        # u = loc.detach()+scale*eps
        # (m*b, k, h, w)
        u = base_dist.rsample(sample_shape).reshape(mc_samples * batch_size, -1, h//4, w//4).cuda()
        u = F.interpolate(u,size=(h,w),mode="bilinear",align_corners=False)
        # (m*b, c, h, w)
        context = self.maybe_expand(batch["x"], mc_samples) if self.cond_flow else None


        if "y" in batch.keys():  # training
            # (b, k, h, w)
            y = batch["y"].float().permute(0, 3, 1, 2)  # * 2 - 1  # [-1, 1]
            # print(batch["y"].shape)
            # y = F.interpolate(batch["y"].permute(0, 3, 1, 2).float(), size=(h, w), mode="nearest")  # * 2 - 1  # [-1, 1]
            # (m*b, k, h, w)
            y = self.maybe_expand(y, mc_samples)
            y1 = self.maybe_expand(batch["y1"], mc_samples)
            # (m*b)
            t = self.maybe_expand(
                torch.zeros(batch_size, device=y.device).uniform_(0, 1), mc_samples
            )
            # (m*b, k, h, w)

            y_t = self.interpolant(u, y, t)
            # print(y_t.shape)
            # print(t.shape)
            # print(context.shape)
            # print(batch["y1"].shape)
            loss, std,train_out = self.logit_pred_loss(y.permute(0,2,3,1),y1, y_t, t, context=None, mc_samples=mc_samples)
            # aux = F.interpolate(loc, size=(h, w), mode="bilinear",align_corners=False)

            loss = loss + self.criterion(u, y1)# + lovasz_softmax(F.softmax(loc, dim=1), batch["y1"].squeeze(1))
        else:  # test
            if ode_kwargs is None:
                ode_kwargs = {
                    "method": "euler",
                    "t": torch.tensor([0.0, 1.0]).to(u.device),
                    "options": dict(step_size=1.0 / self.eval_T),
                }
            # (m*b, k, h, w)
            y_hat = ode_solve(
                self.flow_net, u, context, field="categorical", **ode_kwargs
            )
            # train_out = y_hat.reshape(mc_samples, batch_size, self.num_classes, h, w)
            # std = y_hat.reshape(mc_samples, batch_size, self.num_classes, h, w).std(dim=0)
            # shifts to min of 0
            probs = y_hat - y_hat.min(dim=1, keepdim=True).values  # type: ignore
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-7)  # renorm
            train_out = probs.reshape(mc_samples, batch_size, self.num_classes, h, w)
            # (m, b, h, w, k)
            probs = probs.reshape(mc_samples, batch_size, self.num_classes, h, w).mean(dim=0)
            probs = F.interpolate(probs, size=(480, 640), mode="bilinear",align_corners=False)
            # print(probs.shape)
        if logits is not None:
            # (m, b, h, w, k)
            logits = logits.permute(0, 2, 3, 1).reshape(
                mc_samples, batch_size, h, w, -1
            )

        return dict(loss=loss, logits=logits, probs=probs,train_out = train_out,rgb_features = rgb_features,loc=loc, std=std)

    def get_base_dist(self, x: Optional[torch.Tensor] = None,ir: Optional[torch.Tensor] = None):
        if not self.cond_base:
            loc, scale = self.base_loc, self.base_scale
        else:
            assert self.base_net is not None
            # (b, k, h, w), (b, k, h, w)
            normdist,rgb_features = self.base_net(x,ir)
            loc, log_scale = normdist.chunk(2, dim=1)
            if self.base_std:

                scale = self.base_std  # will broadcast
            else:
                # print(1)
                scale = torch.exp(F.softplus(log_scale))
        return dist.Normal(loc, scale) ,loc,rgb_features

    def interpolant(self, u: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        # (m*b, k, h, w)
        t = t.reshape(-1, *([1] * (u.ndim - 1)))  # for spatial broadcasting
        y_t = (1 - t) * u + t * y
        return y_t

    def logit_pred_loss(
        self,
        y: torch.Tensor,
        y1: torch.Tensor,
        y_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mc_samples: int = 1,
    ):

        std = 0
        # (m*b, k, h, w)
        logits = self.flow_net(y_t, t, context)
        train_out = logits.reshape(mc_samples, -1, *logits.shape[1:]).mean(dim=0)
        # (m, b, h, w, k)
        logits_reshape = logits.reshape(mc_samples, -1, *logits.shape[1:]).permute(
            0, 1, 3, 4, 2
        )
        y_reshaped = y.reshape(mc_samples, -1, *y.shape[1:])
        y1_reshaped = y1.reshape(mc_samples, -1, *y1.shape[1:])
        # (m, b, h, w), broadcasting y to m
        log_py_n = dist.OneHotCategorical(logits=logits_reshape).log_prob(y_reshaped)
        # (m, b)
        self.class_weight = self.class_weight.to(y.device)
        pixel_weight = (y_reshaped*self.class_weight).sum(dim=-1)
        log_py_n = log_py_n*pixel_weight
        log_prob = log_py_n.mean(dim=(-2, -1))

        if mc_samples > 1:
            std = log_prob.exp().std(dim=0)#.mean()
            # print(std.shape)
            log_prob = torch.logsumexp(log_prob, dim=0) - np.log(mc_samples)

        loss_ce = -torch.mean(log_prob)  # / (h * w)
        # y_expanded = self.maybe_expand(y,mc_samples)
        loss_lov = lovasz_softmax(F.softmax(logits,dim=1),y1)
        # loss = loss_ce+loss_lov
        loss = loss_ce+loss_lov
        return loss, std,train_out

    def maybe_expand(self, x: torch.Tensor, mc_samples: int = 1):
        # (m*b, ...) <- (b, ...)
        return (
            x[None, ...].expand(mc_samples, *x.shape).reshape(-1, *x.shape[1:])
            if mc_samples > 1
            else x
        )


def euler_solver(func: Callable, y0: torch.Tensor, t: torch.Tensor, **kwargs):
    y = torch.zeros(len(t), *y0.shape, device=y0.device)
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        y[i] = y[i - 1] + h * func(t[i - 1], y[i - 1])
    return y


def ode_solve(
    model: nn.Module,
    u: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    field: Literal["unconstrained", "categorical"] = "categorical",
    **kwargs: Any,
):
    if field == "unconstrained":
        fn = lambda t, x: model(x, t.repeat(x.shape[0]), context)
    elif field == "categorical":
        fn = lambda t, x: model(x, t.repeat(x.shape[0]), context).softmax(dim=1) - u
    else:
        raise NotImplementedError
    return odeint(fn, u, **kwargs)[-1]



class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.dwconv.in_channels
        # print(dim//16*3)
        self.mona = MonaUncertaintyPrompt(dim,6)

    def forward(self, x,ir):

        rgb_feat,un = self.mona(x,ir)
        rgb_feat = self.block(rgb_feat)
        return rgb_feat


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

    # [128, 256, 512, 1024]
class SegFormerHead(nn.Module):
    def __init__(self, num_classes=12, in_channels=[128, 256, 512, 1024], embedding_dim=256, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_fuse = nn.Sequential(nn.Conv2d(embedding_dim * 4, embedding_dim, 1), nn.BatchNorm2d(embedding_dim),
                                         nn.ReLU(inplace=True))
        # 输出通道改为条件特征的维度
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.up2 = DySample(in_channels=embedding_dim, scale=2)
        self.up4 = DySample(in_channels=embedding_dim, scale=4)
        self.up8 = DySample(in_channels=embedding_dim, scale=8)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.up8(_c4)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.up4(_c3)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.up2(_c2)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(_c)
        x = self.dropout(_c)
        return self.linear_pred(x)

class ThermalModulator(nn.Module):
    def __init__(self,in_chans=3,embed_dims=[128,256,512,1024]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans,32,kernel_size=3,padding=1,stride=2),
            nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.heads = nn.ModuleList()
        for dim in embed_dims:
            head = nn.Conv2d(64,dim*2,kernel_size=1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.heads.append(head)
    def forward(self,x_ir):
        stem_feat = self.stem(x_ir)
        return stem_feat
    def get_params(self,stem_feat,stage_idx,target_shape):
        head = self.heads[stage_idx]
        if stem_feat.shape[-2:]!=target_shape:
            feat_resized = F.interpolate(
                stem_feat,size=target_shape,mode='bilinear',align_corners=False
            )
        else:
            feat_resized = stem_feat
        style = head(feat_resized)
        gamma,beta = style.chunk(2,dim=1)
        return gamma,beta
class DummyBaseNet(nn.Module):
    """一个虚拟的 base_net，用于测试。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        # ContinuousFlowSSN 的 __init__ 需要这个属性
        # --- DINOv3 Backbone (与原来相同) ---
        self.dino = torch.hub.load(
            repo_or_dir="./", model="dinov3_convnext_base",
            source="local", pretrained=False, trust_repo=True
        )
        checkpoint = torch.load('/home/yuride/lyb/bb/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth', map_location='cpu')
        self.dino.load_state_dict(checkpoint, strict=True)
        print("✓ Local weights successfully loaded")

        for param in self.dino.parameters():
            param.requires_grad = False
        # self.ir_encoder = EdgeFeatureGuidance()
        self.ir_encoder = convnextv2_femto()#48,96,192,384
        self.ir_encoder.load_state_dict(torch.load("/home/yuride/lyb/bb/convnextv2_femto_1k_224_ema.pt", map_location="cpu")['model'],strict=False)
        self.adapter_blocks = nn.ModuleList()
        for stage in self.dino.stages:
            for rgb_block in stage:
                adapter = Adapter(rgb_block)
                self.adapter_blocks.append(adapter)
        # self.end_indices = [2, 5, 32, 35]
        self.end_indices = [2, 5, 32, 35]
        # self.thermal_modulator = ThermalModulator(in_channels)
        self.de = SegFormerHead(num_classes=self.out_channels)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=4)S
    def forward(self, img,ir):
        rgb_x = self.dino.downsample_layers[0](img)
        rgb_features = []
        r1,r2,r3,r4 = self.ir_encoder(ir)
        irf = r1, r2, r3, r4
        ds = 0
        # gamma, beta = self.thermal_modulator.get_params(ir_stem, stage_idx=0, target_shape=rgb_x.shape[-2:])
        for i, adapter_block in enumerate(self.adapter_blocks):
            # print(gamma.shape)
            # print(rgb_x.shape)
            rgb_x = adapter_block(rgb_x,irf[ds])
            if i in self.end_indices:
                rgb_features.append(rgb_x)
                if ds < len(self.dino.downsample_layers) - 1:
                    ds = ds + 1
                    rgb_x = self.dino.downsample_layers[ds](rgb_x)
                    # gamma, beta = self.thermal_modulator.get_params(ir_stem, stage_idx=ds, target_shape=rgb_x.shape[-2:])
        feat_dino = self.de(rgb_features)
        return  feat_dino,rgb_features
# class stuNet(nn.Module):
#     def __init__(self):
#         super(stuNet, self).__init__()
#         self.backbone = convnextv2_atto()
#         self.backbone.load_state_dict(torch.load("/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/SAM_/bb/convnextv2_atto_1k_224_ema.pt")['model'],strict=False)
#         # self.deup = SegFormerHeadup()
#         # self.dedown = SegFormerHeaddown()
#         # self.FiLM = FiLMLayer(320,320)
#         # self.out = nn.Conv2d(320, 6, kernel_size=1)
#         # self.pag1 = Pag(40,80,2)
#         # self.pag2 = Pag(80,160,4)
#         # self.pag3 = Pag(160,320,8)
#         self.pa = EvidenceGen(320,6)
#     def forward(self, r):
#         r1, r2, r3, r4 = self.backbone(r)
#         # un,aux = self.pa(r4)
#         p12 = self.pag1(r1,r2)
#         p23 = self.pag2(p12,r3)
#         p34 = self.pag3(p23,r4)
#         un, aux,umap = self.pa(p34)
#         # outup = self.deup((r1,r2,r3,r4))
#         # outdown,loss = self.dedown((r1,r2,r3,r4))
#         # tf = self.clip_proj(text_features)
#         # out, aux = self.of(x, tf)
#         out = self.out(self.FiLM(un,r4))
#         out = F.interpolate(out, size=(480,640), mode='bilinear')
#         aux = F.interpolate(aux, size=(480,640), mode='bilinear')
#         umap = F.interpolate(umap, size=(480,640), mode='bilinear')
#
#         return out,aux,umap

# ===============================================================
# 3. 运行测试
# ===============================================================

if __name__ == "__main__":
    # ---- 测试参数 ----
    BATCH_SIZE = 3
    H, W = 480, 640
    NUM_CLASSES = 6
    IMG_CHANNELS = 3
    MC_SAMPLES = 1

    # ---- 创建虚拟数据 ----
    dummy_x = torch.randn(BATCH_SIZE, IMG_CHANNELS, H, W).cuda()
    dummy_ir = torch.randn(BATCH_SIZE, IMG_CHANNELS, H, W).cuda()
    # dummy_b = torch.randn(BATCH_SIZE, 1, H, W).cuda()
    # 创建一个虚拟的分割掩码 (long类型)
    dummy_y_long = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, H, W)).cuda()
    # print(dummy_y_long.shape)
    # 转换为 one-hot 编码，这是损失函数所期望的格式
    # batch['y'] 的形状应为 (b, h, w, k)
    dummy_y_onehot = nn.functional.one_hot(dummy_y_long, num_classes=NUM_CLASSES).float()

    assert dummy_y_onehot.shape == (BATCH_SIZE, H, W, NUM_CLASSES)

    train_batch = {"x": dummy_x,"ir": dummy_ir , "y": dummy_y_onehot, "y1": dummy_y_long}
    test_batch = {"x": dummy_x,"ir": dummy_ir}

    print("=" * 20)
    print("开始测试 ContinuousFlowSSN ...")
    print("=" * 20)

    # ---- 测试 1: 训练模式，完全非条件 ----
    print("\n--- 测试 1: ")
    # flow_net_unc = UNetModel(input_shape=[6,480,640],model_channels=16).cuda()
    flow_net_unc = Unet(dim=16).cuda()
    base_net_cond = DummyBaseNet(in_channels=IMG_CHANNELS, out_channels=NUM_CLASSES * 2)
    model_unc = ContinuousFlowSSN(
        base_net= base_net_cond,
        flow_net=flow_net_unc,
        num_classes=6,
        cond_base=True,
        cond_flow=True
    ).cuda()

    optimizer = torch.optim.Adam(model_unc.parameters(), lr=1e-3)
    optimizer.zero_grad()

    output = model_unc(train_batch, mc_samples=MC_SAMPLES)
    loss = output['loss']

    assert torch.is_tensor(loss) and loss.numel() == 1, "Loss 应该是标量张量"
    assert output['logits'] is None, "训练时 logits 应为 None"
    assert output['probs'] is None, "训练时 probs 应为 None"

    print("前向传播成功，损失值为:", loss.item())

    loss.backward()
    print("反向传播成功")
    optimizer.step()
    print("优化器步进成功")
    print("--- 测试 1 通过 ---")
    # ---- 测试 4: 推理模式 ----
    print("\n--- 测试 4: 推理模式 (使用测试3的模型) ---")
    model_unc.eval()

    # 定义 ODE 求解器的参数
    ode_kwargs = {
        "t": torch.linspace(0.0, 1.0, steps=10).to(dummy_x.device),
    }

    with torch.no_grad():
        output = model_unc(test_batch, mc_samples=MC_SAMPLES, ode_kwargs=ode_kwargs)

    probs = output['probs']
    print(probs.shape)
    expected_shape = (MC_SAMPLES, BATCH_SIZE, H, W, NUM_CLASSES)

    assert output['loss'] is None, "推理时 loss 应为 None"
    assert output['logits'] is None, "推理时 logits 应为 None"
    assert torch.is_tensor(probs), "Probs 应该是一个张量"
    # assert probs.shape == expected_shape, f"Probs 形状错误, 应该是 {expected_shape}, 实际是 {probs.shape}"
    # 检查概率和是否为1
    # assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0])), "概率和应为1"

    print("前向传播成功")