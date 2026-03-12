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

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        # ContinuousFlowSSN 的 __init__ 需要这个属性
        # --- DINOv3 Backbone ---
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
