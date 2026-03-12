# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from mmcv.ops import DeformConv2dPack

class Adapter(nn.Module):
    """
    简单的瓶颈结构 Adapter:
    Linear/Conv 1x1 (dim -> dim/r) -> GELU -> Linear/Conv 1x1 (dim/r -> dim)
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        # 保持空间维度不变，只做通道变换，适合插入在Block残差连接旁
        self.down = nn.Linear(dim, dim // reduction)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // reduction, dim)

        # 初始化：将 up 层的权重和偏置初始化为0，
        # 这样在训练初期 Adapter 输出为0，不会破坏预训练主干的特征分布
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # x shape: (N, H, W, C) - ConvNeXt Block 内部大部分时间是 Channel Last
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x


# class RGBAdapter(nn.Module):
#
#     def __init__(self, dim, reduction=4):
#         super().__init__()
#         mid_dim = dim//reduction
#         self.down = nn.Linear(dim, mid_dim)
#         self.down1 = nn.Linear(dim, mid_dim)
#         self.dcn = nn.Sequential(DeformConv2dPack(in_channels=mid_dim,out_channels=mid_dim,kernel_size=3,stride=1,padding=1,groups=mid_dim,deform_groups=1),nn.BatchNorm2d(mid_dim),nn.GELU())
#         self.dcn1 = nn.Sequential(DeformConv2dPack(in_channels=mid_dim,out_channels=mid_dim,kernel_size=3,stride=1,padding=1,groups=mid_dim,deform_groups=1),nn.BatchNorm2d(mid_dim),nn.GELU())
#         # self.norm = nn.BatchNorm2d(mid_dim)
#         # self.act = nn.GELU()
#         self.up = nn.Linear(mid_dim, dim)
#         self.adapter_scale = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x,t):
#         x_emb = self.down(x).permute(0,3,1,2).contiguous()
#         t_emb = self.down1(t).permute(0,3,1,2).contiguous()
#         x_dcn1 = self.dcn(x_emb*t_emb)
#         x_dcn2 = self.dcn1(torch.abs(x_emb-t_emb))
#         # x_dcn = torch.concat((x_dcn1,x_dcn2),dim=1)
#         x_dcn = x_dcn1+x_dcn2
#         out = self.up(x_dcn.permute(0,2,3,1).contiguous())
#         return out*self.adapter_scale + x
class FrequencyDynamicSelection(nn.Module):
    """
    单个频率动态选择单元 (FDS Unit)
    对应 Figure 5 中蓝色的 "FDS" 模块
    """

    def __init__(self, channels, k=4, kernel_size=7):
        super(FrequencyDynamicSelection, self).__init__()
        self.channels = channels
        self.k = k  # 基滤波器的数量 (number of learnable filters)
        self.kernel_size = kernel_size

        # 1. 权重生成网络 (MLP)
        # 对应 Eq(2): Softmax(MLP(Pooling(A_I)))
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, k),
            nn.Softmax(dim=1)
        )
        self.shift = kernel_size//2
        # 2. 可学习的频域滤波器基 (Learnable Filters)
        # 对应 Eq(2) 中的 G。
        # 论文提到 M 是卷积核的空间维度。这里我们在空间域定义基，
        # 在前向传播时通过 FFT 转换到频域，这利用了卷积定理，
        # 既保证了 "Spatial Dimension" 的定义，又实现了 "Frequency Domain" 的滤波。
        self.basis_filters = nn.Parameter(
            torch.randn(k, channels, kernel_size, kernel_size)
        )

    def forward(self, x, guidance):
        """
        x: 输入特征 (RGB 或 NIR 经过初始 Conv 后的特征) [B, C, H, W]
        guidance: 聚合特征 (Ar 或 An) [B, C, H, W]
        """
        b, c, h, w = x.shape

        # --- Step 1: 生成动态权重 ---
        # Pooling(A_I) -> Global Average Pooling
        guidance_vec = torch.mean(guidance, dim=[2, 3])  # [B, C]
        weights = self.mlp(guidance_vec)  # [B, k]

        # --- Step 2: 组合基滤波器生成动态滤波器 ---
        # weights: [B, k] -> [B, k, 1, 1, 1]
        weights = weights.view(b, self.k, 1, 1, 1)

        # basis_filters: [k, C, KH, KW] -> [1, k, C, KH, KW]
        bases = self.basis_filters.unsqueeze(0)

        # 线性组合: Sum(w_i * G_i) -> [B, C, KH, KW]
        dynamic_kernel = torch.sum(weights * bases, dim=1)

        # --- Step 3: 频域应用 (Frequency Domain Application) ---
        # 对应 Eq(3): F^-1( F(I) * DF )

        # 3.1 将动态卷积核 Pad 到与输入特征图相同大小，以便在频域点乘
        # 注意：为了使卷积核中心对齐，通常需要 shift，这里简化为直接 pad
        padding_h = h - self.kernel_size
        padding_w = w - self.kernel_size
        # Pad 格式: (left, right, top, bottom)
        dynamic_kernel_padded = F.pad(dynamic_kernel, (0, padding_w, 0, padding_h))
        dynamic_kernel_centered = torch.roll(
            dynamic_kernel_padded,
            shifts=(-self.shift,-self.shift),
            dims=(-2,-1)
        )
        # 3.2 转换到频域 (使用 rfft2 节省计算量)
        # [B, C, H, W] -> [B, C, H, W//2 + 1] (Complex)
        filter_fft = torch.fft.rfft2(dynamic_kernel_centered, s=(h, w))
        x_fft = torch.fft.rfft2(x.to(torch.float32), s=(h, w))

        # 3.3 频域点乘 (Element-wise multiplication)
        out_fft = x_fft * filter_fft

        # 3.4 逆变换回空域
        out = torch.fft.irfft2(out_fft, s=(h, w))

        return out
class RGBAdapter(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.mid_dim = dim//reduction
        self.dim = dim
        self.kernel_size = 3
        # self.reduce = nn.Sequential(nn.Conv2d(self.mid_dim*2,self.mid_dim,1),nn.GELU(),nn.Conv2d(self.mid_dim,self.mid_dim,3,1,1),nn.GELU())
        self.head = nn.Sequential(
            nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, groups=self.mid_dim), nn.SiLU(),
            nn.Conv2d(self.mid_dim, self.mid_dim*2, kernel_size=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.down = nn.Sequential(LayerNorm(dim),nn.Linear(dim, self.mid_dim))
        self.down1 = nn.Sequential(LayerNorm(dim),nn.Linear(dim, self.mid_dim))
        self.FDS = FrequencyDynamicSelection(self.mid_dim)
        self.up = nn.Linear(self.mid_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        # self.adapter_scale = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x,t):

        x_emb = self.down(x).permute(0,3,1,2).contiguous()
        t_emb = self.down1(t).permute(0,3,1,2).contiguous()
        B, C, H, W = x_emb.shape
        style = self.head(t_emb)
        gamma, beta = style.chunk(2, dim=1)
        common = x_emb * (1.0 + gamma) + beta
        # common = self.reduce(torch.concat((x_emb,t_emb),dim=1))

        out = self.FDS(common,x_emb).permute(0,2,3,1).contiguous()
        return self.up(out) + x


class TAdapter(nn.Module):
    """
    可微形态学注意力适配器 (DMA - Differentiable Morphological Attention)

    原理:
        利用 Soft-Max 和 Soft-Min 算子近似形态学的膨胀(Dilation)和腐蚀(Erosion)。
        相比普通卷积，它具有显著的物理意义：
        1. 膨胀: 捕捉高亮热源区域，填补热断裂。
        2. 腐蚀: 抑制背景噪声，收缩热源骨架。
        3. 梯度: 膨胀 - 腐蚀，直接提取热边界。

    Args:
        dim (int): 输入通道数
        reduction (int): 瓶颈层降维比例
        kernel_size (int): 形态学核大小，建议 3
    """

    def __init__(self, dim, reduction=8, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.mid_dim = dim // reduction
        self.k = kernel_size
        self.pad = kernel_size // 2

        # 1. 降维投影
        self.down = nn.Sequential(LayerNorm(dim),nn.Linear(dim, self.mid_dim))

        # 2. 可学习的结构元素 (Structure Element)
        # 形状: (1, mid_dim, 1, K*K) -> 用于对 unfold 后的窗口进行加权
        # 这相当于形态学操作中的 "Kernel形状"
        self.se_weight = nn.Parameter(torch.ones(1, self.mid_dim, 1, kernel_size * kernel_size))

        # 3. 温度系数 (控制 SoftMax 的锐度)
        # 初始值设为 1/sqrt(k*k)，类似 Attention 的 scaling
        self.temperature = nn.Parameter(torch.ones(1) * (kernel_size ** -0.5))

        # 4. 特征融合层
        # 输入: 原始 + 膨胀 + 腐蚀 + 梯度 (共4倍通道)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.mid_dim * 3, self.mid_dim, kernel_size=1),
            nn.BatchNorm2d(self.mid_dim),  # BN 有助于平衡不同形态学特征的量纲
            nn.GELU()
        )

        # 5. 升维投影
        self.up = nn.Linear(self.mid_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        # self.adapter_scale = nn.Parameter(torch.zeros(1))
        # self.adapter_scale = nn.Parameter(torch.tensor([0.1]))
    def _morph_op(self, x_unfold, mode='dilation'):
        """
        执行软形态学操作
        x_unfold: (N, C, L, K*K)  其中 L=H*W
        """
        # 应用结构元素加权 (Broadcasting)
        # se_weight: (1, C, 1, K*K)
        # x_weighted: 代表窗口内的特征值与结构元素的交互
        x_weighted = x_unfold * self.se_weight

        # 计算注意力分数
        # Dilation 关注最大值 (SoftMax)
        # Erosion 关注最小值 (SoftMin -> SoftMax(-x))
        if mode == 'dilation':
            # scaling 提高数值稳定性
            attn = F.softmax(x_weighted / self.temperature.abs().clamp(min=1e-6), dim=-1)
        elif mode == 'erosion':
            attn = F.softmax(-x_weighted / self.temperature.abs().clamp(min=1e-6), dim=-1)

        # 加权求和 (Expectation)
        # (N, C, L, K*K) * (N, C, L, K*K) -> sum -> (N, C, L)
        out = (x_unfold * attn).sum(dim=-1)
        return out

    def forward(self, x):
        # x: (N, H, W, C)
        N, H, W, C = x.shape

        # 1. 降维 & 维度变换 -> (N, mid_dim, H, W)
        x_emb = self.down(x).permute(0, 3, 1, 2).contiguous()

        # 2. Unfold 提取滑动窗口
        # 输出: (N, mid_dim * K*K, L)
        x_unfold = F.unfold(x_emb, kernel_size=self.k, padding=self.pad)

        # Reshape 为 (N, mid_dim, L, K*K) 以便在最后一个维度做 Attention
        # 这里的 L = H * W
        x_unfold = x_unfold.view(N, self.mid_dim, self.k * self.k, H * W).permute(0, 1, 3, 2)

        # 3. 形态学特征计算
        # A. 软膨胀 (Soft Dilation) -> 补全高亮物体
        feat_dilate = self._morph_op(x_unfold, mode='dilation')  # (N, C, L)

        # B. 软腐蚀 (Soft Erosion) -> 抑制背景噪声
        feat_erode = self._morph_op(x_unfold, mode='erosion')  # (N, C, L)

        # D. 原始特征 (保持恒等映射能力)
        feat_orig = x_emb.flatten(2)  # (N, C, L)

        # 4. 多流特征拼接
        # (N, 3*mid_dim, L)
        feat_cat = torch.cat([feat_orig, feat_dilate, feat_erode], dim=1)

        # 变回图片维度 (N, 4*mid_dim, H, W)
        feat_cat = feat_cat.view(N, -1, H, W)

        # 5. 融合与激活
        out = self.fusion(feat_cat)

        # 6. 变回 Channel Last 并升维
        out = out.permute(0, 2, 3, 1)  # (N, H, W, mid_dim)
        out = self.up(out) + x

        return out

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# class Block(nn.Module):
#     """ ConvNeXtV2 Block.
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#     """
#     def __init__(self, dim, drop_path=0.):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.grn = GRN(4 * dim)
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.ad = Adapter(dim)
#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.grn(x)
#         x = self.pwconv2(x)
#         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
#
#         x = input + self.drop_path(x)
#         return x
class Block(nn.Module):
    """ ConvNeXtV2 Block with Modality-Specific Adapters """

    def __init__(self, dim, drop_path=0., use_adapter=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 分别为 RGB 和 Thermal 定义 Adapter
        self.adapter_rgb = RGBAdapter(dim, reduction=8)
        self.adapter_t = TAdapter(dim, reduction=8)
        # # 这里的 scaling 可以是学习的参数，也可以是固定值，这里简单处理为相加
         # self.adapter_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, t):
        # Adapter 通常作用于 Input 或者 Output。
        # 这里选择 Parallel Adapter 结构：作用于 Block 的输入（input），加到输出上
        # 注意 input 是 (N, C, H, W), Adapter 内部是用 Linear 实现的，需要 (N, H, W, C)
        x_permuted = x.permute(0, 2, 3, 1)
        t_permuted = t.permute(0, 2, 3, 1)
        t_permuted = self.adapter_t(t_permuted)
        x_permuted = self.adapter_rgb(x_permuted,t_permuted)

        input_x = x_permuted.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input_x + self.drop_path(x)
        # --- Adapter 注入逻辑 ---
        # 原始逻辑: x = input + self.drop_path(x)
        # 现在的逻辑: x = input + drop_path(main_feat) + adapter(input_permuted)
        # with torch.no_grad():
        input_t = t_permuted.permute(0, 3, 1, 2)
        t = self.dwconv(t)
        t = t.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        t = self.norm(t)
        t = self.pwconv1(t)
        t = self.act(t)
        t = self.grn(t)
        t = self.pwconv2(t)
        t = t.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        t = input_t + self.drop_path(t)


        return x,t


class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.stemt = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            # 注意：这里使用修改后的 Block
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], use_adapter=True) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward_features(self, x,t):
        # 我们需要返回多尺度的特征用于分割 (Stage 0, 1, 2, 3)
        outs = []
        x = self.downsample_layers[0](x)
        t = self.stemt(t)
        for blk in self.stages[0]:
            x,t = blk(x,t)
        outs.append(x)

        for i in range(1,4):
            x = self.downsample_layers[i](x)
            t = self.downsample_layers[i](t)
            # 手动遍历 Sequential 中的 Block 以传递 modality 参数
            for blk in self.stages[i]:
                x,t = blk(x, t)
            outs.append(x)

        # 如果是做分类返回: self.norm(x.mean([-2, -1]))
        # 但既然是做分割，我们返回特征列表
        return outs

    def forward(self, x,t):
        # 仅用于分类头，如果你做分割，主要用 forward_features
        x = self.forward_features(x,t)

        return x

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model