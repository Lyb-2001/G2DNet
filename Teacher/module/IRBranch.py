import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from module.DySample import DySample


def conv1x1_bn_relu(in_planes, out_planes, k=1, s=1, p=0, b=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(),
    )


class GradientProcessor(nn.Module):
    """梯度处理器，用于提取垂直和水平边缘信息"""

    def __init__(self):
        super(GradientProcessor, self).__init__()
        # 定义垂直和水平梯度滤波器
        self.register_buffer('K_v', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('K_h', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入IRSI图像 [B, 3, H, W]
        Returns:
            u_v: 垂直边缘特征 [B, 3, H, W]
            u_h: 水平边缘特征 [B, 3, H, W]
        """
        B, C, H, W = x.shape

        # 对每个通道分别应用梯度滤波器
        u_v_channels = []
        u_h_channels = []

        for i in range(C):
            channel = x[:, i:i + 1, :, :]  # [B, 1, H, W]

            # 应用垂直梯度滤波器
            v_filtered = F.conv2d(channel, self.K_v, padding=1)
            u_v_channels.append(v_filtered)

            # 应用水平梯度滤波器
            h_filtered = F.conv2d(channel, self.K_h, padding=1)
            u_h_channels.append(h_filtered)

        u_v = torch.cat(u_v_channels, dim=1)  # [B, 3, H, W]
        u_h = torch.cat(u_h_channels, dim=1)  # [B, 3, H, W]

        return u_v, u_h


class MixedGradientEquation(nn.Module):
    """混合梯度方程，整合多方向边缘信息"""

    def __init__(self, eps: float = 1e-6):
        super(MixedGradientEquation, self).__init__()
        self.eps = eps

    def forward(self, u_v: torch.Tensor, u_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u_v: 垂直边缘特征 [B, 3, H, W]
            u_h: 水平边缘特征 [B, 3, H, W]
        Returns:
            gradient_magnitude: 梯度幅值 [B, 3, H, W]
            gradient_direction: 梯度方向 [B, 3, H, W]
        """
        # 计算每个通道的梯度幅值
        gradient_magnitude = torch.sqrt(u_v ** 2 + u_h ** 2 + self.eps)
        gradient_direction = torch.atan2(u_v, u_h + 1e-8)
        return gradient_magnitude, gradient_direction


class ConvU(nn.Module):
    """Conv + Norm + ReLU的组合单元"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super(ConvU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class FAM(nn.Module):
    """特征聚合模块，增强边缘特征表达"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super(FAM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 边缘增强注意力
        self.edge_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_conv = self.norm(self.conv(x))
        x_mean = self.pool(x).repeat(1, 1, H, W)

        # 边缘增强
        edge_enhanced = x_conv * (x - x_mean)

        # 注意力机制
        attention = self.edge_attention(edge_enhanced)
        edge_enhanced = edge_enhanced * attention

        out = self.relu(self.norm1(edge_enhanced + x))
        return out


class DirectionPreservationModule(nn.Module):
    """梯度方向保持模块"""

    def __init__(self, channels: int):
        super(DirectionPreservationModule, self).__init__()
        self.channels = channels
        self.direction_conv = nn.Conv2d(channels + 3, channels, 3, 1, 1)  # +3 for direction channels
        self.direction_norm = nn.BatchNorm2d(channels)

        # 方向编码器
        self.direction_encoder = nn.Sequential(
            nn.Conv2d(3, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, gradient_direction: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [B, C, H, W]
            gradient_direction: 梯度方向 [B, 3, H, W]
        """
        B, C, H, W = features.shape

        # 调整方向特征尺寸匹配
        if gradient_direction.shape[2:] != (H, W):
            gradient_direction = F.interpolate(gradient_direction, size=(H, W), mode='bilinear', align_corners=False)

        # 编码方向信息
        direction_weight = self.direction_encoder(gradient_direction)

        # 融合特征和方向信息
        combined = torch.cat([features, gradient_direction], dim=1)
        direction_feat = self.direction_norm(self.direction_conv(combined))

        # 使用方向权重调制特征
        enhanced_feat = direction_feat * direction_weight + features

        return enhanced_feat

class ScaleBlock(nn.Module):
    def __init__(self, pool_factor, in_channels, out_channels, num_fam=1, num_dp=1):
        super(ScaleBlock, self).__init__()
        # 下采样层
        self.pool = nn.MaxPool2d(pool_factor, pool_factor)
        self.convU = ConvU(in_channels, out_channels, 3, 1, 1)
        # 注意：我们假设每个ScaleBlock包含一个DP和一个FAM，但也可以扩展为多个
        self.dp = DirectionPreservationModule(out_channels)
        # self.fam = FAM(out_channels, out_channels)

    def forward(self, x, d):
        x = self.pool(x)
        x = self.convU(x)
        # x = self.fam(x)
        x = self.dp(x, d)  # 需要梯度方向d
        return x

class MultiScaleFeatureEncoder(nn.Module):
    """多尺度特征编码器"""

    def __init__(self, input_channels: int = 3):
        super(MultiScaleFeatureEncoder, self).__init__()

        # 不同尺度的编码层
        self.scale_1 = ScaleBlock(pool_factor=4,in_channels=input_channels,out_channels=32)

        self.scale_2 = ScaleBlock(pool_factor=2,in_channels=32,out_channels=64)

        self.scale_4 = ScaleBlock(pool_factor=2,in_channels=64,out_channels=128)

        self.scale_8 = ScaleBlock(pool_factor=2,in_channels=128,out_channels=256)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 边缘特征 [B, 3, H, W]
            d: 梯度方向 [B, 3, H, W]
        Returns:
            multi_scale_features: 多尺度边缘特征
        """
        # 1倍尺度 (实际是4倍下采样)
        feat_1 = self.scale_1(x,d)

        # 2倍下采样 (相对于feat_1)
        feat_2 = self.scale_2(feat_1,d)

        # 4倍下采样 (相对于feat_1)
        feat_4 = self.scale_4(feat_2,d)

        # 8倍下采样 (相对于feat_1)
        feat_8 = self.scale_8(feat_4,d)

        return feat_1, feat_2, feat_4, feat_8


class EdgeFeatureGuidance(nn.Module):
    """边缘特征引导模块"""

    def __init__(self):
        super(EdgeFeatureGuidance, self).__init__()
        self.gp = GradientProcessor()
        self.mge = MixedGradientEquation()
        self.msfe = MultiScaleFeatureEncoder()

    def forward(self, irsi: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            irsi: 红外图像 [B, 3, H, W]
        Returns:
            bound: 边界预测结果
            attn_features: 注意力特征列表
        """
        # 梯度处理
        u_v, u_h = self.gp(irsi)

        # 混合梯度方程
        edge_repr, direct = self.mge(u_v, u_h)

        # 多尺度特征编码
        r1, r2, r3, r4 = self.msfe(edge_repr, direct)

        return r1,r2,r3,r4

# class MLP(nn.Module):
#     """
#     Linear Embedding
#     """
#     def __init__(self, input_dim=2048, embed_dim=768):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, embed_dim)
#
#     def forward(self, x):
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x
# class SegFormerHead(nn.Module):  # 定义SegFormer的头部模块类，继承自nn.Module
#     def __init__(self, num_classes=96, in_channels=[128, 256, 512, 1024], embedding_dim=512, dropout_ratio=0.1):
#         super(SegFormerHead, self).__init__()  # 调用父类nn.Module的初始化函数
#         # 对每一层的输入通道数进行解构
#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
#
#         # 为每一层定义一个MLP模块，用于学习抽象表示
#         self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
#         # self.linear_out = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
#         # self.gbc = GBC(embedding_dim)
#         # 定义一个卷积模块，用于融合四层的特征表示
#         self.linear_fuse = conv1x1_bn_relu(embedding_dim * 4, embedding_dim)
#         # 定义一个卷积层，用于最终的预测
#         self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
#         # self.bird_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)
#         # self.bound_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)
#         # 定义一个dropout层，用于防止过拟合
#         self.dropout = nn.Dropout2d(dropout_ratio)
#         self.up2 = DySample(in_channels=embedding_dim,scale=2)
#         self.up4 = DySample(in_channels=embedding_dim,scale=4)
#         self.up8 = DySample(in_channels=embedding_dim,scale=8)
#     def forward(self, inputs):  # 定义SegFormer头部模块类的前向传播函数
#         c1, c2, c3, c4 = inputs  # 对输入特征进行解构
#         # 对每一层的特征进行解码
#
#         n, _, h, w = c4.shape  # 从c4的形状中获取batch大小n，高度h和宽度w
#         # 对c4特征进行MLP处理，并改变维度顺序
#         # 对c4特征进行MLP处理，并改变维度顺序
#         # print(c4.shape)
#         _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         _c4 = self.up8(_c4)
#         _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         _c3 = self.up4(_c3)
#         # 对c2特征进行MLP处理，并改变维度顺序
#         _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
#
#         _c2 = self.up2(_c2)
#         # 对c1特征进行MLP处理，并改变维度顺序
#         _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
#         # 将四层的特征进行拼接，然后通过卷积模块进行融合
#         _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
#         _c = self.linear_fuse(_c)
#
#         x = self.dropout(_c)# 对融合后的特征进行dropout操作
#
#         out = self.linear_pred(x)  # 对dropout后的特征进行最终预测
#         # aux = self.bird_pred(x)
#         return out  # 返回预测结果