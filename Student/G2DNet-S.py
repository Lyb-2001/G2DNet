import torch.nn as nn
import torch
import torch.nn.functional as F
from bb.convnextv2_dual1 import convnextv2_femto,LayerNorm
from module.DySample import DySample
# from models.paper2.dee.offset_learning import Offset_Learning
# from toolbox.losses.diffkd.diffkd import DiffKD
import math
from mmcv.ops import DeformConv2d
def conv1x1_bn_relu(in_planes, out_planes, k=1, s=1, p=0, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU()
            )
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class h_swish(nn.Module):
    def __init__(self,inplace = True):
        super(h_swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self,x):
        return x * self.relu(x+3) / 6

class EvidenceGen(nn.Module):

    def __init__(self, in_channels,num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.coarse_head = nn.Conv2d(in_channels,num_classes,kernel_size=1)
        self.refine = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,padding=1,groups=in_channels),nn.BatchNorm2d(in_channels),nn.GELU(),conv1x1_bn_relu(in_channels,in_channels))
        # self.gate_gen = nn.Sequential(conv1x1_bn_relu(1,in_channels//4),conv1x1_bn_relu(in_channels//4,in_channels))
    def forward(self, x):
        logits = self.coarse_head(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1
        S = torch.sum(alpha,dim=1,keepdim=True)
        uncertainty_map = self.num_classes/(S+1e-6)
        # gates = self.gate_gen(uncertainty_map)
        out = self.refine(uncertainty_map * x)+x
        return out,logits,uncertainty_map

class SegFormerHeadup(nn.Module):  # 定义SegFormer的头部模块类，继承自nn.Module
    def __init__(self, num_classes=6, in_channels=[48, 96, 192, 384], embedding_dim=256, dropout_ratio=0.1):
        super(SegFormerHeadup, self).__init__()  # 调用父类nn.Module的初始化函数
        # 对每一层的输入通道数进行解构
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # 为每一层定义一个MLP模块，用于学习抽象表示
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        # 定义一个卷积模块，用于融合四层的特征表示
        self.linear_fuse = conv1x1_bn_relu(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        # self.coa = CoordAtt(embedding_dim, embedding_dim)
        self.up2 = DySample(in_channels=embedding_dim,scale=2)
        self.up4 = DySample(in_channels=embedding_dim,scale=4)
        self.up8 = DySample(in_channels=embedding_dim,scale=8)
    def forward(self, inputs):  # 定义SegFormer头部模块类的前向传播函数
        c1, c2, c3, c4 = inputs  # 对输入特征进行解构
        # 对每一层的特征进行解码

        n, _, h, w = c4.shape  # 从c4的形状中获取batch大小n，高度h和宽度w
        # 对c4特征进行MLP处理，并改变维度顺序
        # 对c4特征进行MLP处理，并改变维度顺序
        # print(c4.shape)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.up8(_c4)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.up4(_c3)
        # 对c2特征进行MLP处理，并改变维度顺序
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c2 = self.up2(_c2)
        # 对c1特征进行MLP处理，并改变维度顺序
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # 将四层的特征进行拼接，然后通过卷积模块进行融合
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(_c)

        x = self.dropout(_c)# 对融合后的特征进行dropout操作
        return self.linear_pred(x)  # 返回预测结果

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer.

    A FiLM layer takes two inputs:
    1. A high-resolution feature map to be modulated (F_detail).
    2. A low-resolution conditioning feature map (F_global).

    It generates a pair of parameters (gamma, beta) from F_global and
    applies them to F_detail in a feature-wise affine transformation.
    """

    def __init__(self, detail_channels, global_channels):
        """
        Args:
            detail_channels (int): Number of channels in the high-res feature map.
            global_channels (int): Number of channels in the low-res feature map.
        """
        super().__init__()
        self.detail_channels = detail_channels

        # This layer generates the modulation parameters.
        # It takes F_global and outputs 2 * C_d channels (for gamma and beta).
        self.param_generator = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),
                                             nn.Linear(global_channels,detail_channels//4),nn.ReLU(),
                                             nn.Linear(detail_channels//4,detail_channels*2))

        # self.up8 = DySample(in_channels=detail_channels * 2, scale=8)
    def forward(self, f_detail, f_global):
        """
        Args:
            f_detail (torch.Tensor): The high-resolution feature map.
                                     Shape: (B, C_d, H, W)
            f_global (torch.Tensor): The low-resolution conditioning feature map.
                                     Shape: (B, C_g, h, w)

        Returns:
            torch.Tensor: The modulated feature map. Shape: (B, C_d, H, W)
        """
        # 1. Get the target spatial size from f_detail
        target_size = f_detail.shape[2:]  # (H, W)


        params = self.param_generator(f_global)


        gamma, beta = torch.chunk(params, chunks=2, dim=1)

        f_final = f_detail * (1 + gamma.unsqueeze(2).unsqueeze(3)) + beta.unsqueeze(2).unsqueeze(3)

        return f_final
class stuNet(nn.Module):
    def __init__(self):
        super(stuNet, self).__init__()
        self.backbone = convnextv2_femto()
        self.backbone.load_state_dict(torch.load("/home/yuride/lyb/bb/convnextv2_femto_1k_224_ema.pt", map_location="cpu")['model'],strict=False)
        self.backbone.stemt[0].weight.data=self.backbone.downsample_layers[0][0].weight.data
        self.backbone.stemt[0].bias.data=self.backbone.downsample_layers[0][0].bias.data
        self.backbone.stemt[1].weight.data=self.backbone.downsample_layers[0][1].weight.data
        self.backbone.stemt[1].bias.data=self.backbone.downsample_layers[0][1].bias.data
        # self.fu1 = conv1x1_bn_relu(96,48)
        # self.fu2 = conv1x1_bn_relu(192,96)
        # self.fu3 = conv1x1_bn_relu(384,192)
        # self.fu4 = conv1x1_bn_relu(768,384)
        self.de = SegFormerHeadup()
    def forward(self, r,t):
        # r1, r2, r3, r4 = self.backbone.forward_features(r,modality='rgb')
        x1, x2, x3, x4 = self.backbone.forward_features(r,t)
        # t1, t2, t3, t4 = self.backbone.forward_features(t,modality='t')
        # x1 = self.fu1(torch.concat((r1,t1),dim=1))
        # x2 = self.fu2(torch.concat((r2,t2),dim=1))
        # x3 = self.fu3(torch.concat((r3,t3),dim=1))
        # x4 = self.fu4(torch.concat((r4,t4),dim=1))
        # x1 = r1+t1
        # x2 = r2+t2
        # x3 = r3+t3
        # x4 = r4+t4
        out = self.de((x1,x2,x3,x4))
        out = F.interpolate(out, size=(480,640), mode='bilinear')
        feat = [x1,x2,x3,x4]

        return out,feat

if __name__ == "__main__":
    with torch.no_grad():
        model = stuNet().cuda()
        print(model)
        print("Model created successfully!")
        # print(model.ir_encoder)
        # print(model)
        # Test with dummy inputs
        rgb_input = torch.randn(1, 3, 480, 640).cuda()  # RGB image
        ir_input = torch.randn(1, 3, 480, 640).cuda()  # IR image
        # tf = torch.randn(6,768).cuda()  # IR image

        out = model(rgb_input,ir_input)[0]
        print(f"Output shapes: {out.shape}")
        # print(f"Output shapes: {out[1].shape}")

        # Check trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")