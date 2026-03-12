import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from models.mine.dee.freqfusion import FreqFusion
from models.mine.dee.offset_learning import Offset_Learning


class OffSegHead(nn.Module):
    """
    OffSeg decode head (Standalone nn.Module version).
    不再继承自 BaseDecodeHead，作为一个独立的 PyTorch 模块使用。
    """

    def __init__(self,
                 in_channels: list,
                 new_channels: list,
                 align_channels: int,  # 替换原来隐藏在 kwargs 里的 'channels'
                 num_classes: int,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()  # 调用 nn.Module 的构造函数

        # 检查输入参数的合法性
        if not isinstance(in_channels, list) or not isinstance(new_channels, list):
            raise TypeError('in_channels and new_channels must be lists.')
        if len(in_channels) != len(new_channels):
            raise ValueError('in_channels and new_channels must have the same length.')

        self.in_channels = in_channels
        self.new_channels = new_channels
        self.num_classes = num_classes
        self.align_channels = align_channels  # 明确存储 align_channels

        # --- 以下的模块定义逻辑基本保持不变 ---

        self.pre = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.pre.append(
                ConvModule(
                    self.in_channels[i],
                    self.new_channels[i],
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            )

        self.freqfusions = nn.ModuleList()
        # 注意：这里的 in_channels 是 new_channels 翻转后的
        fusion_in_channels = new_channels[::-1]
        pre_c = fusion_in_channels[0]
        for c in fusion_in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c, lr_channels=pre_c,
                compressed_channels=(pre_c + c) // 4,
            )
            self.freqfusions.append(freqfusion)
            pre_c += c

        self.align = ConvModule(
            sum(self.new_channels),
            self.align_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.offset_learning = Offset_Learning(self.num_classes, self.align_channels)

    def forward(self, inputs: list):
        """
        Forward function.
        Args:
            inputs (list[Tensor]): A list of feature maps from the backbone,
                                   e.g., [feature_map_stage1, feature_map_stage2, ...].
        """
        # 不再需要 self._transform_inputs(inputs)，因为我们直接接收一个列表
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} feature maps, but got {len(inputs)}")

        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(self.pre[i](inputs[i]))

        # --- 以下的 forward 逻辑完全保持不变 ---

        fused_inputs = new_inputs[::-1]  # 翻转列表，从深层特征开始处理
        lowres_feat = fused_inputs[0]
        for idx, (hires_feat, freqfusion) in enumerate(zip(fused_inputs[1:], self.freqfusions)):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat([hires_feat.reshape(b * 4, -1, h, w),
                                     lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)

        final_feat = lowres_feat

        output = self.align(final_feat)
        output = self.offset_learning(output)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)

        return output