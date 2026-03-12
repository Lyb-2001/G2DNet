import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from mmcv.ops import DeformConv2d
from mmcv.cnn import ConvModule
# class FeatureCommunicationModule(nn.Module):
#     def __init__(self, dim=768, num_heads=2, num_points=4):
#         super().__init__()
#         inter_dim = dim // 8  # 压缩到 1/8
#
#         # QKV 降维 + 标准化
#         self.q_proj1 = nn.Linear(dim, inter_dim)
#         self.kv_proj1 = nn.Linear(dim, inter_dim)
#         self.norm_q1 = nn.LayerNorm(inter_dim)
#         self.norm_kv1 = nn.LayerNorm(inter_dim)
#         # Deformable Cross Attention
#         self.cross_attn1 = MSDeformAttn(
#             embed_dims=inter_dim, num_heads=num_heads,
#             num_levels=1, num_points=num_points, batch_first=True
#         )
#         # FFN 恢复维度
#         self.norm_ffn = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, inter_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(inter_dim, dim)
#         )
#
#         self.proj_back_to_dim1 = nn.Linear(inter_dim, dim)
#     def _build_ref_points(self, B, H, W, device):
#         """生成归一化参考点 [B, H*W, 1, 2]"""
#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=device),
#             torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=device),
#             indexing='ij'
#         )
#         ref = torch.stack((ref_x, ref_y), -1)  # [H, W, 2]
#         ref = ref.reshape(1, H * W, 1, 2).repeat(B, 1, 1, 1)
#         return ref
#
#     def forward(self, x, y):
#         """
#         fvit_bhwc: [B, H_vit, W_vit, C]
#         fsp_bhwc:  [B, H_sp,  W_sp,  C]
#         """
#         assert x.ndim == 4 and y.ndim == 4, \
#             "输入必须是 BHWC 格式"
#
#         B, H, W, C = x.shape
#         device = y.device
#
#         # 展平成 BNC
#         x = x.view(B, H * W, C)
#         y = y.view(B, -1, C)
#         # print(x.shape)
#         # ---------- Cross-Attn 1: sp <- vit ----------
#         q = self.norm_q1(self.q_proj1(y))
#         kv = self.norm_kv1(self.kv_proj1(x))
#         ref_sp = self._build_ref_points(B, H, W, device)
#         spatial_shapes_vit = torch.as_tensor([[H, W]], dtype=torch.long, device=device)
#         level_start_index_vit = torch.as_tensor([0], dtype=torch.long, device=device)
#
#         y_new = y + self.proj_back_to_dim1(self.cross_attn1(
#             query=q,
#             reference_points=ref_sp,
#             value=kv,
#             spatial_shapes=spatial_shapes_vit,
#             level_start_index=level_start_index_vit
#         ))
#
#         # ---------- FFN ----------
#         y_new = y_new + self.ffn(self.norm_ffn(y_new))
#
#         # 还原回 BHWC
#         y_new = y_new.view(B, H, W, C)
#
#         return y_new
class FeatureCommunicationModule(nn.Module):
    def __init__(self, dim=768, num_heads=2, num_points=4):
        super().__init__()
        inter_dim = dim // 8  # 压缩到 1/8

        # QKV 降维 + 标准化
        self.q_proj1 = nn.Linear(dim, inter_dim)  # SAM 特征 -> query
        self.kv_proj1 = nn.Linear(dim, inter_dim)  # IR 特征 -> key/value
        self.norm_q1 = nn.LayerNorm(inter_dim)
        self.norm_kv1 = nn.LayerNorm(inter_dim)

        # Deformable Cross Attention
        self.cross_attn1 = MSDeformAttn(
            embed_dims=inter_dim, num_heads=num_heads,
            num_levels=1, num_points=num_points, batch_first=True
        )

        # FFN 恢复维度
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(inter_dim, dim)
        )

        self.proj_back_to_dim1 = nn.Linear(inter_dim, dim)

    def _build_ref_points(self, B, H, W, device):
        """生成归一化参考点 [B, H*W, 1, 2]"""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=device),
            torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_x, ref_y), -1)  # [H, W, 2]
        ref = ref.reshape(1, H * W, 1, 2).repeat(B, 1, 1, 1)
        return ref

    def forward(self, sam_feat, ir_feat):
        """
        sam_feat: [B, H_sam, W_sam, C] - SAM 高分辨率特征，作为 query，会被更新
        ir_feat: [B, H_ir, W_ir, C] - IR 低分辨率特征，作为 key/value，提供引导信息

        Returns: 增强后的 SAM 特征，尺寸与输入的 sam_feat 相同
        """
        assert sam_feat.ndim == 4 and ir_feat.ndim == 4, \
            "输入必须是 BHWC 格式"

        B, H_sam, W_sam, C = sam_feat.shape
        B, H_ir, W_ir, C = ir_feat.shape
        device = sam_feat.device

        # 展平成 BNC
        sam_flat = sam_feat.view(B, H_sam * W_sam, C)  # [B, H_sam*W_sam, C]
        ir_flat = ir_feat.view(B, H_ir * W_ir, C)  # [B, H_ir*W_ir, C]

        # ---------- Cross-Attn: SAM特征(query) <- IR特征(key/value) ----------
        q = self.norm_q1(self.q_proj1(sam_flat))  # [B, H_sam*W_sam, inter_dim]
        kv = self.norm_kv1(self.kv_proj1(ir_flat))  # [B, H_ir*W_ir, inter_dim]

        # 参考点对应 query 的空间位置 (SAM 特征的高分辨率位置)
        ref_points = self._build_ref_points(B, H_sam, W_sam, device)  # [B, H_sam*W_sam, 1, 2]

        # 空间形状对应 key/value 的空间形状 (IR 特征的低分辨率)
        spatial_shapes = torch.as_tensor([[H_ir, W_ir]], dtype=torch.long, device=device)
        level_start_index = torch.as_tensor([0], dtype=torch.long, device=device)

        # Deformable attention: 在低分辨率IR特征图上采样，更新高分辨率SAM特征
        sam_enhanced = sam_flat + self.proj_back_to_dim1(self.cross_attn1(
            query=q,
            reference_points=ref_points,
            value=kv,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        ))

        # ---------- FFN ----------
        sam_enhanced = sam_enhanced + self.ffn(self.norm_ffn(sam_enhanced))

        # 还原回 BHWC (SAM 的原始高分辨率)
        sam_enhanced = sam_enhanced.view(B, H_sam, W_sam, C)

        return sam_enhanced
class MonaOp1(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

        self.prompt = torch.nn.parameter.Parameter(torch.randn(in_features, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(in_features), requires_grad=True)


    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        identity = x
        x = x.permute(0, 2, 3, 1).reshape(B, N, C)  # 调整为 [B, N, C]
        cos_sim = F.normalize(x, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        # B, N, 1
        mask = cos_sim.clamp(0, 1)
        x = x * mask
        x = x @ self.top_down_transform

        x = x.reshape(B, H, W, C)  # 恢复为 [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # 调整回 [B, C, H, W]
        x = self.projector(x)
        # x = x.permute(0, 2, 3, 1)
        x = identity + x

        return x


class MonaOpMultiClass(nn.Module):
    def __init__(self, in_features, num_classes, prompt_dim=64):
        super().__init__()
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

        # 多类别 prompt，降维存储，节省内存
        self.prompts = nn.Parameter(torch.randn(num_classes, prompt_dim))
        self.prompt_mapper = nn.Linear(prompt_dim, in_features, bias=False)
        self.prompt_mapper1 = nn.Linear(prompt_dim, in_features, bias=False)

        # 共享变换矩阵，避免每类一个矩阵（暴涨内存）
        self.top_down_transform = nn.Parameter(torch.eye(in_features))

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        identity = x
        x = x.reshape(B, N, C)
        x = x @ self.top_down_transform
        # 映射到特征维度
        prompts_full = self.prompt_mapper(self.prompts)  # [num_classes, C]

        # 计算所有类的相似度
        cos_sim = torch.einsum(
            'bnc,kc->bnk',
            F.normalize(x, dim=-1),
            F.normalize(prompts_full, dim=-1)
        )  # [B, N, num_classes]

        # 用 max 或 softmax 汇聚多类响应（可切换）
        # mask, _ = cos_sim.max(dim=-1, keepdim=True)  # [B, N, 1]
        prompts1 = self.prompt_mapper1(self.prompts)
        mask = torch.einsum(
            'bnk,kc->bnc',
            cos_sim,
            prompts1
        )  # [B, N, num_classes]
        mask = mask.clamp(0, 1)

        # 加权 + 共享变换
        x = x * mask


        # 恢复形 + 投影 + 残差
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.projector(x).permute(0, 2, 3, 1)
        x = identity + x
        return x,self.prompts
class MonaOpClip(nn.Module):
    def __init__(self, in_features, prompt_dim=96):
        super().__init__()
        # self.irconv = nn.Conv2d(in_features//4,in_features,1)
        self.projector = nn.Linear(in_features, in_features)
        self.prompt_mapper = nn.Linear(prompt_dim, in_features)
        self.prompt_mapper1 = nn.Linear(prompt_dim, in_features)

    def forward(self, x,irf,tf):
        # print(x)
        # irf = self.irconv(irf)
        B, C, H, W= x.shape
        # print(x.shape)
        N = H * W
        x = x.permute(0, 2, 3, 1)
        x1 = x + irf.permute(0, 2, 3, 1)*x
        identity = x
        x1 = x1.reshape(B, N, C)
        # 映射到特征维度
        prompts_full = self.prompt_mapper(tf)  # [num_classes, C]

        # 计算所有类的相似度
        cos_sim = torch.einsum(
            'bnc,kc->bnk',
            F.normalize(x1, dim=-1),
            F.normalize(prompts_full, dim=-1)
        )  # [B, N, num_classes]

        prompts1 = self.prompt_mapper1(tf)
        x1 = torch.einsum(
            'bnk,kc->bnc',
            cos_sim,
            prompts1
        )  # [B, N, num_classes]

        x1 = identity * x1.reshape(B, H, W, C)
        out = self.projector(x1)+identity
        return out.permute(0, 3, 1, 2)
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
class MonaOpClipDowm(nn.Module):
    def __init__(self, in_features, prompt_dim=96):
        super().__init__()
        # self.irconv = nn.Conv2d(in_features//4,in_features,1)
        # self.projector = nn.Linear(in_features, in_features)
        self.tf = torch.randn(6,96).cuda()
        self.projector = nn.Linear(in_features, in_features)
        self.prompt_mapper = nn.Linear(prompt_dim, in_features)
        self.prompt_mapper1 = nn.Linear(prompt_dim, in_features)
        self.dowm = nn.Linear(in_features,64)
        self.up = nn.Linear(64,in_features)
        self.irfu = ir_rgbfuse(64)
    def forward(self, x,irf):
        # print(x)
        # irf = self.irconv(irf)
        B, C, H, W= x.shape
        # print(x.shape)
        N = H * W
        x = x.permute(0, 2, 3, 1)
        # x1 = x + irf.permute(0, 2, 3, 1)*x
        x1 = self.up(self.irfu(self.dowm(x),irf)) + x
        identity = x
        x1 = x1.reshape(B, N, C)
        # 映射到特征维度
        prompts_full = self.prompt_mapper(self.tf)  # [num_classes, C]

        # 计算所有类的相似度
        cos_sim = torch.einsum(
            'bnc,kc->bnk',
            F.normalize(x1, dim=-1),
            F.normalize(prompts_full, dim=-1)
        )  # [B, N, num_classes]

        prompts1 = self.prompt_mapper1(self.tf)
        x1 = torch.einsum(
            'bnk,kc->bnc',
            cos_sim,
            prompts1
        )  # [B, N, num_classes]

        x1 = identity * x1.reshape(B, H, W, C)
        out = self.projector(x1)+identity
        return out.permute(0, 3, 1, 2)


class MonaUncertaintyPrompt(nn.Module):
    def __init__(self, in_features, num_classes, prompt_dim=64):
        super().__init__()
        # self.projector = nn.Sequential(nn.Conv2d(in_features//4, in_features, kernel_size=1),nn.BatchNorm2d(in_features),nn.GELU())
        self.projector = nn.Linear(in_features//8, in_features)
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
        # 1. 语义锚点 (Semantic Anchors / Prompts)
        # 代表每个类别的"理想特征中心"
        self.prompts = nn.Parameter(torch.randn(num_classes, prompt_dim))
        self.head = nn.Sequential(nn.Conv2d(in_features//8*3,in_features//8*3,kernel_size=3,padding=1,groups=in_features//8*3),nn.SiLU(),nn.Conv2d(in_features//8*3,in_features//4,kernel_size=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # 将 Prompt 映射回特征空间的 Key 和 Value
        self.prompt_key = nn.Linear(prompt_dim, in_features//8)  # 用于计算相似度
        self.prompt_value = nn.Linear(prompt_dim, in_features//8)  # 用于重构特征

        # 共享变换矩阵
        # self.top_down_transform = nn.Parameter(torch.eye(in_features))
        self.top_down_transform = nn.Sequential(LayerNorm(in_features),nn.Linear(in_features, in_features//8))
        # self.top_down_transform = nn.Sequential(LayerNorm(in_features),nn.Linear(in_features, in_features//4),nn.GELU(),GRN(in_features//4))
        self.q_transform = nn.Linear(in_features//8, in_features//8)
        # self._init_weights()
        # # 不确定性调节系数 (可选，让网络自己学温度系数)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

        # self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x,ir):
        B, C, H, W = x.shape
        N = H * W
        identity = x
        # print(ir.shape)
        style = self.head(ir)
        gamma,beta = style.chunk(2,dim=1)
        gamma = gamma.permute(0, 2, 3, 1).reshape(B, N, C//8)
        beta = beta.permute(0, 2, 3, 1).reshape(B, N, C//8)
        # --- A. 特征变换 ---
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        # 你的线性变换逻辑保持不变
        # x_trans = x_flat @ self.top_down_transform
        x_trans = self.top_down_transform(x_flat)
        # print(x_trans.shape)
        x_trans = x_trans*(1.0+gamma)+beta
        q_trans = self.q_transform(x_trans)

        # --- B. 原型交互 (Prototype Interaction) ---
        # 1. 准备 Key 和 Value
        # [num_classes, C]
        prompt_k = self.prompt_key(self.prompts)
        prompt_v = self.prompt_value(self.prompts)

        # 2. 计算语义相似度 (Semantic Similarity)
        # 归一化非常重要，这保证了距离度量的几何意义
        x_norm = F.normalize(q_trans, dim=-1)
        p_norm = F.normalize(prompt_k, dim=-1)

        # cos_sim: [B, N, num_classes]
        # 值域 [-1, 1], 1代表完全重合
        cos_sim = torch.einsum('bnc,kc->bnk', x_norm, p_norm)

        # --- C. 计算不确定性 (Uncertainty Estimation) ---
        # 理论核心：
        # 如果一个特征属于任何已知类，它的 max(cos_sim) 应该接近 1。
        # 如果 max(cos_sim) 很小，说明这个特征既不像A，也不像B，它是"未知的" (OOD)。

        # max_sim: [B, N, 1]
        max_sim, _ = cos_sim.max(dim=-1, keepdim=True)

        uncertainty = 1.0 - (max_sim+1.0)/2.0  # 简单有效的距离不确定性

        # --- D. 基于原型的重构 (Prototype Reconstruction) ---
        # 计算 Attention 权重
        attn = F.softmax(cos_sim / self.temperature, dim=-1)  # [B, N, num_classes]

        # 根据相似度，组合出一个"完美的"、"去噪的"特征
        # x_proto: [B, N, C]
        x_proto = torch.einsum('bnk,kc->bnc', attn, prompt_v)

        # --- E. 不确定性引导的动态校准 (Dynamic Calibration) ---
        # 你的原版是 x * mask，这是一种硬性的注意力。
        # 这里改为：根据不确定性，在"原始观察"和"先验知识"之间做权衡。

        # Logic:
        # - 如果 uncertainty 小: 我很自信，保留 x_trans (原始细节)
        # - 如果 uncertainty 大: 我看不清，用 x_proto (原型先验) 来代替/补充
        # 混合特征
        x_calibrated = (1 - uncertainty) * x_trans + uncertainty * x_proto
        # 这里推荐加法混合，梯度更稳：
        out_flat = x_calibrated

        # --- F. 恢复与输出 ---
        out = out_flat.reshape(B, H, W, C//8)
        out = self.projector(out).permute(0, 3, 1, 2) + identity
        # 返回 output 和 不确定性图 (B, 1, H, W)
        return out, uncertainty.reshape(B, 1, H, W)
# class MonaProjectionUncertainty(nn.Module):
#     def __init__(self, in_features, prompt_dim=64, num_prototypes=16):
#         super().__init__()
#
#         # 1. 可学习的原型 (Knowledge Basis)
#         # 代表语义空间中的基向量 (Basis Vectors)
#         self.prototypes = nn.Parameter(torch.randn(num_prototypes, prompt_dim))
#
#         # 2. 映射器
#         self.prompt_key = nn.Linear(prompt_dim, in_features)
#         self.prompt_value = nn.Linear(prompt_dim, in_features)
#
#         # 3. 基础变换 (保持轻量)
#         self.down = nn.Linear(in_features, 64)
#         self.act = nn.GELU()
#         self.up = nn.Linear(64, in_features)
#
#         # 4. 不确定性调制器 (Uncertainty Modulator)
#         # 将计算出的物理残差(标量)映射为融合权重
#         self.uncertainty_gate = nn.Sequential(
#             nn.Linear(in_features, in_features),
#             nn.Sigmoid()
#         )
#
#         self.projector = nn.Linear(in_features, in_features)
#
#     def forward(self, x):
#         # x: [B, C, H, W]
#         B, C, H, W = x.shape
#         N = H * W
#
#         # --- A. 基础特征变换 ---
#         x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
#         identity = x_perm
#
#         # 瓶颈层处理
#         x_local = self.up(self.act(self.down(x_perm))) + x_perm
#         x_local_flat = x_local.reshape(B, N, C)  # [B, N, C]
#
#         # --- B. 基于原型的流形投影 (Manifold Projection) ---
#         # 生成 Key 和 Value
#         p_keys = self.prompt_key(self.prototypes)  # [K, C]
#         p_values = self.prompt_value(self.prototypes)  # [K, C]
#
#         # 计算相似度 (Projection Coefficients)
#         # 注意：这里我们归一化，模拟“方向”上的对齐
#         x_norm = F.normalize(x_local_flat, dim=-1)
#         k_norm = F.normalize(p_keys, dim=-1)
#
#         # Cosine Similarity [B, N, K]
#         scores = torch.einsum('bnc,kc->bnk', x_norm, k_norm)
#
#         # # Attention Weights [B, N, K]
#         # attn = F.softmax(scores * (C ** -0.5), dim=-1)
#
#         # 重构特征 (Reconstructed Feature) - 这是特征在"安全流形"上的投影
#         # [B, N, K] x [K, C] -> [B, N, C]
#         x_rec = torch.einsum('bnk,kc->bnc', scores, p_values)
#
#         # --- C. 计算正交投影误差 (Orthogonal Projection Error) ---
#         # 理论核心：无法被原型解释的部分就是不确定性
#         # 残差向量 [B, N, C]
#         residual_vector = x_local_flat - x_rec
#
#         # 计算残差的能量 (L2 Norm) 作为不确定性度量
#         # uncertainty_raw: [B, N, 1]
#         # uncertainty_raw = torch.norm(residual_vector, p=2, dim=-1, keepdim=True)
#
#         # 归一化或门控，将其转化为 0~1 的系数
#         # gate 越接近 1，表示不确定性越高；接近 0，表示特征很置信
#         gate = self.uncertainty_gate(residual_vector.detach())
#
#         # --- D. 不确定性引导的校准 ---
#         # 策略：
#         # 如果 gate 小 (置信): 我们信任 x_local (保留局部细节)
#         # 如果 gate 大 (不确信): 我们信任 x_rec (依赖先验知识/原型)
#
#         out_flat = (1 - gate) * x_local_flat + gate * x_rec
#
#         # 最终投影
#         out = self.projector(out_flat) + identity.reshape(B, N, C)
#
#         # reshape back
#         out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
#
#         # 返回 output 和 不确定性图(用于可视化分析)
#         return out, gate.reshape(B, C, H, W)
# class MonaOpClipDowm(nn.Module):
#     def __init__(self, in_features, prompt_dim=96):
#         super().__init__()
#         # self.irconv = nn.Conv2d(in_features//4,in_features,1)
#         # self.projector = nn.Linear(in_features, in_features)
#         self.tf = torch.randn(6,96).cuda()
#         self.projector = nn.Linear(in_features, in_features)
#         self.prompt_mapper = nn.Linear(prompt_dim, in_features)
#         self.prompt_mapper1 = nn.Linear(prompt_dim, in_features)
#         self.dowm = nn.Linear(in_features,64)
#         self.up = nn.Linear(64,in_features)
#         self.irfu = ir_rgbfuse(64)
#     def forward(self, x,irf):
#         # print(x)
#         # irf = self.irconv(irf)
#         B, C, H, W= x.shape
#         # print(x.shape)
#         N = H * W
#         x = x.permute(0, 2, 3, 1)
#         # x1 = x + irf.permute(0, 2, 3, 1)*x
#         x1 = self.up(self.irfu(self.dowm(x),irf)) + x
#         identity = x
#         x1 = x1.reshape(B, N, C)
#         # 映射到特征维度
#         prompts_full = self.prompt_mapper(self.tf)  # [num_classes, C]
#
#         # 计算所有类的相似度
#         cos_sim = torch.einsum(
#             'bnc,kc->bnk',
#             F.normalize(x1, dim=-1),
#             F.normalize(prompts_full, dim=-1)
#         )  # [B, N, num_classes]
#
#         prompts1 = self.prompt_mapper1(self.tf)
#         x1 = torch.einsum(
#             'bnk,kc->bnc',
#             cos_sim,
#             prompts1
#         )  # [B, N, num_classes]
#
#         x1 = identity * x1.reshape(B, H, W, C)
#         out = self.projector(x1)+identity
#         return out.permute(0, 3, 1, 2)

class ir_rgbfuse(nn.Module):
    def __init__(self, channels,
                 deform_groups=1,
                 kernel_size=3,
                 dilation=1,
                 padding=1):
        super().__init__()
        self.offset_generator = nn.Conv2d(channels * 2, deform_groups * 2 * kernel_size * kernel_size, 3, padding=1)
        self.modulation = nn.Sequential(nn.Conv2d(channels,channels,1),nn.Sigmoid())
        self.deform_conv = DeformConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=deform_groups,
            deform_groups=deform_groups,
        )
    def forward(self, x,irf):
        #x BHWC
        #irf BCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        offset = self.offset_generator(torch.concat((x,irf),dim=1))
        x1 = self.deform_conv(x, offset) * self.modulation(irf)
        return x1.permute(0, 2, 3, 1).contiguous()
# class MonaOpMultiClass(nn.Module):
#     def __init__(self, in_features, num_classes, prompt_dim=32,rank = 64):
#         super().__init__()
#         self.proj_down = nn.Linear(in_features, rank, bias=False)
#         self.proj_up = nn.Linear(rank, in_features, bias=False)
#
#         # 多类别 prompt，降维存储，节省内存
#         self.prompts = nn.Parameter(torch.randn(num_classes, prompt_dim))
#         self.prompt_mapper = nn.Linear(prompt_dim, rank, bias=False)
#
#         # 共享变换矩阵，避免每类一个矩阵（暴涨内存）
#         self.top_down_transform = nn.Parameter(torch.eye(rank))
#
#     def forward(self, x):
#         B, H, W, C = x.shape
#         N = H * W
#         identity = x
#         x = x.reshape(B, N, C)
#         x_low = self.proj_down(x)
#         # 映射到特征维度
#         prompts_full = self.prompt_mapper(self.prompts)  # [num_classes, C]
#
#         # 计算所有类的相似度
#         cos_sim = torch.einsum(
#             'bnc,kc->bnk',
#             F.normalize(x_low, dim=-1),
#             F.normalize(prompts_full, dim=-1)
#         )  # [B, N, num_classes]
#
#         # 用 max 或 softmax 汇聚多类响应（可切换）
#         mask, _ = cos_sim.max(dim=-1, keepdim=True)  # [B, N, 1]
#         mask = mask.clamp(0, 1)
#
#         # 加权 + 共享变换
#         x = x_low * mask
#         x = x @ self.top_down_transform
#
#         # 恢复形状 + 投影 + 残差
#         x = x.reshape(B, H, W, -1)
#         x = self.proj_up(x)
#         x = identity + x
#         return x

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x
class Mona(nn.Module):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        x = x.permute(0, 2, 3, 1)
        identity = x
        # print(x.shape)
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        # b, n, c = project1.shape
        # h, w = hw_shapes
        project1 = project1.permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return (identity + project2).permute(0, 3, 1, 2)


class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim, window_size=8, k=3, ratio=0.5):
        super().__init__()

        self.window_size = window_size
        self.ratio = ratio
        cdim = dim + k
        embed_dim = window_size**2

        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            # LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//8, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(cdim//8, 2, 1),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )


    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):

        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        # offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)

        x = torch.mean(x, keepdim=True, dim=1)

        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if self.training or train_mode:
            return mask, offsets, ca, sa
        else:
            score = pred_score[:, : , 0]
            B, N = score.shape
            r = torch.mean(mask,dim=(0,1))
            num_keep_node = int(N * r)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], offsets, ca, sa


class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, shift_size=0, ratio=0.5):
        super().__init__()

        self.dim = dim
        self.window_size = window_size

        self.ratio = ratio
        k = 5
        d = 3

        self.shift_size = shift_size

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Linear(dim, dim, bias = bias)
        self.project_k = nn.Linear(dim, dim, bias = bias)

        # Conv
        # self.conv_sptial = nn.Conv2d(dim, dim, kernel_size=3, bias=True, groups=dim, padding=1)
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d)
        )

        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        self.act = nn.GELU()
        # Predictor
        self.route = PredictorLG(dim,window_size,ratio=ratio)

    def forward(self,x,condition_global=None, mask=None, train_mode=False):
        N,C,H,W = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        v = self.project_v(shifted_x)

        if True:
            condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size),torch.linspace(-1,1,self.window_size)))\
                    .type_as(x).unsqueeze(0).repeat(N, 1, H//self.window_size, W//self.window_size)
            if self.shift_size > 0:
                condition_wind = torch.roll(condition_wind, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)

        mask, offsets, ca, sa = self.route(_condition,ratio=self.ratio,train_mode=train_mode)

        # cyclic shift
        x_warped = flow_warp(x, offsets.permute(0,2,3,1), interp_mode='bilinear', padding_mode='border')

        if self.shift_size > 0:
            shifted_x_warped = torch.roll(x_warped, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x_warped = x_warped


        q = shifted_x_warped
        k = shifted_x_warped
        qk = torch.cat([q,k],dim=1)

        # Attn branch
        vs = v*sa

        v  = rearrange(v,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        if self.training or train_mode:
            N_ = v.shape[1]
            v1,v2 = v*mask, vs*(1-mask)
            qk1 = qk*mask
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1,v2 = batch_index_select(v,idx1),batch_index_select(vs,idx2)
            qk1 = batch_index_select(qk,idx1)

        v1 = rearrange(v1,'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1,'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1,k1 = torch.chunk(qk1,2,dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)


        #calculate attention: Softmax(Q@K)@V
        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn @ v1

        f_attn = rearrange(f_attn,'(b n) (dh dw) c -> b n (dh dw c)',
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)',
            h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size
        )

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            attn_out = attn_out

        out = attn_out
        out = self.act(self.conv_sptial(out))*ca + out
        out = self.project_out(out)

        if self.training:
            return out, torch.mean(mask,dim=1)
        return out


class IRRGBFuse(nn.Module):
    """优化的红外-RGB特征融合模块"""

    def __init__(self, channels, deform_groups=1, kernel_size=3, dilation=1, padding=1):
        super().__init__()
        # 使用更轻量的offset生成器
        self.offset_generator = nn.Sequential(
            nn.Conv2d(channels * 2, deform_groups * 2 * kernel_size * kernel_size, 3, 1, 1)
        )

        # 简化的调制模块
        self.modulation = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.deform_conv = DeformConv2d(
            channels, channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            deform_groups=deform_groups,
        )

        # 添加残差连接的权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, irf):
        """
        Args:
            x: [B, H, W, C] 主特征
            irf: [B, C, H, W] 红外特征
        """
        x = x.permute(0, 3, 1, 2).contiguous()

        # 生成offset和mask
        offset = self.offset_generator(torch.cat([x, irf], dim=1))

        # 可变形卷积 + 调制
        x_deform = self.deform_conv(x, offset)
        modulation = self.modulation(irf)
        x_out = x_deform * modulation

        # # 残差连接
        x_out = x_deform + self.alpha * x_out

        return x_out.permute(0, 2, 3, 1).contiguous()


class MonaOpClipAdapter(nn.Module):
    """优化的CLIP Adapter模块"""

    def __init__(self, in_features, prompt_dim=96, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim

        # Bottleneck结构 - 使用LayerNorm提升稳定性
        self.projdown = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, bottleneck_dim),
            nn.GELU()
        )

        self.projup = nn.Sequential(
            nn.Linear(bottleneck_dim, in_features),
            nn.Dropout(dropout)
        )

        # 红外融合模块
        self.irfuse = IRRGBFuse(bottleneck_dim)

        # # Prompt映射 - 共享部分参数减少冗余
        # self.prompt_mapper_shared = nn.Sequential(
        #     nn.LayerNorm(prompt_dim),
        #     nn.Linear(prompt_dim, in_features // 2),
        #     nn.GELU()
        # )

        self.prompt_mapper_key = nn.Linear(prompt_dim, in_features)
        self.prompt_mapper_value = nn.Linear(prompt_dim, in_features)

        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

        # Gating机制
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.GELU(),
            nn.Linear(in_features // 4, 1),
            nn.Sigmoid()
        )

        # 残差缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, irf, tf):
        """
        Args:
            x: [B, C, H, W] 输入特征
            irf: [B, C', H, W] 红外特征
            tf: [K, prompt_dim] 文本prompt特征
        """
        B, C, H, W = x.shape
        N = H * W
        K = tf.shape[0]

        # 保存输入用于残差连接
        identity = x

        # 转换为 [B, H, W, C]
        x_permuted = x.permute(0, 2, 3, 1).contiguous()

        # === Stage 1: Bottleneck + IR Fusion ===
        x_down = self.projdown(x_permuted)  # [B, H, W, bottleneck_dim]
        x_fused = self.irfuse(x_down, irf)  # [B, H, W, bottleneck_dim]
        x_up = self.projup(x_fused)  # [B, H, W, C]

        # 第一次残差
        x_res1 = x_permuted + self.scale * x_up
        x_flat = x_res1.reshape(B, N, C)  # [B, N, C]

        # === Stage 2: Prompt-guided Attention ===
        # 共享prompt映射
        # prompt_shared = self.prompt_mapper_shared(tf)  # [K, C//2]
        prompt_key = self.prompt_mapper_key(tf)  # [K, C]
        prompt_value = self.prompt_mapper_value(tf)  # [K, C]

        # 计算注意力分数
        x_norm = F.normalize(x_flat, dim=-1)
        prompt_key_norm = F.normalize(prompt_key, dim=-1)

        cos_sim = torch.einsum('bnc,kc->bnk', x_norm, prompt_key_norm)  # [B, N, K]
        attn_weights = F.softmax(cos_sim / self.temperature, dim=-1)  # [B, N, K]

        # 应用注意力
        x_prompted = torch.einsum('bnk,kc->bnc', attn_weights, prompt_value)  # [B, N, C]

        # === Stage 3: Gating Mechanism ===
        x_prompted = x_prompted.reshape(B, H, W, C)
        gate_weights = self.gate(x_res1)  # [B, H, W, 1]
        x_gated = x_res1 + gate_weights * x_prompted

        # 最终残差连接
        out = identity + self.scale * x_gated.permute(0, 3, 1, 2)

        return out

