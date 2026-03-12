import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PPA(nn.Module):
    """
    目标：对齐 Student 和 Teacher 特征在每一类上的 一阶矩(均值) 和 二阶矩(方差)。
    """

    def __init__(self, s_dim=320, t_dim=256, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index

        # 1. 仅保留 Student 端的投影层
        # 负责将学生特征维度 s_dim 映射到 教师特征维度 t_dim
        self.proj_s = nn.Sequential(
            nn.Conv2d(s_dim, t_dim, 1, bias=False),
            nn.BatchNorm2d(t_dim),
            nn.ReLU(inplace=True)
        )
        # 注意：彻底删除了 proj_t，防止特征坍塌

    def get_distribution_proto(self, feat, mask):
        """
        计算每个类别的均值 (mu) 和 方差 (var)
        feat: (B, C, H, W)
        mask: (B, H, W)
        """
        # 1. 对齐 Mask 尺寸 (Nearest 插值)
        if feat.shape[-2:] != mask.shape[-2:]:
            mask = F.interpolate(mask.float().unsqueeze(1), size=feat.shape[-2:], mode='nearest').squeeze(1).long()

        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)  # (B, C, N)
        mask_flat = mask.view(B, -1)  # (B, N)

        dists = []
        eps = 1e-6  # 防止数值不稳定

        for b in range(B):
            unique_c = torch.unique(mask_flat[b])
            p_dict = {}
            for c in unique_c:
                c_item = c.item()

                # 【关键修改】跳过背景/忽略区域
                if c_item == self.ignore_index:
                    continue

                m = (mask_flat[b] == c)  # Bool Mask (N,)

                # 至少需要2个像素点才能计算有意义的方差
                if m.sum() > 1:
                    selected_feat = feat_flat[b, :, m]  # (C, Num_pixels)

                    # 计算均值 (Mean) - 一阶矩
                    mu = selected_feat.mean(dim=1)

                    # 计算方差 (Variance) - 二阶矩
                    # 使用无偏估计 (unbiased=True)，反应类内不确定性/多样性
                    var = selected_feat.var(dim=1) + eps

                    p_dict[c_item] = {'mu': mu, 'var': var}
            dists.append(p_dict)
        return dists

    def moment_matching_loss(self, mu_s, var_s, mu_t, var_t):
        """
        矩匹配损失 (Moment Matching Loss)
        显式对齐均值和方差。
        """
        # 均值对齐 (MSE)
        loss_mu = F.mse_loss(mu_s, mu_t)
        # loss_var = F.relu(var_s-var_t).mean()
        # 方差对齐 (MSE)
        # 能够让学生学习教师的"不确定性"分布宽度
        # loss_var = F.mse_loss(var_s, var_t)


        return 0.5*loss_mu + 0.5*loss_var

    def forward(self, s_feat, t_feat, gt):
        """
        s_feat: 学生原始特征
        t_feat: 教师原始特征 (Fixed)
        gt: Ground Truth Label
        """
        # 1. 投影学生特征到教师空间
        s_feat = self.proj_s(s_feat)

        # 2. 教师特征必须 detach (不传梯度，且不经过任何 Learnable Layer)
        t_feat = t_feat.detach()
        # mse_total = 0.0
        s_feat = F.normalize(s_feat,p=2,dim=1)
        t_feat = F.normalize(t_feat,p=2,dim=1)
        # for s_layer, t_layer in zip(s_feat, t_feat):
        mse_total = F.mse_loss(s_feat, t_feat)
            # mse_total += layer_loss
        # f_loss = F.mse_loss(s_feat,t_feat)
        # 【关键修改】去掉 F.normalize
        # 我们在欧氏空间对齐分布，保留特征的模长信息

        # 3. 计算分布原型 (跳过背景类)
        s_dists = self.get_distribution_proto(s_feat, gt)
        t_dists = self.get_distribution_proto(t_feat, gt)

        loss = 0.0
        cnt = 0

        for b in range(len(s_dists)):
            # 找到 Teacher 和 Student 都有的有效类别 (交集)
            common = set(s_dists[b].keys()) & set(t_dists[b].keys())
            for c in common:
                s_d = s_dists[b][c]
                t_d = t_dists[b][c]

                # 计算矩匹配损失
                mm_loss = self.moment_matching_loss(
                    s_d['mu'], s_d['var'],
                    t_d['mu'], t_d['var']
                )

                loss += mm_loss
                cnt += 1

        # 防止除零
        return loss / (cnt + 1e-7)+0.5*mse_total

class MultiLayerFeatureAligner(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.projs = nn.ModuleDict()
        for i, (in_c, out_c) in enumerate(zip(in_channels_list, out_channels_list)):

            self.projs[f'layer_{i}'] =PPA(in_c,out_c)

    def forward(self, feats_src, feats_target,gt):
        total_loss = 0
        for i, (f_src, f_target) in enumerate(zip(feats_src, feats_target)):
            # 1. 投影转换
            layer_loss = self.projs[f'layer_{i}'](f_src,f_target,gt)

            total_loss += layer_loss
        return total_loss


class DALD(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, temperature=2.0):
        super(DALD, self).__init__()
        self.alpha = alpha  # 控制KL散度的抑制程度
        self.beta = beta  # 控制不确定性边界的惩罚强度
        self.T = temperature
        self.kl_div = nn.KLDivLoss(reduction='none')

    def entropy(self, probs):
        """计算熵 H(p) = -sum(p * log(p))"""
        # 加 epsilon 防止 log(0)
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    def forward(self, student_logits, teacher_samples):
        """
        Args:
            student_logits: 学生模型的输出 logits [B, C, H, W]
            teacher_samples: 教师生成模型的多个采样结果 [M, B, C, H, W] (Softmax后的概率)
                             注意：输入应该是概率值
        """
        # 1. 准备数据
        M, B, C, H, W = teacher_samples.shape

        # 学生预测概率
        student_probs = F.softmax(student_logits / self.T, dim=1)
        # 学生的熵 H(q) [B, H, W]
        student_entropy = self.entropy(student_probs)

        # 2. 计算教师的统计量
        # 教师均值 p_bar [B, C, H, W]
        teacher_mean = torch.mean(teacher_samples, dim=0)

        # 教师均值的熵 H(p_bar) [B, H, W]
        entropy_of_mean = self.entropy(teacher_mean)

        # 教师采样点熵的均值 E[H(p)] [B, H, W]
        mean_of_entropies = torch.mean(torch.stack([self.entropy(sample) for sample in teacher_samples]), dim=0)

        # 计算认知不确定性 (Epistemic Uncertainty) = Mutual Information
        # U_epi = H(p_bar) - E[H(p)]
        u_epi = entropy_of_mean - mean_of_entropies

        # 归一化 U_epi 到 [0, 1] 用于加权 (Min-Max normalization per batch)
        # 这里为了数值稳定，加一个小的平滑项
        u_epi_flat = u_epi.view(B, -1)
        u_min = u_epi_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
        u_max = u_epi_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)
        u_epi_norm = (u_epi - u_min) / (u_max - u_min + 1e-8)

        # ------------------------------------------------------
        # Loss Part 1: Epistemic-Weighted KL Divergence
        # ------------------------------------------------------
        # 在认知不确定性高的地方，降低 KL Loss 的权重
        # student_logits 需要 log_softmax 用于 KLDivLoss
        student_log_probs = F.log_softmax(student_logits / self.T, dim=1)

        # KL Loss [B, H, W] (sum over classes)
        kl_loss = torch.sum(teacher_mean * (torch.log(teacher_mean + 1e-8) - student_log_probs), dim=1)

        # 权重 w = 1 - alpha * u_epi_norm
        # 如果 u_epi 很大，权重变小，不仅拟合均值
        weights = self.alpha * u_epi_norm
        weighted_kl_loss = (weights * kl_loss).mean()

        # ------------------------------------------------------
        # Loss Part 2: Uncertainty Boundary Loss (ReLU Penalty)
        # ------------------------------------------------------
        # 惩罚：如果 U_epi > H(q)，说明学生过于自信，需要惩罚
        # 我们希望 H(q) >= U_epi (大致趋势)

        # # 注意：这两个熵的量级可能不同，建议先 detach 教师的不确定性
        uncertainty_gap = u_epi.detach() - student_entropy
        #
        # # 只惩罚 student_entropy 过小的部分
        uncertainty_loss = F.relu(uncertainty_gap).mean()
        # uncertainty_loss = F.mse_loss(student_entropy,u_epi.detach())
        # ------------------------------------------------------
        # Total Loss
        # ------------------------------------------------------
        total_loss = weighted_kl_loss + self.beta * uncertainty_loss

        return total_loss

