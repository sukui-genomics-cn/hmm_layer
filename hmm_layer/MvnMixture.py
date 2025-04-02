import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import math
from Utility import DefaultDiagBijector, FillScaleTriL

class MvnMixture(nn.Module):
    """
    多元正态混合分布。
    在 R^d 上的多元正态分布, 由 d 个位置和 dxd 下三角尺度矩阵参数化。
    """

    def __init__(self,
                 dim,
                 kernel,
                 mixture_coeff_kernel=None,
                 diag_only=True,  # 设置为 False 是实验性的, 目前可能无法正常工作
                 diag_bijector=DefaultDiagBijector(1.),
                 precomputed=False,
                 **kwargs):
        """
        初始化 MvnMixture。

        参数 : 
            dim: 多元正态分布的维度。
            kernel: 与模型参数密切相关的 4D 核矩阵。形状 (k1, k2, num_components, num_param)。
                  其中 k1 是具有匹配输入序列的模型数, k2 是将针对所有输入评估的模型数, 
                  num_param = 2*dim 如果 diag_only=True, 否则 num_param = dim + dim*(dim+1)//2。
            mixture_coeff_kernel: 与混合系数密切相关的 3D 核矩阵。形状 (k1, k2, num_components)。
                                  如果为 None, 则假定为单个组件, 并且 kernel.shape[2] 必须为 1。
            diag_only: 如果为 True, 则假定尺度矩阵为对角矩阵。否则, 使用完整的尺度矩阵。
            diag_bijector: 用于将尺度矩阵的对角线条目投影到正值的 bijector。
            precomputed: 如果为 True, 则不会在每次调用层时重新计算尺度矩阵。
                         训练期间应为 False, 推理期间应为 True。
            **kwargs: 其他参数。
        """
        super(MvnMixture, self).__init__(**kwargs)
        self.dim = dim
        self.kernel = torch.tensor(kernel, dtype=torch.float32)
        if mixture_coeff_kernel is not None:
            self.mixture_coeff_kernel = torch.tensor(mixture_coeff_kernel, dtype=torch.float32)
        else:
            self.mixture_coeff_kernel = None
        self.num_components = self.kernel.shape[2]
        self.diag_only = diag_only
        self.diag_bijector = diag_bijector
        self.precomputed = precomputed
        self.scale_tril = FillScaleTriL(diag_bijector=diag_bijector)
        self.constant = self.dim * math.log(2 * math.pi)
        self.scale = None
        self.pinv = None
        # 验证
        assert len(self.kernel.shape) == 4
        if diag_only:
            assert self.kernel.shape[-1] == 2 * dim
        else:
            assert self.kernel.shape[-1] == dim + dim * (dim + 1) // 2
        if self.mixture_coeff_kernel is not None:
            assert len(self.mixture_coeff_kernel.shape) == 3
            assert self.mixture_coeff_kernel.shape == self.kernel.shape[:3]
        else:
            assert self.num_components == 1

    def component_expectations(self):
        """
        计算混合组件的期望值。

        返回 : 
            形状 (k1, k2, num_components, dim)。
        """
        mu = self.kernel[..., :self.dim]
        return mu

    def expectation(self):
        """
        计算期望值。

        返回 : 
            形状 (k1, k2, dim)。
        """
        if self.num_components == 1:
            return self.component_expectations()[..., 0, :]
        else:
            comp_exp = self.component_expectations()
            mix_coeff = self.mixture_coefficients()
            return torch.sum(comp_exp * mix_coeff.unsqueeze(-1), -2)

    def component_scales(self, return_scale_diag=False, return_inverse=False):
        """
        计算混合组件的尺度矩阵。协方差矩阵可以计算为 scale * scale^T。

        返回 : 
            形状 (k1, k2, num_components, dim) 如果 return_scale_diag 为 True, 否则 (k1, k2, num_components, dim, dim)。
        """
        if not self.precomputed or self.scale is None:
            if self.diag_only:
                scale_diag = self.diag_bijector.forward(self.kernel[..., self.dim:])
                scale_diag += 1e-8
                scale = scale_diag if return_scale_diag else torch.eye(self.dim, device=self.kernel.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) * scale_diag.unsqueeze(-1)
                if return_inverse:
                    pinv = 1. / scale_diag
            else:
                scale_kernel = self.kernel[..., self.dim:]
                scale = self.scale_tril.forward(scale_kernel)
                if return_inverse:
                    pinv = torch.linalg.pinv(scale)
                if return_scale_diag:
                    scale = torch.diagonal(scale, dim1=-2, dim2=-1)
        return (scale, pinv) if return_inverse else scale

    def component_covariances(self):
        """
        计算混合组件的协方差矩阵。

        返回 : 
            形状 (k1, k2, num_components, dim) 如果 self.diag_only 为 True, 否则 (k1, k2, num_components, dim, dim)。
        """
        scale = self.component_scales(return_scale_diag=self.diag_only)
        if self.diag_only:
            return torch.square(scale)
        else:
            return torch.matmul(scale, scale.transpose(-1, -2))

    def component_log_pdf(self, inputs):
        """
        计算每个混合分布的组件级对数概率密度函数。
        此方法执行 k1 个批次输入和 k2 个模型之间的全对全评估。

        参数 : 
            inputs: 形状 (k1, batch, dim)。

        返回 : 
            形状 (k1, batch, k2, num_components)。
        """
        mu = self.component_expectations()
        scale_diag, pinv = self.component_scales(return_scale_diag=True, return_inverse=True)
        log_det = 2 * torch.sum(torch.log(scale_diag), -1)  # (k1, k2, c, 1)
        diff = inputs.unsqueeze(1).unsqueeze(2) - mu.unsqueeze(-2)  # (k1, k2, c, b, d)
        if self.diag_only:
            pinv_sq = torch.square(pinv)  # (k1, k2, c, d)
            diff_sq = torch.square(diff)
            MD_sq_components = torch.sum(diff_sq * pinv_sq.unsqueeze(-2), -1)  # (k1, k2, c, b)
        else:
            y = torch.matmul(diff, pinv.transpose(-1, -2))  # (k1, k2, c, b, d)
            MD_sq_components = torch.sum(torch.square(y), -1)
        MD_sq_components = MD_sq_components.transpose(1, 3)
        log_pdf = -0.5 * (self.constant + log_det.unsqueeze(1) + MD_sq_components)
        return log_pdf

    def mixture_coefficients(self):
        """
        计算混合系数。

        返回 : 
            形状 (k1, k2, num_components)。
        """
        return torch.softmax(self.mixture_coeff_kernel, dim=-1)

    def log_pdf(self, inputs):
        """
        计算每个混合分布的对数概率密度函数。
        此方法执行 k1 个批次输入和 k2 个模型之间的全对全评估。

        参数 : 
            inputs: 形状 (k1, batch, dim)。

        返回 : 
            形状 (k1, batch, k2)。
        """
        log_pdf_components = self.component_log_pdf(inputs)
        if self.num_components == 1:
            return log_pdf_components[..., 0]
        else:
            return torch.logsumexp(log_pdf_components + torch.log(self.mixture_coefficients().unsqueeze(1)), -1)

    def get_regularization_L2_loss(self):
        """
        计算方差核的 L2 损失, 防止方差过小或过大。
        """
        return torch.mean(torch.sum(torch.square(self.kernel[..., self.dim:]), dim=-1))