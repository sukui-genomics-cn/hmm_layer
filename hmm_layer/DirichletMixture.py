import torch
import torch.nn as nn
import numpy as np
import os

def dirichlet_log_pdf(p, alpha, q):
    """
    计算由 alpha 和 q 给出的混合的 p 处的对数狄利克雷密度。

    参数 : 
        p: 概率分布。形状 : (b, s)
        alpha: 狄利克雷分量参数。形状 : (k, s)
        q: 狄利克雷混合参数。形状 : (k)

    返回 : 
        狄利克雷密度的对数。输出形状 : (b)
    """
    # 每个分量的归一化常数的对数
    logZ = torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha.sum(-1))
    log_p_alpha = torch.log(p).unsqueeze(1) * (alpha - 1).unsqueeze(0)
    log_p_alpha = log_p_alpha.sum(-1) - logZ
    log_pdf = torch.logsumexp(log_p_alpha + torch.log(q), -1)
    return log_pdf

class DirichletMixtureLayer(nn.Module):
    """
    一个狄利克雷混合层，用于计算一批概率分布的似然性。

    参数 : 
        num_components: 混合中的狄利克雷分量数。
        alphabet_size: 离散分类概率分布的大小。
        use_dirichlet_process: 如果为 True，则在训练模式下使用狄利克雷过程近似。参见
            狄利克雷混合、狄利克雷过程和蛋白质空间的结构，Nguyen 等人，2013 年
        number_of_examples: 如果 use_dirichlet_process == True，则必须指定的训练数据集大小。
        alpha_init: 分量分布的初始化器。
        mix_init: 混合分布的初始化器。
        trainable: 可用于在用作先验时冻结层。
    """
    def __init__(self,
                 num_components,
                 alphabet_size,
                 use_dirichlet_process=True,
                 number_of_examples=-1,
                 alpha_init="random_normal",
                 mix_init="random_normal",
                 background_init=None,
                 trainable=True,
                 **kwargs):
        super(DirichletMixtureLayer, self).__init__(**kwargs)
        self.num_components = num_components
        self.alphabet_size = alphabet_size
        self.use_dirichlet_process = use_dirichlet_process
        self.number_of_examples = number_of_examples
        self.alpha_init = alpha_init
        self.mix_init = mix_init
        self.background_init = background_init
        self.trainable = trainable

    def build(self, input_shape=None):
        """构建模型参数。"""
        self.alpha_kernel = nn.Parameter(torch.randn(self.num_components, self.alphabet_size), requires_grad=self.trainable)
        self.mix_kernel = nn.Parameter(torch.randn(self.num_components), requires_grad=self.trainable)
        if self.use_dirichlet_process:
            self.gamma_kernel = nn.Parameter(torch.tensor([50.0]), requires_grad=self.trainable)
            self.beta_kernel = nn.Parameter(torch.tensor([100.0]), requires_grad=self.trainable)
            self.lambda_kernel = nn.Parameter(torch.ones(1), requires_grad=self.trainable)
            self.background_kernel = nn.Parameter(torch.randn(20), requires_grad=self.trainable)

    def make_alpha(self):
        return torch.nn.functional.softplus(self.alpha_kernel, name="alpha")

    def make_mix(self):
        return torch.nn.functional.softmax(self.mix_kernel, dim=-1, name="mix")

    def make_gamma(self):
        return torch.nn.functional.softplus(self.gamma_kernel, name="gamma")

    def make_beta(self):
        return torch.nn.functional.softplus(self.beta_kernel, name="beta")

    def make_lambda(self):
        return torch.nn.functional.softplus(self.lambda_kernel, name="lambda")

    def make_background(self):
        return torch.nn.functional.softmax(self.background_kernel, dim=-1, name="background")

    def log_pdf(self, p):
        return dirichlet_log_pdf(p, self.make_alpha(), self.make_mix())

    def component_distributions(self):
        alpha = self.make_alpha()
        return alpha / alpha.sum(-1, keepdim=True)

    def expectation(self):
        return (self.component_distributions() * self.make_mix().unsqueeze(-1)).sum(0)

    def forward(self, p, training=False):
        alpha = self.make_alpha()
        mix = self.make_mix()
        loglik = dirichlet_log_pdf(p, alpha, mix).mean()
        if training:
            if self.use_dirichlet_process:
                sum_alpha = alpha.sum(-1, keepdim=True)
                lamb = self.make_lambda()
                sum_alpha_prior = torch.log(lamb) - lamb * sum_alpha  # exponential
                sum_alpha_prior = sum_alpha_prior.sum()
                mix_dist = torch.ones_like(mix) * self.make_gamma() / self.num_components
                mix_prior = dirichlet_log_pdf(mix.unsqueeze(0), mix_dist.unsqueeze(0), torch.ones(1))
                comp_dist = self.make_background() * self.make_beta()
                comp_prior = dirichlet_log_pdf(alpha / sum_alpha, comp_dist.unsqueeze(0), torch.ones(1)).sum()
                joint_density = loglik + (sum_alpha_prior + mix_prior + comp_prior) / self.number_of_examples
                self.loss = -joint_density
            else:
                self.loss = -loglik
        return loglik

def make_model(dirichlet_mixture_layer):
    """构建 DirichletMixtureLayer 上的 PyTorch 模型。"""
    class Model(nn.Module):
        def __init__(self, layer):
            super(Model, self).__init__()
            self.layer = layer

        def forward(self, p):
            return self.layer(p)
    return Model(dirichlet_mixture_layer)

def load_mixture_model(model_path, num_components, alphabet_size, trainable=False, dtype=torch.float32):
    dm = DirichletMixtureLayer(num_components, alphabet_size, trainable=trainable)
    model = make_model(dm)
    # 假设 model_path 是一个 .pt 文件
    model.load_state_dict(torch.load(model_path))
    return model