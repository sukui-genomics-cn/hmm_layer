import torch
import torch.nn as nn
from enum import Enum

from . import Initializers as initializers
from . import Priors as priors

class ProfileHMMEmitter(nn.Module):
    """
    Profile HMM 发射概率层。
    定义发射概率分布和先验。
    默认配置实现了氨基酸字母表上的多项式匹配分布和狄利克雷先验。
    """

    def __init__(self,
                 emission_init=initializers.make_default_emission_init(),
                 insertion_init=initializers.make_default_insertion_init(),
                 prior=None,
                 frozen_insertions=True,
                 device=None,
                 **kwargs):
        """
        初始化 ProfileHMMEmitter。

        参数 : 
            emission_init: 匹配状态的初始化器列表, 每个模型一个初始化器。
            insertion_init: 插入状态的初始化器列表, 每个模型一个初始化器。
            prior: 先验分布, 默认为氨基酸先验。
            frozen_insertions: 是否冻结插入状态的训练, 默认为 True。
            **kwargs: 其他参数。
        """
        super(ProfileHMMEmitter, self).__init__(**kwargs)
        self.emission_init = [emission_init] if not hasattr(emission_init, '__iter__') else emission_init
        self.insertion_init = [insertion_init] if not hasattr(insertion_init, '__iter__') else insertion_init
        self.prior = priors.AminoAcidPrior() if prior is None else prior
        self.frozen_insertions = frozen_insertions

    def set_lengths(self, lengths):
        """
        设置模型长度。

        参数 : 
            lengths: 模型长度列表。
        """
        self.lengths = lengths
        self.num_models = len(lengths)

        # 确保长度有效
        assert len(self.lengths) == len(self.emission_init), \
            f"发射概率初始化器数量 ({len(self.emission_init)}) 应与模型数量 ({len(self.lengths)}) 匹配。"
        assert len(self.lengths) == len(self.insertion_init), \
            f"插入概率初始化器数量 ({len(self.insertion_init)}) 应与模型数量 ({len(self.lengths)}) 匹配。"

    def build(self, input_shape):
        """
        构建模型参数。

        参数 : 
            input_shape: 输入张量的形状。
        """
        if hasattr(self, "built") and self.built:
            return
        s = input_shape[-1] - 1  # 减去终止符
        self.emission_kernel = nn.ParameterList([
            nn.Parameter(init(torch.Size([length, s])))
            for length, init in zip(self.lengths, self.emission_init)
        ])
        self.insertion_kernel = nn.ParameterList([
            nn.Parameter(init(torch.Size([s])))
            for init in self.insertion_init
        ])
        if self.frozen_insertions:
            for param in self.insertion_kernel:
                param.requires_grad = False
        self.prior.build()
        self.built = True

    def recurrent_init(self):
        """
        在每次循环运行前自动调用。用于设置每次循环应用只需要一次的参数。
        """
        self.B = self.make_B()
        self.B_transposed = torch.transpose(self.B, 1, 2)

    def make_emission_matrix(self, i):
        """
        根据核构建发射概率矩阵。

        参数 : 
            i: 模型索引。

        返回 : 
            发射概率矩阵。
        """
        em, ins = self.emission_kernel[i], self.insertion_kernel[i]
        length = self.lengths[i]
        return self.make_emission_matrix_from_kernels(em, ins, length)

    def make_emission_matrix_from_kernels(self, em, ins, length):
        """
        根据核构建发射概率矩阵。

        参数 : 
            em: 匹配状态的核。
            ins: 插入状态的核。
            length: 模型长度。

        返回 : 
            发射概率矩阵。
        """
        s = em.shape[-1]
        i1 = ins.unsqueeze(0)
        i2 = torch.stack([ins] * (length + 1))
        emissions = torch.cat([i1, em, i2], dim=0)
        emissions = torch.softmax(emissions, dim=-1)
        emissions = torch.cat([emissions, torch.zeros_like(emissions[:, :1])], dim=-1)
        end_state_emission = torch.nn.functional.one_hot(torch.tensor([s]), num_classes=s + 1, dtype=em.dtype)
        emissions = torch.cat([emissions, end_state_emission], dim=0)
        return emissions

    def make_B(self):
        """
        构建所有模型的发射概率矩阵。
        """
        emission_matrices = []
        max_num_states = max([len(self.lengths)+2] * self.num_models) #get_num_states(self.lengths)
        for i in range(self.num_models):
            em_mat = self.make_emission_matrix(i)
            padding = max_num_states - em_mat.shape[0]
            em_mat_pad = torch.nn.functional.pad(em_mat, (0, 0, 0, padding))
            emission_matrices.append(em_mat_pad)
        B = torch.stack(emission_matrices, dim=0)
        return B

    def make_B_amino(self):
        """
        用于绘制 HMM 的 make_B 变体。
        """
        return self.make_B()

    def forward(self, inputs, end_hints=None, training=False):
        """
        前向传播。

        参数 : 
            inputs: 输入张量, 形状 (k, ..., s)。
            end_hints: 结束状态提示, 形状 (num_models, batch_size, 2, num_states)。
            training: 是否为训练模式。

        返回 : 
            发射概率张量, 形状 (k, ..., q)。
        """
        input_shape = inputs.shape
        inputs = inputs.reshape(inputs.shape[0], -1, input_shape[-1])
        B = self.B_transposed[..., :input_shape[-1], :]
        emit = torch.einsum("kbs,ksq->kbq", inputs, B)
        emit_shape = torch.Size([B.shape[0]] + list(input_shape[1:-1]) + [B.shape[-1]])
        emit = emit.reshape(emit_shape)
        return emit

    def get_aux_loss(self):
        """
        获取辅助损失。
        """
        return torch.tensor(0., dtype=self.dtype)

    def get_prior_log_density(self):
        """
        获取先验对数密度。
        """
        return self.prior(self.B, lengths=self.lengths)

    def duplicate(self, model_indices=None, share_kernels=False):
        """
        复制发射概率层。

        参数 : 
            model_indices: 要复制的模型索引列表。
            share_kernels: 是否共享核。

        返回 : 
            复制后的发射概率层。
        """
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [initializers.ConstantInitializer(self.emission_kernel[i].detach().numpy()) for i in model_indices]
        sub_insertion_init = [initializers.ConstantInitializer(self.insertion_kernel[i].detach().numpy()) for i in model_indices]
        emitter_copy = ProfileHMMEmitter(
            emission_init=sub_emission_init,
            insertion_init=sub_insertion_init,
            prior=self.prior,
            frozen_insertions=self.frozen_insertions,
            dtype=self.dtype
        )
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.built = True
        return emitter_copy

    def get_config(self):
        """
        获取配置信息。
        """
        config = super(ProfileHMMEmitter, self).get_config()
        config.update({
            "lengths": self.lengths,
            "emission_init": self.emission_init,
            "insertion_init": self.insertion_init,
            "prior": self.prior,
            "frozen_insertions": self.frozen_insertions
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        从配置信息创建发射概率层。
        """
        config["emission_init"] = [initializers.deserialize(e) for e in config["emission_init"]]
        config["insertion_init"] = [initializers.deserialize(i) for i in config["insertion_init"]]
        config["prior"] = initializers.deserialize(config["prior"])
        lengths = config.pop("lengths")
        emitter = cls(**config)
        emitter.set_lengths(lengths)
        return emitter

    def __repr__(self):
        """
        返回对象的字符串表示。
        """
        return f"ProfileHMMEmitter(\n emission_init={self.emission_init[0]},\n insertion_init={self.insertion_init[0]},\n prior={self.prior},\n frozen_insertions={self.frozen_insertions}, )"

class TemperatureMode(Enum):
    TRAINABLE = 1
    LENGTH_NORM = 2
    COLD_TO_WARM = 3
    WARM_TO_COLD = 4
    CONSTANT = 5
    NONE = 6

    @staticmethod
    def from_string(name):
        return {"trainable": TemperatureMode.TRAINABLE,
                "length_norm": TemperatureMode.LENGTH_NORM,
                "cold_to_warm": TemperatureMode.COLD_TO_WARM,
                "warm_to_cold": TemperatureMode.WARM_TO_COLD,
                "constant": TemperatureMode.CONSTANT,
                "none": TemperatureMode.NONE}[name.lower()]

# 注意 : 在 PyTorch 中, 不需要使用 get_custom_objects。
# 如果您需要自定义模块, 请确保它们已定义。