import torch
import torch.nn as nn
import numpy as np

class EmissionInitializer(nn.Module):
    """
    发射概率初始化器。
    用于初始化发射概率张量, 使其符合给定的分布。
    """

    def __init__(self, dist):
        """
        初始化发射概率初始化器。

        参数 : 
            dist: 初始分布, 一个 NumPy 数组或 PyTorch 张量。
        """
        super(EmissionInitializer, self).__init__()
        self.dist = torch.tensor(dist) if isinstance(dist, np.ndarray) else dist

    def forward(self, shape, dtype=None, device=None):
        """
        生成初始化后的发射概率张量。

        参数 : 
            shape: 输出张量的形状。
            dtype: 输出张量的数据类型 (可选) 。
            device: 输出张量所在的设备 (可选) 。

        返回 : 
            初始化后的发射概率张量。
        """
        assert shape[-1] == self.dist.size(0), f"形状的最后一个维度必须与初始分布的大小匹配。形状={shape} dist.size={self.dist.size(0)}"

        if dtype is not None:
            dist = self.dist.to(dtype)
        else:
            dist = self.dist

        if device is not None:
            dist = dist.to(device)

        prod_shape = torch.prod(torch.tensor(shape[:-1]))
        tiled_dist = dist.repeat(prod_shape)
        return tiled_dist.reshape(shape)

    def __repr__(self):
        """
        返回对象的字符串表示。
        """
        return f"EmissionInitializer(dist={self.dist.tolist()})"

    def get_config(self):
        """
        返回对象的配置信息, 用于序列化。
        """
        return {"dist": self.dist.tolist()}

    @classmethod
    def from_config(cls, config):
        """
        从配置信息创建对象。
        """
        return cls(np.array(config["dist"]))
    

class ConstantInitializer(torch.nn.Module):
    """
    常量初始化器。
    用于初始化张量, 使其所有元素都为给定的常量值。
    """

    def __init__(self, value):
        """
        初始化常量初始化器。

        参数 : 
            value: 常量值, 可以是标量、列表或 NumPy 数组。
        """
        super(ConstantInitializer, self).__init__()
        self.value = torch.tensor(value) if isinstance(value, np.ndarray) else torch.tensor([value]) if np.isscalar(value) else torch.tensor(value)

    def forward(self, shape, dtype=None, device=None):
        """
        生成初始化后的常量张量。

        参数 : 
            shape: 输出张量的形状。
            dtype: 输出张量的数据类型 (可选) 。
            device: 输出张量所在的设备 (可选) 。

        返回 : 
            初始化后的常量张量。
        """
        if dtype is not None:
            value = self.value.to(dtype)
        else:
            value = self.value

        if device is not None:
            value = value.to(device)

        return value.repeat(shape)

    def __repr__(self):
        """
        返回对象的字符串表示。
        """
        if self.value.numel() == 1:
            return f"Const({self.value.item()})"
        elif self.value.ndim == 1:
            return f"Const(size={self.value.size(0)})"
        else:
            return f"Const(shape={self.value.shape})"

    def get_config(self):
        """
        返回对象的配置信息, 用于序列化。
        """
        return {"value": self.value.tolist() if isinstance(self.value, torch.Tensor) else self.value}

    @classmethod
    def from_config(cls, config):
        """
        从配置信息创建对象。
        """
        return cls(np.array(config["value"]))

# 暂时注释
'''

R, p = 0, 0
# R, p = parse_paml(LG_paml, SequenceDataset.alphabet[:-1])
exchangeability_init = inverse_softplus(R + 1e-32).numpy()


prior_path = os.path.dirname(__file__)+"/trained_prior/"
model_path = prior_path+"_".join([str(1), "True", "float32", "_dirichlet.h5"])
model = dm.load_mixture_model(model_path, 1, 20, trainable=False)
dirichlet = model.layers[-1]
background_distribution = dirichlet.expectation()
#the prior was trained on example distributions over the 20 amino acid alphabet
#the additional frequencies for 'B', 'Z',  'X', 'U', 'O' were derived from Pfam
extra = [7.92076933e-04, 5.84256792e-08, 1e-32]
background_distribution = np.concatenate([background_distribution, extra], axis=0)
background_distribution /= np.sum(background_distribution)

def make_default_anc_probs_init(num_models):
    exchangeability_stack = np.stack([exchangeability_init]*num_models, axis=0)
    log_p_stack = np.stack([np.log(p)]*num_models, axis=0)
    exchangeability_stack = np.expand_dims(exchangeability_stack, axis=1) #"k" in AncProbLayer
    log_p_stack = np.expand_dims(log_p_stack, axis=1) #"k" in AncProbLayer
    return [ConstantInitializer(-3), 
            ConstantInitializer(exchangeability_stack), 
            ConstantInitializer(log_p_stack)]
'''

def make_15_class_emission_kernel(smoothing=0.1, num_copies=1, num_models=1, noise_strength=0.001):
    # input classes: IR, I, E0, E1, E2
    # states: Ir, I0, I1, I2, E0, E1, E2, START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
    # Returns: shape (num_models, 1 + num_copies*(14 - 2*introns_shared), 15) 
    assert smoothing > 0, "Smoothing can not be exactly zero to prevent numerical issues."
    n = 15
    probs = np.eye(n)
    probs += -probs * smoothing + (1-probs)*smoothing/(n-1)
    if num_copies > 1:
        repeats = [1] + [num_copies]*(probs.shape[-2]-1)
        probs = np.repeat(probs, repeats, axis=-2)
    # make multiple copies of the emission matrix, one for each model
    probs = np.repeat(probs[np.newaxis, ...], num_models, axis=0)
    # add random noise to each model

    return np.log(probs).astype(np.float32)

background_distribution = make_15_class_emission_kernel()
def make_default_emission_init():
    return EmissionInitializer(np.log(background_distribution))


def make_default_insertion_init():
    return ConstantInitializer(np.log(background_distribution))


class EntryInitializer(nn.Module):
    """
    条目初始化器。
    用于初始化张量, 使其第一个条目接近 0.5, 其余条目均匀分布。
    """

    def __init__(self):
        """
        初始化条目初始化器。
        """
        super(EntryInitializer, self).__init__()

    def forward(self, shape, dtype=None, device=None):
        """
        生成初始化后的张量。

        参数 : 
            shape: 输出张量的形状。
            dtype: 输出张量的数据类型 (可选) 。
            device: 输出张量所在的设备 (可选) 。

        返回 : 
            初始化后的张量。
        """
        if dtype is None:
            dtype = torch.float32 # Default dtype

        p0 = torch.zeros([1] + list(shape[1:]), dtype=dtype, device=device) # 第一个条目为0
        p = torch.log(1 / (shape[0] - 1)) * torch.ones([shape[0] - 1] + list(shape[1:]), dtype=dtype, device=device) # 其余条目均匀分布

        return torch.cat([p0, p], dim=0)

    def __repr__(self):
        """
        返回对象的字符串表示。
        """
        return "DefaultEntry()"
    
class ExitInitializer(nn.Module):
    """
    退出概率初始化器。
    用于初始化张量, 使其所有元素都等于一个特定的退出概率。
    """

    def __init__(self):
        """
        初始化退出概率初始化器。
        """
        super(ExitInitializer, self).__init__()

    def forward(self, shape, dtype=None, device=None):
        """
        生成初始化后的张量。

        参数 : 
            shape: 输出张量的形状。
            dtype: 输出张量的数据类型 (可选) 。
            device: 输出张量所在的设备 (可选) 。

        返回 : 
            初始化后的张量。
        """
        if dtype is None:
            dtype = torch.float32  # 默认数据类型

        return torch.zeros(shape, dtype=dtype, device=device) + torch.log(0.5 / (shape[0] - 1))

    def __repr__(self):
        """
        返回对象的字符串表示。
        """
        return "DefaultExit()"


class MatchTransitionInitializer(nn.Module):
    """
    匹配状态转移概率初始化器。
    用于初始化匹配状态的转移概率张量。
    """

    def __init__(self, val, i, scale):
        """
        初始化匹配状态转移概率初始化器。

        参数 : 
            val: 包含转移概率初始值的列表。
            i: 要提取的转移概率的索引。
            scale: 随机正态分布的尺度。
        """
        super(MatchTransitionInitializer, self).__init__()
        self.val = torch.tensor(val)
        self.i = i
        self.scale = scale

    def forward(self, shape, dtype=None, device=None):
        """
        生成初始化后的转移概率张量。

        参数 : 
            shape: 输出张量的形状。
            dtype: 输出张量的数据类型 (可选) 。
            device: 输出张量所在的设备 (可选) 。

        返回 : 
            初始化后的转移概率张量。
        """
        if dtype is None:
            dtype = torch.float32  # 默认数据类型

        val = self.val.to(dtype).unsqueeze(0)  # [1, len(val)]
        z = torch.normal(mean=0, std=self.scale, size=(shape[0], 1), dtype=dtype, device=device)  # [shape[0], 1]
        val_z = val + z  # [shape[0], len(val)]

        p_exit_desired = 0.5 / (shape[0] - 1)
        prob = torch.softmax(val_z, dim=-1) * (1 - p_exit_desired)  # [shape[0], len(val)]
        return torch.log(prob[:, self.i])  # [shape[0]]

    def __repr__(self):
        """
        返回对象的字符串表示。
        """
        return f"DefaultMatchTransition({self.val[self.i]})"

    def get_config(self):
        """
        返回对象的配置信息, 用于序列化。
        """
        return {"val": self.val.tolist(), "i": self.i, "scale": self.scale}


    
class RandomNormalInitializer(nn.Module):
    """
    随机正态分布初始化器。
    用于初始化张量, 使其元素服从给定的正态分布。
    """

    def __init__(self, mean=0.0, stddev=0.05):
        """
        初始化随机正态分布初始化器。

        参数 : 
            mean: 正态分布的均值。
            stddev: 正态分布的标准差。
        """
        super(RandomNormalInitializer, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, shape, dtype=None, device=None):
        """
        生成初始化后的正态分布张量。

        参数 : 
            shape: 输出张量的形状。
            dtype: 输出张量的数据类型 (可选) 。
            device: 输出张量所在的设备 (可选) 。

        返回 : 
            初始化后的正态分布张量。
        """
        if dtype is None:
            dtype = torch.float32  # 默认数据类型

        return torch.normal(mean=self.mean, std=self.stddev, size=shape, dtype=dtype, device=device)

    def __repr__(self):
        """
        返回对象的字符串表示。
        """
        return f"Norm({self.mean}, {self.stddev})"

    def get_config(self):
        """
        返回对象的配置信息, 用于序列化。
        """
        return {"mean": self.mean, "stddev": self.stddev}
    
    
def make_default_flank_init():
    return ConstantInitializer(0.)

    
def make_default_transition_init(MM=1, 
                                 MI=-1, 
                                 MD=-1, 
                                 II=-0.5, 
                                 IM=0, 
                                 DM=0, 
                                 DD=-0.5,
                                 FC=0, 
                                 FE=-1,
                                 R=-9, 
                                 RF=0, 
                                 T=0,
                                 scale=0.1):
    """
    创建默认的转移概率初始化器字典。

    参数 : 
        MM: 匹配状态到匹配状态的初始值。
        MI: 匹配状态到插入状态的初始值。
        MD: 匹配状态到删除状态的初始值。
        II: 插入状态到插入状态的初始值。
        IM: 插入状态到匹配状态的初始值。
        DM: 删除状态到匹配状态的初始值。
        DD: 删除状态到删除状态的初始值。
        FC: 侧翼和未注释片段循环的初始值。
        FE: 侧翼和未注释片段退出的初始值。
        R: 结束状态到未注释片段的初始值。
        RF: 结束状态到右侧翼的初始值。
        T: 结束状态到终止状态的初始值。
        scale: 随机正态分布的尺度。

    返回 : 
        转移概率初始化器字典。
    """
    transition_init_kernel = {
        "begin_to_match": EntryInitializer(),
        "match_to_end": ExitInitializer(), #需要定义ExitInitializer
        "match_to_match": MatchTransitionInitializer([MM, MI, MD], 0, scale), #需要定义MatchTransitionInitializer
        "match_to_insert": MatchTransitionInitializer([MM, MI, MD], 1, scale), #需要定义MatchTransitionInitializer
        "insert_to_match": RandomNormalInitializer(IM, scale),
        "insert_to_insert": RandomNormalInitializer(II, scale),
        "match_to_delete": MatchTransitionInitializer([MM, MI, MD], 2, scale), #需要定义MatchTransitionInitializer
        "delete_to_match": RandomNormalInitializer(DM, scale),
        "delete_to_delete": RandomNormalInitializer(DD, scale),
        "left_flank_loop": RandomNormalInitializer(FC, scale),
        "left_flank_exit": RandomNormalInitializer(FE, scale),
        "right_flank_loop": RandomNormalInitializer(FC, scale),
        "right_flank_exit": RandomNormalInitializer(FE, scale),
        "unannotated_segment_loop": RandomNormalInitializer(FC, scale),
        "unannotated_segment_exit": RandomNormalInitializer(FE, scale),
        "end_to_unannotated_segment": RandomNormalInitializer(R, scale),
        "end_to_right_flank": RandomNormalInitializer(RF, scale),
        "end_to_terminal": RandomNormalInitializer(T, scale)
    }
    return transition_init_kernel

