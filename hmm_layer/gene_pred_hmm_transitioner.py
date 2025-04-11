import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import show_value
from .Transitioner import make_transition_matrix_from_indices


class SimpleGenePredHMMTransitioner(nn.Module):
    """
    定义 HMM 状态之间允许的转换以及如何初始化它们。
    假设状态顺序 : Ir, I0, I1, I2, E0, E1, E2
    """

    def __init__(self,
                 num_models=1,
                 initial_exon_len=100,
                 initial_intron_len=10000,
                 initial_ir_len=10000,
                 init=None,
                 starting_distribution_init="zeros",
                 starting_distribution_trainable=True,
                 transitions_trainable=True,
                 init_component_sd=0,  # 0.2
                 device=None,
                 **kwargs):
        super(SimpleGenePredHMMTransitioner, self).__init__(**kwargs)
        self.num_models = num_models
        self.initial_exon_len = initial_exon_len
        self.initial_intron_len = initial_intron_len
        self.initial_ir_len = initial_ir_len
        if not hasattr(self, "num_states"):
            self.num_states = 7
        self.indices = self.make_transition_indices()
        self.starting_distribution_init = starting_distribution_init
        self.starting_distribution_trainable = starting_distribution_trainable
        self.transitions_trainable = transitions_trainable
        self.num_transitions = len(self.indices)
        self.reverse = False
        if init is None:
            self.init = self.make_transition_init(1, init_component_sd)
        else:
            self.init = init
        self.transition_kernel = nn.Parameter(
            torch.tensor(self.init).to(torch.float32).unsqueeze(0),
            requires_grad=self.transitions_trainable,
        )
        self.starting_distribution_kernel = nn.Parameter(
            torch.zeros(1, 1, self.num_states).to(torch.float32),
            requires_grad=self.starting_distribution_trainable) if starting_distribution_init == "zeros" else nn.Parameter(
            torch.ones(1, 1, self.num_states).float(), requires_grad=self.starting_distribution_trainable)
        self.A = None
        self.A_transposed = None
        self.device = device

    def is_intergenic_loop(self, edge):
        return edge[1] == edge[2] and edge[1] == 0

    def is_intron_loop(self, edge, k=1):
        return edge[1] == edge[2] and edge[1] > 0 and edge[1] < 1 + 3 * k

    def is_exon_transition(self, edge, k=1):
        found_any = False
        exon_offset = 1 + 3 * k
        for i in range(k):
            found = edge[2] - exon_offset == (edge[1] - exon_offset + k) % (3 * k) and edge[1] >= exon_offset and edge[
                1] < exon_offset + 3 * k
            found_any = found_any or found
        return found_any

    def is_exon_1_out_transition(self, edge, k=1):
        return edge[1] >= 1 + 4 * k and edge[1] < 1 + 5 * k and edge[1] != edge[2]

    def is_intergenic_out_transition(self, edge, k=1):
        return edge[1] == 0 and edge[2] != 0

    def recurrent_init(self):
        """
        在每次递归运行之前自动调用。应将其用于每个递归层应用仅需要一次的设置。
        """
        A = self.make_A()
        self.A = nn.Parameter(A, requires_grad=self.transitions_trainable)
        A_transposed = torch.transpose(self.A, 1, 2)
        self.A_transposed = nn.Parameter(A_transposed, requires_grad=self.transitions_trainable)
        show_value(A, "02.transitioner.A")
        show_value(A_transposed, "02.transitioner.A_transposed")

    def make_A_sparse(self, values=None):
        """
        从内核计算状态转移概率。对状态的所有传出边的内核值应用 softmax。

        参数：
            values: 如果不为 None，则稀疏张量的值将设置为此值。否则，将使用内核。

        返回：
            将转换类型映射到概率的字典。
        """
        if values is None:
            values = self.transition_kernel.reshape(-1)
        # 按行主序重新排序，假设是单个模型，因此所有索引三元组都以 (0,..) 开头
        row_major_order = np.argsort([i * self.num_states + j for _, i, j in self.indices])
        ordered_indices = self.indices[row_major_order]
        ordered_values = values[row_major_order]
        dense_probs = make_transition_matrix_from_indices(ordered_indices[:, 1:], ordered_values, self.num_states)
        dense_probs = dense_probs.unsqueeze(0)
        probs_vec = (dense_probs[:, ordered_indices[:, 1], ordered_indices[:, 2]]).squeeze(0)
        A_sparse = torch.sparse_coo_tensor(
            indices=ordered_indices.T,  # PyTorch 稀疏张量需要转置索引
            values=probs_vec,
            size=(1, self.num_states, self.num_states),
        )
        return A_sparse

    def make_A(self):
        A = self.make_A_sparse().to_dense()
        A = A.repeat(self.num_models, 1, 1)
        return A

    def make_log_A(self):
        A_sparse = self.make_A_sparse()
        log_A_sparse = torch.sparse.log_softmax(A_sparse, dim=-1)
        log_A = log_A_sparse.to_dense()
        log_A = log_A.repeat(self.num_models, 1, 1)
        # log_A = nn.Parameter(log_A, requires_grad=self.transitions_trainable)
        return log_A

    def make_initial_distribution(self):
        return F.softmax(self.starting_distribution_kernel, dim=-1).repeat(1, self.num_models, 1)

    def forward(self, inputs):
        """
        参数 : 
            inputs: 形状 (k, b, q)
        返回 : 
            形状 (k, b, q)
        """
        # k 个输入与 k 个矩阵的批矩阵乘法
        if self.reverse:
            return torch.matmul(inputs, self.A_transposed)
        else:
            return torch.matmul(inputs, self.A)

    def get_prior_log_densities(self):
        # 可以在将来用于正则化。
        return {"none": 0.}

    def make_transition_indices(self, model_index=0):
        """
        返回稀疏转换矩阵内核的 3D 索引 (model_index, from_state, to_state)。
        假设状态顺序 : Ir, I0, I1, I2, E0, E1, E2
        """
        Ir = 0
        I = list(range(1, 4))
        E = list(range(4, 7))
        indices = [(Ir, Ir), (Ir, E[0]), (E[2], Ir)]
        for cds in range(3):
            indices.append((E[cds], E[(cds + 1) % 3]))
            indices.append((E[cds], I[cds]))
            indices.append((I[cds], I[cds]))
            indices.append((I[cds], E[(cds + 1) % 3]))
        indices = np.concatenate([np.full((len(indices), 1), model_index, dtype=np.int64), indices], axis=1)
        assert len(indices) == 15
        return indices

    def make_transition_init(self, k=1, sd=0.05):
        # 使用大致真实的初始长度分布
        sd = 0  # TODO: for testing, keeping small change.
        init = []
        for edge in self.indices:
            # edge = (model, from, to), ingore model for now
            if self.is_intergenic_loop(edge):
                p_loop = 1 - 1. / self.initial_ir_len
                init.append(-np.log(1 / p_loop - 1))
            elif self.is_intron_loop(edge, k):
                p_loop = 1 - 1. / self.initial_intron_len
                init.append(-np.log(1 / p_loop - 1))
            elif self.is_exon_transition(edge, k):
                p_next_exon = 1 - 1. / self.initial_exon_len
                init.append(-np.log(1 / p_next_exon - 1))
            elif self.is_exon_1_out_transition(edge, k):
                init.append(np.log(1. / (2)))
            elif self.is_intergenic_out_transition(edge, k):
                init.append(np.log(1. / k) + np.random.normal(0., sd))
            else:
                init.append(0)
        return np.array(init)

    def get_config(self):
        return {"initial_exon_len": self.initial_exon_len,
                "initial_intron_len": self.initial_intron_len,
                "initial_ir_len": self.initial_ir_len,
                "starting_distribution_init": self.starting_distribution_init,
                "starting_distribution_trainable": self.starting_distribution_trainable,
                "transitions_trainable": self.transitions_trainable}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GenePredHMMTransitioner(SimpleGenePredHMMTransitioner):
    """
    使用强制生物学结构的起始和终止状态扩展简单 HMM。
    假设状态顺序 : Ir, I0, I1, I2, E0, E1, E2,
                     START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
    """

    def __init__(self, use_experimental_prior=False, **kwargs):
        if not hasattr(self, "num_states"):
            self.num_states = 15
        if not hasattr(self, "k"):
            self.k = 1
        super(GenePredHMMTransitioner, self).__init__(**kwargs)
        self.use_experimental_prior = use_experimental_prior
        if use_experimental_prior:
            self.alpha = self.make_prior_alpha()

    def make_transition_indices(self, model_index=0):
        """
        返回稀疏转换矩阵内核的 3D 索引 (model_index, from_state, to_state)。
        """
        Ir = 0
        I = list(range(1, 4))
        E = list(range(4, 7))
        START = 7
        EI = list(range(8, 11))
        IE = list(range(11, 14))
        STOP = 14
        indices = [(Ir, Ir), (Ir, START), (STOP, Ir), (START, E[1]), (E[1], STOP)]
        for cds in range(3):
            indices.append((E[cds], E[(cds + 1) % 3]))
            indices.append((E[cds], EI[cds]))
            indices.append((EI[cds], I[cds]))
            indices.append((I[cds], I[cds]))
            indices.append((I[cds], IE[cds]))
            indices.append((IE[cds], E[cds]))
        indices = np.concatenate([np.full((len(indices), 1), model_index, dtype=np.int64), indices], axis=1)
        assert len(indices) == 23
        return indices

    def gather_binary_probs_for_prior(self, A):
        """
        从先验中使用的转换矩阵中提取二元分布。
        """
        # 自循环状态
        m = 1 + 3 * self.k
        diag = torch.diag(A[:m, :m])
        probs_ir_intron = torch.stack([diag, torch.sum(A[:m, :], dim=-1) - diag], dim=1)
        probs_exon = []
        # 外显子
        for i in range(3):
            for j in range(self.k):
                e = 1 + (i + 3) * self.k + j
                next_e = 1 + 3 * self.k + ((i + 1) % 3) * self.k + j
                probs_exon.extend([A[e, next_e], torch.sum(A[e, :]) - A[e, next_e]])
        probs_exon = torch.stack(probs_exon, dim=0)
        probs_exon = probs_exon.reshape(3 * self.k, 2)
        probs = torch.cat([probs_ir_intron, probs_exon], dim=0)
        return probs

    def make_prior_alpha(self, n=1e3):
        # 假设先验绘制次数
        # 我们根据我们看到每个转换的预期次数选择 alpha
        # 较高的值使先验更严格
        p0 = torch.tensor(self.make_transition_init(self.k, self.init_component_sd)).float().unsqueeze(0)
        A0_sparse = self.make_A_sparse(values=p0.reshape(-1))
        A0 = A0_sparse.to_dense()[0]
        return self.gather_binary_probs_for_prior(A0) * n

    def get_prior_log_densities(self):
        # 根据作为初始分布给出的值正则化转换概率。
        # Dirichlet 参数是基于初始分布的 n 个先验绘制选择的。
        if self.use_experimental_prior:
            self.binary_probs = self.gather_binary_probs_for_prior(self.A[0])
            log_p = torch.log(self.binary_probs)
            priors = torch.sum((self.alpha - 1) * log_p, dim=-1)
            return {i: priors[i].item() for i in range(1 + 6 * self.k)}
        else:
            return {"none": 0.}


class GenePredMultiHMMTransitioner(GenePredHMMTransitioner):
    """
    与 GenePredHMMTransitioner 相同, 但具有共享相同架构但参数不同的多个 (子) HMM。
    与 GenePredHMMTransitioner 相同的状态顺序, 但 Ir 以外的每个状态都乘以 k 次 : 
    Ir, I0*k, I1*k, I2*k, E0*k, E1*k, E2*k, START*k, EI0*k, EI1*k, EI2*k, IE0*k, IE1*k, IE2*k, STOP*k
    参数 : 
        k: 共享 IR 状态的基因模型副本数。
        init_component_sd: 用于初始化转换 IR -> 组件的噪声的标准差。
    """

    def __init__(self, k=1, init_component_sd=0.2, **kwargs):
        self.k = k
        self.num_states = 1 + 14 * k
        self.init_component_sd = init_component_sd
        super(GenePredMultiHMMTransitioner, self).__init__(**kwargs)
        self.init = self.make_transition_init(k, init_component_sd)

    def make_transition_indices(self, model_index=0):
        """
        返回稀疏转换矩阵内核的 3D 索引 (model_index, from_state, to_state)。
        """
        Ir = 0
        I = list(range(1, 1 + 3 * self.k))
        E = list(range(1 + 3 * self.k, 1 + 6 * self.k))
        START = list(range(1 + 6 * self.k, 1 + 7 * self.k))
        EI = list(range(1 + 7 * self.k, 1 + 10 * self.k))
        IE = list(range(1 + 10 * self.k, 1 + 13 * self.k))
        STOP = list(range(1 + 13 * self.k, 1 + 14 * self.k))
        indices = [(Ir, Ir)]
        for hmm in range(self.k):
            indices.extend([(Ir, START[hmm]), (STOP[hmm], Ir),
                            (START[hmm], E[self.k + hmm]), (E[self.k + hmm], STOP[hmm])])
            for cds in range(3):
                indices.extend([(E[self.k * cds + hmm], E[self.k * ((cds + 1) % 3) + hmm]),
                                (E[self.k * cds + hmm], EI[self.k * cds + hmm]),
                                (EI[self.k * cds + hmm], I[self.k * cds + hmm]),
                                (I[self.k * cds + hmm], I[self.k * cds + hmm]),
                                (I[self.k * cds + hmm], IE[self.k * cds + hmm]),
                                (IE[self.k * cds + hmm], E[self.k * cds + hmm])])
        indices = np.concatenate([np.full((len(indices), 1), model_index, dtype=np.int64), indices], axis=1)
        assert len(indices) == 1 + 22 * self.k
        return indices

    def get_config(self):
        config = super(GenePredMultiHMMTransitioner, self).get_config()
        config.update({"k": self.k})
        return config


if __name__ == '__main__':
    torch.manual_seed(42)  # 使用与 TensorFlow 相同的种子值

    model = SimpleGenePredHMMTransitioner()
    # model.build(input_shape=(1, 2, 7))
    model.recurrent_init()
    # input_shape = (1, 2, 15)
    inputs = torch.tensor([[[0.6645621, 0.44100678, 0.3528825, 0.46448255, 0.03366041, 0.68467236, 0.74011743],
                            [0.8724445, 0.22632635, 0.22319686, 0.3103881, 0.7223358, 0.13318717, 0.5480639]]])
    print(inputs)
    outputs = model(inputs)
    print(outputs, outputs.shape)
