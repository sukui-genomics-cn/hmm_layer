import copy

import torch
import torch.nn as nn

from .Emitter import ProfileHMMEmitter  # 假设这些类已转换为 PyTorch
from .Transitioner import ProfileHMMTransitioner
from .Utility import get_num_states  # 假设这些函数已转换为 PyTorch

class HmmCell(nn.Module):
    """
    通用单元, 在其 forward 方法中计算前向算法的一个递归步骤, 也可以计算后向算法。
    它旨在与通用 RNN 层一起使用, 以计算一批序列的似然。它还包装了一个先验, 并提供
    功能 (通过注入的 emitter 和 transitioner) 来构建 emission- 和 transition-matrices, 
    这些矩阵也在其他地方使用, 例如在 Viterbi 期间。
    基于 https://github.com/mslehre/classify-seqs/blob/main/HMMCell.py。

    参数 : 
        num_states: 每个模型的状态数列表。
        dim: 输入序列的维度数。
        emitter: 遵循 emitter 接口的对象或对象列表 (参见 MultinomialAminoAcidEmitter) 。
        transitioner: 遵循 transitioner 接口的对象 (参见 ProfileHMMTransitioner) 。
        use_step_counter: 是否使用步长计数器。
        use_fake_step_counter: 仅用于 Tiberius 的向后兼容性, 否则永远不要将其设置为 True。
        **kwargs: 其他关键字参数。
    """
    def __init__(self, num_states, dim, emitter, transitioner, use_step_counter=False, use_fake_step_counter=False, device=None, **kwargs):
        super(HmmCell, self).__init__(**kwargs)
        self.num_states = num_states
        self.num_models = len(self.num_states)
        self.max_num_states = max(self.num_states)
        self.dim = dim
        emitter = [emitter] if not isinstance(emitter, list) else emitter
        self.emitter = nn.ModuleList(emitter)
        self.transitioner = transitioner
        self.epsilon = 1e-16
        self.reverse = False
        self.use_step_counter = use_step_counter
        self.use_fake_step_counter = use_fake_step_counter

        # 初始化参数
        self.recurrent_init()

    def recurrent_init(self):
        """初始化递归计算中使用的参数。"""
        self.transitioner.recurrent_init()
        for em in self.emitter:
            em.recurrent_init()
        self.log_A_dense = self.transitioner.make_log_A()
        self.log_A_dense_t = torch.transpose(self.log_A_dense, 1, 2)
        self.init_dist = self.make_initial_distribution()
        if not self.reverse and self.use_step_counter:
            self.step_counter = torch.tensor(-1, dtype=torch.int32)

    def make_initial_distribution(self):
        """构造初始状态分布, 该分布取决于转移概率。
        参见 ProfileHMMTransitioner。

        返回 : 
            形状为 (1, num_model, q) 的概率分布。
        """
        return self.transitioner.make_initial_distribution()

    def emission_probs(self, inputs, end_hints=None, training=False):
        """计算给定观测的每个状态的发射概率。多个发射器相乘。

        参数 : 
            inputs: 一批序列位置。
            end_hints: 形状为 (..., 2, num_states) 的张量, 其中包含每个块的左右端的正确状态。
        """
        em_probs = self.emitter[0](inputs, end_hints=end_hints, training=training)
        for em in self.emitter[1:]:
            em_probs *= em(inputs, end_hints=end_hints, training=training)
        return em_probs

    def forward(self, emission_probs, states, training=None, init=False):
        """计算前向 DP 的一个递归步骤。"""
        old_scaled_forward, old_loglik = states
        old_scaled_forward = old_scaled_forward.view(self.num_models, -1, self.max_num_states)
        if init:
            R = old_scaled_forward
        else:
            R = self.transitioner(old_scaled_forward)
        E = emission_probs.view(self.num_models, -1, self.max_num_states)
        # 如果并行, 允许将输入广播到前向概率
        q = R.shape[1] // E.shape[1]  # q == 1 如果不并行, 否则 q = num_states
        R = R.view(self.num_models, -1, q, self.max_num_states)
        E = E.view(self.num_models, -1, 1, self.max_num_states)
        old_loglik = old_loglik.view(self.num_models, -1, q, 1)
        E = torch.maximum(E, torch.tensor(self.epsilon))
        R = torch.maximum(R, torch.tensor(self.epsilon))
        scaled_forward = E * R
        S = torch.sum(scaled_forward, dim=-1, keepdim=True)
        loglik = old_loglik + torch.log(S)
        scaled_forward /= S
        scaled_forward = scaled_forward.view(-1, q * self.max_num_states)
        loglik = loglik.view(-1, q)
        new_state = [scaled_forward, loglik]
        if self.reverse:
            output = torch.log(R)
            output = output.view(-1, q * self.max_num_states)
            old_loglik = old_loglik.view(-1, q)
            output = torch.cat([output, old_loglik], dim=-1)
        else:
            output = torch.log(scaled_forward)
            output = torch.cat([output, loglik], dim=-1)
        if not self.reverse and self.use_step_counter:
            self.step_counter += 1
        return output, new_state

    def get_initial_state(self, inputs=None, batch_size=None, parallel_factor=1, device=None):
        """返回初始递归状态, 该状态是一对张量 : 缩放的前向概率 (形状为 (num_models*batch, num_states)) 
        和对数似然 (形状为 (num_models*batch, 1)) 。
        返回值可以安全地重塑为 (num_models, batch, ...)。
        如果并行, 则返回的张量具有形状 (num_models*batch, num_states*num_states) 和 (num_models*batch, num_states)。
        """
        if parallel_factor == 1:
            if self.reverse:
                init_dist = torch.ones((self.num_models * batch_size, self.max_num_states), dtype=torch.float32, device=device)
            else:
                init_dist = self.make_initial_distribution().repeat(batch_size, 1, 1).transpose(0, 1).reshape(-1, self.max_num_states).to(device)
            loglik = torch.zeros((self.num_models * batch_size, 1), dtype=torch.float32, device=device)
            return [init_dist.to(device), loglik.to(device)]
        else:
            indices = torch.arange(self.max_num_states).repeat(self.num_models * batch_size)
            init_dist = torch.nn.functional.one_hot(indices, num_classes=self.max_num_states).float().to(device)
            if self.reverse:
                init_dist_chunk = init_dist.clone().view(self.num_models * batch_size, self.max_num_states, self.max_num_states)
                first_emissions = inputs[:, 0, :].view(self.num_models, batch_size // parallel_factor, parallel_factor, self.max_num_states)
                first_emissions = torch.roll(first_emissions, shifts=-1, dims=2).view(self.num_models * batch_size, 1, self.max_num_states)
                init_dist_chunk *= first_emissions
            else:
                init_dist_chunk = init_dist
            init_dist_chunk = init_dist_chunk.view(self.num_models, batch_size * self.max_num_states, self.max_num_states).to(device)
            init_dist_trans = self.transitioner(init_dist_chunk).view(self.num_models, batch_size // parallel_factor, parallel_factor, self.max_num_states * self.max_num_states)
            is_first_chunk = torch.zeros((self.num_models, batch_size // parallel_factor, parallel_factor - 1, self.max_num_states * self.max_num_states), dtype=torch.float32, device=device)
            if self.reverse:
                is_first_chunk = torch.cat([is_first_chunk, torch.ones_like(is_first_chunk[..., :1, :])], dim=2).to(device)
            else:
                is_first_chunk = torch.cat([torch.ones_like(is_first_chunk[..., :1, :]), is_first_chunk], dim=2).to(device)
            init_dist = init_dist.view(self.num_models, batch_size // parallel_factor, parallel_factor, self.max_num_states * self.max_num_states)
            init_dist = is_first_chunk * init_dist + (1 - is_first_chunk) * init_dist_trans
            init_dist = init_dist.view(self.num_models * batch_size, self.max_num_states * self.max_num_states)
            loglik = torch.zeros((self.num_models * batch_size, self.max_num_states), dtype=torch.float32, device=device)
            return [init_dist.to(device), loglik.to(device)]

    def get_aux_loss(self):
        return sum([em.get_aux_loss() for em in self.emitter])

    def get_prior_log_density(self):
        em_priors = [torch.sum(em.get_prior_log_density(), dim=1) for em in self.emitter]
        trans_priors = self.transitioner.get_prior_log_densities()
        prior = sum(em_priors) + sum(trans_priors.values())
        return prior

    # @classmethod
    def make_reverse_direction_offspring(self):
        """返回一个共享此单元参数的单元, 该单元配置为计算后向递归。"""
        # copy_transitioner =
        reverse_cell = copy.deepcopy(HmmCell(self.num_states, self.dim, self.emitter, self.transitioner))
        reverse_cell.reverse = True
        reverse_cell.transitioner.reverse = True
        reverse_cell.recurrent_init()
        return reverse_cell

    def reverse_direction(self, reverse=True):
        self.reverse = reverse
        self.transitioner.reverse = reverse

class MsaHmmCell(HmmCell):
    """用于 profile HMM 的单元, 计算前向算法的前向递归。

    参数 : 
        length: 模型长度/匹配状态数或长度列表。
        dim: 输入序列的维度数。
        emitter: 遵循 emitter 接口的对象或对象列表 (参见 MultinomialAminoAcidEmitter) 。
        transitioner: 遵循 transitioner 接口的对象 (参见 ProfileHMMTransitioner) 。
    """
    def __init__(self, length, dim=24, emitter=None, transitioner=None, **kwargs):
        if emitter is None:
            emitter = ProfileHMMEmitter()
        if transitioner is None:
            transitioner = ProfileHMMTransitioner()
        self.length = [length] if not isinstance(length, list) else length
        super(MsaHmmCell, self).__init__(get_num_states(self.length), dim, emitter, transitioner, **kwargs)
        for em in self.emitter:
            em.set_lengths(self.length)
        self.transitioner.set_lengths(self.length)
        

