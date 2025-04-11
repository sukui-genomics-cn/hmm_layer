import numpy as np
import torch
import torch.nn as nn

from . import Initializers as initializers
from . import Priors as priors
from .Utility import get_num_states, get_num_states_implicit




class ProfileHMMTransitioner(nn.Module):
    """
    一个 transitioner 定义了 HMM 状态之间允许的转换, 如何初始化它们, 
    以及如何表示转移矩阵 (稠密、稀疏或其他) 。
    transitioner 还保存转移分布的先验。
    此 transitioner 使用附加的 Plan7 状态实现默认的 profile HMM 逻辑。

    参数 : 
        transition_init: 每个边类型的初始化器字典列表, 每个模型一个。
        flank_init: 左侧侧翼状态的初始概率的初始化器字典列表。
        prior: 规范化每种转移类型的兼容先验。
        frozen_kernels: 一个字典, 可用于通过添加 "kernel_id": False 来省略某些内核的参数更新。
    """
    def __init__(self,
                 transition_init=initializers.make_default_transition_init(),
                 flank_init=initializers.make_default_flank_init(),
                 prior=None,
                 frozen_kernels={},
                 **kwargs):
        super(ProfileHMMTransitioner, self).__init__(**kwargs)
        transition_init = [transition_init] if isinstance(transition_init, dict) else transition_init
        self.transition_init = transition_init  # 假设 NoDependency 已移除
        self.flank_init = [flank_init] if not isinstance(flank_init, list) else flank_init
        self.prior = priors.ProfileHMMTransitionPrior() if prior is None else prior
        self.frozen_kernels = frozen_kernels
        self.approx_log_zero = -1000.
        self.reverse = False

    def set_lengths(self, lengths):
        """
        设置模型长度。

        参数 : 
            lengths: 模型长度列表。
        """
        self.lengths = lengths
        self.num_states = get_num_states(lengths)
        self.num_states_implicit = get_num_states_implicit(lengths)
        self.max_num_states = max(self.num_states)
        self.num_models = len(lengths)
        self.explicit_transition_kernel_parts = [_make_explicit_transition_kernel_parts(length) for length in self.lengths]
        self.implicit_transition_parts = [_make_implicit_transition_parts(length) for length in self.lengths]
        self.sparse_transition_indices_implicit = [_make_sparse_transition_indices_implicit(length) for length in self.lengths]
        self.sparse_transition_indices_explicit = [_make_sparse_transition_indices_explicit(length) for length in self.lengths]
        # 确保参数有效
        assert len(self.lengths) == len(self.transition_init), \
            f"转移初始化器数量 ({len(self.transition_init)}) 应与模型数量 ({len(self.lengths)}) 匹配。"
        assert len(self.lengths) == len(self.flank_init), \
            f"侧翼初始化器数量 ({len(self.flank_init)}) 应与模型数量 ({len(self.lengths)}) 匹配。"
        for init, parts in zip(self.transition_init, self.explicit_transition_kernel_parts):
            _assert_transition_init_kernel(init, parts)

    def build(self, input_shape=None):
        """构建模型参数。"""
        if hasattr(self, 'transition_kernel'):
            return
        # (稀疏) 内核被分为几组转换。
        # 为了避免将长数组切片为较小部分时容易出错, 
        # 我们将这些部分存储为字典, 并在以后按正确的顺序连接它们。
        # 内核与具有删除状态的隐式模型中的转移矩阵密切相关。
        self.transition_kernel = []
        for model_kernel_parts in self._get_kernel_parts_init_list():
            model_transition_kernel = {}
            for i, (part_name, length, init, frozen, shared_with) in enumerate(model_kernel_parts):
                if (shared_with is None or all(s not in model_transition_kernel for s in shared_with)):
                    k = nn.Parameter(torch.tensor(init).float(), requires_grad=not frozen)
                else:
                    for s in shared_with:
                        if s in model_transition_kernel:
                            k = model_transition_kernel[s]
                            break
                model_transition_kernel[part_name] = k
            self.transition_kernel.append(model_transition_kernel)

        # 与左侧侧翼状态的初始概率密切相关
        self.flank_init_kernel = [nn.Parameter(torch.tensor(init).float()) for init in self.flank_init]
        self.prior.build()
        self.built = True

    def _get_kernel_parts_init_list(self):
        """
        返回一个列表的列表，该列表指定所有转移内核的单元初始化数据。
        外部列表包含每个模型一个列表。内部列表包含 5 元组 : 
        (part_name: str, length: int, init: torch.Tensor, frozen: bool, shared_with: list 或 None)
        """
        # 假设 shared_kernels 最多包含每个名称一次
        shared_kernels = [["right_flank_loop", "left_flank_loop"],
                          ["right_flank_exit", "left_flank_exit"]]
        # 将每个名称映射到它所在的列表
        shared_kernel_dict = {}
        for shared in shared_kernels:
            for name in shared:
                shared_kernel_dict[name] = shared
        kernel_part_list = []
        for init, parts in zip(self.transition_init, self.explicit_transition_kernel_parts):
            kernel_part_list.append([(part_name,
                                       length,
                                       init[part_name],
                                       self.frozen_kernels.get(part_name, False),
                                       shared_kernel_dict.get(part_name, None))
                                      for part_name, length in parts])
        return kernel_part_list

    def recurrent_init(self):
        """在每次递归运行之前自动调用。应将其用于仅在递归层应用一次的设置。"""
        self.A_sparse, self.implicit_log_probs, self.log_probs, self.probs = self.make_A_sparse(return_probs=True)
        self.A = self.A_sparse.to_dense()
        self.A_t = torch.transpose(self.A, 1, 2)

    def make_flank_init_prob(self):
        return torch.sigmoid(torch.stack([k for k in self.flank_init_kernel]))

    def make_initial_distribution(self):
        """构造每个模型的初始状态分布, 该分布取决于转移概率。
        返回 : 
            每个模型的概率分布。形状 : (1, k, q)
        """
        # 状态顺序 : LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL
        init_flank_probs = self.make_flank_init_prob()
        log_init_flank_probs = torch.log(init_flank_probs)
        log_complement_init_flank_probs = torch.log(1 - init_flank_probs)
        log_init_dists = []
        for i in range(self.num_models):
            log_init_match = (self.implicit_log_probs[i]["left_flank_to_match"]
                              + log_complement_init_flank_probs[i]
                              - self.log_probs[i]["left_flank_exit"])
            log_init_right_flank = (self.implicit_log_probs[i]["left_flank_to_right_flank"]
                                    + log_complement_init_flank_probs[i]
                                    - self.log_probs[i]["left_flank_exit"])
            log_init_unannotated_segment = (self.implicit_log_probs[i]["left_flank_to_unannotated_segment"]
                                            + log_complement_init_flank_probs[i]
                                            - self.log_probs[i]["left_flank_exit"])
            log_init_terminal = (self.implicit_log_probs[i]["left_flank_to_terminal"]
                                 + log_complement_init_flank_probs[i]
                                 - self.log_probs[i]["left_flank_exit"])
            log_init_insert = torch.zeros((self.lengths[i] - 1), dtype=torch.float32) + self.approx_log_zero
            log_init_dist = torch.cat([log_init_flank_probs[i],
                                       log_init_match,
                                       log_init_insert,
                                       log_init_unannotated_segment,
                                       log_init_right_flank,
                                       log_init_terminal], dim=0)
            log_init_dist = torch.nn.functional.pad(log_init_dist,
                                                   (0, self.max_num_states - self.num_states[i]),
                                                   value=self.approx_log_zero)
            log_init_dists.append(log_init_dist)
        log_init_dists = torch.stack(log_init_dists, dim=0)
        log_init_dists = log_init_dists.unsqueeze(0)
        init_dists = torch.exp(log_init_dists)
        return init_dists

    def make_transition_kernel(self):
        """以一致的顺序连接所有转换类型 (例如, 匹配到匹配) 的内核。
        返回 : 
            连接的内核向量。
        """
        concat_transition_kernels = []
        for part_names, kernel in zip(self.explicit_transition_kernel_parts, self.transition_kernel):
            concat_kernel = torch.cat([kernel[part_name] for part_name, _ in part_names], dim=0)
            concat_transition_kernels.append(concat_kernel)
        return concat_transition_kernels

    def make_probs(self):
        """从内核计算所有转移概率。将 softmax 应用于状态的所有传出边的内核值。
        返回 : 
            将转移类型映射到概率的字典。
        """
        model_prob_dicts = []
        for indices_explicit, parts, num_states, kernel in zip(self.sparse_transition_indices_explicit,
                                                            self.explicit_transition_kernel_parts,
                                                            self.num_states_implicit,
                                                            self.make_transition_kernel()):
            probs_dict = {}
            indices_explicit = np.concatenate([indices_explicit[part_name] for part_name, _ in parts], axis=0)
            dense_probs = make_transition_matrix_from_indices(indices_explicit, kernel, num_states)
            probs_vec = torch.gather(dense_probs, 1, torch.tensor(indices_explicit).long())
            lsum = 0
            for part_name, length in parts:
                probs_dict[part_name] = probs_vec[lsum:lsum + length]
                lsum += length
            model_prob_dicts.append(probs_dict)
        return model_prob_dicts

    def make_log_probs(self):
        """计算概率的log值。
        返回 : 
            将转移类型映射到log概率的字典。
        """
        probs = self.make_probs()
        log_probs = [{key: torch.log(p) for key, p in model_probs.items()} for model_probs in probs]
        return log_probs, probs

    def make_implicit_log_probs(self):
        """计算隐式模型中的所有对数转移概率。
        返回 : 
            将转移类型映射到概率的字典。
        """
        log_probs, probs = self.make_log_probs()
        implicit_log_probs = []
        for p, length in zip(log_probs, self.lengths):
            # 计算 match_skip(i,j) = P(Mj+2 | Mi) , L x L
            # 将 "begin" 视为 M0, 将 "end" 视为 ML
            MD = p["match_to_delete"].unsqueeze(-1)
            DD = torch.cat([torch.tensor([0.0]), p["delete_to_delete"]], dim=0)
            DD_cumsum = torch.cumsum(DD, dim=0)
            DD = DD_cumsum.unsqueeze(0) - DD_cumsum.unsqueeze(1)
            DM = p["delete_to_match"].unsqueeze(0)
            M_skip = MD + DD + DM
            upper_triangle = torch.tril(torch.ones([length - 2, length - 2], dtype=torch.float32), diagonal=0)
            entry_add = _logsumexp(p["begin_to_match"],
                                    torch.cat([torch.tensor([self.approx_log_zero]), M_skip[0, :-1]], dim=0))
            exit_add = _logsumexp(p["match_to_end"],
                                    torch.cat([M_skip[1:, -1], torch.tensor([self.approx_log_zero])], dim=0))
            imp_probs = {}
            imp_probs["match_to_match"] = p["match_to_match"]
            imp_probs["match_to_insert"] = p["match_to_insert"]
            imp_probs["insert_to_match"] = p["insert_to_match"]
            imp_probs["insert_to_insert"] = p["insert_to_insert"]
            imp_probs["left_flank_loop"] = p["left_flank_loop"]
            imp_probs["right_flank_loop"] = p["right_flank_loop"]
            imp_probs["right_flank_exit"] = p["right_flank_exit"]
            imp_probs["match_skip"] = torch.masked_select(M_skip[1:-1, 1:-1], upper_triangle.bool())
            imp_probs["left_flank_to_match"] = p["left_flank_exit"] + entry_add
            imp_probs["left_flank_to_right_flank"] = (p["left_flank_exit"] + M_skip[0, -1] + p["end_to_right_flank"])
            imp_probs["left_flank_to_unannotated_segment"] = (p["left_flank_exit"] + M_skip[0, -1] + p["end_to_unannotated_segment"])
            imp_probs["left_flank_to_terminal"] = (p["left_flank_exit"] + M_skip[0, -1] + p["end_to_terminal"])
            imp_probs["match_to_unannotated"] = exit_add + p["end_to_unannotated_segment"]
            imp_probs["match_to_right_flank"] = exit_add + p["end_to_right_flank"]
            imp_probs["match_to_terminal"] = exit_add + p["end_to_terminal"]
            imp_probs["unannotated_segment_to_match"] = p["unannotated_segment_exit"] + entry_add
            imp_probs["unannotated_segment_loop"] = _logsumexp(p["unannotated_segment_loop"],
                                                              (p["unannotated_segment_exit"] + M_skip[0, -1] + p["end_to_unannotated_segment"]))
            imp_probs["unannotated_segment_to_right_flank"] = (p["unannotated_segment_exit"] + M_skip[0, -1] + p["end_to_right_flank"])
            imp_probs["unannotated_segment_to_terminal"] = (p["unannotated_segment_exit"] + M_skip[0, -1] + p["end_to_terminal"])
            imp_probs["terminal_self_loop"] = torch.zeros((1,), dtype=torch.float32)
            implicit_log_probs.append(imp_probs)
        return implicit_log_probs, log_probs, probs

    def make_log_A_sparse(self, return_probs=False):
        """
        返回 : 
            表示 k 个模型的对数转移矩阵的形状为 (k, q, q) 的 3D 稀疏张量。
        """
        implicit_log_probs, log_probs, probs = self.make_implicit_log_probs()
        values_all_models, indices_all_models = [], []
        for i, (p, parts, indices, num_states) in enumerate(zip(implicit_log_probs,
                                                            self.implicit_transition_parts,
                                                            self.sparse_transition_indices_implicit,
                                                            self.num_states_implicit)):
            # 按模型顺序获取值和索引
            values = torch.cat([p[part_name] for part_name, _ in parts], dim=0)
            indices_concat = np.concatenate([indices[part_name] for part_name, _ in parts], axis=0)
            # 按行主序重新排序
            row_major_order = np.argsort([i * num_states + j for i, j in indices_concat])
            indices_concat = indices_concat[row_major_order]
            values = torch.gather(values, 0, torch.tensor(row_major_order).long())
            indices_concat = np.pad(indices_concat, ((0, 0), (1, 0)), constant_values=i)
            values_all_models.append(values)
            indices_all_models.append(indices_concat)
        values_all_models = torch.cat(values_all_models, dim=0)  # "model major" order
        indices_all_models = np.concatenate(indices_all_models, axis=0)
        log_A_sparse = torch.sparse_coo_tensor(
            torch.tensor(indices_all_models).t(),
            values_all_models,
            (self.num_models, self.max_num_states, self.max_num_states)
        )
        if return_probs:
            return log_A_sparse, implicit_log_probs, log_probs, probs
        else:
            return log_A_sparse

    def make_log_A(self):
        """
        返回 : 
            表示 k 个模型的对数转移矩阵的形状为 (k, q, q) 的 3D 稠密张量。
        """
        log_A = self.make_log_A_sparse()
        log_A = log_A.to_dense()
        log_A = torch.where(log_A == 0, torch.full_like(log_A, self.approx_log_zero), log_A)
        return log_A

    def make_A_sparse(self, return_probs=False):
        """
        返回 : 
            表示 k 个模型的转移矩阵的形状为 (k, q, q) 的 3D 稀疏张量。
        """
        if return_probs:
            log_A_sparse, *p = self.make_log_A_sparse(True)
        else:
            log_A_sparse = self.make_log_A_sparse(False)
        A_sparse = torch.sparse_coo_tensor(
            log_A_sparse.indices(),
            torch.exp(log_A_sparse.values()),
            log_A_sparse.shape
        )
        if return_probs:
            return A_sparse, *p
        else:
            return A_sparse

    def make_A(self):
        """
        返回 : 
            表示 k 个模型的转移矩阵的形状为 (k, q, q) 的 3D 稠密张量。
        """
        A = self.make_A_sparse()
        A = A.to_dense()
        return A

    def forward(self, inputs):
        """
        参数 : 
            inputs: 形状 (k, b, q)
        返回 : 
            形状 (k, b, q)
        """
        # k 个输入与 k 个矩阵的批矩阵乘法
        if self.reverse:
            return torch.matmul(inputs, self.A_t)
        else:
            return torch.matmul(inputs, self.A)

    def get_prior_log_densities(self):
        return self.prior(self.make_probs(), self.make_flank_init_prob())


def make_transition_matrix_from_indices(indices, kernel, num_states, approx_log_zero=-1000.):
    """Constructs a dense probabilistic transition matrix from a sparse index list and a kernel.
    Args:
        indices: A 2D tensor of shape (num_transitions, 2) that specifies the indices of the kernel.
        kernel: A 1D tensor of shape (num_transitions) that contains the kernel values.
        num_states: The number of states in the model.
    Returns:
        A dense probabilistic transition matrix of shape (num_states, num_states).
    """
    # Convert indices to torch tensor if it's not already
    if not torch.is_tensor(indices):
        indices = torch.from_numpy(np.array(indices))

    # torch.sparse requires a strict row-major ordering of the indices
    row_major_order = torch.argsort(torch.tensor([i * num_states + j for i, j in indices]))
    # reorder row-major
    indices_row_major = indices[row_major_order]
    kernel_row_major = kernel[row_major_order]

    # don't allow too small values in the kernel
    kernel_row_major = torch.maximum(kernel_row_major, torch.tensor(approx_log_zero + 1, device=kernel.device))
    kernel_row_major[kernel_row_major == 0] = 1e-12

    # create sparse tensor - need to transpose indices for COO format
    sparse_kernel = torch.sparse_coo_tensor(
        indices=indices_row_major.t().to(kernel.device),
        values=kernel_row_major,
        size=[num_states, num_states],
        device=kernel.device
    )

    # convert to dense
    dense_kernel = sparse_kernel.to_dense()
    # fill default value for non-specified indices
    dense_kernel[dense_kernel == 0] = approx_log_zero

    # softmax that ignores non-existing transitions
    dense_probs = torch.nn.functional.softmax(dense_kernel, dim=-1)

    # mask out non-existing transitions and rescale for numerical stability
    mask = (dense_kernel > approx_log_zero).to(dense_probs.dtype)
    dense_probs += 1e-16
    dense_probs = dense_probs * mask
    dense_probs /= torch.sum(dense_probs, dim=-1, keepdim=True)

    return dense_probs

# def make_transition_matrix_from_indices(indices, kernel, num_states, approx_log_zero=-1000.):
#     """
#     从稀疏索引列表和内核构建稠密概率转移矩阵。
#
#     参数 :
#         indices: 形状为 (num_transitions, 2) 的 2D 张量, 指定内核的索引。
#         kernel: 形状为 (num_transitions) 的 1D 张量, 包含内核值。
#         num_states: 模型中的状态数。
#         approx_log_zero: 用于近似日志零的值, 用于处理不存在的转换。
#
#     返回 :
#         形状为 (num_states, num_states) 的稠密概率转移矩阵。
#     """
#     # 确保索引按行主序排列
#     row_major_order = np.argsort([i * num_states + j for i, j in indices])
#     indices_row_major = indices[row_major_order]
#     kernel_row_major = kernel[row_major_order]
#
#     # 将内核值限制在 approx_log_zero + 1 以上, 以避免过小的数值
#     kernel_row_major = torch.maximum(kernel_row_major, torch.tensor(approx_log_zero + 1.0))
#     kernel_row_major[kernel_row_major == 0] = 1e-10
#
#     # 创建稀疏张量
#     sparse_kernel = torch.sparse_coo_tensor(
#         torch.tensor(indices_row_major).t().to(kernel_row_major.device),
#         kernel_row_major,
#         (num_states, num_states)
#     )
#
#     # 将稀疏张量转换为稠密张量, 默认值为 approx_log_zero
#     dense_kernel = torch.where(sparse_kernel.to_dense() == 0, torch.full_like(sparse_kernel.to_dense(), approx_log_zero), sparse_kernel.to_dense())
#
#     # 对稠密内核应用 softmax, 忽略不存在的转换
#     dense_probs = torch.nn.functional.softmax(dense_kernel, dim=-1)
#
#     # 创建掩码, 用于掩盖不存在的转换
#     mask = (dense_kernel > approx_log_zero).float()
#
#     # 添加一个很小的数值以增加数值稳定性, 并将不存在的转换置零, 然后进行归一化
#     dense_probs += 1e-16
#     dense_probs = dense_probs * mask
#     dense_probs /= (torch.sum(dense_probs, dim=-1, keepdim=True)+1e-16)
#
#     return dense_probs
#

def _make_explicit_transition_kernel_parts(length): 
    return [("begin_to_match", length), 
             ("match_to_end", length),
             ("match_to_match", length-1), 
             ("match_to_insert", length-1),
             ("insert_to_match", length-1), 
             ("insert_to_insert", length-1),
            #consider begin and end states as additional match states:
             ("match_to_delete", length), 
             ("delete_to_match", length),
             ("delete_to_delete", length-1),
             ("left_flank_loop", 1), 
             ("left_flank_exit", 1),
             ("unannotated_segment_loop", 1), 
             ("unannotated_segment_exit", 1),
             ("right_flank_loop", 1), 
             ("right_flank_exit", 1),
             ("end_to_unannotated_segment", 1), 
             ("end_to_right_flank", 1), 
             ("end_to_terminal", 1)]


def _make_implicit_transition_parts(length):
    return ([("left_flank_loop", 1),
               ("left_flank_to_match", length),
               ("left_flank_to_right_flank", 1),
               ("left_flank_to_unannotated_segment", 1),
               ("left_flank_to_terminal", 1),
               ("match_to_match", length-1),
               ("match_skip", int((length-1) * (length-2) / 2)),
               ("match_to_unannotated", length),
               ("match_to_right_flank", length),
               ("match_to_terminal", length),
               ("match_to_insert", length-1),
               ("insert_to_match", length-1),
               ("insert_to_insert", length-1),
               ("unannotated_segment_to_match", length),
               ("unannotated_segment_loop", 1),
               ("unannotated_segment_to_right_flank", 1),
               ("unannotated_segment_to_terminal", 1),
               ("right_flank_loop", 1),
               ("right_flank_exit", 1),
               ("terminal_self_loop", 1)])



def _make_sparse_transition_indices_implicit(length):
    """ Returns 2D indices for the kernel of a sparse (2L+3 x 2L+3) transition matrix without silent states.
        Assumes the following ordering of states: 
        LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, RIGHT_FLANK, TERMINAL
    """
    a = np.arange(length+1, dtype=np.int64)
    left_flank = 0
    first_insert = length + 1
    unanno_segment = 2*length
    right_flank = 2*length + 1
    terminal = 2*length + 2
    zeros = np.zeros(length, dtype=a.dtype)
    indices_dict = {
        "left_flank_loop" : [[left_flank, left_flank]],
        "left_flank_to_match" : np.stack([zeros+left_flank, a[1:]], axis=1),
        "left_flank_to_right_flank" : [[left_flank, right_flank]],
        "left_flank_to_unannotated_segment" : [[left_flank, unanno_segment]],
        "left_flank_to_terminal" : [[left_flank, terminal]],
        "match_to_match" : np.stack([a[1:-1], a[1:-1]+1], axis=1),
        "match_skip" : np.concatenate([np.stack([zeros[:-i-1]+i, 
                                     np.arange(i+2, length+1)], axis=1)
            for i in range(1, length-1)
                ], axis=0),
        "match_to_unannotated" : np.stack([a[1:], zeros+unanno_segment], axis=1),
        "match_to_right_flank" : np.stack([a[1:], zeros+right_flank], axis=1),
        "match_to_terminal" : np.stack([a[1:], zeros+terminal], axis=1),
        "match_to_insert" : np.stack([a[1:-1], a[:-2]+first_insert], axis=1),
        "insert_to_match" : np.stack([a[:-2]+first_insert, a[2:]], axis=1),
        "insert_to_insert" : np.stack([a[:-2]+first_insert]*2, axis=1),
        "unannotated_segment_to_match" : np.stack([zeros+unanno_segment, a[1:]], axis=1),
        "unannotated_segment_loop" : [[unanno_segment, unanno_segment]],
        "unannotated_segment_to_right_flank" : [[unanno_segment, right_flank]],
        "unannotated_segment_to_terminal" : [[unanno_segment, terminal]],
        "right_flank_loop" : [[right_flank, right_flank]],
        "right_flank_exit" : [[right_flank, terminal]],
        "terminal_self_loop" : [[terminal, terminal]]}
    return indices_dict

def _make_sparse_transition_indices_explicit(length):
    """ Returns 2D indices for the (linear) kernel of a sparse (3L+3 x 3L+3) transition matrix with silent states.
        Assumes the following ordering of states:
        LEFT_FLANK, MATCH x length, INSERT x length-1, UNANNOTATED_SEGMENT, 
        RIGHT_FLANK, TERMINAL, BEGIN, END, DELETE x length
    """
    a = np.arange(length+1, dtype=np.int64)
    left_flank = 0
    first_insert = length + 1
    unanno_segment = 2*length
    right_flank = 2*length + 1
    terminal = 2*length + 2
    begin = 2*length + 3
    end = 2*length + 4
    first_delete = 2*length + 5
    zeros = np.zeros(length, dtype=a.dtype)
    indices_dict = {
        "begin_to_match" : np.stack([zeros+begin, a[1:]], axis=1),
        "match_to_end" : np.stack([a[1:], zeros+end], axis=1),
        "match_to_match" : np.stack([a[1:-1], a[1:-1]+1], axis=1),
        "match_to_insert" : np.stack([a[1:-1], a[:-2]+first_insert], axis=1),
        "insert_to_match" : np.stack([a[:-2]+first_insert, a[2:]], axis=1),
        "insert_to_insert" : np.stack([a[:-2]+first_insert]*2, axis=1),
        "match_to_delete" : np.stack([np.insert(a[1:-1], 0, begin), a[:-1]+first_delete], axis=1),
        "delete_to_match" : np.stack([a[:-1]+first_delete, np.append(a[:-2]+2, end)], axis=1),
        "delete_to_delete" : np.stack([a[:-2]+first_delete, a[:-2]+first_delete+1], axis=1),
        "left_flank_loop" : [[left_flank, left_flank]],
        "left_flank_exit" : [[left_flank, begin]],
        "unannotated_segment_loop" : [[unanno_segment, unanno_segment]],
        "unannotated_segment_exit" : [[unanno_segment, begin]],
        "right_flank_loop" : [[right_flank, right_flank]],
        "right_flank_exit" : [[right_flank, terminal]],
        "end_to_unannotated_segment" : [[end, unanno_segment]],
        "end_to_right_flank" : [[end, right_flank]],
        "end_to_terminal" : [[end, terminal]] }
    return indices_dict
            
def _assert_transition_init_kernel(kernel_init, parts):
    for part_name,_ in parts:
        assert part_name in kernel_init, "No initializer found for kernel " + part_name + "."
    for part_name in kernel_init.keys():
        assert part_name in [part[0] for part in parts], part_name + " is in the kernel init dict but there is no kernel part matching it. Wrong spelling?"
        import torch

def _logsumexp(x, y):
    """
    计算两个对数概率的和的对数。

    参数 : 
        x: 第一个对数概率。
        y: 第二个对数概率。

    返回 : 
        两个对数概率的和的对数。
    """
    # 找到 x 和 y 的最大值, 以提高数值稳定性
    max_val = torch.max(x, y)
    # 计算 log(exp(x) + exp(y)) = log(exp(x - max_val) + exp(y - max_val)) + max_val
    return max_val + torch.log(torch.exp(x - max_val) + torch.exp(y - max_val))
