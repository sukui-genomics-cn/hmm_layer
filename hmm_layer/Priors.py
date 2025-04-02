import torch
import torch.nn as nn
import os
import numpy as np

# 假设 dm.load_mixture_model 已转换为 PyTorch
# 假设 dm.DirichletMixture 已转换为 PyTorch

class ProfileHMMTransitionPrior(nn.Module):
    """
    profileHMM 转换的默认狄利克雷混合先验。

    参数 : 
        match_comp: 匹配先验的组件数。
        insert_comp: 插入先验的组件数。
        delete_comp: 删除先验的组件数。
        alpha_flank: 偏向于保持 3 个侧翼状态之一的高概率。
        alpha_single: 偏向于单个主模型命中的高概率 (避免循环路径) 。
        alpha_global: 偏向于在第一个匹配项处进入并在最后一个匹配项后退出的高概率模型。
        epsilon: 用于数值稳定性的一个小常数。
    """
    def __init__(self,
                 match_comp=1,
                 insert_comp=1,
                 delete_comp=1,
                 alpha_flank=7000,
                 alpha_single=1e9,
                 alpha_global=1e4,
                 alpha_flank_compl=1,
                 alpha_single_compl=1,
                 alpha_global_compl=1,
                 epsilon=1e-16,
                 **kwargs):
        super(ProfileHMMTransitionPrior, self).__init__(**kwargs)
        self.match_comp = match_comp
        self.insert_comp = insert_comp
        self.delete_comp = delete_comp
        # 相当于狄利克雷混合 -1 的 alpha 参数。
        # 在将主模型与附加的 "Plan7" 状态和转换联合优化时，这些值至关重要。
        self.alpha_flank = alpha_flank
        self.alpha_single = alpha_single
        self.alpha_global = alpha_global
        self.alpha_flank_compl = alpha_flank_compl
        self.alpha_single_compl = alpha_single_compl
        self.alpha_global_compl = alpha_global_compl
        self.epsilon = epsilon

    def build(self, input_shape=None):
        """构建模型参数。"""
        if hasattr(self, 'match_dirichlet'):
            return
        prior_path = os.path.dirname(__file__) + "/trained_prior/transition_priors/"

        match_model_path = prior_path + "_".join(["match_prior", str(self.match_comp), str(torch.float32)]) + ".h5"
        match_model = dm.load_mixture_model(match_model_path, self.match_comp, 3, trainable=False)
        self.match_dirichlet = match_model.layers[-1]

        insert_model_path = prior_path + "_".join(["insert_prior", str(self.insert_comp), str(torch.float32)]) + ".h5"
        insert_model = dm.load_mixture_model(insert_model_path, self.insert_comp, 2, trainable=False)
        self.insert_dirichlet = insert_model.layers[-1]

        delete_model_path = prior_path + "_".join(["delete_prior", str(self.delete_comp), str(torch.float32)]) + ".h5"
        delete_model = dm.load_mixture_model(delete_model_path, self.delete_comp, 2, trainable=False)
        self.delete_dirichlet = delete_model.layers[-1]

        self.built = True

    def forward(self, probs_list, flank_init_prob):
        """计算每个转换先验的对数 pdf 值。

        参数 : 
            probs_list: 一个字典列表，该字典将转换类型映射到每个模型的概率。
            flank_init_prob: 每个模型的侧翼初始概率。

        返回 : 
            一个字典，该字典将先验名称映射到每个模型的先验值列表。
        """
        match_dirichlet = []
        insert_dirichlet = []
        delete_dirichlet = []
        flank_prior = []
        hit_prior = []
        global_prior = []
        for i, probs in enumerate(probs_list):
            log_probs = {part_name: torch.log(p) for part_name, p in probs.items()}
            # 匹配状态转换
            p_match = torch.stack([probs["match_to_match"],
                                     probs["match_to_insert"],
                                     probs["match_to_delete"][1:]], dim=-1) + self.epsilon
            p_match_sum = torch.sum(p_match, dim=-1, keepdim=True)
            match_dirichlet.append(torch.sum(self.match_dirichlet.log_prob(p_match / p_match_sum)))
            # 插入状态转换
            p_insert = torch.stack([probs["insert_to_match"],
                                      probs["insert_to_insert"]], dim=-1)
            insert_dirichlet.append(torch.sum(self.insert_dirichlet.log_prob(p_insert)))
            # 删除状态转换
            p_delete = torch.stack([probs["delete_to_match"][:-1],
                                      probs["delete_to_delete"]], dim=-1)
            delete_dirichlet.append(torch.sum(self.delete_dirichlet.log_prob(p_delete)))
            # 其他转换
            flank = (self.alpha_flank - 1) * log_probs["unannotated_segment_loop"]  # todo: handle as extra case?
            flank += (self.alpha_flank - 1) * log_probs["right_flank_loop"]
            flank += (self.alpha_flank - 1) * log_probs["left_flank_loop"]
            flank += (self.alpha_flank - 1) * log_probs["end_to_right_flank"]
            flank += (self.alpha_flank - 1) * torch.log(flank_init_prob[i])
            flank += (self.alpha_flank_compl - 1) * log_probs["unannotated_segment_exit"]  # todo: handle as extra case?
            flank += (self.alpha_flank_compl - 1) * log_probs["right_flank_exit"]
            flank += (self.alpha_flank_compl - 1) * log_probs["left_flank_exit"]
            flank += (self.alpha_flank_compl - 1) * torch.log(probs["end_to_unannotated_segment"] + probs["end_to_terminal"])
            flank += (self.alpha_flank_compl - 1) * torch.log(1 - flank_init_prob[i])
            flank_prior.append(torch.squeeze(flank))
            # 单次命中
            hit = (self.alpha_single - 1) * torch.log(probs["end_to_right_flank"] + probs["end_to_terminal"])
            hit += (self.alpha_single_compl - 1) * torch.log(probs["end_to_unannotated_segment"])
            hit_prior.append(torch.squeeze(hit))
            # 均匀的进入/退出先验
            # 将 begin_to_match 重新缩放为总和为 1
            div = torch.maximum(torch.tensor(self.epsilon), 1 - probs["match_to_delete"][0])
            btm = probs["begin_to_match"] / div
            enex = torch.unsqueeze(btm, 1) * torch.unsqueeze(probs["match_to_end"], 0)
            enex = torch.tril(enex)
            log_enex = torch.log(torch.maximum(torch.tensor(self.epsilon), 1 - enex))
            log_enex_compl = torch.log(torch.maximum(torch.tensor(self.epsilon), enex))
            glob = (self.alpha_global - 1) * (torch.sum(log_enex) - log_enex[0, -1])
            glob += (self.alpha_global_compl - 1) * (torch.sum(log_enex_compl) - log_enex_compl[0, -1])
            global_prior.append(glob)
        prior_val = {
            "match_prior": match_dirichlet,
            "insert_prior": insert_dirichlet,
            "delete_prior": delete_dirichlet,
            "flank_prior": flank_prior,
            "hit_prior": hit_prior,
            "global_prior": global_prior
        }
        prior_val = {k: torch.stack(v) for k, v in prior_val.items()}
        return prior_val

    def get_config(self):
        config = {
            "match_comp": self.match_comp,
            "insert_comp": self.insert_comp,
            "delete_comp": self.delete_comp,
            "alpha_flank": self.alpha_flank,
            "alpha_single": self.alpha_single,
            "alpha_global": self.alpha_global,
            "alpha_flank_compl": self.alpha_flank_compl,
            "alpha_single_compl": self.alpha_single_compl,
            "alpha_global_compl": self.alpha_global_compl,
            "epsilon": self.epsilon
        }
        return config

    def __repr__(self):
        return f"ProfileHMMTransitionPrior(match_comp={self.match_comp}, insert_comp={self.insert_comp}, delete_comp={self.delete_comp}, alpha_flank={self.alpha_flank}, alpha_single={self.alpha_single}, alpha_global={self.alpha_global}, alpha_flank_compl={self.alpha_flank_compl}, alpha_single_compl={self.alpha_single_compl}, alpha_global_compl={self.alpha_global_compl})"