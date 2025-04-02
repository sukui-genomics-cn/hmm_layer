import torch
import torch.nn as nn
import torch.nn.functional as F


class HMMCell(nn.Module):
    def __init__(self, num_state, obs_dim):
        super(HMMCell, self).__init__()
        self.num_state = num_state
        self.obs_dim = obs_dim

        self.trans_logits = nn.Parameter(torch.randn(num_state, num_state))
        self.emit_net = nn.Sequential(
            nn.Linear(obs_dim, num_state * 4),
            nn.ReLU(),
            nn.Linear(num_state * 4, num_state)
        )

    def forward(self, obs, prev_state):
        """
        Args:
            obs: (batch_size, obs_dim)
            prev_state: (batch_size, num_state)

        Returns:
            (batch_size, num_state
        """

        emit_logits = self.emit_net(obs)  # (batch_size, num_states)
        trans_probs = F.log_softmax(self.trans_logits, dim=-1)  # (num_states, num_states)
        # forward algorithm
        # logsumexp(prev_state + trans_probs) + emit_logits
        next_state = torch.logsumexp(
            prev_state.unsqueeze(-1) + trans_probs.unsqueeze(0),
            dim=1
        ) + emit_logits

        return next_state


class BidirectionalHMM(nn.Module):
    def __init__(self, cell, num_states):
        super(BidirectionalHMM, self).__init__()
        self.fwd_cell = cell
        self.bwd_cell = self._create_backward_cell(cell, num_states)  # create bwd cell
        self.num_states = num_states

    def _create_backward_cell(self, fwd_cell, num_states):
        bwd_cell = type(fwd_cell)(num_states, fwd_cell.obs_dim)

        # copy params
        bwd_cell.emit_net.load_state_dict(fwd_cell.emit_net.state_dict())

        # 转移矩阵取转置
        with torch.no_grad():
            bwd_cell.trans_logits.copy_(fwd_cell.trans_logits.t())
        return bwd_cell

    def forward(self, observations):
        """
        Args:
            observations: (seq_len, batch, obs_dim)
        Returns:
            log_alphas: 前向概率 (seq_len, batch, num_states
            log_betas: 后向概率 (seq_len, batch, num_states)
        """
        seq_len, batch_size, _ = observations.shape

        # forward pass
        fwd_states = []
        h = torch.zeros(batch_size, self.num_states, device=observations.device)
        for t in range(seq_len):
            h = self.fwd_cell(observations[t], h)
            fwd_states.append(h)
        log_alphas = torch.stack(fwd_states, dim=0)

        # backward pass
        bwd_states = []
        h = torch.zeros(batch_size, self.num_states, device=observations.device)
        for t in reversed(range(seq_len)):
            h = self.bwd_cell(observations[t], h)
            bwd_states.insert(0, h)
        log_betas = torch.stack(bwd_states, dim=0)

        return log_alphas, log_betas


class HMMModel(nn.Module):
    def __init__(self, num_states, obs_dim):
        super(HMMModel, self).__init__()
        self.num_states = num_states
        self.obs_dim = obs_dim

        # init probability
        self.init_logits = nn.Parameter(torch.randn(num_states))

        # bidirectional Model
        base_cell = HMMCell(num_states, obs_dim)
        self.bi_hmm = BidirectionalHMM(base_cell, num_states)

    def forward(self, observations):
        """
        Args:
            observations: (batch, seq_len, obs_dim)

        Returns:
            posteriors: 状态后验概率 (batch, seq_len, num_states)
        """
        if observations.dim() == 3 and observations.size(0) != self.num_states:
            observations = observations.transpose(0, 1)  # transform to (seq_len, batch, obs_dim)

        # calculate log_alphas and log_betas
        log_alphas, log_betas = self.bi_hmm(observations)

        # 计算配分函数(序列概率)
        log_Z = torch.logsumexp(log_alphas[-1], dim=-1)

        # calculate 后验概率
        log_gammas = log_alphas + log_betas - log_Z.view(1, -1, 1)
        posteriors = F.softmax(log_gammas, dim=-1)

        # transpose to (batch, seq_len, num_states)
        return posteriors.transpose(0, 1)


if __name__ == '__main__':
    # 参数设置
    num_states = 5
    obs_dim = 10
    batch_size = 3
    seq_len = 7

    # 创建模型
    model = HMMModel(num_states, obs_dim)

    # 测试转移矩阵的对称性
    fwd_trans = model.bi_hmm.fwd_cell.trans_logits  # (i→j)
    bwd_trans = model.bi_hmm.bwd_cell.trans_logits  # (j→i)
    print("前向转移矩阵示例:\n", fwd_trans[:2, :2].detach())
    print("反向转移矩阵示例:\n", bwd_trans[:2, :2].detach())

    # 测试输入
    obs = torch.randn(batch_size, seq_len, obs_dim)  # (batch, seq, obs_dim)

    # 前向计算
    posteriors = model(obs)
    print("后验概率形状:", posteriors.shape)  # 应为 (3, 7, 5)


    # 计算损失函数示例
    def hmm_loss(posteriors, true_states):
        """负对数似然损失"""
        return -torch.sum(torch.log(posteriors.gather(-1, true_states.unsqueeze(-1))))


    true_labels = torch.randint(0, num_states, (batch_size, seq_len))
    loss = hmm_loss(posteriors, true_labels)
    print("损失值:", loss.item())
