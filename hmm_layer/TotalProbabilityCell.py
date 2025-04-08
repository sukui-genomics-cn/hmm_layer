import torch
import torch.nn as nn


class TotalProbabilityCell(nn.Module):
    """
    A utility RNN Cell that computes the total forward probabilities based on the conditional forward
    probabilities of chunked observations given the state at their left border in lograrithmic scale to avoid underflow.
    Args:
        cell: HMM cell whose forward recursion is used.
        reverse: If True, the cell is configured for computing the backward recursion.
    """

    def __init__(self, cell, reverse=False):
        super(TotalProbabilityCell, self).__init__()
        self.cell = cell
        self.reverse = reverse
        self.device = cell.device

    @property
    def sate_size(self):
        return (torch.Size([self.cell.max_num_states]), torch.Size([]))

    def make_initial_distribution(self):
        """
        Returns:
            A probability distribution over the states per model. Shape: (1, k, q)
        """
        return self.cell.transitioner.make_initial_distribution()

    def forward(self, conditional_forward, states=None, training=None, init=False):
        """

        Args:
            conditional_forward: A batch of conditional logarithmic forward probabilities. Shape: (b, q*q)
            states: Tuple of (old_forward, loglik) The rows correspond to the states that are conditioned on.
        Returns:
            The log of the total forward probabilities. Shape: (b, q)
        """
        old_forward, _loglik = states
        batch_size = conditional_forward.size(0)
        conditional_forward = conditional_forward.view(batch_size, self.cell.max_num_states, self.cell.max_num_states)

        # expand dimension for broadcasting
        forward = old_forward.unsqueeze(-1) + conditional_forward
        forward = torch.logsumexp(forward, dim=-2)
        loglik = torch.logsumexp(forward, dim=-1)
        new_state = (forward, loglik)
        output = forward
        return output, new_state

    def get_initial_state(self, batch_size=None, inputs=None, dtype=None):
        if self.reverse:
            init_dist = torch.zeros(batch_size, self.cell.max_num_states, dtype=dtype, device=self.device)
            loglik = torch.zeros(batch_size, dtype=dtype, device=self.device)
            state = (init_dist, loglik)
        else:
            init_dist = self.make_initial_distribution()
            init_dist = init_dist.repeat(batch_size // self.cell.num_models, 1, 1)
            init_dist = init_dist.transpose(0, 1)
            init_dist = init_dist.reshape(-1, self.cell.max_num_states).to(self.device)
            loglike = torch.zeros((batch_size), dtype=dtype, device=self.device)
            state = (torch.log(init_dist), loglike)
        return state

if __name__ == '__main__':
    total_cell = TotalProbabilityCell(cell=None)
