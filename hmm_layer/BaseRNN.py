import torch
import torch.nn as nn


class BaseRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaseRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # define train parameters
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(hidden_size))
        self.max_num_states = hidden_size

        self.recurrent_init()

    def recurrent_init(self):
        # initialize weights
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, input, hidden):
        """
        single time step forward pass
        Args:
            input: (batch_size, input_size)
            hidden: (batch_size, hidden_size)

        Returns:
            next_hidden: (batch_size, hidden_size)
        """
        # calculate hidden state
        igates = torch.matmul(input, self.weight_ih.t()) + self.bias_ih
        hgates = torch.matmul(hidden, self.weight_hh.t()) + self.bias_hh
        next_hidden = torch.tanh(igates + hgates)

        return next_hidden


class BaseHMMCell(nn.Module):
    def __init__(self, n_states, hidden_size):
        super(BaseHMMCell, self).__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.max_num_states = hidden_size

        self.transition = nn.Parameter(torch.randn(n_states, n_states))
        self.emission = nn.Parameter(torch.randn(n_states, hidden_size))
        self.init = nn.Parameter(torch.randn(n_states))

        self.recurrent_init()

    def recurrent_init(self):
        nn.init.xavier_uniform_(self.transition)
        nn.init.xavier_uniform_(self.emission)
        nn.init.zeros_(self.init)

    def get_initial_state(self, batch_size, **kwargs):
        return torch.zeros(batch_size, self.n_states)

    def emission_probs(self, inputs, **kwargs):
        """
        Args:
            inputs: (batch_size, n_observation)

        Returns:
            B: (n_states, n_observation)
        """
        B = torch.softmax(self.emission, dim=1)
        emit = torch.matmul(inputs, B)

        return emit

    def forward(self, inputs, states, **kwargs):
        """
        Args:
            inputs: (batch_size, n_observation)
            states: (batch_size, n_states)

        Returns:
            next_states: (batch_size, n_states)
        """
        # calculate A and B
        A = torch.softmax(self.transition, dim=1)
        B = torch.softmax(self.emission, dim=1)
        # calculate next states
        next_states = torch.matmul(states, A) + torch.matmul(inputs, B.t()) + self.init

        return next_states, next_states


class BaseRNN(nn.Module):
    def __init__(self, cell, batch_first=False):
        super(BaseRNN, self).__init__()
        self.cell = cell
        self.batch_first = batch_first
        self.return_state = True

    def forward(self, inputs, hidden=None, **kwargs):
        """
        how to detail with the whole sequence
        Args:
            inputs: if batch_first, (batch_size, seq_len, input_size)
                    else (seq_len, batch_size, input_size)
            hidden: (batch, hidden_size)

        Returns:
            output: if batch_first, (batch_size, seq_len, hidden_size)
                    else (seq_len, batch_size, hidden_size)
            hidden: (batch_size, hidden_size)
        """
        if self.batch_first:
            # transpose to (seq_len, batch_size, input_size)
            inputs = inputs.transpose(0, 1)

        seq_len, batch_size, _ = inputs.size()
        # hidden_size = self.cell.max_num_states

        # init hidden state
        if hidden is None:
            hidden = self.cell.get_initial_state(inputs=inputs, batch_size=batch_size)

        # save all time step hidden states
        outputs = []
        current_hidden = hidden
        for t in range(seq_len):
            current_input = inputs[t]
            output, hidden = self.cell(
                current_input,
                states=hidden
            )
            outputs.append(output)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, current_hidden


def test_base_rnn():
    input_size = 10
    hidden_size = 20
    batch_size = 3
    seq_len = 5

    cell = nn.RNNCell(input_size, hidden_size)
    rnn = BaseRNN(cell, batch_first=True)

    input_seq = torch.randn(batch_size, seq_len, input_size)
    outputs, final_hidden = rnn(input_seq)

    print(f"outputs shape: {outputs.shape}")  # (3,5, 20)
    print(f"final_hidden shape: {final_hidden.shape}")  # (3,20)


def parallel_rnn_forward():
    from hmm_layer.MsaHMMLayer import _state_posterior_log_probs_impl

    num_model = 1
    input_size = 10
    hidden_size = 20
    batch_size = 3
    seq_len = 5
    input_seq = torch.randn(num_model, batch_size, seq_len, input_size)

    cell = BaseHMMCell(input_size, hidden_size)
    rnn = BaseRNN(cell, batch_first=True)

    _state_posterior_log_probs_impl(
        inputs=input_seq,
        cell=cell,
        reverse_cell=cell,
        bidirectional_rnn=rnn,
        total_prob_rnn=None,
        total_prob_rnn_rev=None,
        parallel_factor=1,
    )


if __name__ == '__main__':
    # test_base_rnn()
    parallel_rnn_forward()
