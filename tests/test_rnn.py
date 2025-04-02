import unittest


class TestRNN(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_base_rnn(self):
        # test_rnn.TestRNN.test_base_rnn

        import torch
        import torch.nn as nn
        from hmm_layer.BaseRNN import BaseRNN, BaseHMMCell

        # Example with LSTM cell
        lstm_cell = nn.LSTMCell(input_size=64, hidden_size=128)
        rnn_layer = BaseRNN(cell=lstm_cell,
                            batch_first=True,
                            return_sequences=True,
                            return_state=True)

        # Input shape (batch, seq, features)
        inputs = torch.randn(32, 10, 64)
        output, (h_n, c_n) = rnn_layer(inputs)

        # Without returning sequences
        rnn_layer = BaseRNN(cell=lstm_cell, return_sequences=True)
        last_output = rnn_layer(inputs)  # shape (32, 128)
        print(last_output)
