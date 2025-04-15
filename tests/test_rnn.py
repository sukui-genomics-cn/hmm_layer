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

    def test_gene_hmm_layer(self):
        # test_rnn.TestRNN.test_gene_hmm_layer
        import torch

        from hmm_layer import GenePredHMMLayer

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        stacked_inputs = torch.randn(size=(2, 9999, 20)).to(device)
        layer = GenePredHMMLayer(
            parallel_factor=1
        ).to(device)

        for key, value in layer.named_parameters():
            print(key, value.shape, value.device, value.requires_grad, value.dtype)

        outputs = layer(
            inputs=stacked_inputs[..., :15],
            nucleotides=stacked_inputs[..., -5:],
            training=True,
        )

        print(f'outputs shape: {outputs.shape}')
        for key, value in layer.named_parameters():
            print(key, value.shape, value.device, value.requires_grad, value.dtype)

