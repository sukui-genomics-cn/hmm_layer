import torch

# from hmm_layer.Initializers import make_15_class_emission_kernel
from hmm_layer.MsaHmmCell import HmmCell
from hmm_layer.gene_pred_hmm_emitter import SimpleGenePredHMMEmitter, GenePredHMMEmitter
from hmm_layer.gene_pred_hmm_transitioner import SimpleGenePredHMMTransitioner, GenePredMultiHMMTransitioner
from hmm_layer.BaseRNN import BaseRNN
from hmm_layer.Bidirectional import Bidirectional
from hmm_layer.TotalProbabilityCell import TotalProbabilityCell
from hmm_layer.MsaHMMLayer import _state_posterior_log_probs_impl

def parallel_rnn_forward():

    num_model = 1
    input_size = 7
    batch_size = 3
    seq_len = 5
    # input_seq = torch.randn(num_model, batch_size, seq_len, input_size)
    dim = 15
    num_states = [dim]
    embedding_inputs = torch.randn(1, 32, 9999, dim)
    nucleotide_inputs = torch.randn(1, 32, 9999, 5)
    stacked_inputs = torch.concat([embedding_inputs, nucleotide_inputs], dim=-1)

    # emitter_init = make_15_class_emission_kernel(smoothing=1e-2, num_copies=1)
    emitter = GenePredHMMEmitter(
        start_codons=[("ATG", 1.)],
        stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
        intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.005), ("NAT", 0.005)],
        intron_end_pattern=[("AGN", 0.99), ("ACN", 0.01)],
        initial_variance=0.05,
        temperature=100.,
        # init=emitter_init,
    )
    transitioner = GenePredMultiHMMTransitioner(
        initial_exon_len=200,
        initial_intron_len=4500,
        initial_ir_len=10000,
        starting_distribution_init="zeros"
    )
    if dim == 7:
        emitter = SimpleGenePredHMMEmitter()
        transitioner = SimpleGenePredHMMTransitioner()

        stacked_inputs = embedding_inputs

    emitter.build(embedding_inputs.shape)


    cell = HmmCell(
        num_states=num_states,
        dim=dim,
        emitter=emitter,
        transitioner=transitioner,
    )

    reverse_cell = cell.make_reverse_direction_offspring()
    rnn = BaseRNN(cell, batch_first=True, return_sequences=True, return_state=True)
    reverse_rnn = BaseRNN(reverse_cell, batch_first=True, return_sequences=True, return_state=True, reverse=True)

    bidirectional_rnn = Bidirectional(rnn, merge_mode='concat', backward_layer=reverse_rnn)
    bidirectional_rnn.forward_layer = rnn
    bidirectional_rnn.backward_layer = reverse_rnn

    total_prob_cell = TotalProbabilityCell(cell=cell, )
    total_prob_rnn = BaseRNN(total_prob_cell, batch_first=True, return_sequences=True, return_state=True)
    total_prob_cell_rev = TotalProbabilityCell(cell=reverse_cell)
    total_prob_rnn_rev = BaseRNN(total_prob_cell_rev, batch_first=True, return_sequences=True, return_state=True, reverse=True)

    log_post, prior, _ = _state_posterior_log_probs_impl(
        inputs=stacked_inputs,
        cell=cell,
        reverse_cell=reverse_cell,
        bidirectional_rnn=bidirectional_rnn,
        total_prob_rnn=total_prob_rnn,
        total_prob_rnn_rev=total_prob_rnn_rev,
        parallel_factor=99,
        training=True,
        return_prior=True
    )

    print(f"outputs shape: {log_post.shape}")


if __name__ == '__main__':
    parallel_rnn_forward()
