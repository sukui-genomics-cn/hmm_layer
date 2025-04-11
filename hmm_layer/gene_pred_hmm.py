import torch

from .BaseRNN import BaseRNN
from .MsaHmmCell import HmmCell
from .Bidirectional import Bidirectional
from .Initializers import make_15_class_emission_kernel
from .MsaHMMLayer import MsaHmmLayer, _state_posterior_log_probs_impl
from .TotalProbabilityCell import TotalProbabilityCell
from .gene_pred_hmm_emitter import GenePredHMMEmitter
from .gene_pred_hmm_transitioner import GenePredMultiHMMTransitioner


class GenePredHMMLayer(MsaHmmLayer):
    """A PyTorch implementation of the gene prediction HMM layer."""

    def __init__(self,
                 num_models=1,
                 num_copies=1,
                 start_codons=[("ATG", 1.)],
                 stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
                 intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.005), ("NAT", 0.005)],
                 intron_end_pattern=[("AGN", 0.99), ("ACN", 0.01)],
                 initial_exon_len=200,
                 initial_intron_len=4500,
                 initial_ir_len=10000,
                 emitter_init=make_15_class_emission_kernel(smoothing=1e-2, num_copies=1),
                 starting_distribution_init="zeros",
                 trainable_emissions=False,
                 trainable_transitions=False,
                 trainable_starting_distribution=False,
                 trainable_nucleotides_at_exons=False,
                 emit_embeddings=False,
                 embedding_dim=None,
                 full_covariance=False,
                 embedding_kernel_init="random_normal",
                 initial_variance=0.1,
                 temperature=96.,
                 share_intron_parameters=False,
                 simple=False,
                 variance_l2_lambda=0.01,
                 disable_metrics=True,
                 parallel_factor=1,
                 use_border_hints=False,
                 device=None,
                 **kwargs):

        self.num_models = num_models
        self.num_copies = num_copies
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.intron_begin_pattern = intron_begin_pattern
        self.intron_end_pattern = intron_end_pattern
        self.emitter_init = emitter_init
        self.initial_exon_len = initial_exon_len
        self.initial_intron_len = initial_intron_len
        self.initial_ir_len = initial_ir_len
        self.starting_distribution_init = starting_distribution_init
        self.trainable_emissions = trainable_emissions
        self.trainable_transitions = trainable_transitions
        self.trainable_starting_distribution = trainable_starting_distribution
        self.trainable_nucleotides_at_exons = trainable_nucleotides_at_exons
        self.emit_embeddings = emit_embeddings
        self.embedding_dim = embedding_dim
        self.full_covariance = full_covariance
        self.embedding_kernel_init = embedding_kernel_init
        self.initial_variance = initial_variance
        self.temperature = temperature
        self.share_intron_parameters = share_intron_parameters
        self.simple = simple
        self.variance_l2_lambda = variance_l2_lambda
        self.disable_metrics = disable_metrics
        self.use_border_hints = use_border_hints

        # Placeholder for cell, will be initialized in build()
        self.device = device
        self.dim = 15
        super(GenePredHMMLayer, self).__init__(parallel_factor=parallel_factor)

    def build(self):
        if hasattr(self, 'built') and self.built:
            return
        self.cell, self.reverse_cell = self.create_cell()
        # make a variant of the forward cell configured for backward
        # self.reverse_cell = self.cell.make_reverse_direction_offspring()
        # make forward rnn layer
        self.rnn = BaseRNN(self.cell, batch_first=True, return_sequences=True, return_state=True)

        # make backward rnn layer
        self.rnn_backward = BaseRNN(self.reverse_cell, batch_first=True, return_sequences=True, return_state=True,
                                    reverse=self.reverse_cell.reverse)

        # make bidirectional rnn layer
        self.bidirectional_rnn = Bidirectional(self.rnn,
                                               merge_mode="concat" if self.parallel_factor > 1 else "sum",
                                               backward_layer=self.rnn_backward)

        # Override the copies made by Bidirectional
        self.bidirectional_rnn.forward_layer = self.rnn
        self.bidirectional_rnn.backward_layer = self.rnn_backward

        if self.parallel_factor > 1:
            self.total_prob_cell = TotalProbabilityCell(self.cell)
            self.total_prob_cell_rev = TotalProbabilityCell(self.reverse_cell, reverse=True)
            self.total_prob_rnn = BaseRNN(self.total_prob_cell, batch_first=True, return_sequences=True,
                                          return_state=True)
            self.total_prob_rnn_rev = BaseRNN(self.total_prob_cell_rev, batch_first=True, return_sequences=True,
                                              return_state=True, reverse=True)
        else:
            self.total_prob_rnn = None
            self.total_prob_rnn_rev = None

        self.built = True

    def create_cell(self):
        emitter = GenePredHMMEmitter(
            start_codons=self.start_codons,
            stop_codons=self.stop_codons,
            intron_begin_pattern=self.intron_begin_pattern,
            intron_end_pattern=self.intron_end_pattern,
            l2_lambda=self.variance_l2_lambda,
            num_models=self.num_models,
            num_copies=self.num_copies,
            init=self.emitter_init,
            trainable_emissions=self.trainable_emissions,
            emit_embeddings=self.emit_embeddings,
            embedding_dim=self.embedding_dim,
            full_covariance=self.full_covariance,
            embedding_kernel_init=self.embedding_kernel_init,
            initial_variance=self.initial_variance,
            temperature=self.temperature,
            share_intron_parameters=self.share_intron_parameters,
            trainable_nucleotides_at_exons=self.trainable_nucleotides_at_exons,
            device=self.device
        )

        reverse_emitter = GenePredHMMEmitter(
            start_codons=self.start_codons,
            stop_codons=self.stop_codons,
            intron_begin_pattern=self.intron_begin_pattern,
            intron_end_pattern=self.intron_end_pattern,
            l2_lambda=self.variance_l2_lambda,
            num_models=self.num_models,
            num_copies=self.num_copies,
            init=self.emitter_init,
            trainable_emissions=self.trainable_emissions,
            emit_embeddings=self.emit_embeddings,
            embedding_dim=self.embedding_dim,
            full_covariance=self.full_covariance,
            embedding_kernel_init=self.embedding_kernel_init,
            initial_variance=self.initial_variance,
            temperature=self.temperature,
            share_intron_parameters=self.share_intron_parameters,
            trainable_nucleotides_at_exons=self.trainable_nucleotides_at_exons,
            device=self.device
        )

        transitioner = GenePredMultiHMMTransitioner(
            k=self.num_copies,
            num_models=self.num_models,
            initial_exon_len=self.initial_exon_len,
            initial_intron_len=self.initial_intron_len,
            initial_ir_len=self.initial_ir_len,
            starting_distribution_init=self.starting_distribution_init,
            starting_distribution_trainable=self.trainable_starting_distribution,
            transitions_trainable=self.trainable_transitions,
            device=self.device,
        )

        reverse_transitioner = GenePredMultiHMMTransitioner(
            k=self.num_copies,
            num_models=self.num_models,
            initial_exon_len=self.initial_exon_len,
            initial_intron_len=self.initial_intron_len,
            initial_ir_len=self.initial_ir_len,
            starting_distribution_init=self.starting_distribution_init,
            starting_distribution_trainable=self.trainable_starting_distribution,
            transitions_trainable=self.trainable_transitions,
            device=self.device,
        )
        reverse_transitioner.reverse = True

        # Initialize the cell
        cell = HmmCell(
            num_states=[emitter.num_states] * self.num_models,
            dim=self.dim,
            emitter=emitter,
            transitioner=transitioner,
            use_fake_step_counter=True,
            device=self.device
        )

        reverse_cell = HmmCell(
            num_states=[reverse_emitter.num_states] * self.num_models,
            dim=self.dim,
            emitter=reverse_emitter,
            transitioner=reverse_transitioner,
            use_fake_step_counter=True,
            device=self.device
        )
        reverse_cell.reverse = True
        return cell, reverse_cell

    def forward(self, inputs, nucleotides=None, embeddings=None, end_hints=None, training=False, use_loglik=True):
        """
        Computes the state posterior log-probabilities.
        Args:
                inputs: Shape (batch, len, alphabet_size)
                nucleotides: Shape (batch, len, 5) one-hot encoded nucleotides with N in the last position.
                embeddings: Shape (batch, len, dim) embeddings of the inputs as output by a language model.
                end_hints: A tensor of shape (batch, 2, num_states) that contains the correct state for the left and right ends of each chunk.
        Returns:
                State posterior log-probabilities (without loglik if use_loglik is False). The order of the states is Ir, I0, I1, I2, E0, E1, E2.
                Shape (batch, len, number_of_states) if num_models=1 and (batch, len, num_models, number_of_states) if num_models>1.
        """
        if end_hints is not None:
            end_hints = end_hints.unsqueeze(0)
        assert inputs.shape[-1] == 15, "inputs should be of shape (batch, len, 15) with 15 being the number of nucleotides"
        assert nucleotides.shape[-1] == 5, "nucleotides should be of shape (batch, len, 5) with 5 being the number of nucleotides"

        if self.simple:
            inputs_expanded = inputs.unsqueeze(0)
            log_post, prior, _ = self.state_posterior_log_probs(
                inputs_expanded,
                return_prior=True,
                end_hints=end_hints,
                training=training,
                no_loglik=not use_loglik
            )
        else:
            stacked_inputs = self.concat_inputs(inputs, nucleotides, embeddings)
            # TODO: Remove this hardcoded path for testing
            log_post, prior, _ = _state_posterior_log_probs_impl(
                stacked_inputs,
                self.cell,
                self.reverse_cell,
                self.bidirectional_rnn,
                self.total_prob_rnn,
                self.total_prob_rnn_rev,
                end_hints=end_hints,
                return_prior=True,
                training=training,
                no_loglik=False,
                parallel_factor=self.parallel_factor,
            )

        if training:
            prior = prior.mean()
            self.loss = prior  # Store loss for backward pass
            # For metrics tracking, you might want to use something like:
            if not hasattr(self, 'metrics'):
                self.metrics = {}
            self.metrics['prior'] = prior.item()

        if self.num_models == 1:
            return log_post[0]
        else:
            return log_post.permute(1, 2, 0, 3)  # Equivalent to tf.transpose

    def concat_inputs(self, inputs, nucleotides, embeddings=None):
        assert nucleotides is not None
        inputs = inputs.unsqueeze(0)
        nucleotides = nucleotides.unsqueeze(0)
        input_list = [inputs, nucleotides]
        if self.emit_embeddings:
            assert embeddings is not None
            embeddings = embeddings.unsqueeze(0)
            input_list.insert(1, embeddings)
        stacked_inputs = torch.cat(input_list, dim=-1)
        return stacked_inputs


def create_hmm_layer():
    # Define the forward layer
    dim = 15
    num_states = [dim]
    # stacked_inputs = torch.concat([embedding_inputs, nucleotide_inputs], dim=-1)
    emitter_init = make_15_class_emission_kernel(smoothing=1e-2, num_copies=1)
    emitter = GenePredHMMEmitter(
        start_codons=[("ATG", 1.)],
        stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
        intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.005), ("NAT", 0.005)],
        intron_end_pattern=[("AGN", 0.99), ("ACN", 0.01)],
        initial_variance=0.05,
        temperature=100.,
        init=emitter_init,
    )
    transitioner = GenePredMultiHMMTransitioner(
        initial_exon_len=200,
        initial_intron_len=4500,
        initial_ir_len=10000,
        starting_distribution_init="zeros",
        starting_distribution_trainable=False,
        transitions_trainable=False,
    )
    emitter.build()
    cell = HmmCell(
        num_states=num_states,
        dim=dim,
        emitter=emitter,
        transitioner=transitioner,
        use_fake_step_counter=True,
    )

    hmm_layer = MsaHmmLayer(
        cell=cell,
        parallel_factor=99
    )
    return hmm_layer
