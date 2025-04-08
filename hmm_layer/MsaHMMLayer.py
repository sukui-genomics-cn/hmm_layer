import torch
import torch.nn as nn

from .BaseRNN import BaseRNN
from .Bidirectional import Bidirectional
from .TotalProbabilityCell import TotalProbabilityCell


class MsaHmmLayer(nn.Module):
    """A layer that computes the log-likelihood and posterior state probabilities for batches of observations
    under a number of HMMs.

    Args:
        cell: HMM cell whose forward recursion is used.
        num_seqs: The number of sequences in the dataset. If not provided, the prior is not normalized.
        use_prior: If true, the prior is added to the log-likelihood.
        sequence_weights: A tensor of shape (num_seqs,) that contains the weight of each sequence.
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in 
                        parallel at the cost of memory usage. The parallel factor has to be a divisor 
                        of the sequence length.
    """

    def __init__(self,
                 cell=None,
                 reverse_cell=None,
                 num_seqs=None,
                 use_prior=True,
                 sequence_weights=None,
                 parallel_factor=1):
        super(MsaHmmLayer, self).__init__()
        self.cell = cell
        self.num_seqs = num_seqs
        self.use_prior = use_prior
        self.parallel_factor = parallel_factor

        if sequence_weights is not None:
            self.register_buffer('sequence_weights', torch.tensor(sequence_weights, dtype=torch.float32))
            self.register_buffer('weight_sum', torch.sum(self.sequence_weights))
        else:
            self.sequence_weights = None
            self.weight_sum = None

        # These will be initialized in build()
        self.reverse_cell = reverse_cell
        self.rnn = None
        self.rnn_backward = None
        self.bidirectional_rnn = None
        self.total_prob_rnn = None
        self.total_prob_rnn_rev = None

        self.build()  # Placeholder, will be set in build()

    def build(self):
        if hasattr(self, 'built') and self.built:
            return

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

    def forward_recursion(self, inputs, end_hints=None, return_prior=False, training=False):
        """Computes the forward recursion for multiple models where each model
        receives a batch of sequences as input.

        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
            return_prior: If true, the prior is computed and returned.
            training: If true, the cell is run in training mode.

        Returns:
            forward variables: Shape: (num_model, b, seq_len, q)
            log-likelihoods: Shape: (num_model, b)
        """
        return _forward_recursion_impl(inputs, self.cell, self.rnn, self.total_prob_rnn,
                                       end_hints=end_hints, return_prior=return_prior,
                                       training=training, parallel_factor=self.parallel_factor)

    def backward_recursion(self, inputs, end_hints=None, return_prior=False, training=False):
        """Computes the backward recursion for multiple models where each model
        receives a batch of sequences as input.

        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
            return_prior: If true, the prior is computed and returned.
            training: If true, the cell is run in training mode.

        Returns:
            backward variables: Shape: (num_model, b, seq_len, q)
        """
        return _backward_recursion_impl(inputs, self.cell, self.reverse_cell,
                                        self.rnn_backward, self.total_prob_rnn_rev,
                                        end_hints=end_hints, return_prior=return_prior,
                                        training=training, parallel_factor=self.parallel_factor)

    def state_posterior_log_probs(self, inputs, end_hints=None, return_prior=False, training=False, no_loglik=False):
        """Computes the log-probability of state q at position i given inputs.

        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
            return_prior: If true, the prior is computed and returned.
            training: If true, the cell is run in training mode.
            no_loglik: If true, the loglik is not used in the return value. This can be beneficial for end-to-end training when the
                      normalizing constant of the posteriors is not important and the activation function is the softmax.

        Returns:
            state posterior probabilities: Shape: (num_model, b, seq_len, q)
        """
        return _state_posterior_log_probs_impl(inputs, self.cell, self.reverse_cell,
                                               self.bidirectional_rnn, self.total_prob_rnn,
                                               self.total_prob_rnn_rev,
                                               end_hints=end_hints, return_prior=return_prior,
                                               training=training, no_loglik=no_loglik,
                                               parallel_factor=self.parallel_factor)

    def apply_sequence_weights(self, loglik, indices, aggregate=False):
        if self.sequence_weights is not None:
            weights = self.sequence_weights[indices]
            loglik = loglik * weights
            if aggregate:
                loglik = torch.sum(loglik, dim=1) / torch.sum(weights, dim=1)  # mean over batch
                loglik = torch.mean(loglik)  # mean over models
        elif aggregate:
            loglik = torch.mean(loglik)  # mean over both models and batches
        return loglik

    def compute_prior(self, scaled=True):
        self.cell.recurrent_init()  # initialize all relevant tensors
        prior = self.cell.get_prior_log_density()
        if scaled:
            prior = self._scale_prior(prior)
        return prior

    def _scale_prior(self, prior):
        if self.sequence_weights is not None:
            prior = prior / self.weight_sum
        elif self.num_seqs is not None:
            prior = prior / self.num_seqs
        return prior

    def forward(self, inputs, indices=None, training=False):
        """Computes log-likelihoods per model and sequence.

        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            indices: Optional sequence indices required to assign sequence weights. Shape: (num_model, b)
            training: If true, the cell is run in training mode.

        Returns:
            log-likelihoods: Shape: (num_model, b)
            aggregated loglik: Shape: ()
            prior: Shape: (num_model)
            aux_loss: Shape: ()
        """
        inputs = inputs.to(self.dtype)

        if self.use_prior:
            _, loglik, prior, aux_loss = self.forward_recursion(inputs, return_prior=True, training=training)
            prior = self._scale_prior(prior)
        else:
            _, loglik = self.forward_recursion(inputs, return_prior=False, training=training)

        loglik_mean = self.apply_sequence_weights(loglik, indices, aggregate=True)
        loglik_mean = torch.squeeze(loglik_mean)

        if self.use_prior:
            return loglik, loglik_mean, prior, aux_loss
        else:
            return loglik, loglik_mean

    def get_config(self):
        return {
            "cell": self.cell,
            "num_seqs": self.num_seqs,
            "use_prior": self.use_prior,
            "sequence_weights": self.sequence_weights.numpy() if self.sequence_weights is not None else None,
            "parallel_factor": self.parallel_factor
        }

    @classmethod
    def from_config(cls, config):
        if config["sequence_weights"] is not None:
            config["sequence_weights"] = torch.tensor(config["sequence_weights"], dtype=torch.float32)
        return cls(**config)


def _forward_recursion_impl(inputs, cell, rnn, total_prob_rnn,
                            end_hints=None, return_prior=False,
                            training=False, parallel_factor=1):
    """Computes the forward recursion for multiple models where each model
    receives a batch of sequences as input.

    Args:
        inputs: Sequences. Shape: (num_model, b, seq_len, s)
        cell: HMM cell used for forward recursion.
        rnn: A RNN layer that runs the forward recursion.
        total_prob_rnn: A RNN layer that computes the total probability of the forward variables.
        end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
        return_prior: If true, the prior is computed and returned.
        training: If true, the cell is run in training mode.
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.

    Returns:
        forward variables: Shape: (num_model, b, seq_len, q)
        log-likelihoods: Shape: (num_model, b)
    """
    cell.recurrent_init()
    num_model, b, seq_len, s = inputs.shape
    q = cell.max_num_states

    emission_probs = cell.emission_probs(inputs, end_hints=end_hints, training=training)

    # Reshape to equally sized chunks according to parallel factor
    chunk_size = seq_len // parallel_factor
    emission_probs = emission_probs.reshape(num_model * b * parallel_factor, chunk_size, q)

    # Do one initialization step
    initial_state = cell.get_initial_state(batch_size=b * parallel_factor, parallel_factor=parallel_factor)
    forward_1, step_1_state = cell(emission_probs[:, 0], initial_state, training=training, init=True)

    # Run forward with the output of the first step as initial state
    forward, _, loglik = rnn(emission_probs[:, 1:], initial_state=step_1_state, training=training)

    # Prepend the separate first step to the other forward steps
    forward = torch.cat([forward_1.unsqueeze(1), forward], dim=1)

    if parallel_factor == 1:
        forward = forward.reshape(num_model, b, seq_len, -1)
        forward_scaled = forward[..., :-1]
        forward_scaling_factors = forward[..., -1:]
        forward_result = forward_scaled + forward_scaling_factors
        loglik = loglik.reshape(num_model, b)
    else:
        forward_result, loglik = _get_total_forward_from_chunks(forward, cell, total_prob_rnn, b, seq_len,
                                                                parallel_factor=parallel_factor)

    if return_prior:
        prior = cell.get_prior_log_density()
        aux_loss = cell.get_aux_loss()
        return forward_result, loglik, prior, aux_loss
    else:
        return forward_result, loglik


def _get_total_forward_from_chunks(forward, cell, total_prob_rnn, b, seq_len, parallel_factor=1):
    """Utility method that computes the actual forward probabilities from the chunked forward variables.
    Returns the forward probabilities and the log-likelihood.
    """
    q = cell.max_num_states
    num_model = cell.num_models
    chunk_size = seq_len // parallel_factor

    forward_scaled = forward[..., :-q]
    forward_scaling_factors = forward[..., -q:]

    forward_scaled = forward_scaled.reshape(num_model * b, parallel_factor, chunk_size, q, -1)
    forward_scaling_factors = forward_scaling_factors.reshape(num_model * b, parallel_factor, chunk_size, q, 1)

    forward_chunks = forward_scaled + forward_scaling_factors  # shape: (num_model*b, factor, chunk_size, q (conditional states), q (actual states))

    # Compute the actual forward variables across the chunks via the total probability
    forward_chunks_last = forward_chunks[:, :, -1]  # (num_model*b, factor, q, q)
    forward_chunks_last = forward_chunks_last.reshape(num_model * b, parallel_factor, q * q)

    forward_total, (_, loglik) = total_prob_rnn(forward_chunks_last)  # (num_model*b, factor, q)

    init, _ = cell.get_initial_state(batch_size=b, parallel_factor=1)
    init = torch.log(init + cell.epsilon)

    T = torch.cat([init.unsqueeze(1), forward_total[:, :-1]], dim=1)
    T = T.unsqueeze(2).unsqueeze(4)  # shape: (num_model*b, factor, 1, q, 1)

    forward_result = forward_chunks + T  # shape: (num_model*b, factor, chunk_size, q, q)
    forward_result = forward_result.reshape(num_model, b, seq_len, q, q)
    forward_result = torch.logsumexp(forward_result, dim=-2)

    loglik = loglik.reshape(num_model, b)

    return forward_result, loglik


def _backward_recursion_impl(inputs, cell, reverse_cell,
                             rnn_backward, total_prob_rnn_rev,
                             end_hints=None, return_prior=False,
                             training=False, parallel_factor=1):
    """Computes the backward recursion for multiple models where each model
    receives a batch of sequences as input.

    Args:
        inputs: Sequences. Shape: (num_model, b, seq_len, s)
        cell: HMM cell used for forward recursion.
        reverse_cell: HMM cell used for backward recursion.
        rnn_backward: A RNN layer that runs the backward recursion.
        total_prob_rnn_rev: A RNN layer that computes the total probability of the backward variables.
        end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
        return_prior: If true, the prior is computed and returned.
        training: If true, the cell is run in training mode.
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.

    Returns:
        backward variables: Shape: (num_model, b, seq_len, q)
    """
    cell.recurrent_init()
    reverse_cell.recurrent_init()

    num_model, b, seq_len, s = inputs.shape
    q = cell.max_num_states

    emission_probs = reverse_cell.emission_probs(inputs, end_hints=end_hints, training=training)

    # Reshape to equally sized chunks according to parallel factor
    chunk_size = seq_len // parallel_factor
    emission_probs = emission_probs.reshape(num_model * b * parallel_factor, chunk_size, q)

    # Do one initialization step
    initial_state = reverse_cell.get_initial_state(inputs=emission_probs, batch_size=b * parallel_factor,
                                                   parallel_factor=parallel_factor)
    backward_1, step_1_state = reverse_cell(emission_probs[:, -1], initial_state, training=training, init=True)

    # Run backward with the output of the first step as initial state
    backward, _, _ = rnn_backward(emission_probs[:, :-1], initial_state=step_1_state, training=training)

    # Prepend the separate first step to the other backward steps
    backward = torch.cat([backward_1.unsqueeze(1), backward], dim=1)

    if parallel_factor == 1:
        backward = backward.reshape(num_model, b, seq_len, -1)
        backward_scaled = backward[..., :-1]
        backward_scaling_factors = backward[..., -1:]
        backward_result = backward_scaled + backward_scaling_factors
        backward_result = torch.flip(backward_result, [-2])
    else:
        backward_result = _get_total_backward_from_chunks(backward, cell, reverse_cell, total_prob_rnn_rev,
                                                          b, seq_len, parallel_factor=parallel_factor)

    if return_prior:
        prior = cell.get_prior_log_density()
        aux_loss = cell.get_aux_loss()
        return backward_result, prior, aux_loss
    else:
        return backward_result


def _get_total_backward_from_chunks(backward, cell, reverse_cell, total_prob_rnn_rev,
                                    b, seq_len, revert_chunks=True, parallel_factor=1):
    """Utility method that computes the actual backward probabilities from the chunked backward variables."""
    q = cell.max_num_states
    num_model = cell.num_models
    chunk_size = seq_len // parallel_factor

    backward_scaled = backward[..., :-q]
    backward_scaling_factors = backward[..., -q:]

    backward_scaled = backward_scaled.reshape(num_model * b, parallel_factor, chunk_size, q, -1)
    backward_scaling_factors = backward_scaling_factors.reshape(num_model * b, parallel_factor, chunk_size, q, 1)

    backward_chunks = backward_scaled + backward_scaling_factors  # shape: (num_model*b, factor, chunk_size, q (conditional states), q (actual states))

    if revert_chunks:
        backward_chunks = torch.flip(backward_chunks, [-3])

    # Compute the actual backward variables across the chunks via the total probability
    backward_chunks_last = backward_chunks[:, :, 0]  # (num_model*b, factor, q, q)
    backward_chunks_last = backward_chunks_last.reshape(num_model * b, parallel_factor, q * q)

    backward_total, (_, _) = total_prob_rnn_rev(backward_chunks_last)  # (num_model*b, factor, q)
    backward_total = torch.flip(backward_total, [1])

    init, _ = reverse_cell.get_initial_state(batch_size=b, parallel_factor=1)
    init = torch.log(init + reverse_cell.epsilon)

    T = torch.cat([backward_total[:, 1:], init.unsqueeze(1)], dim=1)
    T = T.unsqueeze(2).unsqueeze(4)  # shape: (num_model*b, factor, 1, q, 1)

    backward_result = backward_chunks + T  # shape: (num_model*b, factor, chunk_size, q, q)
    backward_result = backward_result.reshape(num_model, b, seq_len, q, q)
    backward_result = torch.logsumexp(backward_result, dim=-2)

    return backward_result


def _state_posterior_log_probs_impl(inputs, cell, reverse_cell,
                                    bidirectional_rnn, total_prob_rnn,
                                    total_prob_rnn_rev,
                                    end_hints=None, return_prior=False,
                                    training=False, no_loglik=False, parallel_factor=1):
    """Computes the log-probability of state q at position i given inputs.

    Args:
        inputs: Sequences. Shape: (num_model, b, seq_len, s)
        cell: HMM cell used for forward recursion.
        reverse_cell: HMM cell used for backward recursion.
        bidirectional_rnn: A bidirectional RNN layer that runs forward and backward in parallel.
        total_prob_rnn: A RNN layer that computes the total probability of the forward variables.
        total_prob_rnn_rev: A RNN layer that computes the total probability of the backward variables.
        end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
        return_prior: If true, the prior is computed and returned.
        training: If true, the cell is run in training mode.
        no_loglik: If true, the loglik is not used in the return value. This can be beneficial for end-to-end training when the
                  normalizing constant of the posteriors is not important and the activation function is the softmax.
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.

    Returns:
        state posterior probabilities: Shape: (num_model, b, seq_len, q)
    """
    cell.recurrent_init()
    reverse_cell.recurrent_init()

    num_model, b, seq_len, s = inputs.shape
    q = cell.max_num_states

    emission_probs = cell.emission_probs(inputs, end_hints=end_hints, training=training)

    # Reshape to equally sized chunks according to parallel factor
    chunk_size = seq_len // parallel_factor
    emission_probs = emission_probs.reshape(num_model * b * parallel_factor, chunk_size, q)

    # Make the initial states for both passes
    initial_state = cell.get_initial_state(batch_size=b * parallel_factor, parallel_factor=parallel_factor)
    rev_initial_state = reverse_cell.get_initial_state(inputs=emission_probs, batch_size=b * parallel_factor,
                                                       parallel_factor=parallel_factor)

    # Handle the first observation separately
    forward_1, forward_step_1_state = cell(emission_probs[:, 0], initial_state, training=training, init=True)
    backward_1, backward_step_1_state = reverse_cell(emission_probs[:, -1], rev_initial_state, training=training,
                                                     init=True)

    # Run forward and backward in parallel
    if emission_probs.shape[1] > 2:
        posterior, *states = bidirectional_rnn(emission_probs[:, 1:-1],
                                               initial_state=(*forward_step_1_state, *backward_step_1_state),
                                               training=training)
    else:
        # Posterior as defined here is never used but required to make the code work
        posterior, states = torch.zeros(()), forward_step_1_state + backward_step_1_state
        if cell.use_step_counter:
            cell.step_counter += 1

    # Manually do the last forward and backward step
    forward_last, final_state = cell(emission_probs[:, -1], states[:2], training=training)
    backward_last, _ = reverse_cell(emission_probs[:, 0], states[2:], training=training)

    # forward_last, final_state = cell(emission_probs[:, -1], states, training=training)
    # backward_last, _ = reverse_cell(emission_probs[:, 0], states, training=training)

    posterior_1 = torch.stack([forward_1, backward_last], dim=-2) if parallel_factor > 1 else forward_1 + backward_last
    posterior_last = torch.stack([forward_last, backward_1],
                                 dim=-2) if parallel_factor > 1 else forward_last + backward_1

    if emission_probs.shape[1] > 2:
        if parallel_factor > 1:
            posterior = posterior.reshape(num_model * b * parallel_factor, chunk_size - 2, 2, -1)
        else:
            posterior = posterior.reshape(num_model * b * parallel_factor, chunk_size - 2, 2, -1)
            posterior_1 = posterior_1.unsqueeze(1)
            posterior_last = posterior_last.unsqueeze(1)
        posterior = torch.cat([posterior_1.unsqueeze(1), posterior, posterior_last.unsqueeze(1)], dim=1)
    else:
        posterior = torch.cat([posterior_1.unsqueeze(1), posterior_last.unsqueeze(1)], dim=1)

    if parallel_factor == 1:
        posterior = posterior.reshape(num_model, b, seq_len, -1)
        loglik = final_state[1].reshape(num_model, b)
        posterior = posterior[..., :-1] + posterior[..., -1:]
    else:
        forward_result, loglik = _get_total_forward_from_chunks(posterior[..., 0, :], cell, total_prob_rnn,
                                                                b, seq_len, parallel_factor=parallel_factor)
        backward_result = _get_total_backward_from_chunks(posterior[..., 1, :], cell, reverse_cell,
                                                          total_prob_rnn_rev, b, seq_len,
                                                          revert_chunks=False, parallel_factor=parallel_factor)
        posterior = forward_result + backward_result

    if not no_loglik:
        posterior = posterior - loglik.unsqueeze(-1).unsqueeze(-1)

    if return_prior:
        prior = cell.get_prior_log_density()
        aux_loss = cell.get_aux_loss()
        return posterior, prior, aux_loss
    else:
        return posterior


def parallel_rnn_forward():
    from hmm_layer.MsaHmmCell import HmmCell
    from hmm_layer.gene_pred_hmm_emitter import SimpleGenePredHMMEmitter, GenePredHMMEmitter
    from hmm_layer.gene_pred_hmm_transitioner import SimpleGenePredHMMTransitioner, GenePredMultiHMMTransitioner
    from hmm_layer.BaseRNN import BaseRNN
    from hmm_layer.Bidirectional import Bidirectional
    from hmm_layer.TotalProbabilityCell import TotalProbabilityCell

    num_model = 1
    input_size = 7
    batch_size = 3
    seq_len = 5
    input_seq = torch.randn(num_model, batch_size, seq_len, input_size)
    embedding_inputs = torch.randn(1, 32, 9999, 15)
    nucleotide_inputs = torch.randn(1, 32, 9999, 5)
    stacked_inputs = torch.concat([embedding_inputs, nucleotide_inputs], dim=-1)

    num_states = [15]
    dim = 15
    emitter = GenePredHMMEmitter(
        start_codons=[("ATG", 1.)],
        stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
        intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.005), ("NAT", 0.005)],
        intron_end_pattern=[("AGN", 0.99), ("ACN", 0.01)],
    )
    emitter.build()
    transitioner = GenePredMultiHMMTransitioner()

    cell = HmmCell(
        num_states=num_states,
        dim=dim,
        emitter=emitter,
        transitioner=transitioner,
    )

    reverse_cell = cell.make_reverse_direction_offspring()
    rnn = BaseRNN(cell, batch_first=True)
    reverse_rnn = BaseRNN(reverse_cell, batch_first=True)

    bidirectional_rnn = Bidirectional(rnn, merge_mode='concat', backward_layer=reverse_rnn)
    bidirectional_rnn.forward_layer = rnn
    bidirectional_rnn.backward_layer = reverse_rnn

    total_prob_cell = TotalProbabilityCell(cell=cell)
    total_prob_rnn = BaseRNN(total_prob_cell, batch_first=True)
    total_prob_cell_rev = TotalProbabilityCell(cell=reverse_cell)
    total_prob_rnn_rev = BaseRNN(total_prob_cell_rev, batch_first=True)

    _state_posterior_log_probs_impl(
        inputs=stacked_inputs,
        cell=cell,
        reverse_cell=reverse_cell,
        bidirectional_rnn=bidirectional_rnn,
        total_prob_rnn=total_prob_rnn,
        total_prob_rnn_rev=total_prob_rnn_rev,
        parallel_factor=1,
    )


if __name__ == '__main__':
    parallel_rnn_forward()
