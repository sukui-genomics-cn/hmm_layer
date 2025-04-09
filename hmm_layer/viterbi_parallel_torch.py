import torch


def make_k_mers(sequences, k, pivot_left=True):
    """
    Maps one-hot encoded nucleotide sequences to a k-mer representation.
    Args:
        sequences: A tensor of shape (b, L, 5) representing one-hot encoded sequences of length L.
                   Last dimension: [A, C, G, T, N].
        k: Integer, length of the k-mer.
        pivot_left: Boolean, whether to pivot the k-mer to the left or right.
    Returns:
        A tensor of shape (b, L, 4**(k-1), 4). If pivot_left is True, the last dimension corresponds
        to the 4 possible nucleotides in the leftmost position of the k-mer; otherwise, the rightmost.
        If the k-mer contains 'N', it’s expressed equiprobably among the 4 nucleotides.
    """
    b, L, _ = sequences.shape
    n = sequences.shape[-1] - 1  # Alphabet size excluding 'N' (e.g., 4 for ACGT)
    # n = sequences.dtype.type(n)  # Match dtype of input tensor

    # Handle 'N' by distributing probability uniformly over A, C, G, T
    sequences_no_N = sequences[..., :-1]  # [b, L, 4]
    N_pos = (sequences[..., -1:] == 1).to(sequences.dtype)  # [b, L, 1]
    sequences_no_N = sequences_no_N + (1.0 / n) * N_pos  # Equiprobable for 'N'

    # Padding for k-mers that go beyond sequence boundaries
    pad = torch.ones_like(sequences_no_N[:, :k - 1, :], dtype=sequences.dtype) / n  # [b, k-1, 4]

    if pivot_left:
        # Pad at the end
        sequences_padded_no_N = torch.cat([sequences_no_N, pad], dim=-2)  # [b, L+k-1, 4]
        k_mers = sequences_padded_no_N[:, :L, None, :]  # [b, L, 1, 4]
    else:
        # Pad at the start
        sequences_padded_no_N = torch.cat([pad, sequences_no_N], dim=-2)  # [b, L+k-1, 4]
        k_mers = sequences_padded_no_N[:, k - 1:L + k - 1, None, :]  # [b, L, 1, 4]

    # Build k-mers by shifting and multiplying
    range_iter = range(1, k) if pivot_left else range(k - 2, -1, -1)
    for i in range_iter:
        shift_i = sequences_padded_no_N[:, i:L + i, None, :, None]  # [b, L, 1, 4, 1]
        k_mers = k_mers[..., None, :] * shift_i  # Element-wise multiplication
        shape = [b, L] + ([4 ** i, 4] if pivot_left else [4 ** (k - i - 1), 4])
        k_mers = k_mers.reshape(shape)

    return k_mers


def encode_kmer_string(kmer, pivot_left=True, alphabet="ACGT"):
    """
    Converts a k-mer string to a one-hot encoded class representation (i, j).
    Args:
        kmer: String, e.g., "AAT".
        pivot_left: Boolean, whether to pivot the k-mer to the left or right.
        alphabet: String, default "ACGT".
    Returns:
        A tensor of shape (4**(k-1), 4) representing the one-hot encoded k-mer.
        If 'N' is present, it’s equiprobable among A, C, G, T.
    """
    alphabet_with_unknown = alphabet + "N"
    kmer_indices = [alphabet_with_unknown.index(c) for c in kmer]
    kmer_tensor = torch.tensor(kmer_indices)  # [k]

    # One-hot encode the kmer
    one_hot = torch.nn.functional.one_hot(kmer_tensor, num_classes=len(alphabet_with_unknown)).float()  # [k, 5]

    # Use make_k_mers to encode
    encoded_kmers = make_k_mers(one_hot[None, ...], k=len(kmer), pivot_left=pivot_left)  # [1, k, 4**(k-1), 4]

    # Squeeze and return the appropriate slice
    if pivot_left:
        return encoded_kmers.squeeze(0)[0]  # [4**(k-1), 4]
    else:
        return encoded_kmers.squeeze(0)[-1]  # [4**(k-1), 4]


def viterbi_step(gamma_prev, emission_probs_i, transition_matrix, non_homogeneous_mask=None):
    """ Computes one Viterbi dynamic programming step. z is a helper dimension for parallelization and not used in the final result.
    Args:
        gamma_prev: Viterbi values of the previous recursion. Shape (num_models, b, z, q)
        emission_probs_i: Emission probabilities of the i-th vertical input slice. Shape (num_models, b, q)
        transition_matrix: Logarithmic transition matrices describing the Markov chain. Shape (num_models, q, q)
        non_homogeneous_mask: Optional mask of shape (num_models, b, q, q) that specifies which transitions are allowed.
    Returns:
        Viterbi values of the current recursion (gamma_next). Shape (num_models, b, z, q)
    """
    # Add dimensions to transition_matrix and gamma_prev for broadcasting
    gamma_next = transition_matrix.unsqueeze(1).unsqueeze(1) + gamma_prev.unsqueeze(-1)  # (n, b, z, q, q)

    # Apply non_homogeneous_mask if provided
    if non_homogeneous_mask is not None:
        gamma_next += safe_log(non_homogeneous_mask.unsqueeze(2))  # (n, b, z, q, q)

    # Reduce over the second-to-last dimension (q) to get the maximum values
    gamma_next, _ = torch.max(gamma_next, dim=-2)  # (n, b, z, q)

    # Add emission probabilities
    gamma_next += safe_log(emission_probs_i.unsqueeze(2))  # (n, b, z, q)

    return gamma_next


def viterbi_dyn_prog(emission_probs, init, transition_matrix, use_first_position_emission=True,
                     non_homogeneous_mask_func=None):
    """ Logarithmic (underflow safe) viterbi capable of decoding many sequences in parallel on the GPU.
    z is a helper dimension for parallelization and not used in the final result.
    Args:
        emission_probs: Tensor. Shape (num_models, b, L, q).
        init: Initial state distribution. Shape (num_models, z, q).
        transition_matrix: Logarithmic transition matrices describing the Markov chain. Shape (num_models, q, q)
        use_first_position_emission: If True, the first position of the sequence is considered to have an emission.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_models x q x q mask that specifies which transitions are allowed.
    Returns:
        Viterbi values (gamma) per model. Shape (num_models, b, z, L, q)
    """
    # Initialize gamma_val with safe_log of init
    gamma_val = safe_log(init).unsqueeze(1)  # Shape: (num_models, 1, z, q)
    gamma_val = gamma_val.to(dtype=transition_matrix.dtype)

    # Handle first position emission
    b0 = emission_probs[:, :, 0]  # Shape: (num_models, b, q)
    if use_first_position_emission:
        gamma_val = gamma_val + safe_log(b0).unsqueeze(2)  # Shape: (num_models, b, z, q)
    else:
        gamma_val = gamma_val + torch.zeros_like(b0).unsqueeze(2)  # Shape: (num_models, b, z, q)

    # Get sequence length L
    L = emission_probs.size(2)

    # Initialize a list to store gamma values for each time step
    gamma_list = [gamma_val]

    # Iterate over sequence positions
    for i in range(1, L):
        # Get emission probabilities for the current position
        emission_probs_i = emission_probs[:, :, i]  # Shape: (num_models, b, q)

        # Get non_homogeneous_mask if provided
        non_homogeneous_mask = None
        if non_homogeneous_mask_func is not None:
            non_homogeneous_mask = non_homogeneous_mask_func(i)  # Shape: (num_models, q, q)

        # Compute gamma_val for the current step
        gamma_val = viterbi_step(gamma_val, emission_probs_i, transition_matrix, non_homogeneous_mask)
        gamma_list.append(gamma_val)

    # Stack gamma values along the time dimension (L)
    gamma = torch.stack(gamma_list, dim=3)  # Shape: (num_models, b, z, L, q)

    return gamma


def safe_log(x, log_zero_val=-1e3):
    """ Computes element-wise logarithm with output_i=log_zero_val where x_i=0.
    """
    epsilon = torch.finfo(torch.float32).tiny
    log_x = torch.log(torch.clamp(x, min=epsilon))
    zero_mask = (x == 0).to(dtype=log_x.dtype)
    log_x = (1 - zero_mask) * log_x + zero_mask * log_zero_val
    return log_x


def viterbi_chunk_step(gamma_prev, local_gamma):
    """ A variant of the Viterbi step that is used in the parallel variant of Viterbi.
    Args:
        gamma_prev: Viterbi values of the previous recursion. Shape (num_models, b, q)
        local_gamma: Logarithmic transition matrices describing the transition from chunk start to end. Shape (num_models, b, q, q)
    Returns:
        Viterbi values of the current recursion (gamma_next). Shape (num_models, b, q)
    """
    gamma_next = local_gamma + gamma_prev.unsqueeze(-1)  # Shape: (num_models, b, q, q)
    gamma_next, _ = torch.max(gamma_next, dim=-2)  # Shape: (num_models, b, q)
    return gamma_next


def viterbi_chunk_dyn_prog(emission_probs, init, transition_matrix, local_gamma, non_homogeneous_mask=None):
    """ A variant of Viterbi that computes the gamma values at the begin and end positions of chunks.
    Args:
        emission_probs: Emission probabilities at the starting positions of each chunk. Shape (num_models, b, num_chunks, q).
        init: Initial state distribution. Shape (num_models, q).
        transition_matrix: Logarithmic transition matrices describing the Markov chain. Shape (num_models, q, q)
        local_gamma: Local viterbi values at the end of each chunk. Shape (num_models, b, num_chunks, q, q)
        non_homogeneous_mask: Optional mask of shape (num_models, b, q, q) that specifies which transitions are allowed.
    Returns:
        Viterbi values (gamma) of begin and end positions per chunk. Shape (num_models, b, num_chunks, 2, q)
    """
    gamma_val = safe_log(init).unsqueeze(1)  # Shape: (num_models, 1, q)
    gamma_val = gamma_val.to(dtype=transition_matrix.dtype)
    b0 = emission_probs[:, :, 0]  # Shape: (num_models, b, q)
    gamma_val = gamma_val + safe_log(b0)  # Shape: (num_models, b, q)

    num_chunks = emission_probs.size(2)
    gamma_list = [gamma_val]  # Shape: (num_models, b, 1, q)

    gamma_val = viterbi_chunk_step(gamma_val, local_gamma[:, :, 0])
    gamma_list.append(gamma_val)  # Shape: (num_models, b, 1, q)
    # Iterate over chunks
    for i in range(1, num_chunks):
        # Compute gamma_val for the current chunk start
        gamma_val = viterbi_step(gamma_val.unsqueeze(-2), emission_probs[:, :, i], transition_matrix,
                                 non_homogeneous_mask)[..., 0, :]  # Shape: (num_models, b, q)
        gamma_list.append(gamma_val)  # Shape: (num_models, b, 1, q)

        # Compute gamma_val for the current chunk end
        gamma_val = viterbi_chunk_step(gamma_val, local_gamma[:, :, i])  # Shape: (num_models, b, q)
        gamma_list.append(gamma_val)  # Shape: (num_models, b, 1, q)

    gamma = torch.stack(gamma_list, dim=2)  # Shape: (num_models, b, num_chunks + 1, q)
    gamma = gamma.reshape(gamma.size(0), gamma.size(1), num_chunks, 2,
                       gamma.size(-1))  # Shape: (num_models, b, num_chunks, 2, q)

    return gamma


def viterbi_backtracking_step(prev_states, gamma_state, transition_matrix_transposed, output_type,
                              non_homogeneous_mask=None):
    """ Computes a Viterbi backtracking step in parallel for all models and batch elements.
    Args:
        prev_states: Previously decoded states. Shape: (num_model, b, 1)
        gamma_state: Viterbi values of the previously decoded states. Shape: (num_model, b, q)
        transition_matrix_transposed: Transposed logarithmic transition matrices describing the Markov chain.
                                        Shape (num_models, q, q) or (num_models, b, q, q)
        output_type: Datatype of the output states.
        non_homogeneous_mask: Optional mask of shape (num_models, b, q, q) that specifies which transitions are allowed.
    Returns:
        Next states. Shape: (num_model, b, 1)
    """
    if non_homogeneous_mask is None:
        if transition_matrix_transposed.dim() == prev_states.dim() + 1:
            A_prev_states = torch.take_along_dim(transition_matrix_transposed, prev_states.unsqueeze(-1), -2).squeeze(
                -1)
            if A_prev_states.dim() != gamma_state.dim():
                A_prev_states = A_prev_states.squeeze(-2)
        elif transition_matrix_transposed.dim() == prev_states.dim():
            A_prev_states = torch.take_along_dim(transition_matrix_transposed, prev_states, -2)
        else:
            A_prev_states = torch.gather(transition_matrix_transposed, -2, prev_states)
    else:
        if transition_matrix_transposed.dim() == prev_states.dim() + 1:
            A_prev_states = torch.gather(
                transition_matrix_transposed + safe_log(non_homogeneous_mask.transpose(-1, -2)), -2,
                prev_states.unsqueeze(-1)).squeeze(-1)
        else:
            A_prev_states = torch.gather(
                transition_matrix_transposed + safe_log(non_homogeneous_mask.transpose(-1, -2)), -2,
                prev_states)

    next_states = torch.argmax(A_prev_states + gamma_state, dim=-1, keepdim=True)
    return next_states.to(dtype=output_type)


def viterbi_backtracking(gamma, transition_matrix_transposed, output_type=torch.int64, non_homogeneous_mask_func=None):
    """ Performs backtracking on Viterbi score tables.
    Args:
        gamma: A Viterbi score table per model and batch element. Shape (num_model, b, L, q)
        transition_matrix_transposed: Transposed logarithmic transition matrices describing the Markov chain.
                                            Shape (num_models, q, q)
        output_type: Output type of the state sequences.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_model x batch x q x q mask that specifies which transitions are allowed.
    Returns:
        State sequences. Shape (num_model, b, L).
    """
    cur_states = torch.argmax(gamma[:, :, -1], dim=-1, keepdim=True)  # Shape: (num_model, b, 1)
    L = gamma.size(2)
    state_seqs_max_lik = [cur_states]

    for i in range(L - 2, -1, -1):
        cur_states = viterbi_backtracking_step(cur_states, gamma[:, :, i], transition_matrix_transposed, output_type,
                                               non_homogeneous_mask_func(
                                                   i + 1) if non_homogeneous_mask_func is not None else None)
        state_seqs_max_lik.append(cur_states)

    state_seqs_max_lik = torch.cat(state_seqs_max_lik[::-1], dim=-1)  # Shape: (num_model, b, L)
    return state_seqs_max_lik


def viterbi_chunk_backtracking(gamma, local_gamma_end_transposed, transition_matrix_transposed, output_type=torch.int64,
                               non_homogeneous_mask_func=None):
    """Performs backtracking on chunk-wise Viterbi score tables.
    Args:
        gamma: Viterbi values of begin and end positions per chunk. Shape (num_model, b, num_chunks, 2, q)
        local_gamma_end_transposed: Local viterbi values at the end of each chunk (transposed output of viterbi_chunk_dyn_prog).
                                Shape (num_models, b, num_chunks, q, q)
        transition_matrix_transposed: Transposed logarithmic transition matrices describing the Markov chain.
                                            Shape (num_models, q, q)
        output_type: Output type of the state sequences.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_model x batch x q x q mask that specifies which transitions are allowed.
    Returns:
        Most likely states at the chunk borders. Shape (num_model, b, num_chunks, 2).
    """
    # Initialize current states with the last chunk's end states
    cur_states = torch.argmax(gamma[:, :, -1, 1], dim=-1, keepdim=True)  # Shape: (num_model, b, 1)
    num_chunks = gamma.size(2)

    # Initialize a list to store the most likely states
    state_seqs_max_lik = [cur_states]

    # Backtracking for the last chunk
    cur_states = viterbi_backtracking_step(cur_states, gamma[:, :, -1, 0], local_gamma_end_transposed[:, :, -1],
                                           output_type)
    state_seqs_max_lik.append(cur_states)

    # Backtracking for the remaining chunks
    for i in range(1, num_chunks):
        # Backtrack to the start of the current chunk
        cur_states = viterbi_backtracking_step(cur_states, gamma[:, :, -1 - i, 1], transition_matrix_transposed,
                                               output_type,
                                               non_homogeneous_mask_func(
                                                   num_chunks - i) if non_homogeneous_mask_func is not None else None)
        state_seqs_max_lik.append(cur_states)

        # Backtrack to the end of the previous chunk
        cur_states = viterbi_backtracking_step(cur_states, gamma[:, :, -1 - i, 0],
                                               local_gamma_end_transposed[:, :, -1 - i], output_type)
        state_seqs_max_lik.append(cur_states)

    # Stack and reshape the results
    state_seqs_max_lik = torch.cat(state_seqs_max_lik[::-1], dim=-1)  # Shape: (num_model, b, 2 * num_chunks)
    state_seqs_max_lik = state_seqs_max_lik.reshape(state_seqs_max_lik.size(0), state_seqs_max_lik.size(1), num_chunks, 2)

    return state_seqs_max_lik


def viterbi_full_chunk_backtracking(viterbi_chunk_borders, local_gamma, transition_matrix_transposed,
                                    output_type=torch.int64, non_homogeneous_mask_func=None):
    """ Given the optimal end points for each chunk, determines the full Viterbi state sequence.
    Args:
        viterbi_chunk_borders: Most likely states at the chunk borders. Shape (num_model, b, num_chunks, 2)
        local_gamma: Local viterbi values for all chunks Shape (num_models, b, num_chunks, q, chunk_length, q)
        transition_matrix_transposed: Transposed logarithmic transition matrices describing the Markov chain.
                                            Shape (num_models, q, q)
        output_type: Output type of the state sequences.
        non_homogeneous_mask_func: Optional function that maps a sequence index i to a num_model x batch x q x q mask that specifies which transitions are allowed.
    Returns:
        State sequences. Shape (num_model, b, num_chunks * chunk_length).
    """
    num_model, b, num_chunks, q, chunk_length, _ = local_gamma.shape
    local_gamma = local_gamma.reshape(num_model, b * num_chunks, q, chunk_length, q)
    start_states = viterbi_chunk_borders[:, :, :, 0].reshape(num_model, b * num_chunks, 1)
    end_states = viterbi_chunk_borders[:, :, :, 1].reshape(num_model, b * num_chunks, 1)

    local_gamma = torch.take_along_dim(local_gamma, start_states.unsqueeze(-1).unsqueeze(-1), dim=2)[:, :, 0]

    cur_states = end_states
    state_seqs_max_lik = [cur_states]

    for i in range(chunk_length - 2, -1, -1):
        cur_states = viterbi_backtracking_step(cur_states, local_gamma[:, :, i], transition_matrix_transposed,
                                               output_type,
                                               non_homogeneous_mask_func(
                                                   i + 1) if non_homogeneous_mask_func is not None else None)
        state_seqs_max_lik.append(cur_states)

    state_seqs_max_lik = torch.cat(state_seqs_max_lik[::-1], dim=-1)  # Shape: (num_model, b * num_chunks, chunk_length)
    state_seqs_max_lik = state_seqs_max_lik.reshape(num_model, b, num_chunks * chunk_length)
    return state_seqs_max_lik


def viterbi_parallel(emission_probs, parallel_factor, A, At, init_dist):
    """ Placeholder function for parallel Viterbi decoding. """
    num_model, b, seq_len, q = emission_probs.shape
    chunk_size = seq_len // parallel_factor
    emission_probs = emission_probs.reshape(num_model, b*parallel_factor, chunk_size, q)

    init = init_dist if parallel_factor == 1 else torch.eye(q, device=emission_probs.device).unsqueeze(0)
    z = init.size(1)

    gamma = viterbi_dyn_prog(emission_probs, init, A,
                             use_first_position_emission=parallel_factor == 1,
                             non_homogeneous_mask_func=None)

    gamma = gamma.reshape(num_model, b * parallel_factor * z, chunk_size, q)
    if parallel_factor == 1:
        viterbi_paths = viterbi_backtracking(gamma, At, non_homogeneous_mask_func=None)
        variables_out = gamma
    else:
        emission_probs_at_chunk_start = emission_probs[:, :, 0].reshape(num_model, b, parallel_factor, q)
        gamma_local_at_chunk_end = gamma[:, :, -1].reshape(num_model, b, parallel_factor, q, q)

        gamma_at_chunk_borders = viterbi_chunk_dyn_prog(emission_probs_at_chunk_start, init_dist[:, 0], A,
                                                        gamma_local_at_chunk_end)
        gamma_local_at_chunk_end = gamma_local_at_chunk_end.transpose(-1, -2)

        At_parallel = At.unsqueeze(0)
        viterbi_chunk_borders = viterbi_chunk_backtracking(gamma_at_chunk_borders, gamma_local_at_chunk_end,
                                                           At_parallel)

        gamma = gamma.reshape(num_model, b, parallel_factor, z, chunk_size, q)
        viterbi_paths = viterbi_full_chunk_backtracking(viterbi_chunk_borders, gamma, At)
        num_model, b, num_chunks, q, chunk_length, _ = gamma.shape
        variables_out = gamma.transpose(-2, -3)
        variables_out = variables_out.reshape(num_model, b, num_chunks * chunk_length, q, q)

    return viterbi_paths, variables_out


if __name__ == '__main__':
    # emission_probs = torch.randn((1, 2, 2, 3))
    # A = torch.randn((1, 2, 2))
    # At = A.transpose(1, 2)
    # parallel_factor = 1
    # init_dist = torch.randn((1, 1, 2))

    # emission_probs = torch.Tensor([[[[0.5, 0.4, 0.1], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]]])
    # A = torch.Tensor([[[0.7, 0.2, 0.1], [0.4, 0.5, 0.1], [0.4, 0.5, 0.1]]])
    # At = A.transpose(1, 2)
    # parallel_factor = 1
    # init_dist = torch.Tensor([[[0.6, 0.3, 0.1]]])

    # 初始概率（log形式）
    init_dist = torch.log(torch.tensor([[[0.6, 0.3, 0.1]]]))  # 初始更可能是晴天

    # 转移概率矩阵（log形式）
    A = torch.log(torch.tensor([[
        [0.7, 0.3, 0.1],  # 晴天 -> 晴天/雨天
        [0.4, 0.6, 0.1],  # 雨天 -> 晴天/雨天
        [0.2, 0.3, 0.5]  # 阴天 -> 晴天/雨天
    ]]))
    At = A.transpose(1, 2)

    # 发射概率矩阵（log形式）
    emission_probs = torch.log(torch.tensor([[[
        [0.5, 0.3, 0.2],  # 晴天时的活动概率
        [0.1, 0.4, 0.5],  # 雨天时的活动概率
        [0.1, 0.1, 0.8],  # 阴天时的活动概率
        [0.5, 0.2, 0.3],  # 阴天时的活动概率
        [0.5, 0.0, 0.5],  # 晴天时的活动概率
        [0.1, 0.1, 0.8],  # 雨天时的活动概率
        [0.5, 0.1, 0.8],  # 阴天时的活动概率
        [0.5, 3.2, 0.3],  # 阴天时的活动概率
        [0.5, 0.3, 0.2],  # 晴天时的活动概率
        [0.1, 0.4, 0.5],  # 雨天时的活动概率
        [0.1, 0.1, 0.8],  # 阴天时的活动概率
        [0.5, 0.2, 0.3],  # 阴天时的活动概率
        [0.5, 0.0, 0.5],  # 晴天时的活动概率
        [0.1, 0.1, 0.8],  # 雨天时的活动概率
        [0.5, 0.1, 0.8],  # 阴天时的活动概率
        [0.5, 3.2, 0.3],  # 阴天时的活动概率
    ]]]))
    emission_probs = emission_probs.reshape(1, 4, 4, 3)
    parallel_factor = 1

    viterbi_paths, gamma = viterbi_parallel(emission_probs, parallel_factor, A, At, init_dist)
    print(viterbi_paths)
    print(gamma.shape)
    print(gamma[0])
