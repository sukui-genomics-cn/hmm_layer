import torch

def make_k_mers(sequences, k, pivot_left=True):
    """
    将 one-hot 编码的核苷酸序列映射到 k-mer 表示。

    参数 : 
        sequences: 形状为 (b, L, 5) 的张量, 表示长度为 L 的序列。
                   假定最后一个维度是 one-hot 编码, 其中 "N" 对应于最后一个位置。
        k: 指定 k-mer 长度的整数。
        pivot_left: 指定是否将 k-mer 旋转到左侧或右侧的布尔值。

    返回 : 
        形状为 (b, L, 4**k-1, 4) 的张量。如果 pivot_left 为 True, 
        则最后一个维度对应于 k-mer 最左侧位置的 4 个可能的核苷酸。
        否则, 最后一个维度对应于 k-mer 最右侧的位置。
        如果 k-mer 包含 N, 则在可能位于该位置的 4 个常规核苷酸中等概率表示。
    """
    L = sequences.shape[-2]
    n = sequences.shape[-1] - 1  # 字母表大小是字符数减 1 (N)
    n = torch.tensor(n, dtype=sequences.dtype)
    # 在 N 的情况下, 在字母表上均匀分布
    sequences_no_N = sequences[..., :-1]
    N_pos = (sequences[..., -1:] == 1).to(sequences.dtype)
    sequences_no_N += (1 / n) * N_pos
    # 计算跨越序列边界的 k-mer 的填充
    pad = torch.ones_like(sequences_no_N[:, :k - 1, :], dtype=sequences.dtype) / n
    if pivot_left:
        sequences_padded_no_N = torch.cat([sequences_no_N, pad], dim=-2)
        k_mers = sequences_padded_no_N[:, :L, None, :]
    else:
        sequences_padded_no_N = torch.cat([pad, sequences_no_N], dim=-2)
        k_mers = sequences_padded_no_N[:, k - 1:L + k - 1, None, :]
    if pivot_left:
        iteration_range = range(1, k)
    else:
        iteration_range = range(k - 2, -1, -1)

    for i in iteration_range:
        shift_i = sequences_padded_no_N[:, i:L + i, None, :, None]
        k_mers = k_mers[..., None, :] * shift_i
        if pivot_left:
            shape = [4**i, 4]
        else:
            shape = [4**(k - i - 1), 4]
        k_mers = k_mers.reshape(list(k_mers.shape[:-3]) + shape)
    return k_mers

def encode_kmer_string(kmer, pivot_left=True, alphabet="ACGT"):
    """
    将 k-mer 转换为格式为 (i, j) 的类, 其中 i < n^{k-1} 且 j < n, 其中 n 是字母表大小。
    例如, AAA -> (0, 0), AAT -> (3, 0), TAA -> (0, 3) 如果 pivot_left 为 True, 否则
        AAA -> (0, 0), AAT -> (0, 3), TAA -> (12, 0)
    输出是 A、C、G、T 情况下的这些类的 one-hot 编码。
    如果 k-mer 包含 N, 则在 4 个常规核苷酸中等概率表示。
    """
    alphabet_with_unknown = alphabet + "N"
    kmer = [alphabet_with_unknown.index(x) for x in kmer]
    kmer = torch.tensor(kmer)
    one_hot = torch.nn.functional.one_hot(kmer, num_classes=len(alphabet_with_unknown)).to(torch.float32) #修正了这一行
    encoded_kmers = make_k_mers(one_hot.unsqueeze(0), k=len(kmer), pivot_left=pivot_left)
    if pivot_left:
        return encoded_kmers.squeeze(0)[0]
    else:
        return encoded_kmers.squeeze(0)[-1]