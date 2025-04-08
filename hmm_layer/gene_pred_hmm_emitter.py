import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kmer
from .Initializers import ConstantInitializer, make_15_class_emission_kernel
from .MvnMixture import MvnMixture, DefaultDiagBijector

class SimpleGenePredHMMEmitter(nn.Module):
    """
    定义基因预测 HMM 的发射概率, 使用嵌入或类预测作为输入。

    参数 : 
        num_models: 半独立的基因模型数量 (参见 GenePredHMMLayer) 。
        num_copies: 一个 HMM 中共享 IR 状态的基因模型副本数量。
        init: 发射概率的初始化器。
        emit_embeddings: 如果为 True, 则对嵌入发射进行建模。假定嵌入向量是从多元正态分布中采样的。
        full_covariance: 如果为 True, 则使用完整协方差矩阵对多元正态分布进行建模。否则, 仅对对角线元素进行建模。
        embedding_dim: 嵌入向量的维度。如果 emit_embeddings=True, 则必须给出。
        embedding_kernel_init: 嵌入发射概率的初始化器。请注意, 对角线通过 softplus 函数传递, 以确保正值并偏移 1。
        temperature: 用于调节 mvn pdf 的温度参数。
        share_intron_parameters: 如果为 True, 则内含子状态共享相同的发射参数。
    """
    def __init__(self, 
                 num_models=1,
                 num_copies=1,
                 init=make_15_class_emission_kernel(smoothing=1e-2, num_copies=1), #ConstantInitializer(0.)
                 trainable_emissions=False,
                 emit_embeddings=False, 
                 embedding_dim=None, 
                 full_covariance=False,
                 embedding_kernel_init="random_normal",
                 initial_variance=0.05,
                 temperature=100.,
                 share_intron_parameters=False,
                 device=None,
                 **kwargs):
        super(SimpleGenePredHMMEmitter, self).__init__(**kwargs)
        self.num_models = num_models
        self.num_copies = num_copies
        self.num_states = 1 + 6 * num_copies
        self.init = init
        self.trainable_emissions = trainable_emissions
        self.emit_embeddings = emit_embeddings
        self.embedding_dim = embedding_dim
        self.full_covariance = full_covariance
        self.embedding_kernel_init = embedding_kernel_init
        self.initial_variance = initial_variance
        self.temperature = temperature
        self.share_intron_parameters = share_intron_parameters
        if self.emit_embeddings:
            assert embedding_dim is not None, "如果 emit_embeddings=True, 则必须给出 embedding_dim。"
        else:
            assert embedding_dim is None, "如果 emit_embeddings=False, 则不能给出 embedding_dim。"

        self.device = device
        self.emission_kernel = None
        self.embedding_emission_kernel = None
        self.mvn_mixture = None
        self.B = None
        self.embedding_emit = None
        self.built = False

    def build(self):
        if self.built:
            return
        # s = input_shape[-1]
        # self.emission_kernel = nn.Parameter(torch.full(
        #     (self.num_models, self.num_states - 2 * self.num_copies * int(self.share_intron_parameters), s),
        #     float(self.init)),
        #     requires_grad=self.trainable_emissions
        # )
        self.emission_kernel = nn.Parameter(
            torch.from_numpy(self.init).to(self.device),
            requires_grad=self.trainable_emissions
        )
        if self.emit_embeddings:
            assert self.num_models == 1, "嵌入发射当前仅支持一个模型。您很可能意外地设置了 emit_embeddings=True。"
            d = self.embedding_dim
            num_mvn_param = d + d * (d + 1) // 2 if self.full_covariance else 2 * d
            if self.embedding_kernel_init == "random_normal":
                self.embedding_emission_kernel = nn.Parameter(torch.randn(1, self.num_states - 2 * self.num_copies * int(self.share_intron_parameters), 1, num_mvn_param), requires_grad=True)
            else:
                raise ValueError(f"embedding_kernel_init '{self.embedding_kernel_init}' not supported")
        self.built = True

    def recurrent_init(self):
        """
        在每次递归运行之前自动调用。应将其用于每个递归层应用仅需要一次的设置。
        """
        self.B = self.make_B()
        if self.emit_embeddings:
            self.mvn_mixture = MvnMixture(self.embedding_dim, 
                                            self.embedding_emission_kernel,
                                            diag_only=not self.full_covariance,
                                            diag_bijector=DefaultDiagBijector(self.initial_variance))

    def make_B(self):
        """
        从发射核构造发射概率。
        """
        return F.softmax(self.emission_kernel, dim=-1)

    def forward(self, inputs, end_hints=None, training=False):
        """
        参数 : 
            inputs: 如果 emit_embeddings=False, 则形状为 (num_models, batch_size, length, alphabet_size) 的张量, 否则为 (num_models, ..., alphabet_size+embedding_dim)。
            end_hints: 形状为 (num_models, batch_size, 2, num_states) 的张量, 其中包含每个块的左右端点的正确状态。
        返回 : 
            形状为 (num_models, batch_size, length, num_states) 的发射概率张量。
        """
        if self.emit_embeddings:
            class_inputs = inputs[..., :-self.embedding_dim]
            embedding_inputs = inputs[..., -self.embedding_dim:]
            class_emit = torch.einsum("...s,kqs->k...q", class_inputs, self.B)
            embedding_inputs = embedding_inputs.reshape(1, -1, self.embedding_dim)
            log_pdf = self.mvn_mixture.log_pdf(embedding_inputs)
            log_pdf = log_pdf.reshape(class_emit.shape)
            self.embedding_emit = torch.exp(log_pdf / self.temperature)
            if training:
                class_emit += 1e-10
                self.embedding_emit += 1e-10
            emit = class_emit * self.embedding_emit
        else:
            emit = torch.einsum("...s,kqs->k...q", inputs[0], self.B)
        if self.share_intron_parameters:
            emit = torch.cat([emit[..., :1 + self.num_copies]] + [emit[..., 1:1 + self.num_copies]] * 2 + [emit[..., 1 + self.num_copies:]], dim=-1)
        if end_hints is not None:
            left_end = end_hints[..., :1, :] * emit[..., :1, :]
            right_end = end_hints[..., 1:, :] * emit[..., -1:, :]
            emit = torch.cat([left_end, emit[..., 1:-1, :], right_end], dim=-2)
        return emit

    def get_prior_log_density(self):
        # 可以在将来用于正则化。
        return torch.tensor([[0.]])

    def get_aux_loss(self):
        return torch.tensor(0.)


    def get_config(self):
        return {"num_models": self.num_models,
                "num_copies": self.num_copies,
                "init": self.init,
                "trainable_emissions": self.trainable_emissions,
                "emit_embeddings": self.emit_embeddings,
                "embedding_dim": self.embedding_dim,
                "full_covariance": self.full_covariance,
                "embedding_kernel_init": self.embedding_kernel_init,
                "initial_variance": self.initial_variance,
                "temperature": self.temperature,
                "share_intron_parameters": self.share_intron_parameters}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def assert_codons(codons):
    assert sum(p for _, p in codons) == 1, "密码子概率必须总和为 1, 得到 : " + str(codons)
    for triplet, prob in codons:
        assert len(triplet) == 3, "三联体长度必须为 3, 得到 : " + str(codons)
        assert prob >= 0 and prob <= 1, "概率必须介于 0 和 1 之间, 得到 : " + str(codons)

def make_codon_probs(codons, pivot_left):
    assert_codons(codons)
    codon_probs = sum(prob * kmer.encode_kmer_string(triplet, pivot_left) for triplet, prob in codons) #(16,4)
    codon_probs = codon_probs.reshape(64)
    return codon_probs.unsqueeze(0).unsqueeze(0)


class GenePredHMMEmitter(SimpleGenePredHMMEmitter):
    """
    通过强制执行生物结构的起始和终止状态扩展简单 HMM。
    """
    def __init__(self,
                 start_codons,
                 stop_codons,
                 intron_begin_pattern,
                 intron_end_pattern,
                 l2_lambda=0.01,
                 nucleotide_kernel_init=ConstantInitializer(0.),
                 trainable_nucleotides_at_exons=False,
                 **kwargs):
        """
        参数 : 
            start_codons: 允许的起始密码子。一个对列表。每对的第一个元素是一个字符串，是 ACGTN 字母表上的三联体。
                          第二个条目是该三联体的概率。所有概率之和必须为 1。
            stop_codons: 允许的终止密码子。格式与 `start_codons` 相同。
            intron_begin_pattern: 内允许的起始模式。格式与 `start_codons` 相同。
                                  由于只有内含子的前 2 个核苷酸相关，因此给出形式为 "N.." 的 3-mers，其中 N 表示
                                  允许前一个外显子中最后一个核苷酸的任何核苷酸。
            intron_end_pattern: 内含子中允许的终止模式。格式与 `start_codons` 相同。
                                由于只有内含子的最后 2 个核苷酸相关，因此给出形式为 "..N" 的 3-mers，其中 N 表示
                                允许下一个外显子中第一个核苷酸的任何核苷酸。
            nucleotide_kernel_init: 与外显子状态下的核苷酸分布密切相关的内核的初始化器。
            trainable_nucleotides_at_exons: 如果为 True，则外显子状态下的核苷酸是可训练的。
        """
        super(GenePredHMMEmitter, self).__init__(**kwargs)
        self.num_states = 1 + 14 * self.num_copies
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.intron_begin_pattern = intron_begin_pattern
        self.intron_end_pattern = intron_end_pattern
        self.l2_lambda = l2_lambda
        self.nucleotide_kernel_init = nucleotide_kernel_init
        self.trainable_nucleotides_at_exons = trainable_nucleotides_at_exons
        # 起始和终止密码子/起始和终止模式的发射概率
        self.start_codon_probs = make_codon_probs(start_codons, pivot_left=True)  # START 状态
        self.stop_codon_probs = make_codon_probs(stop_codons, pivot_left=False)  # STOP 状态
        self.intron_begin_codon_probs = make_codon_probs(intron_begin_pattern, pivot_left=True)  # 内含子起始状态
        self.intron_end_codon_probs = make_codon_probs(intron_end_pattern, pivot_left=False)  # 内含子终止状态
        self.any_codon_probs = make_codon_probs([("NNN", 1.)], pivot_left=False)  # 除 E2 和 EI1 之外的任何其他状态
        self.not_stop_codon_probs = self.any_codon_probs * (self.stop_codon_probs == 0).float()  # 任何其他状态
        self.not_stop_codon_probs /= self.not_stop_codon_probs.sum()  # 除终止密码子之外的所有密码子之和为 1
        # 状态顺序 : (Ir, I0, I1, I2, E0, E1), E2, START, EI0, EI1, EI2, IE0, IE1, IE2, STOP
        #  (省略前 6 个状态，因为它们没有密码子限制) 
        self.left_codon_probs = torch.cat([self.any_codon_probs]
                                           + [self.start_codon_probs]
                                           + [self.intron_begin_codon_probs] * 3
                                           + [self.any_codon_probs] * 4, dim=1)
        self.right_codon_probs = torch.cat([self.not_stop_codon_probs]
                                            + [self.any_codon_probs] * 2
                                            + [self.not_stop_codon_probs]
                                            + [self.any_codon_probs]
                                            + [self.intron_end_codon_probs] * 3
                                            + [self.stop_codon_probs], dim=1)
        self.codon_probs = torch.cat([self.left_codon_probs, self.right_codon_probs], dim=0).to(self.device)  # (2, num_states, 64)

    def build(self):
        if self.built:
            return
        super(GenePredHMMEmitter, self).build()
        # s = input_shape[-1]
        if self.trainable_nucleotides_at_exons:
            assert self.num_models == 1, "Trainable nucleotide emissions are currently only supported for one model."
            self.nuc_emission_kernel = nn.Parameter(torch.zeros(self.num_models, 3 * self.num_copies, 4), requires_grad=self.trainable_nucleotides_at_exons)

    def get_nucleotide_probs(self):
        return torch.softmax(self.nuc_emission_kernel, dim=-1)

    def forward(self, inputs, end_hints=None, training=False):
        """
        参数 : 
            inputs: 形状为 (num_models, batch, length, alphabet_size + 5) 的张量
            end_hints: 形状为 (num_models, batch_size, 2, num_states) 的张量，包含每个块的左右端的正确状态。
        返回 : 
            形状为 (num_models, batch, length, num_states) 的发射概率张量。
        """
        nucleotides = inputs[..., -5:]
        inputs = inputs[..., :-5]
        emit = super(GenePredHMMEmitter, self).forward(inputs, end_hints=end_hints, training=training)

        # 计算起始第一个外显子或内含子的概率
        # 以及终止最后一个外显子或内含子的概率
        num_models, batch, length = nucleotides.shape[:3]
        nucleotides = nucleotides.reshape(-1, length, 5)
        left_3mers = kmer.make_k_mers(nucleotides, k=3, pivot_left=True)
        left_3mers = left_3mers.reshape(num_models, batch, length, 64)
        right_3mers = kmer.make_k_mers(nucleotides, k=3, pivot_left=False)
        right_3mers = right_3mers.reshape(num_models, batch, length, 64)
        input_3mers = torch.stack([left_3mers, right_3mers], dim=-2)  # (num_models, batch, length, 2, 64)
        codon_emission_probs = torch.einsum("k...rs,rqs->k...rq", input_3mers, self.codon_probs)
        codon_emission_probs = codon_emission_probs.prod(dim=-2)

        if self.num_copies > 1:
            repeats = [self.num_copies] * codon_emission_probs.shape[-1]
            codon_emission_probs = codon_emission_probs.repeat_interleave(torch.tensor(repeats), dim=-1)
        codon_emission_probs = torch.cat([torch.ones_like(codon_emission_probs[..., :(1 + 5 * self.num_copies)]) / 4096., codon_emission_probs], dim=-1)

        if training:
            codon_emission_probs += 1e-7

        full_emission = emit * codon_emission_probs

        if self.trainable_nucleotides_at_exons:
            nucleotides = inputs[..., -5:]
            nucleotides_no_N = nucleotides[..., :4] + nucleotides[..., 4:] / 4
            nuc_emission_probs = torch.einsum("k...s,kqs->k...q", nucleotides_no_N, self.get_nucleotide_probs())
            nuc_emission_probs = torch.cat([torch.ones_like(full_emission[..., :1 + 3 * self.num_copies]) / 4.,
                                            nuc_emission_probs,
                                            torch.ones_like(full_emission[..., 1 + 6 * self.num_copies:]) / 4.], dim=-1)
            full_emission *= nuc_emission_probs

        if self.emit_embeddings:
            self.loss = self.l2_lambda * self.mvn_mixture.get_regularization_L2_loss()

        return full_emission

    def duplicate(self, model_indices=None, share_kernels=False):
        init = torch.tensor(self.emission_kernel.numpy())
        embedding_kernel_init = torch.tensor(self.embedding_emission_kernel.numpy()) if self.emit_embeddings else torch.zeros(1)
        if self.trainable_nucleotides_at_exons:
            nucleotide_kernel_init = torch.tensor(self.nuc_emission_kernel.numpy())
        else:
            nucleotide_kernel_init = torch.zeros(1)
        emitter_copy = GenePredHMMEmitter(self.start_codons,
                                            self.stop_codons,
                                            self.intron_begin_pattern,
                                            self.intron_end_pattern,
                                            self.l2_lambda,
                                            num_models=self.num_models,
                                            num_copies=self.num_copies,
                                            init=init,
                                            trainable_emissions=self.trainable_emissions,
                                            emit_embeddings=self.emit_embeddings,
                                            embedding_dim=self.embedding_dim,
                                            full_covariance=self.full_covariance,
                                            embedding_kernel_init=embedding_kernel_init,
                                            initial_variance=self.initial_variance,
                                            temperature=self.temperature,
                                            share_intron_parameters=self.share_intron_parameters,
                                            nucleotide_kernel_init=nucleotide_kernel_init,
                                            trainable_nucleotides_at_exons=self.trainable_nucleotides_at_exons)
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            if self.trainable_nucleotides_at_exons:
                emitter_copy.nuc_emission_kernel = self.nuc_emission_kernel
            if self.emit_embeddings:
                emitter_copy.embedding_emission_kernel = self.embedding_emission_kernel
            emitter_copy.built = True
        return emitter_copy

    def get_config(self):
        config = super(GenePredHMMEmitter, self).get_config()
        config.update({"start_codons": self.start_codons,
                       "stop_codons": self.stop_codons,
                       "intron_begin_pattern": self.intron_begin_pattern,
                       "intron_end_pattern": self.intron_end_pattern,
                       "l2_lambda": self.l2_lambda,
                       "nucleotide_kernel_init": self.nucleotide_kernel_init,
                       "trainable_nucleotides_at_exons": self.trainable_nucleotides_at_exons})
        return config

    @classmethod
    def from_config(cls, config):
        config["init"] = torch.tensor(config["init"].numpy())
        config["nucleotide_kernel_init"] = torch.tensor(config["nucleotide_kernel_init"].numpy())
        return cls(**config)
    


if __name__ == '__main__':
    start_codons = [("ATG", 1.)]
    stop_codons = [("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)]
    intron_begin_pattern = [("NGT", 0.99), ("NGC", 0.005), ("NAT", 0.005)]
    intron_end_pattern = [("AGN", 0.99), ("ACN", 0.01)]

    model = GenePredHMMEmitter(
        start_codons,
        stop_codons,
        intron_begin_pattern,
        intron_end_pattern,
        share_intron_parameters=False
    )
    # model = SimpleGenePredHMMEmitter()
    input_shape = (1, 2, 9, 15)
    model.build(input_shape)
    model.recurrent_init()
    print(model)
    inputs = torch.rand((1, 2, 9, 20))
    output = model.forward(inputs)
    print(output.shape)