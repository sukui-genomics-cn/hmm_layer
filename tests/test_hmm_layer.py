import logging
import torch
import numpy as np

from Initializers import make_15_class_emission_kernel
from MsaHMMLayer import MsaHmmLayer
from MsaHmmCell import HmmCell
from gene_pred_hmm_emitter import GenePredHMMEmitter
from gene_pred_hmm_transitioner import GenePredMultiHMMTransitioner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_hmm_layer():
    logger.info("start run hmm_layer")

    hmm_inputs = np.load("hmm_inputs.npy")

    # Define the forward layer
    dim = 15
    num_states = [dim]
    embedding_inputs = torch.randn(1, 32, 9999, dim)
    nucleotide_inputs = torch.randn(1, 32, 9999, 5)
    # stacked_inputs = torch.concat([embedding_inputs, nucleotide_inputs], dim=-1)
    stacked_inputs = torch.from_numpy(hmm_inputs).float()
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
    reverse_cell = HmmCell(
        num_states=num_states,
        dim=dim,
        emitter=emitter,
        transitioner=transitioner,
        use_fake_step_counter=True,
    )
    reverse_cell.reverse = True
    reverse_cell.transitioner.reverse = True

    hmm_layer = MsaHmmLayer(
        cell=cell,
        reverse_cell=reverse_cell,
        parallel_factor=99
    )
    outputs = hmm_layer.state_posterior_log_probs(
        inputs=stacked_inputs, training=True, no_loglik=False, return_prior=True
    )
    logger.info(f"outputs {outputs}")
    logger.info("end run hmm_layer")


def test_gene_hmm_layer():
    from gene_pred_hmm import GenePredHMMLayer

    hmm_inputs = np.load("hmm_inputs.npy")

    # Define the forward layer
    dim = 15
    stacked_inputs = torch.from_numpy(hmm_inputs).float()
    layer = GenePredHMMLayer()
    outputs = layer(
        inputs=stacked_inputs, training=True
    )
    logger.info(f"outputs {outputs}")
    logger.info("end run hmm_layer")

if __name__ == '__main__':
    test_gene_hmm_layer()
