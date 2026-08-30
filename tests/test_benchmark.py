import numpy as np

from wireless_likelihoods.benchmark import unconditional_gaussian_nll
from wireless_likelihoods.dataset import generate_dataset, sample_iid_states
from wireless_likelihoods.simulation import IIDErrorChannel


def test_unconditional_baseline_is_finite():
    states = sample_iid_states(10, np.random.default_rng(40), 32)
    dataset = generate_dataset(IIDErrorChannel(), states, replicates=2, n_symbols=32, seed=41)
    assert np.isfinite(unconditional_gaussian_nll(dataset))
