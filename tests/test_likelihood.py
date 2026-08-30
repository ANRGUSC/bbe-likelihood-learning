import numpy as np
import torch

from wireless_likelihoods.dataset import generate_dataset, sample_fixed_burst_states
from wireless_likelihoods.likelihoods import ConditionalGaussianLikelihood, ConditionalGaussianMixtureLikelihood, train_likelihood
from wireless_likelihoods.simulation import FixedBurstChannel


def test_likelihood_trains_and_returns_finite_log_probabilities():
    dataset = generate_dataset(FixedBurstChannel(), sample_fixed_burst_states(14, np.random.default_rng(11), 64), replicates=6, n_symbols=64, seed=12)
    model = ConditionalGaussianLikelihood(theta_dim=4, hidden_dim=24)
    history = train_likelihood(model, dataset, epochs=25, batch_size=64, seed=13)
    assert history["train"][-1] < history["train"][0]
    values = model.log_prob(dataset["measurement"][:4, 1:], dataset["theta"][:4], dataset["action_features"][:4])
    assert values.shape == (4,)
    assert torch.isfinite(values).all()


def test_mixture_likelihood_has_finite_density_and_distribution():
    dataset = generate_dataset(FixedBurstChannel(), sample_fixed_burst_states(10, np.random.default_rng(14), 48), replicates=3, n_symbols=48, seed=15)
    model = ConditionalGaussianMixtureLikelihood(theta_dim=4, hidden_dim=16)
    train_likelihood(model, dataset, epochs=8, batch_size=32, seed=16)
    distribution = model.distribution(dataset["theta"][:2], dataset["action_features"][:2])
    assert torch.isfinite(distribution.log_prob(torch.as_tensor(dataset["measurement"][:2, 1:]))).all()
    assert np.all(np.isfinite(model.marginal_density(np.linspace(0, 2, 10), dataset["theta"][0], dataset["action_features"][0], 1)))
