import numpy as np

from wireless_likelihoods.actions import ACTIONS
from wireless_likelihoods.dataset import (
    CHANNEL_FAMILIES,
    combine_family_datasets,
    generate_dataset,
    sample_fixed_burst_states,
    sample_gilbert_elliott_states,
    sample_iid_states,
    sample_log_gaussian_fading_states,
    sample_markov_interference_states,
    simulate_once,
)
from wireless_likelihoods.simulation import (
    FixedBurstChannel,
    GilbertElliottChannel,
    IIDErrorChannel,
    LogGaussianFadingChannel,
    MarkovInterferenceChannel,
)


def test_same_seed_reproduces_a_simulated_sample():
    state = sample_fixed_burst_states(7, np.random.default_rng(3), 64)[0]
    first = simulate_once(FixedBurstChannel(), state, ACTIONS[1], np.random.default_rng(9), n_symbols=64)
    second = simulate_once(FixedBurstChannel(), state, ACTIONS[1], np.random.default_rng(9), n_symbols=64)
    assert np.array_equal(first["measurement"], second["measurement"])
    assert np.array_equal(first["raw"]["errors"], second["raw"]["errors"])


def test_dataset_has_repeats_and_grouped_splits():
    states = sample_fixed_burst_states(12, np.random.default_rng(4), 64)
    dataset = generate_dataset(FixedBurstChannel(), states, replicates=3, n_symbols=64, seed=5)
    assert len(dataset["theta"]) == 12 * 3 * len(ACTIONS)
    for state_id in range(12):
        assert len(set(dataset["split"][dataset["state_id"] == state_id])) == 1


def test_actions_change_receiver_measurements():
    state = sample_fixed_burst_states(7, np.random.default_rng(7), 128)[0]
    means = []
    for action in (ACTIONS[0], ACTIONS[2]):
        rng = np.random.default_rng(8)
        means.append(np.mean([simulate_once(FixedBurstChannel(), state, action, rng, n_symbols=128)["measurement"][1] for _ in range(250)]))
    assert means[1] > means[0] + 0.10


def test_all_family_samplers_generate_valid_datasets_and_unified_encoding():
    builders = {
        "iid": (IIDErrorChannel, sample_iid_states),
        "gilbert_elliott": (GilbertElliottChannel, sample_gilbert_elliott_states),
        "fixed_burst": (FixedBurstChannel, sample_fixed_burst_states),
        "markov_interference": (MarkovInterferenceChannel, sample_markov_interference_states),
        "log_gaussian_fading": (LogGaussianFadingChannel, sample_log_gaussian_fading_states),
    }
    datasets = {}
    for index, family in enumerate(CHANNEL_FAMILIES):
        channel, sampler = builders[family]
        states = sampler(8, np.random.default_rng(20 + index), 32)
        datasets[family] = generate_dataset(channel(), states, replicates=1, n_symbols=32, seed=30 + index)
    unified = combine_family_datasets(datasets)
    assert unified["theta"].shape == (8 * 3 * len(CHANNEL_FAMILIES), len(CHANNEL_FAMILIES) + 4)
    assert set(unified["family_id"]) == set(range(len(CHANNEL_FAMILIES)))
    assert np.allclose(unified["theta"][:, : len(CHANNEL_FAMILIES)].sum(axis=1), 1.0)
