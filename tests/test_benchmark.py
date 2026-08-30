import numpy as np

import pandas as pd

from wireless_likelihoods.benchmark import summarize_repeated_results, unconditional_gaussian_nll
from wireless_likelihoods.dataset import generate_dataset, sample_iid_states
from wireless_likelihoods.simulation import IIDErrorChannel


def test_unconditional_baseline_is_finite():
    states = sample_iid_states(10, np.random.default_rng(40), 32)
    dataset = generate_dataset(IIDErrorChannel(), states, replicates=2, n_symbols=32, seed=41)
    assert np.isfinite(unconditional_gaussian_nll(dataset))


def test_repeated_summary_uses_paired_seed_differences():
    rows = []
    for seed in range(4):
        for family in ("iid", "gilbert_elliott", "fixed_burst", "markov_interference", "log_gaussian_fading"):
            separate = -1.0 - 0.01 * seed
            unified = separate - (0.18 + 0.01 * seed)
            rows.append({
                "seed": seed, "family": family, "baseline_test_nll": 0.0,
                "separate_test_nll": separate, "unified_test_nll": unified,
                "joint_gain": separate - unified,
            })
    summary = summarize_repeated_results(pd.DataFrame(rows))
    assert len(summary) == 6
    assert np.allclose(summary["joint_gain_mean"], 0.195)
    assert np.all(summary["n_seeds"] == 4)
