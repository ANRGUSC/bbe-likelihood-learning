import numpy as np

from wireless_likelihoods.actions import ACTIONS
from wireless_likelihoods.simulation import (
    FixedBurstChannel,
    GilbertElliottChannel,
    IIDErrorChannel,
    LogGaussianFadingChannel,
    MarkovInterferenceChannel,
)
from wireless_likelihoods.types import ChannelState


def test_iid_empirical_error_rate_matches_configuration():
    state = ChannelState("iid", {"error_prob": 0.12})
    output = IIDErrorChannel().simulate(state, ACTIONS[0], 30_000, np.random.default_rng(1))
    assert abs(output["error_rate"] - 0.12) < 0.01


def test_gilbert_elliott_bad_occupancy_matches_stationary_probability():
    state = ChannelState("gilbert_elliott", {"p_good_to_bad": 0.02, "p_bad_to_good": 0.20, "error_good": 0.005, "error_bad": 0.25})
    output = GilbertElliottChannel().simulate(state, ACTIONS[0], 30_000, np.random.default_rng(2))
    assert abs(output["latent_state"].mean() - (0.02 / 0.22)) < 0.025


def test_fixed_burst_reports_requested_length():
    state = ChannelState("fixed_burst", {"background_error_prob": 0.01, "burst_probability": 1.0, "burst_length": 20.0, "burst_error_prob": 0.4})
    output = FixedBurstChannel().simulate(state, ACTIONS[1], 100, np.random.default_rng(3))
    assert output["burst_length"] == 20
    assert 0 <= output["burst_start"] <= 80


def test_markov_interference_transition_rates():
    state = ChannelState("markov_interference", {"p_on": 0.04, "p_off": 0.20, "error_off": 0.01, "error_on": 0.3})
    latent = MarkovInterferenceChannel().simulate(state, ACTIONS[0], 50_000, np.random.default_rng(4))["latent_state"]
    off_to_on = latent[:-1][latent[:-1] == 0]
    on_to_off = latent[:-1][latent[:-1] == 1]
    assert abs(latent[1:][latent[:-1] == 0].mean() - 0.04) < 0.015
    assert abs(1.0 - latent[1:][latent[:-1] == 1].mean() - 0.20) < 0.025
    assert len(off_to_on) > 100 and len(on_to_off) > 100


def test_log_gaussian_fading_has_positive_short_lag_correlation():
    state = ChannelState("log_gaussian_fading", {"mean_log_snr": 0.5, "std_log_snr": 0.7, "length_scale": 12.0, "snr_offset": 1.0})
    output = LogGaussianFadingChannel().simulate(state, ACTIONS[0], 100, np.random.default_rng(5))
    latent = output["latent_log_snr"]
    assert np.corrcoef(latent[:-1], latent[1:])[0, 1] > 0.5
