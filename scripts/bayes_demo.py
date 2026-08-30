"""Run a grid posterior update from a saved model."""

from __future__ import annotations

import argparse

import numpy as np

from wireless_likelihoods.actions import ACTIONS
from wireless_likelihoods.dataset import simulate_once, theta_from_state
from wireless_likelihoods.likelihoods import load_checkpoint
from wireless_likelihoods.plotting import plot_posterior_updates
from wireless_likelihoods.receiver import AbstractReceiverModel
from wireless_likelihoods.simulation import FixedBurstChannel
from wireless_likelihoods.types import ChannelState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/likelihood.pt")
    parser.add_argument("--output", default="results/posterior_update.png")
    args = parser.parse_args()
    state = ChannelState("fixed_burst", {"background_error_prob": 0.02, "burst_probability": 0.25, "burst_length": 24.0, "burst_error_prob": 0.35})
    action = ACTIONS[1]
    rng = np.random.default_rng(2)
    theta_grid = np.repeat(theta_from_state(state)[None, :], 100, axis=0)
    theta_grid[:, 1] = np.linspace(0.01, 0.5, len(theta_grid))
    observations = np.asarray([simulate_once(FixedBurstChannel(), state, action, rng, receiver=AbstractReceiverModel())["measurement"][1:] for _ in range(10)])
    path = plot_posterior_updates(load_checkpoint(args.model), theta_grid, action, observations, state.params["burst_probability"], path=args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
