"""Create a held-out-style likelihood figure from a saved fixed-burst model."""

from __future__ import annotations

import argparse

from wireless_likelihoods.actions import ACTIONS
from wireless_likelihoods.likelihoods import load_checkpoint
from wireless_likelihoods.plotting import plot_action_conditioned_likelihoods
from wireless_likelihoods.receiver import AbstractReceiverModel
from wireless_likelihoods.simulation import FixedBurstChannel
from wireless_likelihoods.types import ChannelState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/likelihood.pt")
    parser.add_argument("--output", default="results/likelihood_fit.png")
    args = parser.parse_args()
    state = ChannelState("fixed_burst", {"background_error_prob": 0.02, "burst_probability": 0.25, "burst_length": 24.0, "burst_error_prob": 0.35})
    path = plot_action_conditioned_likelihoods(load_checkpoint(args.model), FixedBurstChannel(), state, ACTIONS, receiver=AbstractReceiverModel(), path=args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
