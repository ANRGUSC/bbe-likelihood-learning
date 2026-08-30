"""Generate the default 60,000-sample fixed-burst dataset."""

from __future__ import annotations

import argparse

import numpy as np

from wireless_likelihoods.dataset import generate_dataset, sample_fixed_burst_states, save_dataset
from wireless_likelihoods.simulation import FixedBurstChannel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dataset.npz")
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--symbols", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    states = sample_fixed_burst_states(args.states, np.random.default_rng(args.seed), args.symbols)
    dataset = generate_dataset(FixedBurstChannel(), states, replicates=args.replicates, n_symbols=args.symbols, seed=args.seed)
    path = save_dataset(dataset, args.output)
    print(f"Saved {len(dataset['theta']):,} samples to {path}")


if __name__ == "__main__":
    main()
