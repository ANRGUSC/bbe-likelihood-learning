"""Train and compare family-specific and unified neural likelihoods."""

from __future__ import annotations

import argparse

from wireless_likelihoods.benchmark import run_channel_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/channel_benchmark")
    parser.add_argument("--data-dir", default="data/channel_benchmark")
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--symbols", type=int, default=64)
    parser.add_argument("--separate-epochs", type=int, default=100)
    parser.add_argument("--unified-epochs", type=int, default=140)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()
    metrics = run_channel_benchmark(
        output_dir=args.output_dir, data_dir=args.data_dir, n_states=args.states,
        replicates=args.replicates, n_symbols=args.symbols,
        separate_epochs=args.separate_epochs, unified_epochs=args.unified_epochs, seed=args.seed,
    )
    columns = ["family", "baseline_test_nll", "separate_test_nll", "unified_test_nll"]
    print(metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Wrote benchmark artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
