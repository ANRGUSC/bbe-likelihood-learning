"""Repeat the separate-versus-unified benchmark across independent seeds."""

from __future__ import annotations

import argparse

from wireless_likelihoods.benchmark import run_repeated_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/repeated_benchmark")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=21)
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--symbols", type=int, default=64)
    parser.add_argument("--separate-epochs", type=int, default=100)
    parser.add_argument("--unified-epochs", type=int, default=140)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    _, summary = run_repeated_benchmark(
        output_dir=args.output_dir, n_seeds=args.seeds, start_seed=args.start_seed,
        n_states=args.states, replicates=args.replicates, n_symbols=args.symbols,
        separate_epochs=args.separate_epochs, unified_epochs=args.unified_epochs,
        resume=not args.no_resume,
    )
    columns = ["family", "separate_mean", "unified_mean", "joint_gain_mean", "joint_gain_ci_low", "joint_gain_ci_high"]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
