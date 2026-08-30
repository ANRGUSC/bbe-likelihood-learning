"""Train the baseline conditional Gaussian likelihood."""

from __future__ import annotations

import argparse

import numpy as np

from wireless_likelihoods.likelihoods import ConditionalGaussianLikelihood, ConditionalGaussianMixtureLikelihood, save_checkpoint, train_likelihood


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset.npz")
    parser.add_argument("--output", default="results/likelihood.pt")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--mixture", action="store_true", help="Use a 3-component mixture-density network.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    archive = np.load(args.dataset)
    dataset = {key: archive[key] for key in archive.files}
    model_class = ConditionalGaussianMixtureLikelihood if args.mixture else ConditionalGaussianLikelihood
    model = model_class(theta_dim=dataset["theta"].shape[1])
    history = train_likelihood(model, dataset, epochs=args.epochs, seed=args.seed)
    path = save_checkpoint(model, args.output)
    print(f"Saved {path}; validation NLL {history['validation'][-1]:.3f}")


if __name__ == "__main__":
    main()
