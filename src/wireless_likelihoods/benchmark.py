"""Training and evaluation utilities for per-family and unified likelihoods."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .dataset import (
    CHANNEL_FAMILIES,
    combine_family_datasets,
    generate_dataset,
    sample_fixed_burst_states,
    sample_gilbert_elliott_states,
    sample_iid_states,
    sample_log_gaussian_fading_states,
    sample_markov_interference_states,
    save_dataset,
)
from .likelihoods import ConditionalGaussianMixtureLikelihood, save_checkpoint, train_likelihood
from .simulation import (
    FixedBurstChannel,
    GilbertElliottChannel,
    IIDErrorChannel,
    LogGaussianFadingChannel,
    MarkovInterferenceChannel,
)


FAMILY_BUILDERS = {
    "iid": (IIDErrorChannel, sample_iid_states),
    "gilbert_elliott": (GilbertElliottChannel, sample_gilbert_elliott_states),
    "fixed_burst": (FixedBurstChannel, sample_fixed_burst_states),
    "markov_interference": (MarkovInterferenceChannel, sample_markov_interference_states),
    "log_gaussian_fading": (LogGaussianFadingChannel, sample_log_gaussian_fading_states),
}


def generate_family_datasets(
    *, n_states: int = 48, replicates: int = 20, n_symbols: int = 64, seed: int = 21,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate equally sized, independently seeded datasets for every family."""
    datasets = {}
    for family_id, family in enumerate(CHANNEL_FAMILIES):
        channel_class, sampler = FAMILY_BUILDERS[family]
        family_seed = seed + 1009 * family_id
        states = sampler(n_states, np.random.default_rng(family_seed), n_symbols)
        datasets[family] = generate_dataset(
            channel_class(), states, replicates=replicates, n_symbols=n_symbols, seed=family_seed + 1,
        )
    return datasets


def mean_nll(model, dataset: dict[str, np.ndarray], split: int = 2, extra_mask: np.ndarray | None = None) -> float:
    mask = dataset["split"] == split
    if extra_mask is not None:
        mask &= extra_mask
    with torch.no_grad():
        value = -model.log_prob(
            dataset["measurement"][mask, 1:], dataset["theta"][mask], dataset["action_features"][mask]
        ).mean()
    return float(value.detach())


def unconditional_gaussian_nll(dataset: dict[str, np.ndarray], split: int = 2) -> float:
    """Reference NLL from a single diagonal Gaussian fitted to training telemetry."""
    target = dataset["measurement"][:, 1:]
    train = dataset["split"] == 0
    evaluate = dataset["split"] == split
    mean = target[train].mean(axis=0)
    std = np.maximum(target[train].std(axis=0), 1e-6)
    row_nll = 0.5 * ((((target[evaluate] - mean) / std) ** 2) + np.log(2.0 * np.pi) + 2.0 * np.log(std)).sum(axis=1)
    return float(row_nll.mean())


def run_channel_benchmark(
    *, output_dir: str | Path = "results/channel_benchmark", data_dir: str | Path = "data/channel_benchmark",
    n_states: int = 48, replicates: int = 20, n_symbols: int = 64,
    separate_epochs: int = 100, unified_epochs: int = 140, seed: int = 21,
) -> pd.DataFrame:
    """Train five family models and one unified model, then save held-out metrics."""
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets = generate_family_datasets(n_states=n_states, replicates=replicates, n_symbols=n_symbols, seed=seed)
    for family, dataset in datasets.items():
        save_dataset(dataset, data_dir / f"{family}.npz")

    rows: list[dict[str, float | int | str]] = []
    separate_models = {}
    for family_id, family in enumerate(CHANNEL_FAMILIES):
        dataset = datasets[family]
        model = ConditionalGaussianMixtureLikelihood(dataset["theta"].shape[1], hidden_dim=64, n_components=3)
        history = train_likelihood(model, dataset, epochs=separate_epochs, seed=seed + family_id)
        save_checkpoint(model, output_dir / f"{family}.pt")
        separate_models[family] = model
        rows.append({
            "family": family,
            "samples": len(dataset["theta"]),
            "theta_dim": dataset["theta"].shape[1],
            "baseline_test_nll": unconditional_gaussian_nll(dataset),
            "separate_validation_nll": mean_nll(model, dataset, split=1),
            "separate_test_nll": mean_nll(model, dataset, split=2),
            "separate_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_validation_nll": min(history["validation"]),
        })

    unified = combine_family_datasets(datasets)
    save_dataset(unified, data_dir / "unified.npz")
    unified_model = ConditionalGaussianMixtureLikelihood(unified["theta"].shape[1], hidden_dim=96, n_components=3)
    unified_history = train_likelihood(unified_model, unified, epochs=unified_epochs, seed=seed + 99, batch_size=512)
    save_checkpoint(unified_model, output_dir / "unified.pt")
    for family_id, row in enumerate(rows):
        family_mask = unified["family_id"] == family_id
        row["unified_validation_nll"] = mean_nll(unified_model, unified, split=1, extra_mask=family_mask)
        row["unified_test_nll"] = mean_nll(unified_model, unified, split=2, extra_mask=family_mask)
        row["unified_parameters"] = sum(parameter.numel() for parameter in unified_model.parameters())
    frame = pd.DataFrame(rows)
    frame["separate_gain_vs_baseline"] = frame["baseline_test_nll"] - frame["separate_test_nll"]
    frame["unified_gain_vs_baseline"] = frame["baseline_test_nll"] - frame["unified_test_nll"]
    frame.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "configuration": {
            "states_per_family": n_states, "replicates": replicates, "actions": 3,
            "symbols": n_symbols, "separate_epochs": separate_epochs, "unified_epochs": unified_epochs,
            "seed": seed,
        },
        "unified_best_validation_nll": min(unified_history["validation"]),
        "metrics": frame.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_benchmark(frame, output_dir / "comparison.png")
    return frame


def plot_benchmark(frame: pd.DataFrame, path: str | Path) -> Path:
    """Compare baseline, separate, and unified test NLL by channel family."""
    labels = [name.replace("_", "\n") for name in frame["family"]]
    x = np.arange(len(frame))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(x - width, frame["baseline_test_nll"], width, label="unconditional Gaussian")
    axis.bar(x, frame["separate_test_nll"], width, label="family-specific network")
    axis.bar(x + width, frame["unified_test_nll"], width, label="unified network")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("held-out test NLL (lower is better)")
    axis.set_title("Neural likelihoods across channel families")
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return destination
