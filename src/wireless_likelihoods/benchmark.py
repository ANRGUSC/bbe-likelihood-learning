"""Training and evaluation utilities for per-family and unified likelihoods."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

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
        torch.manual_seed(seed + family_id)
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
    torch.manual_seed(seed + 99)
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


def run_comparison_seed(
    *, seed: int, n_states: int = 48, replicates: int = 20, n_symbols: int = 64,
    separate_epochs: int = 100, unified_epochs: int = 140,
) -> pd.DataFrame:
    """Train all models for one independently generated experimental seed."""
    datasets = generate_family_datasets(
        n_states=n_states, replicates=replicates, n_symbols=n_symbols, seed=seed,
    )
    rows = []
    for family_id, family in enumerate(CHANNEL_FAMILIES):
        dataset = datasets[family]
        torch.manual_seed(seed + family_id)
        model = ConditionalGaussianMixtureLikelihood(dataset["theta"].shape[1], hidden_dim=64, n_components=3)
        train_likelihood(model, dataset, epochs=separate_epochs, seed=seed + family_id)
        rows.append({
            "seed": seed,
            "family": family,
            "baseline_test_nll": unconditional_gaussian_nll(dataset),
            "separate_test_nll": mean_nll(model, dataset, split=2),
        })
    unified = combine_family_datasets(datasets)
    torch.manual_seed(seed + 99)
    unified_model = ConditionalGaussianMixtureLikelihood(unified["theta"].shape[1], hidden_dim=96, n_components=3)
    train_likelihood(unified_model, unified, epochs=unified_epochs, seed=seed + 99, batch_size=512)
    for family_id, row in enumerate(rows):
        family_mask = unified["family_id"] == family_id
        row["unified_test_nll"] = mean_nll(unified_model, unified, split=2, extra_mask=family_mask)
        row["joint_gain"] = row["separate_test_nll"] - row["unified_test_nll"]
    return pd.DataFrame(rows)


def summarize_repeated_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Compute Student-t 95% confidence intervals and paired significance tests."""
    required = {"seed", "family", "baseline_test_nll", "separate_test_nll", "unified_test_nll", "joint_gain"}
    if not required.issubset(per_seed.columns):
        raise ValueError(f"per_seed is missing columns: {sorted(required - set(per_seed.columns))}")
    family_order = list(CHANNEL_FAMILIES) + ["macro_average"]
    macro = per_seed.groupby("seed", as_index=False)[
        ["baseline_test_nll", "separate_test_nll", "unified_test_nll", "joint_gain"]
    ].mean()
    macro["family"] = "macro_average"
    augmented = pd.concat([per_seed, macro], ignore_index=True)
    rows = []
    for family in family_order:
        subset = augmented[augmented["family"] == family]
        row: dict[str, float | int | str] = {"family": family, "n_seeds": int(subset["seed"].nunique())}
        for output_name, column in (
            ("baseline", "baseline_test_nll"),
            ("separate", "separate_test_nll"),
            ("unified", "unified_test_nll"),
            ("joint_gain", "joint_gain"),
        ):
            values = subset[column].to_numpy(dtype=float)
            mean = float(values.mean())
            half_width = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)) if len(values) > 1 else float("nan")
            row[f"{output_name}_mean"] = mean
            row[f"{output_name}_ci_low"] = mean - half_width
            row[f"{output_name}_ci_high"] = mean + half_width
            row[f"{output_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        gains = subset["joint_gain"].to_numpy(dtype=float)
        row["joint_gain_p_value_two_sided"] = float(stats.ttest_1samp(gains, popmean=0.0).pvalue)
        row["joint_win_rate"] = float(np.mean(gains > 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def run_repeated_benchmark(
    *, output_dir: str | Path = "results/repeated_benchmark", n_seeds: int = 30,
    start_seed: int = 21, n_states: int = 48, replicates: int = 20, n_symbols: int = 64,
    separate_epochs: int = 100, unified_epochs: int = 140, resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a resumable multi-seed comparison and write raw and aggregate results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / "per_seed.csv"
    expected_seeds = list(range(start_seed, start_seed + n_seeds))
    if resume and per_seed_path.exists():
        per_seed = pd.read_csv(per_seed_path)
    else:
        per_seed = pd.DataFrame()
    completed = set(per_seed["seed"].unique()) if len(per_seed) else set()
    for run_index, seed in enumerate(expected_seeds, start=1):
        if seed in completed:
            print(f"[{run_index:02d}/{n_seeds}] seed {seed} already complete", flush=True)
            continue
        result = run_comparison_seed(
            seed=seed, n_states=n_states, replicates=replicates, n_symbols=n_symbols,
            separate_epochs=separate_epochs, unified_epochs=unified_epochs,
        )
        per_seed = pd.concat([per_seed, result], ignore_index=True)
        per_seed = per_seed.sort_values(["seed", "family"])
        per_seed.to_csv(per_seed_path, index=False)
        macro_gain = result["joint_gain"].mean()
        print(f"[{run_index:02d}/{n_seeds}] seed {seed} complete, mean joint gain {macro_gain:+.4f}", flush=True)
    per_seed = per_seed[per_seed["seed"].isin(expected_seeds)].copy()
    counts = per_seed.groupby("seed")["family"].nunique()
    if len(counts) != n_seeds or not np.all(counts.to_numpy() == len(CHANNEL_FAMILIES)):
        raise RuntimeError("Repeated benchmark is incomplete; rerun with resume enabled.")
    summary = summarize_repeated_results(per_seed)
    summary.to_csv(output_dir / "summary.csv", index=False)
    n_train = max(1, int(round(0.70 * n_states)))
    n_validation = max(1, int(round(0.15 * n_states)))
    n_train = min(n_train, n_states - 2)
    configuration = {
        "n_seeds": n_seeds, "start_seed": start_seed, "states_per_family": n_states,
        "replicates_per_state_action": replicates, "actions": 3, "symbols_per_frame": n_symbols,
        "separate_epochs": separate_epochs, "unified_epochs": unified_epochs,
        "split": {
            "train": n_train,
            "validation": n_validation,
            "test": n_states - n_train - n_validation,
        },
        "confidence_interval": "two-sided 95% Student-t interval across independent seeds",
    }
    (output_dir / "summary.json").write_text(
        json.dumps({"configuration": configuration, "summary": summary.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    plot_repeated_benchmark(summary, output_dir / "comparison_30_seeds.png")
    return per_seed, summary


def plot_repeated_benchmark(summary: pd.DataFrame, path: str | Path) -> Path:
    """Plot mean NLLs and paired joint-learning gains with 95% intervals."""
    plot_data = summary[summary["family"] != "macro_average"].reset_index(drop=True)
    labels = [name.replace("gilbert_elliott", "Gilbert-Elliott").replace("_", "\n") for name in plot_data["family"]]
    x = np.arange(len(plot_data))
    width = 0.25
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.7, 1.0]})
    for offset, key, label in (
        (-width, "baseline", "unconditional Gaussian"),
        (0.0, "separate", "family-specific network"),
        (width, "unified", "unified network"),
    ):
        mean = plot_data[f"{key}_mean"].to_numpy()
        error = np.vstack((mean - plot_data[f"{key}_ci_low"], plot_data[f"{key}_ci_high"] - mean))
        left.bar(x + offset, mean, width, yerr=error, capsize=3, label=label)
    left.axhline(0.0, color="black", linewidth=0.8)
    left.set_xticks(x, labels)
    left.set_ylabel("held-out test NLL (lower is better)")
    n_seeds = int(plot_data["n_seeds"].iloc[0])
    left.set_title(f"Mean performance across {n_seeds} seeds")
    left.legend(frameon=False, fontsize=8)
    gain = plot_data["joint_gain_mean"].to_numpy()
    gain_error = np.vstack((gain - plot_data["joint_gain_ci_low"], plot_data["joint_gain_ci_high"] - gain))
    right.errorbar(gain, x, xerr=gain_error, fmt="o", capsize=4, color="tab:green")
    right.axvline(0.0, color="black", linewidth=0.8)
    right.set_yticks(x, labels)
    right.set_xlabel("separate NLL minus unified NLL")
    right.set_title("Paired gain from joint training")
    figure.tight_layout()
    destination = Path(path)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
