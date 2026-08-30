"""Matplotlib visualizations used by scripts and the quick demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import simulate_once
from .inference import posterior_sequence
from .receiver import ReceiverModel
from .types import ActionSpec, ChannelState


def plot_action_conditioned_likelihoods(
    model,
    channel_model,
    state: ChannelState,
    actions: tuple[ActionSpec, ...],
    *,
    receiver: ReceiverModel,
    n_samples: int = 1000,
    n_symbols: int = 128,
    seed: int = 100,
    path: str | Path = "results/likelihood_fit.png",
) -> Path:
    """Compare held-out simulator effort distributions to learned marginals."""
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(1, len(actions), figsize=(4.2 * len(actions), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    theta = simulate_once(channel_model, state, actions[0], rng, receiver=receiver, n_symbols=n_symbols)["theta"]
    x = np.linspace(0.0, 2.5, 300, dtype=np.float32)
    for axis, action in zip(axes, actions):
        observations = np.asarray(
            [simulate_once(channel_model, state, action, rng, receiver=receiver, n_symbols=n_symbols)["measurement"][2] for _ in range(n_samples)]
        )
        action_features = np.asarray(action.features(), dtype=np.float32)
        density = model.marginal_density(x, theta, action_features, dimension=1)
        axis.hist(observations, bins=30, density=True, alpha=0.55, label="simulator")
        axis.plot(x, density, color="black", linewidth=2, label="learned likelihood")
        axis.set_title(action.name.replace("_", " "))
        axis.set_xlabel("log1p(effort)")
    axes[0].set_ylabel("density")
    axes[-1].legend(frameon=False)
    figure.suptitle("Action-conditioned held-out likelihoods", y=1.02)
    figure.tight_layout()
    return _save(figure, path)


def plot_posterior_updates(
    model,
    theta_grid: np.ndarray,
    action: ActionSpec,
    measurements: np.ndarray,
    true_value: float,
    *,
    path: str | Path = "results/posterior_update.png",
) -> Path:
    """Plot prior and selected sequential posteriors for a one-dimensional grid."""
    sequence = posterior_sequence(model, theta_grid, action, measurements)
    figure, axis = plt.subplots(figsize=(7, 3.8))
    x = theta_grid[:, 1]
    selected = [(0, "prior"), (1, "after 1"), (min(5, len(measurements)), "after 5"), (len(measurements), f"after {len(measurements)}")]
    for index, label in selected:
        axis.plot(x, sequence[index], label=label)
    axis.axvline(true_value, color="black", linestyle="--", label="true burst probability")
    axis.set(xlabel="burst probability", ylabel="posterior mass", title="Bayesian update using learned likelihood")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    return _save(figure, path)


def _save(figure, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination
