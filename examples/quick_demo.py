"""Run the complete action-conditioned neural likelihood MVP on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wireless_likelihoods.actions import ACTIONS
from wireless_likelihoods.dataset import generate_dataset, sample_fixed_burst_states, save_dataset, simulate_once, theta_from_state
from wireless_likelihoods.likelihoods import ConditionalGaussianMixtureLikelihood, save_checkpoint, train_likelihood
from wireless_likelihoods.plotting import plot_action_conditioned_likelihoods, plot_posterior_updates
from wireless_likelihoods.receiver import AbstractReceiverModel
from wireless_likelihoods.simulation import FixedBurstChannel


def main() -> None:
    seed = 7
    n_symbols = 128
    rng = np.random.default_rng(seed)
    channel = FixedBurstChannel()
    receiver = AbstractReceiverModel()
    states = sample_fixed_burst_states(60, rng, n_symbols)
    dataset = generate_dataset(channel, states, replicates=40, n_symbols=n_symbols, seed=seed, receiver=receiver)
    save_dataset(dataset, "data/quick_demo_dataset.npz")

    model = ConditionalGaussianMixtureLikelihood(theta_dim=dataset["theta"].shape[1], n_components=3)
    history = train_likelihood(model, dataset, epochs=120, batch_size=256, seed=seed)
    save_checkpoint(model, "results/likelihood.pt")

    test_ids = np.flatnonzero(dataset["state_split"] == 2)
    # A moderate burst probability makes the sequential grid update easy to read.
    test_state_id = int(min(test_ids, key=lambda index: abs(states[index].params["burst_probability"] - 0.13)))
    held_out_state = states[test_state_id]
    likelihood_plot = plot_action_conditioned_likelihoods(
        model, channel, held_out_state, ACTIONS, receiver=receiver, n_samples=700,
        n_symbols=n_symbols, path="results/likelihood_fit.png",
    )

    theta_grid = np.repeat(theta_from_state(held_out_state, n_symbols)[None, :], 100, axis=0)
    theta_grid[:, 1] = np.linspace(0.01, 0.5, len(theta_grid))
    observation_rng = np.random.default_rng(seed + 1)
    inference_action = ACTIONS[1]
    observations = np.asarray([
        simulate_once(channel, held_out_state, inference_action, observation_rng, receiver=receiver, n_symbols=n_symbols)["measurement"][1:]
        for _ in range(10)
    ])
    posterior_plot = plot_posterior_updates(
        model, theta_grid, inference_action, observations,
        float(held_out_state.params["burst_probability"]), path="results/posterior_update.png",
    )
    print(f"Completed CPU demo. Final validation NLL: {history['validation'][-1]:.3f}")
    print(f"Wrote {likelihood_plot} and {posterior_plot}")


if __name__ == "__main__":
    main()
