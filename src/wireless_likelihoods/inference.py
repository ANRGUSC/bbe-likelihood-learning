"""Small grid-based Bayesian updates using a learned likelihood."""

from __future__ import annotations

import numpy as np
import torch

from .types import ActionSpec


def bayesian_update(
    model,
    theta_grid: np.ndarray,
    action: ActionSpec,
    measurement: np.ndarray,
    log_prior: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Apply one stable Bayesian update over a discrete theta grid.

    ``theta_grid`` is shape ``(n_grid, theta_dim)``. The returned posterior is
    normalized in probability space and can be supplied as the next prior in
    log form.
    """
    theta_grid = np.asarray(theta_grid, dtype=np.float32)
    if theta_grid.ndim != 2:
        raise ValueError("theta_grid must have shape (n_grid, theta_dim).")
    if log_prior is None:
        log_prior_tensor = torch.full((len(theta_grid),), -np.log(len(theta_grid)), dtype=torch.float32)
    else:
        log_prior_tensor = torch.as_tensor(log_prior, dtype=torch.float32)
        if log_prior_tensor.shape != (len(theta_grid),):
            raise ValueError("log_prior must have one entry per grid point.")
    action_features = np.repeat(np.asarray(action.features(), dtype=np.float32)[None, :], len(theta_grid), axis=0)
    observations = np.repeat(np.asarray(measurement, dtype=np.float32)[None, :], len(theta_grid), axis=0)
    model.eval()
    with torch.no_grad():
        log_posterior = log_prior_tensor + model.log_prob(observations, theta_grid, action_features)
        return torch.softmax(log_posterior, dim=0).cpu().numpy()


def posterior_sequence(model, theta_grid: np.ndarray, action: ActionSpec, measurements: np.ndarray) -> list[np.ndarray]:
    """Return prior and posterior after every observation."""
    prior = np.full(len(theta_grid), 1.0 / len(theta_grid), dtype=np.float32)
    outputs = [prior]
    for measurement in measurements:
        prior = bayesian_update(model, theta_grid, action, measurement, np.log(np.maximum(prior, 1e-30)))
        outputs.append(prior)
    return outputs
