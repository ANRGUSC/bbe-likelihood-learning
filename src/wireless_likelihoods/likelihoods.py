"""Torch likelihood baselines and a compact CPU training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


@dataclass(frozen=True)
class Normalizer:
    """Train-split normalization statistics for contexts and measurements."""

    context_mean: np.ndarray
    context_std: np.ndarray
    measurement_mean: np.ndarray
    measurement_std: np.ndarray

    @classmethod
    def fit(cls, context: np.ndarray, measurement: np.ndarray) -> "Normalizer":
        return cls(
            context.mean(axis=0).astype(np.float32),
            np.maximum(context.std(axis=0), 1e-6).astype(np.float32),
            measurement.mean(axis=0).astype(np.float32),
            np.maximum(measurement.std(axis=0), 1e-6).astype(np.float32),
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "context_mean": self.context_mean,
            "context_std": self.context_std,
            "measurement_mean": self.measurement_mean,
            "measurement_std": self.measurement_std,
        }


class ConditionalGaussianLikelihood(nn.Module):
    """Diagonal Gaussian model for `q(m | theta, action)` in raw units."""

    def __init__(self, theta_dim: int, measurement_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.theta_dim = theta_dim
        self.measurement_dim = measurement_dim
        self.hidden_dim = hidden_dim
        self.context_dim = theta_dim + 4
        self.net = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * measurement_dim),
        )
        self.register_buffer("context_mean", torch.zeros(self.context_dim))
        self.register_buffer("context_std", torch.ones(self.context_dim))
        self.register_buffer("measurement_mean", torch.zeros(measurement_dim))
        self.register_buffer("measurement_std", torch.ones(measurement_dim))

    def set_normalizer(self, normalizer: Normalizer) -> None:
        if normalizer.context_mean.shape != (self.context_dim,) or normalizer.measurement_mean.shape != (self.measurement_dim,):
            raise ValueError("Normalizer dimensions do not match this likelihood model.")
        self.context_mean.copy_(torch.as_tensor(normalizer.context_mean))
        self.context_std.copy_(torch.as_tensor(normalizer.context_std))
        self.measurement_mean.copy_(torch.as_tensor(normalizer.measurement_mean))
        self.measurement_std.copy_(torch.as_tensor(normalizer.measurement_std))

    def _as_tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.context_mean.device, dtype=torch.float32)
        return torch.as_tensor(value, device=self.context_mean.device, dtype=torch.float32)

    def context(self, theta: np.ndarray | torch.Tensor, action_features: np.ndarray | torch.Tensor) -> torch.Tensor:
        theta_tensor = self._as_tensor(theta)
        action_tensor = self._as_tensor(action_features)
        if theta_tensor.ndim == 1:
            theta_tensor = theta_tensor.unsqueeze(0)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if theta_tensor.shape[-1] != self.theta_dim or action_tensor.shape[-1] != 4:
            raise ValueError("theta or action_features has the wrong final dimension.")
        theta_batch = theta_tensor.shape[:-1]
        action_batch = action_tensor.shape[:-1]
        batch_shape = torch.broadcast_shapes(theta_batch, action_batch)
        theta_tensor = theta_tensor.expand(*batch_shape, self.theta_dim)
        action_tensor = action_tensor.expand(*batch_shape, 4)
        raw_context = torch.cat((theta_tensor, action_tensor), dim=-1)
        return (raw_context - self.context_mean) / self.context_std

    def _normalized_distribution(self, theta: np.ndarray | torch.Tensor, action_features: np.ndarray | torch.Tensor) -> Normal:
        out = self.net(self.context(theta, action_features))
        mean = out[..., : self.measurement_dim]
        log_std = torch.clamp(out[..., self.measurement_dim :], -5.0, 3.0)
        return Normal(mean, torch.exp(log_std))

    def distribution(self, theta: np.ndarray | torch.Tensor, action_features: np.ndarray | torch.Tensor) -> Normal:
        """Return the learned distribution in the original measurement units."""
        distribution = self._normalized_distribution(theta, action_features)
        return Normal(
            distribution.loc * self.measurement_std + self.measurement_mean,
            distribution.scale * self.measurement_std,
        )

    def log_prob(
        self,
        measurement: np.ndarray | torch.Tensor,
        theta: np.ndarray | torch.Tensor,
        action_features: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        measurement_tensor = self._as_tensor(measurement)
        if measurement_tensor.ndim == 1:
            measurement_tensor = measurement_tensor.unsqueeze(0)
        standardized = (measurement_tensor - self.measurement_mean) / self.measurement_std
        return self._normalized_distribution(theta, action_features).log_prob(standardized).sum(-1) - torch.log(self.measurement_std).sum()

    def marginal_density(self, values: np.ndarray, theta: np.ndarray, action_features: np.ndarray, dimension: int) -> np.ndarray:
        """Evaluate a one-dimensional learned marginal in raw measurement units."""
        distribution = self.distribution(theta, action_features)
        mean = distribution.loc.detach().cpu().numpy().reshape(-1, self.measurement_dim)[0, dimension]
        std = distribution.scale.detach().cpu().numpy().reshape(-1, self.measurement_dim)[0, dimension]
        values = np.asarray(values, dtype=np.float32)
        return np.exp(-0.5 * ((values - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


class ConditionalGaussianMixtureLikelihood(ConditionalGaussianLikelihood):
    """A compact mixture-density network for multimodal burst/fading telemetry."""

    def __init__(self, theta_dim: int, measurement_dim: int = 2, hidden_dim: int = 64, n_components: int = 3):
        nn.Module.__init__(self)
        self.theta_dim = theta_dim
        self.measurement_dim = measurement_dim
        self.hidden_dim = hidden_dim
        self.context_dim = theta_dim + 4
        self.n_components = n_components
        self.net = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_components * (1 + 2 * measurement_dim)),
        )
        self.register_buffer("context_mean", torch.zeros(self.context_dim))
        self.register_buffer("context_std", torch.ones(self.context_dim))
        self.register_buffer("measurement_mean", torch.zeros(measurement_dim))
        self.register_buffer("measurement_std", torch.ones(measurement_dim))

    def _mixture_parameters(self, theta, action_features):
        output = self.net(self.context(theta, action_features))
        batch = output.shape[:-1]
        output = output.reshape(*batch, self.n_components, 1 + 2 * self.measurement_dim)
        logits = output[..., 0]
        means = output[..., 1 : 1 + self.measurement_dim]
        log_stds = torch.clamp(output[..., 1 + self.measurement_dim :], -5.0, 3.0)
        return logits, means, torch.exp(log_stds)

    def distribution(self, theta, action_features) -> MixtureSameFamily:
        logits, means, stds = self._mixture_parameters(theta, action_features)
        raw_means = means * self.measurement_std + self.measurement_mean
        raw_stds = stds * self.measurement_std
        return MixtureSameFamily(Categorical(logits=logits), Independent(Normal(raw_means, raw_stds), 1))

    def log_prob(self, measurement, theta, action_features) -> torch.Tensor:
        measurement_tensor = self._as_tensor(measurement)
        if measurement_tensor.ndim == 1:
            measurement_tensor = measurement_tensor.unsqueeze(0)
        standardized = (measurement_tensor - self.measurement_mean) / self.measurement_std
        logits, means, stds = self._mixture_parameters(theta, action_features)
        component_log_prob = Normal(means, stds).log_prob(standardized.unsqueeze(-2)).sum(-1)
        return torch.logsumexp(torch.log_softmax(logits, dim=-1) + component_log_prob, dim=-1) - torch.log(self.measurement_std).sum()

    def marginal_density(self, values: np.ndarray, theta: np.ndarray, action_features: np.ndarray, dimension: int) -> np.ndarray:
        logits, means, stds = self._mixture_parameters(theta, action_features)
        weights = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1, self.n_components)[0]
        means = (means * self.measurement_std + self.measurement_mean).detach().cpu().numpy().reshape(-1, self.n_components, self.measurement_dim)[0, :, dimension]
        stds = (stds * self.measurement_std).detach().cpu().numpy().reshape(-1, self.n_components, self.measurement_dim)[0, :, dimension]
        values = np.asarray(values, dtype=np.float32)[:, None]
        component_density = np.exp(-0.5 * ((values - means) / stds) ** 2) / (stds * np.sqrt(2.0 * np.pi))
        return (component_density * weights).sum(axis=1)


def train_likelihood(
    model: ConditionalGaussianLikelihood,
    dataset: dict[str, np.ndarray],
    *,
    epochs: int = 120,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Fit the baseline with validation checkpoint selection on CPU."""
    torch.manual_seed(seed)
    context = np.concatenate((dataset["theta"], dataset["action_features"]), axis=1)
    target = dataset["measurement"][:, 1:]
    train_mask = dataset["split"] == 0
    val_mask = dataset["split"] == 1
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Dataset needs non-empty training and validation splits.")
    model.set_normalizer(Normalizer.fit(context[train_mask], target[train_mask]))
    train_indices = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_val = float("inf")
    history = {"train": [], "validation": []}
    theta = dataset["theta"]
    actions = dataset["action_features"]
    for _ in range(epochs):
        model.train()
        shuffled = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            optimizer.zero_grad()
            loss = -model.log_prob(target[indices], theta[indices], actions[indices]).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(-model.log_prob(target[val_mask], theta[val_mask], actions[val_mask]).mean())
        history["train"].append(float(np.mean(losses)))
        history["validation"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_checkpoint(model: ConditionalGaussianLikelihood, path: str | Path) -> Path:
    """Save architecture metadata and parameters for later plotting/inference."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"theta_dim": model.theta_dim, "measurement_dim": model.measurement_dim, "hidden_dim": model.hidden_dim, "model_type": type(model).__name__, "n_components": getattr(model, "n_components", None), "state_dict": model.state_dict()}, destination)
    return destination


def load_checkpoint(path: str | Path) -> ConditionalGaussianLikelihood:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") == "ConditionalGaussianMixtureLikelihood":
        model = ConditionalGaussianMixtureLikelihood(checkpoint["theta_dim"], checkpoint["measurement_dim"], hidden_dim=checkpoint.get("hidden_dim", 64), n_components=checkpoint["n_components"])
    else:
        model = ConditionalGaussianLikelihood(checkpoint["theta_dim"], checkpoint["measurement_dim"], hidden_dim=checkpoint.get("hidden_dim", 64))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
