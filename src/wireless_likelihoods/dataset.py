"""Reproducible simulation datasets for conditional likelihood learning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .actions import ACTIONS
from .receiver import AbstractReceiverModel, ReceiverModel
from .simulation import ChannelModel
from .types import ActionSpec, ChannelState


THETA_KEYS = {
    "iid": ("error_prob",),
    "gilbert_elliott": ("p_good_to_bad", "p_bad_to_good", "error_good", "error_bad"),
    "fixed_burst": ("background_error_prob", "burst_probability", "burst_length", "burst_error_prob"),
    "markov_interference": ("p_on", "p_off", "error_off", "error_on"),
    "log_gaussian_fading": ("mean_log_snr", "std_log_snr", "length_scale", "snr_offset"),
}

CHANNEL_FAMILIES = tuple(THETA_KEYS)
MAX_THETA_DIM = max(len(keys) for keys in THETA_KEYS.values())


def theta_from_state(state: ChannelState, n_symbols: int = 256) -> np.ndarray:
    """Return the stable numeric theta representation for one channel family."""
    if state.model not in THETA_KEYS:
        raise ValueError(f"Unsupported channel model {state.model!r}.")
    values = [float(state.params[key]) for key in THETA_KEYS[state.model]]
    if state.model == "fixed_burst":
        values[2] /= n_symbols
    elif state.model == "log_gaussian_fading":
        values[2] /= n_symbols
    return np.asarray(values, dtype=np.float32)


def simulate_once(
    channel_model: ChannelModel,
    channel_state: ChannelState,
    action: ActionSpec,
    rng: np.random.Generator,
    *,
    receiver: ReceiverModel | None = None,
    n_symbols: int = 256,
) -> dict:
    """Simulate one `(theta, action, measurement)` sample."""
    receiver = receiver or AbstractReceiverModel()
    raw = channel_model.simulate(channel_state, action, n_symbols, rng)
    measurement = receiver.measure(raw, action, rng)
    return {
        "theta": theta_from_state(channel_state, n_symbols),
        "action_id": action.action_id,
        "action_features": np.asarray(action.features(), dtype=np.float32),
        "measurement": np.asarray(
            [measurement.success, *measurement.continuous_features()], dtype=np.float32
        ),
        "raw": raw,
    }


def sample_fixed_burst_states(n_states: int, rng: np.random.Generator, n_symbols: int = 256) -> list[ChannelState]:
    """Sample broad but valid fixed-burst states for the default MVP dataset."""
    if n_states < 3:
        raise ValueError("n_states must be at least 3 to support train/validation/test splits.")
    states = []
    for _ in range(n_states):
        states.append(
            ChannelState(
                "fixed_burst",
                {
                    "background_error_prob": float(rng.uniform(0.002, 0.04)),
                    "burst_probability": float(rng.uniform(0.01, 0.5)),
                    "burst_length": float(rng.integers(max(2, n_symbols // 32), max(3, n_symbols // 4) + 1)),
                    "burst_error_prob": float(rng.uniform(0.15, 0.55)),
                },
            )
        )
    return states


def sample_iid_states(n_states: int, rng: np.random.Generator, n_symbols: int = 256) -> list[ChannelState]:
    """Sample IID error probabilities over the intended MVP range."""
    _validate_state_count(n_states)
    return [ChannelState("iid", {"error_prob": float(rng.uniform(0.001, 0.3))}) for _ in range(n_states)]


def sample_gilbert_elliott_states(n_states: int, rng: np.random.Generator, n_symbols: int = 256) -> list[ChannelState]:
    """Sample stable good/bad Markov channels with separated error regimes."""
    _validate_state_count(n_states)
    return [
        ChannelState(
            "gilbert_elliott",
            {
                "p_good_to_bad": float(rng.uniform(0.005, 0.08)),
                "p_bad_to_good": float(rng.uniform(0.08, 0.4)),
                "error_good": float(rng.uniform(0.001, 0.03)),
                "error_bad": float(rng.uniform(0.12, 0.45)),
            },
        )
        for _ in range(n_states)
    ]


def sample_markov_interference_states(n_states: int, rng: np.random.Generator, n_symbols: int = 256) -> list[ChannelState]:
    """Sample ON/OFF interferer dynamics and state-conditioned error rates."""
    _validate_state_count(n_states)
    return [
        ChannelState(
            "markov_interference",
            {
                "p_on": float(rng.uniform(0.01, 0.15)),
                "p_off": float(rng.uniform(0.08, 0.4)),
                "error_off": float(rng.uniform(0.001, 0.03)),
                "error_on": float(rng.uniform(0.12, 0.5)),
            },
        )
        for _ in range(n_states)
    ]


def sample_log_gaussian_fading_states(n_states: int, rng: np.random.Generator, n_symbols: int = 256) -> list[ChannelState]:
    """Sample RBF-correlated log-Gaussian fading configurations."""
    _validate_state_count(n_states)
    return [
        ChannelState(
            "log_gaussian_fading",
            {
                "mean_log_snr": float(rng.uniform(-0.2, 1.5)),
                "std_log_snr": float(rng.uniform(0.15, 0.8)),
                "length_scale": float(rng.uniform(2.0, max(2.1, n_symbols / 3))),
                "snr_offset": float(rng.uniform(0.8, 1.5)),
            },
        )
        for _ in range(n_states)
    ]


def _validate_state_count(n_states: int) -> None:
    if n_states < 3:
        raise ValueError("n_states must be at least 3.")


def unified_theta(theta: np.ndarray, family: str) -> np.ndarray:
    """Encode native theta with family one-hot plus zero-padded parameters."""
    if family not in CHANNEL_FAMILIES:
        raise ValueError(f"Unknown channel family {family!r}.")
    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim != 2 or theta.shape[1] != len(THETA_KEYS[family]):
        raise ValueError("theta width does not match the selected channel family.")
    encoded = np.zeros((len(theta), len(CHANNEL_FAMILIES) + MAX_THETA_DIM), dtype=np.float32)
    encoded[:, CHANNEL_FAMILIES.index(family)] = 1.0
    encoded[:, len(CHANNEL_FAMILIES) : len(CHANNEL_FAMILIES) + theta.shape[1]] = theta
    return encoded


def combine_family_datasets(datasets: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Combine native-family datasets for a single family-conditioned model."""
    missing = set(CHANNEL_FAMILIES) - set(datasets)
    if missing:
        raise ValueError(f"Missing channel-family datasets: {', '.join(sorted(missing))}")
    combined: dict[str, list[np.ndarray]] = {
        "theta": [], "action_id": [], "action_features": [], "measurement": [],
        "state_id": [], "split": [], "family_id": [],
    }
    state_offset = 0
    for family_id, family in enumerate(CHANNEL_FAMILIES):
        dataset = datasets[family]
        combined["theta"].append(unified_theta(dataset["theta"], family))
        for key in ("action_id", "action_features", "measurement", "split"):
            combined[key].append(dataset[key])
        combined["state_id"].append(dataset["state_id"] + state_offset)
        combined["family_id"].append(np.full(len(dataset["theta"]), family_id, dtype=np.int8))
        state_offset += int(dataset["state_id"].max()) + 1
    result = {key: np.concatenate(values, axis=0) for key, values in combined.items()}
    result["model"] = np.asarray("unified")
    result["n_symbols"] = np.asarray(next(iter(datasets.values()))["n_symbols"])
    return result


def grouped_split(n_states: int, rng: np.random.Generator) -> np.ndarray:
    """Assign whole theta points to train (0), validation (1), or test (2)."""
    if n_states < 7:
        raise ValueError("Use at least 7 states for a 70/15/15 grouped split.")
    indices = rng.permutation(n_states)
    n_train = max(1, int(round(0.70 * n_states)))
    n_val = max(1, int(round(0.15 * n_states)))
    n_train = min(n_train, n_states - 2)
    split = np.empty(n_states, dtype=np.int8)
    split[indices[:n_train]] = 0
    split[indices[n_train : n_train + n_val]] = 1
    split[indices[n_train + n_val :]] = 2
    return split


def generate_dataset(
    channel_model: ChannelModel,
    states: Iterable[ChannelState],
    *,
    actions: tuple[ActionSpec, ...] = ACTIONS,
    replicates: int = 100,
    n_symbols: int = 256,
    seed: int = 0,
    receiver: ReceiverModel | None = None,
) -> dict[str, np.ndarray]:
    """Generate a grouped, repeated-simulation dataset in memory."""
    state_list = list(states)
    if not state_list:
        raise ValueError("At least one channel state is required.")
    if replicates < 1:
        raise ValueError("replicates must be positive.")
    if len({state.model for state in state_list}) != 1:
        raise ValueError("A dataset contains one channel family so theta has a fixed dimension.")
    rng = np.random.default_rng(seed)
    receiver = receiver or AbstractReceiverModel()
    state_splits = grouped_split(len(state_list), rng)
    total = len(state_list) * len(actions) * replicates
    theta_dim = len(theta_from_state(state_list[0], n_symbols))
    theta = np.empty((total, theta_dim), dtype=np.float32)
    action_ids = np.empty(total, dtype=np.int64)
    action_features = np.empty((total, 4), dtype=np.float32)
    measurement = np.empty((total, 3), dtype=np.float32)
    state_ids = np.empty(total, dtype=np.int64)
    split = np.empty(total, dtype=np.int8)
    index = 0
    for state_id, state in enumerate(state_list):
        for action in actions:
            for _ in range(replicates):
                sample = simulate_once(channel_model, state, action, rng, receiver=receiver, n_symbols=n_symbols)
                theta[index] = sample["theta"]
                action_ids[index] = sample["action_id"]
                action_features[index] = sample["action_features"]
                measurement[index] = sample["measurement"]
                state_ids[index] = state_id
                split[index] = state_splits[state_id]
                index += 1
    return {
        "theta": theta,
        "action_id": action_ids,
        "action_features": action_features,
        "measurement": measurement,
        "state_id": state_ids,
        "split": split,
        "state_split": state_splits,
        "model": np.asarray(state_list[0].model),
        "n_symbols": np.asarray(n_symbols),
    }


def save_dataset(dataset: dict[str, np.ndarray], path: str | Path) -> Path:
    """Persist a generated dataset as a portable compressed NPZ archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **dataset)
    return destination
