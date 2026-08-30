"""Analytical channel and interference simulators used by the MVP."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import ActionSpec, ChannelState


class ChannelModel(ABC):
    """Interface for a simulator that produces raw channel outcomes."""

    @abstractmethod
    def simulate(
        self,
        state: ChannelState,
        action: ActionSpec,
        n_symbols: int,
        rng: np.random.Generator,
    ) -> dict:
        """Return errors and model-specific latent outcomes."""
        raise NotImplementedError


def _require(state: ChannelState, model: str, names: tuple[str, ...]) -> dict[str, float]:
    if state.model != model:
        raise ValueError(f"{model} received state for {state.model!r}.")
    missing = [name for name in names if name not in state.params]
    if missing:
        raise ValueError(f"Missing {model} parameters: {', '.join(missing)}")
    if any(not np.isfinite(float(state.params[name])) for name in names):
        raise ValueError("Channel parameters must be finite.")
    return {name: float(state.params[name]) for name in names}


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")


class IIDErrorChannel(ChannelModel):
    """Independent symbol errors with configurable probability."""

    def simulate(self, state: ChannelState, action: ActionSpec, n_symbols: int, rng: np.random.Generator) -> dict:
        params = _require(state, "iid", ("error_prob",))
        _validate_probability(params["error_prob"], "error_prob")
        _validate_n_symbols(n_symbols)
        errors = rng.random(n_symbols) < params["error_prob"]
        return _outcome(errors, effective_snr=_snr_from_error(params["error_prob"]))


class GilbertElliottChannel(ChannelModel):
    """Two-state good/bad Markov error process."""

    def simulate(self, state: ChannelState, action: ActionSpec, n_symbols: int, rng: np.random.Generator) -> dict:
        p = _require(state, "gilbert_elliott", ("p_good_to_bad", "p_bad_to_good", "error_good", "error_bad"))
        for name, value in p.items():
            _validate_probability(value, name)
        if p["p_good_to_bad"] + p["p_bad_to_good"] == 0:
            raise ValueError("At least one Gilbert-Elliott transition probability must be positive.")
        _validate_n_symbols(n_symbols)
        stationary_bad = p["p_good_to_bad"] / (p["p_good_to_bad"] + p["p_bad_to_good"])
        latent = np.empty(n_symbols, dtype=np.int8)
        latent[0] = int(rng.random() < stationary_bad)
        for index in range(1, n_symbols):
            if latent[index - 1]:
                latent[index] = int(not (rng.random() < p["p_bad_to_good"]))
            else:
                latent[index] = int(rng.random() < p["p_good_to_bad"])
        probs = np.where(latent == 1, p["error_bad"], p["error_good"])
        errors = rng.random(n_symbols) < probs
        return _outcome(errors, latent_state=latent, effective_snr=_snr_from_error(float(probs.mean())))


class FixedBurstChannel(ChannelModel):
    """IID background errors with at most one contiguous high-error burst per frame."""

    def simulate(self, state: ChannelState, action: ActionSpec, n_symbols: int, rng: np.random.Generator) -> dict:
        p = _require(state, "fixed_burst", ("background_error_prob", "burst_probability", "burst_length", "burst_error_prob"))
        for name in ("background_error_prob", "burst_probability", "burst_error_prob"):
            _validate_probability(p[name], name)
        _validate_n_symbols(n_symbols)
        burst_length = int(p["burst_length"])
        if burst_length != p["burst_length"] or not 1 <= burst_length <= n_symbols:
            raise ValueError("burst_length must be an integer in [1, n_symbols].")
        probs = np.full(n_symbols, p["background_error_prob"], dtype=float)
        burst_start: int | None = None
        if rng.random() < p["burst_probability"]:
            burst_start = int(rng.integers(0, n_symbols - burst_length + 1))
            probs[burst_start : burst_start + burst_length] = p["burst_error_prob"]
        errors = rng.random(n_symbols) < probs
        return _outcome(
            errors,
            burst_start=-1 if burst_start is None else burst_start,
            burst_length=burst_length if burst_start is not None else 0,
            burst_fraction=0.0 if burst_start is None else burst_length / n_symbols,
            effective_snr=_snr_from_error(float(probs.mean())),
        )


class MarkovInterferenceChannel(ChannelModel):
    """ON/OFF interferer, where p_on is OFF→ON and p_off is ON→OFF."""

    def simulate(self, state: ChannelState, action: ActionSpec, n_symbols: int, rng: np.random.Generator) -> dict:
        p = _require(state, "markov_interference", ("p_on", "p_off", "error_off", "error_on"))
        for name, value in p.items():
            _validate_probability(value, name)
        if p["p_on"] + p["p_off"] == 0:
            raise ValueError("At least one ON/OFF transition probability must be positive.")
        _validate_n_symbols(n_symbols)
        stationary_on = p["p_on"] / (p["p_on"] + p["p_off"])
        latent = np.empty(n_symbols, dtype=np.int8)
        latent[0] = int(rng.random() < stationary_on)
        for index in range(1, n_symbols):
            if latent[index - 1]:
                latent[index] = int(not (rng.random() < p["p_off"]))
            else:
                latent[index] = int(rng.random() < p["p_on"])
        probs = np.where(latent == 1, p["error_on"], p["error_off"])
        errors = rng.random(n_symbols) < probs
        return _outcome(errors, latent_state=latent, effective_snr=_snr_from_error(float(probs.mean())))


class LogGaussianFadingChannel(ChannelModel):
    """Log-SNR fading process with squared-exponential temporal correlation.

    The latent log-SNR trajectory has covariance
    ``std_log_snr**2 * exp(-0.5 * (lag / length_scale)**2)``.  This is a
    lightweight analytical proxy for temporally correlated fading, not a
    waveform-level propagation simulator.
    """

    def __init__(self) -> None:
        self._cholesky_cache: dict[tuple[int, float, float], np.ndarray] = {}

    def simulate(self, state: ChannelState, action: ActionSpec, n_symbols: int, rng: np.random.Generator) -> dict:
        p = _require(state, "log_gaussian_fading", ("mean_log_snr", "std_log_snr", "length_scale", "snr_offset"))
        _validate_n_symbols(n_symbols)
        if p["std_log_snr"] < 0 or p["length_scale"] <= 0:
            raise ValueError("std_log_snr must be nonnegative and length_scale must be positive.")
        cache_key = (n_symbols, p["std_log_snr"], p["length_scale"])
        cholesky = self._cholesky_cache.get(cache_key)
        if cholesky is None:
            positions = np.arange(n_symbols, dtype=float)
            lag = positions[:, None] - positions[None, :]
            kernel = (p["std_log_snr"] ** 2) * np.exp(-0.5 * (lag / p["length_scale"]) ** 2)
            # The tiny diagonal term also covers exactly zero fading variance.
            cholesky = np.linalg.cholesky(kernel + 1e-10 * np.eye(n_symbols))
            self._cholesky_cache[cache_key] = cholesky
        latent_log_snr = p["mean_log_snr"] + cholesky @ rng.standard_normal(n_symbols)
        snr = np.exp(latent_log_snr)
        error_probability = 1.0 / (1.0 + np.exp(np.clip(1.15 * (snr - p["snr_offset"]), -30.0, 30.0)))
        errors = rng.random(n_symbols) < error_probability
        return _outcome(
            errors,
            latent_log_snr=latent_log_snr,
            effective_snr=float(10.0 * np.log10(np.maximum(snr.mean(), 1e-8))),
            fading_correlation_length=p["length_scale"],
        )


def _validate_n_symbols(n_symbols: int) -> None:
    if not isinstance(n_symbols, int) or n_symbols < 1:
        raise ValueError("n_symbols must be a positive integer.")


def _snr_from_error(error_rate: float) -> float:
    return float(-10.0 * np.log10(max(error_rate, 1e-6)))


def _outcome(errors: np.ndarray, **extra: object) -> dict:
    error_rate = float(errors.mean())
    transitions = float(np.mean(errors[1:] != errors[:-1])) if len(errors) > 1 else 0.0
    return {"errors": errors, "error_rate": error_rate, "correlation_proxy": 1.0 - transitions, **extra}
