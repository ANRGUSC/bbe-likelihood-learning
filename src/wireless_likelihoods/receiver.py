"""Decoder-agnostic stochastic receiver telemetry model."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import ActionSpec, ReceiverMeasurement


class ReceiverModel(ABC):
    """Maps raw simulator outcomes to observable receiver telemetry."""

    @abstractmethod
    def measure(self, raw_outcome: dict, action: ActionSpec, rng: np.random.Generator) -> ReceiverMeasurement:
        raise NotImplementedError


class AbstractReceiverModel(ReceiverModel):
    """A generic, stochastic decoder/receiver telemetry model.

    This is an abstract receiver telemetry model used to validate the
    simulation-to-likelihood architecture. It is not intended to model any
    particular decoder.
    """

    def __init__(self, difficulty_noise: float = 0.35, reliability_noise: float = 0.25, effort_noise: float = 0.15):
        self.difficulty_noise = difficulty_noise
        self.reliability_noise = reliability_noise
        self.effort_noise = effort_noise

    def measure(self, raw_outcome: dict, action: ActionSpec, rng: np.random.Generator) -> ReceiverMeasurement:
        error_rate = float(raw_outcome["error_rate"])
        burst_fraction = float(raw_outcome.get("burst_fraction", 0.0))
        correlation = float(raw_outcome.get("correlation_proxy", 0.0))
        difficulty = (
            10.0 * error_rate
            + 1.2 * burst_fraction
            + 0.35 * correlation
            - 3.0 * action.redundancy
            - 0.8 * action.decoder_strength
            - 0.4 * np.log1p(action.interleaver_depth)
            + rng.normal(0.0, self.difficulty_noise)
        )
        success_probability = _sigmoid(-difficulty)
        success = float(rng.random() < success_probability)
        reliability = float(np.clip(_sigmoid(-difficulty + rng.normal(0.0, self.reliability_noise)), 0.0, 1.0))
        effort_location = -0.4 + 5.0 * error_rate + 0.7 * action.decoder_strength + 0.2 * action.decoder_budget
        effort = float(np.logaddexp(0.0, effort_location) + rng.normal(0.0, self.effort_noise))
        return ReceiverMeasurement(success=success, reliability=reliability, effort=max(effort, 1e-6))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))
