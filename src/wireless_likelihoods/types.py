"""Small public data types shared by simulation, learning, and inference."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ChannelState:
    """Physical/environmental state known to the simulator."""

    model: str
    params: Mapping[str, float]


@dataclass(frozen=True)
class ActionSpec:
    """Structured, decoder-agnostic communication action."""

    action_id: int
    name: str
    redundancy: float
    interleaver_depth: int
    decoder_strength: float
    decoder_budget: float

    def features(self) -> tuple[float, float, float, float]:
        """Numeric action representation consumed by likelihood models."""
        return (
            self.redundancy,
            float(self.interleaver_depth),
            self.decoder_strength,
            self.decoder_budget,
        )


@dataclass(frozen=True)
class ReceiverMeasurement:
    """Telemetry observable at a generic receiver."""

    success: float
    reliability: float
    effort: float

    def continuous_features(self) -> tuple[float, float]:
        """Continuous likelihood target: reliability and log receiver effort."""
        import math

        return (self.reliability, math.log1p(self.effort))
