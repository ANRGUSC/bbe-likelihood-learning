# MVP Implementation Brief: Action-Conditioned Neural Likelihoods for Wireless Adaptation

## Goal

Build a small, clean, open-source Python repository that demonstrates the full pipeline

\[
(\theta, a) \rightarrow \text{simulator} \rightarrow m
\rightarrow q_\phi(m\mid \theta,a)
\rightarrow \text{Bayesian update}.
\]

The repository should be credible as an early implementation of **simulation-based action-conditioned neural likelihood estimation** for adaptive wireless systems.

The MVP does **not** require proprietary decoder software, Sionna, ns-3, real 5G NR, or any external communications simulator.

The purpose of v0.1 is to show that the statistical and software interfaces are correct and extensible.

---

# 1. Core Scientific Object

Let

- \(\theta\): channel/interference state parameters
- \(a\): selected communication action
- \(m\): receiver-side measurement/telemetry
- \(q_\phi(m\mid\theta,a)\): learned conditional likelihood

The simulator provides samples

\[
m \sim p_{\text{sim}}(m\mid\theta,a).
\]

The neural model learns

\[
q_\phi(m\mid\theta,a)
\approx
p_{\text{sim}}(m\mid\theta,a).
\]

The learned likelihood can then be used in a Bayesian update

\[
p(\theta\mid m,a)
\propto
q_\phi(m\mid\theta,a)p(\theta).
\]

This is the only concept the MVP must establish convincingly.

---

# 2. Scope for v0.1

Implement:

1. A common `ChannelModel` interface.
2. A common `ActionSpec` abstraction.
3. A common `ReceiverModel` abstraction.
4. Four simple channel/interference models.
5. Three abstract communication actions.
6. A receiver that produces realistic-looking but generic telemetry.
7. Dataset generation for tuples `(theta, action, measurement)`.
8. A conditional Gaussian neural likelihood.
9. Optionally, a conditional Gaussian mixture likelihood.
10. One Bayesian posterior-update example.
11. Unit tests.
12. A README with one end-to-end figure.

Do **not** implement:

- proprietary decoder integrations
- Sionna
- 5G NR
- LDPC / Polar / Turbo coding
- ns-3
- ray tracing
- normalizing flows
- SBI libraries
- distributed simulation
- GPU requirements

Those belong on the roadmap.

---

# 3. Recommended Repository Name

Suggested working name:

`wireless-neural-likelihoods`

Alternative:

`acnl-wireless`

where ACNL = Action-Conditioned Neural Likelihoods.

Avoid DARPA-specific naming in the public repository.

---

# 4. Minimal Dependencies

`requirements.txt`:

```text
numpy
scipy
torch
matplotlib
pandas
pytest
```

Optional later:

```text
pyyaml
```

No other dependencies should be required for the MVP.

---

# 5. Repository Structure

```text
wireless-neural-likelihoods/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── src/
│   └── wireless_likelihoods/
│       ├── __init__.py
│       ├── types.py
│       ├── simulation.py
│       ├── actions.py
│       ├── receiver.py
│       ├── dataset.py
│       ├── likelihoods.py
│       └── inference.py
│
├── scripts/
│   ├── generate_dataset.py
│   ├── train_likelihood.py
│   ├── plot_likelihood.py
│   └── bayes_demo.py
│
├── examples/
│   └── quick_demo.py
│
├── tests/
│   ├── test_channels.py
│   ├── test_actions.py
│   ├── test_dataset.py
│   ├── test_likelihood.py
│   └── test_bayes.py
│
└── results/
    └── .gitkeep
```

Keep the source tree shallow for v0.1.

---

# 6. Core Data Types

Create `types.py`.

Use dataclasses.

```python
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ChannelState:
    """Physical/environmental state known to the simulator."""
    model: str
    params: Dict[str, float]


@dataclass(frozen=True)
class ActionSpec:
    """Abstract code/decoder action.

    The MVP deliberately does not require a real FEC implementation.
    """
    action_id: int
    name: str

    # Generic structured descriptors.
    redundancy: float
    interleaver_depth: int
    decoder_strength: float
    decoder_budget: float


@dataclass
class ReceiverMeasurement:
    """Telemetry assumed observable at the receiver."""
    success: float
    reliability: float
    effort: float
```

Important design principle:

`ActionSpec` must **not** be just an integer.

The structured features allow future replacement by

- code family
- block length
- rate
- interleaver depth
- decoder family
- query budget
- stopping rule

without changing the likelihood-learning API.

---

# 7. Abstract Channel Interface

Create `simulation.py`.

```python
from abc import ABC, abstractmethod
import numpy as np

from .types import ChannelState, ActionSpec


class ChannelModel(ABC):

    @abstractmethod
    def simulate(
        self,
        state: ChannelState,
        action: ActionSpec,
        n_symbols: int,
        rng: np.random.Generator,
    ) -> dict:
        """Return latent/raw channel outcomes used by the receiver model."""
        raise NotImplementedError
```

The returned dictionary may contain generic quantities such as

```python
{
    "error_rate": ...,
    "burst_fraction": ...,
    "effective_snr": ...,
    "correlation": ...
}
```

The likelihood learner must never depend on the internals of the channel model.

---

# 8. Initial Channel / Interference Models

Implement exactly four for v0.1.

## 8.1 IID Channel

Parameters:

```text
error_prob ∈ [0.001, 0.3]
```

Generate IID bit-error indicators.

This is the simplest sanity test.

Class:

```python
IIDErrorChannel
```

---

## 8.2 Gilbert-Elliott Channel

Two states:

```text
GOOD
BAD
```

Parameters:

```text
p_good_to_bad
p_bad_to_good
error_good
error_bad
```

Reasonable defaults:

```text
p_good_to_bad = 0.02
p_bad_to_good = 0.20
error_good = 0.005
error_bad = 0.25
```

Generate a Markov state sequence and then errors conditioned on state.

Class:

```python
GilbertElliottChannel
```

This is the most important non-IID MVP model.

---

## 8.3 Fixed Burst Channel

Parameters:

```text
background_error_prob
burst_probability
burst_length
burst_error_prob
```

For each frame:

1. generate background IID errors;
2. with probability `burst_probability`, choose a random burst start;
3. replace error probability over `burst_length` symbols by `burst_error_prob`.

Class:

```python
FixedBurstChannel
```

This makes interleaving effects visually obvious.

---

## 8.4 Markov Interference Channel

Hidden state:

```text
OFF
ON
```

Parameters:

```text
p_on
p_off
error_off
error_on
```

Equivalent to an ON/OFF interferer.

Class:

```python
MarkovInterferenceChannel
```

This can later evolve naturally into multiple interferers, varying INR, fading, or real waveform simulation.

---

# 9. Abstract Actions

The MVP should have three actions.

They are **not** claimed to be real codes.

They represent generic code/decoder configurations with different redundancy, interleaving, decoding strength, and computational cost.

Example:

```python
ACTIONS = [
    ActionSpec(
        action_id=0,
        name="fast_low_redundancy",
        redundancy=0.10,
        interleaver_depth=1,
        decoder_strength=0.7,
        decoder_budget=0.5,
    ),
    ActionSpec(
        action_id=1,
        name="interleaved",
        redundancy=0.20,
        interleaver_depth=8,
        decoder_strength=1.0,
        decoder_budget=1.0,
    ),
    ActionSpec(
        action_id=2,
        name="robust",
        redundancy=0.35,
        interleaver_depth=16,
        decoder_strength=1.5,
        decoder_budget=2.0,
    ),
]
```

The key requirement is that actions change the **distribution of receiver measurements**.

---

# 10. Generic Receiver Model

Create `receiver.py`.

This provides abstract decoder behavior for the MVP.

It should be explicitly described as an **abstract decoder/receiver telemetry model**.

Input:

```text
raw channel outcome
action
```

Output:

```text
ReceiverMeasurement
```

Recommended telemetry:

```text
success
reliability
effort
```

where:

### success

Binary decoded-success indicator.

### reliability

Continuous value in `[0,1]`.

Interpretation:

generic confidence / soft decoder reliability.

### effort

Positive continuous scalar.

Interpretation:

generic normalized decoder work, analogous to

- decoder query count,
- iteration count,
- list size,
- search effort,
- or decoding latency.

---

# 11. Receiver Mathematics

Let the simulated raw channel produce an effective frame error fraction

\[
e.
\]

Let the action have redundancy \(r_a\), decoder strength \(d_a\), and interleaver depth \(I_a\).

Define an effective difficulty score

\[
z
=
\alpha e
-
\beta r_a
-
\gamma d_a
-
\eta \log(1+I_a)
+
\epsilon,
\]

where

\[
\epsilon\sim\mathcal N(0,\sigma^2).
\]

Then define success probability

\[
P(\text{success}=1)
=
\sigma(-z)
\]

using the logistic sigmoid.

Reliability can be

\[
m_{\text{rel}}
=
\operatorname{clip}
\left(
\sigma(-z+\epsilon_r),0,1
\right).
\]

Effort can be

\[
m_{\text{effort}}
=
\log\left(
1+\exp(
c_0+c_1 e+c_2 d_a
)
\right)
+\epsilon_q.
\]

The exact constants are not scientifically important for v0.1.

What matters is:

1. measurements are stochastic;
2. measurements depend on channel state;
3. measurements depend on action;
4. distributions overlap rather than being deterministic;
5. different actions produce visibly different likelihoods.

Include a clear comment in source:

```python
# This is an abstract receiver telemetry model used to validate the
# simulation-to-likelihood architecture. It is not intended to model
# any particular decoder.
```

---

# 12. Dataset Generation

Create a single universal function:

```python
sample = simulate_once(
    channel_model,
    channel_state,
    action,
    rng,
)
```

It should return:

```python
{
    "theta": np.ndarray,
    "action_id": int,
    "action_features": np.ndarray,
    "measurement": np.ndarray,
}
```

Suggested measurement vector:

```text
[
    success,
    reliability,
    log1p(effort)
]
```

For the first likelihood demo, it is acceptable to train only on

```text
[
    reliability,
    log1p(effort)
]
```

and treat `success` separately.

---

# 13. Training Dataset

Generate approximately:

```text
200 channel-state points
×
3 actions
×
100 stochastic replicates
=
60,000 samples
```

This is enough for a convincing MVP.

Do not sample only one observation per state/action pair.

Repeated simulations at identical \((\theta,a)\) are essential because the goal is to learn a **conditional distribution**, not a deterministic mapping.

Save to:

```text
data/dataset.npz
```

No HDF5, Zarr, Parquet, database, or cloud storage for v0.1.

---

# 14. State Sampling

For each model, define a low-dimensional numeric vector.

Examples.

IID:

```text
theta = [error_prob]
```

Gilbert-Elliott:

```text
theta = [
    p_good_to_bad,
    p_bad_to_good,
    error_good,
    error_bad
]
```

Fixed burst:

```text
theta = [
    background_error_prob,
    burst_probability,
    normalized_burst_length,
    burst_error_prob
]
```

For the first neural demo it is acceptable to use **one channel family only**, preferably fixed burst or Gilbert-Elliott.

The repository can contain all four simulators while the README demo uses one.

---

# 15. Conditional Gaussian Neural Likelihood

Create `likelihoods.py`.

Start with the simplest correct neural likelihood:

\[
q_\phi(m\mid\theta,a)
=
\mathcal N
\left(
\mu_\phi(\theta,a),
\operatorname{diag}
\sigma_\phi^2(\theta,a)
\right).
\]

Input:

```text
theta vector
+
action structured features
```

Output:

```text
mean
log standard deviation
```

Implementation:

```python
import torch
from torch import nn
from torch.distributions import Normal


class ConditionalGaussianLikelihood(nn.Module):

    def __init__(self, context_dim: int, measurement_dim: int):
        super().__init__()

        self.measurement_dim = measurement_dim

        self.net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * measurement_dim),
        )

    def distribution(self, context):
        out = self.net(context)

        mean = out[..., :self.measurement_dim]
        log_std = out[..., self.measurement_dim:]

        log_std = torch.clamp(log_std, -5.0, 3.0)
        std = torch.exp(log_std)

        return Normal(mean, std)

    def log_prob(self, measurement, context):
        return self.distribution(context).log_prob(measurement).sum(-1)
```

Train using negative log likelihood:

```python
loss = -model.log_prob(measurement, context).mean()
```

This is the core ML demonstration.

---

# 16. Optional Second Model: Gaussian Mixture

If time permits, add a conditional mixture-density network.

For \(K=3\):

\[
q_\phi(m\mid\theta,a)
=
\sum_{k=1}^{3}
\pi_k(\theta,a)
\mathcal N(
m;
\mu_k(\theta,a),
\sigma_k^2(\theta,a)
).
\]

This is useful because bursty channels may generate multimodal telemetry.

Do not make this necessary for the first release.

---

# 17. Train / Validation Split

Do **not** randomly split all individual rows.

Instead split by channel-state point.

For example:

```text
70% theta points → train
15% theta points → validation
15% theta points → test
```

All stochastic replicates from one exact \(\theta\) point stay in the same split.

This prevents leakage and makes the evaluation much more credible.

---

# 18. Main Figure

The README should contain one high-value figure.

Use one fixed held-out channel state \(\theta^\star\).

For each of the three actions:

1. generate 2,000 simulator measurements;
2. plot empirical histogram of one scalar measurement;
3. overlay the learned conditional Gaussian density.

Recommended scalar:

```text
log1p(effort)
```

or:

```text
reliability
```

The figure should visibly demonstrate

\[
p(m\mid\theta^\star,a_1)
\neq
p(m\mid\theta^\star,a_2)
\neq
p(m\mid\theta^\star,a_3).
\]

That is the central idea of the repository.

---

# 19. Bayesian Update Demo

Create `inference.py`.

For the MVP, use a discrete grid over one channel parameter.

Example:

\[
\theta = p_{\text{burst}}
\]

with

```python
theta_grid = np.linspace(0.01, 0.5, 100)
```

Start with uniform prior:

\[
p_0(\theta)=1/N.
\]

Given an observed measurement \(m_t\) under action \(a_t\):

\[
p_t(\theta)
\propto
q_\phi(m_t\mid\theta,a_t)p_{t-1}(\theta).
\]

Implementation:

```python
log_posterior = (
    log_prior
    + model.log_prob(
        measurement.expand(len(theta_grid), -1),
        context,
    )
)

posterior = torch.softmax(log_posterior, dim=0)
```

Repeat for 5-10 observations.

Plot:

```text
prior
posterior after 1 observation
posterior after 5 observations
posterior after 10 observations
true theta
```

This demonstrates directly why the learned likelihood is useful.

---

# 20. `quick_demo.py`

One command should execute the entire workflow:

```bash
python examples/quick_demo.py
```

It should:

```text
1. create a fixed-burst simulator
2. define three actions
3. generate a small dataset
4. train the likelihood
5. save likelihood_fit.png
6. perform Bayesian updates
7. save posterior_update.png
```

The demo should run entirely on CPU.

Target runtime should be modest enough for an ordinary laptop.

---

# 21. Required Tests

## Channel tests

IID:

```text
empirical error rate ≈ configured error probability
```

Gilbert-Elliott:

```text
bad-state occupancy roughly agrees with stationary probability
```

Fixed burst:

```text
generated bursts have requested length
```

Markov interference:

```text
ON/OFF transition frequencies roughly match configuration
```

## Reproducibility

Same random seed must produce the same simulated sample.

## Action conditioning

At a fixed channel state:

```text
mean measurement under action 0
!=
mean measurement under action 2
```

within a robust statistical tolerance.

## Likelihood

Training loss must decrease on a tiny synthetic dataset.

## Bayesian update

Posterior must:

```text
sum to one
```

and, under a controlled toy experiment, place more mass near the true parameter after multiple observations.

---

# 22. README Structure

The README should be short and visually clear.

## Header

```text
# Wireless Neural Likelihoods

Simulation-based action-conditioned neural likelihoods for
adaptive communication systems.
```

## First paragraph

Use approximately:

> This repository explores simulation-based learning of receiver measurement likelihoods for adaptive communication systems. Given channel/interference state \(\theta\) and an abstract communication action \(a\), a simulator generates receiver telemetry \(m\). A neural conditional density model learns \(q_\phi(m\mid\theta,a)\), which can be used directly in Bayesian channel-state inference and adaptive action selection.

## Show immediately

\[
(\theta,a)
\rightarrow
\text{simulation}
\rightarrow
m
\rightarrow
q_\phi(m\mid\theta,a)
\rightarrow
\text{Bayesian update}.
\]

Then:

```text
pip install -e .
python examples/quick_demo.py
```

Then embed:

```text
results/likelihood_fit.png
results/posterior_update.png
```

---

# 23. Important README Disclaimer

Add:

> **Current scope.** The initial release uses lightweight analytical and synthetic channel models together with an abstract receiver telemetry model. The receiver interface is deliberately decoder-agnostic: it is intended to support future integrations with conventional FEC decoders and higher-fidelity communications simulators without changing the likelihood-learning interface.

This makes the deliberately abstract receiver scope clear.

---

# 24. Software Abstraction to Preserve

The most important interface in the repository should effectively be:

```python
measurement = simulator.simulate(theta, action, rng)
```

and the model should expose:

```python
log_likelihood = model.log_prob(
    measurement,
    theta,
    action,
)
```

Everything else should be replaceable.

Future decoder integration should therefore require only a new simulator/receiver backend, not changes to the neural likelihood or Bayesian inference code.

---

# 25. Roadmap in README

Use four milestones.

## v0.1 — Lightweight synthetic environments

- IID errors
- Gilbert-Elliott
- fixed bursts
- Markov interference
- abstract receiver telemetry
- conditional Gaussian likelihood
- Bayesian update

## v0.2 — Richer likelihood models

- Gaussian mixtures
- mixed discrete/continuous observations
- calibration diagnostics
- posterior predictive checks

## v0.3 — Communications backends

- real coding families
- advanced decoder adapters
- AFF3CT or other FEC backends
- waveform-level AWGN/fading models

## v0.4 — High-fidelity environments

- Sionna adapters
- measured channel datasets
- multi-interferer environments
- standardized channel models
- ray-traced environments

---

# 26. What Not to Overengineer

Do not add:

```text
Docker
Hydra
Weights & Biases
Lightning
Ray
Dask
Zarr
HDF5
MLflow
Sionna
sbi
normalizing flows
GPU CI
large datasets
```

until the MVP pipeline works.

A repository with 1,000 clean lines of understandable Python is preferable to a large framework dependency graph.

---

# 27. MVP Acceptance Criteria

The repository is ready to show publicly when all of the following are true:

- [ ] `pip install -e .` works.
- [ ] `pytest` passes.
- [ ] Four simple channel/interference models exist.
- [ ] Three abstract actions exist.
- [ ] The same channel state produces different measurement distributions under different actions.
- [ ] Dataset generation produces repeated `(theta, action, measurement)` samples.
- [ ] Conditional Gaussian likelihood trains successfully.
- [ ] Held-out likelihood plot is generated.
- [ ] Bayesian posterior update runs using the learned likelihood.
- [ ] README explains \(q_\phi(m\mid\theta,a)\) clearly.
- [ ] README explicitly says the receiver is abstract and decoder-agnostic.
- [ ] High-fidelity simulators are listed only as future integrations.
- [ ] Repository includes LICENSE and basic contribution instructions.
- [ ] GitHub Actions runs `pytest`.

---

# 28. Recommended Implementation Order

Implement in exactly this order:

```text
1. types.py
2. actions.py
3. simulation.py
4. receiver.py
5. one fixed-burst environment
6. dataset.py
7. quick_demo.py without ML
8. ConditionalGaussianLikelihood
9. training loop
10. likelihood plot
11. Bayesian update
12. tests
13. remaining three channel models
14. README cleanup
15. CI
```

Do not start by implementing all channel models.

Get the complete vertical slice working first.

---

# 29. The First Vertical Slice

The very first working version can use only:

```text
Channel:
    fixed burst

State:
    theta = [burst_probability, burst_length]

Actions:
    fast
    interleaved
    robust

Measurements:
    reliability
    effort

Likelihood:
    diagonal Gaussian MLP

Inference:
    grid posterior over burst_probability
```

This alone is sufficient to demonstrate:

\[
(\theta,a)
\rightarrow
p_{\rm sim}(m\mid\theta,a)
\rightarrow
q_\phi(m\mid\theta,a)
\rightarrow
p(\theta\mid m,a).
\]

Once this works, the rest of the repository is expansion rather than architectural uncertainty.

---

# 30. Suggested Initial Commit Sequence

Use small commits so the repository visibly shows active development.

```text
Initial package skeleton and scientific interfaces

Add fixed-burst simulation and abstract action model

Add decoder-agnostic receiver telemetry interface

Add reproducible dataset generation

Add conditional Gaussian neural likelihood baseline

Add action-conditioned likelihood visualization

Add Bayesian posterior update example

Add Gilbert-Elliott and Markov interference models

Add tests and GitHub Actions

Document roadmap toward real decoders and higher-fidelity simulators
```

---

# 31. Bottom Line

The MVP should make one claim and demonstrate it cleanly:

> **We can generate stochastic receiver measurements from parameterized communication simulations, condition those measurements on selectable code/decoder-like actions, learn an explicit neural likelihood \(q_\phi(m\mid\theta,a)\), and use that likelihood in Bayesian inference.**

No real decoder implementation is necessary to establish this architecture.

The receiver/action API should simply be designed so that a real decoder can later replace the abstract receiver without changing the rest of the pipeline.
