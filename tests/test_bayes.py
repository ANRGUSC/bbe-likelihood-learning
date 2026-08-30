import numpy as np
import torch

from wireless_likelihoods.actions import ACTIONS
from wireless_likelihoods.inference import bayesian_update


class _ToyLikelihood:
    def eval(self):
        return self

    def log_prob(self, measurement, theta, action_features):
        theta = torch.as_tensor(theta, dtype=torch.float32)
        measurement = torch.as_tensor(measurement, dtype=torch.float32)
        return -0.5 * ((measurement[:, 0] - theta[:, 0]) / 0.03) ** 2


def test_bayesian_update_normalizes_and_favors_true_grid_value():
    grid = np.linspace(0.01, 0.50, 100, dtype=np.float32)[:, None]
    posterior = bayesian_update(_ToyLikelihood(), grid, ACTIONS[1], np.array([0.26], dtype=np.float32))
    assert np.isclose(posterior.sum(), 1.0)
    assert abs(grid[posterior.argmax(), 0] - 0.26) < 0.02
