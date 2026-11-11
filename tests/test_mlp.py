import numpy as np
import pytest

from mlp_equal_compute.mlp import MLP, mean_loss


def numeric_grad(model, theta, X, y, task, eps=1e-6):
    g = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = eps
        g[i] = (mean_loss(model.forward(theta + e, X), y, task) - mean_loss(model.forward(theta - e, X), y, task)) / (2 * eps)
    return g


@pytest.mark.parametrize("task,hidden,k", [("classification", (5,), 3), ("classification", (4, 3), 2),
                                           ("regression", (6,), 1), ("regression", (3, 4), 1)])
def test_gradient_matches_finite_differences(task, hidden, k):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 4))
    y = rng.integers(0, k, 30) if task == "classification" else rng.standard_normal(30)
    model = MLP(4, hidden, k)
    theta = model.init(rng) + 0.05 * rng.standard_normal(model.n_params)
    loss, grad = model.loss_grad(theta, X, y, task)
    assert np.isclose(loss, mean_loss(model.forward(theta, X), y, task))
    assert np.allclose(grad, numeric_grad(model, theta, X, y, task), rtol=1e-4, atol=1e-7)


def test_parameter_count_and_layout():
    model = MLP(7, (5, 3), 2)
    assert model.n_params == 7 * 5 + 5 + 5 * 3 + 3 + 3 * 2 + 2
    theta = model.init(np.random.default_rng(0))
    for _, _, b in model.slices:
        assert np.all(theta[b] == 0.0)


def test_nonfinite_loss_is_capped():
    model = MLP(2, (2,), 2)
    theta = np.full(model.n_params, 1e200)
    X = np.ones((4, 2))
    y = np.array([0, 1, 0, 1])
    loss, grad = model.loss_grad(theta, X, y, "classification")
    assert np.isfinite(loss) and np.all(grad == 0.0)
