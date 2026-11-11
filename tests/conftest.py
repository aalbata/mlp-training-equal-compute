import numpy as np
import pytest

from mlp_equal_compute.problem import Problem


def make_problem(task="classification", n=240, d=6, k=3, hidden=(8,), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if task == "classification":
        W = rng.standard_normal((d, k))
        y = np.argmax(X @ W + 0.3 * rng.standard_normal((n, k)), axis=1).astype(np.int64)
    else:
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(n)
        y = (y - y.mean()) / y.std()
    a, b = int(0.6 * n), int(0.8 * n)
    arrays = {"X_train": X[:a], "y_train": y[:a], "X_val": X[a:b], "y_val": y[a:b], "X_test": X[b:], "y_test": y[b:]}
    meta = {"task": task, "n_classes": k if task == "classification" else 0, "data_id": 0}
    return Problem(arrays, meta, hidden)


@pytest.fixture
def cls_problem():
    return make_problem("classification")


@pytest.fixture
def reg_problem():
    return make_problem("regression", hidden=(5, 4))
