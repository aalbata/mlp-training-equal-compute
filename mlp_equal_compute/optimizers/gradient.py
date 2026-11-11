"""Gradient-based training: mini-batch Adam, RMSprop and SGD with momentum, and full-batch L-BFGS."""
from __future__ import annotations

import numpy as np


class Adam:
    """Adam update (Kingma and Ba) applied in place to ``theta``."""

    def __init__(self, theta: np.ndarray, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.theta, self.lr, self.beta1, self.beta2, self.eps = theta, lr, beta1, beta2, eps
        self.m = np.zeros_like(theta)
        self.v = np.zeros_like(theta)
        self.t = 0

    def step(self, g: np.ndarray) -> None:
        self.t += 1
        self.m *= self.beta1
        self.m += (1.0 - self.beta1) * g
        self.v *= self.beta2
        self.v += (1.0 - self.beta2) * g * g
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        self.theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop:
    """RMSprop update with decay 0.9 applied in place to ``theta``."""

    def __init__(self, theta: np.ndarray, lr: float, rho: float = 0.9, eps: float = 1e-8):
        self.theta, self.lr, self.rho, self.eps = theta, lr, rho, eps
        self.v = np.zeros_like(theta)

    def step(self, g: np.ndarray) -> None:
        self.v *= self.rho
        self.v += (1.0 - self.rho) * g * g
        self.theta -= self.lr * g / (np.sqrt(self.v) + self.eps)


class SGDMomentum:
    """Stochastic gradient descent with heavy-ball momentum applied in place to ``theta``."""

    def __init__(self, theta: np.ndarray, lr: float, momentum: float):
        self.theta, self.lr, self.momentum = theta, lr, momentum
        self.buf = np.zeros_like(theta)

    def step(self, g: np.ndarray) -> None:
        self.buf *= self.momentum
        self.buf += g
        self.theta -= self.lr * self.buf


def train_minibatch(ev, opt, batch_size: int, rng: np.random.Generator, until: float | None = None) -> None:
    """Shuffled mini-batch epochs until the budget is spent, or until ``until`` units when given."""
    n = ev.n_train
    bs = max(1, min(int(batch_size), n))
    while until is None or ev.units < until:
        perm = rng.permutation(n)
        for start in range(0, n, bs):
            if until is not None and ev.units >= until:
                return
            _, g = ev.loss_grad_batch(opt.theta, perm[start:start + bs])
            opt.step(g)


def run_adam(ev, theta0, cfg, rng):
    opt = Adam(theta0.copy(), cfg["lr"])
    ev.set_incumbent(lambda: opt.theta)
    train_minibatch(ev, opt, cfg["batch_size"], rng)


def run_rmsprop(ev, theta0, cfg, rng):
    opt = RMSprop(theta0.copy(), cfg["lr"])
    ev.set_incumbent(lambda: opt.theta)
    train_minibatch(ev, opt, cfg["batch_size"], rng)


def run_sgd_momentum(ev, theta0, cfg, rng):
    opt = SGDMomentum(theta0.copy(), cfg["lr"], cfg["momentum"])
    ev.set_incumbent(lambda: opt.theta)
    train_minibatch(ev, opt, cfg["batch_size"], rng)


def run_lbfgs(ev, theta0, cfg, rng):
    """Full-batch L-BFGS-B without bounds; each loss and gradient evaluation costs 3 units."""
    from scipy.optimize import minimize

    best = {"x": theta0.copy(), "f": np.inf}
    ev.set_incumbent(lambda: best["x"])

    def fun(x):
        f, g = ev.loss_grad_full(x)
        if f < best["f"]:
            best["f"], best["x"] = f, x.copy()
        return f, g

    minimize(fun, theta0.copy(), jac=True, method="L-BFGS-B",
             options={"maxcor": int(cfg["memory"]), "maxiter": 10 ** 9, "maxfun": 10 ** 9,
                      "ftol": 0.0, "gtol": 0.0})
