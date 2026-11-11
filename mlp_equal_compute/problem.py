"""Training problem and compute accounting.

Compute is counted in forward-pass equivalents over the training split. Scoring one
candidate network on the whole training split costs one unit. A gradient computed on
a mini-batch of b rows out of n costs 3 b / n units, because a backward pass is
counted as two forward passes. Every run receives the same budget of units; the
last evaluation of a run may exceed it by at most its own cost, which is at most
3 units.

The evaluator also implements the common selection rule. The shared initial network
is the first candidate. At the first evaluation after every multiple of
``budget / checks`` units the evaluator scores the current network (the incumbent)
on the validation split, and the network with the lowest validation loss among the
initial network and these validation checks is the one that is reported. A run ends when its budget is
spent; only a method that cannot continue, such as L-BFGS after a failed line
search, ends earlier.
"""
from __future__ import annotations

import time

import numpy as np

from .mlp import MLP, mean_loss


class BudgetExhausted(Exception):
    """Raised when a run has spent its budget."""


class Problem:
    def __init__(self, arrays: dict, meta: dict, hidden: tuple):
        self.task = meta["task"]
        self.n_classes = int(meta["n_classes"])
        self.X_train, self.y_train = arrays["X_train"], arrays["y_train"]
        self.X_val, self.y_val = arrays["X_val"], arrays["y_val"]
        self.X_test, self.y_test = arrays["X_test"], arrays["y_test"]
        self.n_train = int(self.X_train.shape[0])
        n_out = self.n_classes if self.task == "classification" else 1
        self.model = MLP(self.X_train.shape[1], hidden, n_out)

    def loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        return mean_loss(self.model.forward(theta, X), y, self.task)

    def train_loss(self, theta: np.ndarray) -> float:
        return self.loss(theta, self.X_train, self.y_train)

    def val_loss(self, theta: np.ndarray) -> float:
        return self.loss(theta, self.X_val, self.y_val)

    def metrics(self, theta: np.ndarray, split: str) -> dict:
        X, y = getattr(self, f"X_{split}"), getattr(self, f"y_{split}")
        Z = self.model.forward(theta, X)
        out = {"loss": mean_loss(Z, y, self.task)}
        if self.task == "classification":
            out["accuracy"] = float(np.mean(np.argmax(Z, axis=1) == y))
        else:
            r = Z[:, 0] - y
            out["rmse"] = float(np.sqrt(np.mean(r * r))) if np.all(np.isfinite(r)) else float("inf")
            ss = float(np.sum((y - y.mean()) ** 2))
            out["r2"] = float(1.0 - np.sum(r * r) / ss) if ss > 0 and np.all(np.isfinite(r)) else float("-inf")
        return out


class Evaluator:
    """Charges every loss evaluation against the budget and keeps the best network on validation."""

    def __init__(self, problem: Problem, budget: float, checks: int = 1000, log_every: int = 10):
        self.problem = problem
        self.budget = float(budget)
        self.check_step = self.budget / checks
        self.next_check = self.check_step
        self.log_every = log_every
        self.units = 0.0
        self.n_checks = 0
        self.best_val = np.inf
        self.best_theta = None
        self.best_units = 0.0
        self.history = []
        self.stop_reason = None
        self.handoff_units = None
        self._incumbent = None
        self._start = time.perf_counter()

    @property
    def n_train(self) -> int:
        return self.problem.n_train

    def set_incumbent(self, getter) -> None:
        """Register a function that returns the network the method would report now."""
        self._incumbent = getter

    def remaining(self) -> float:
        return max(0.0, self.budget - self.units)

    def consider_initial(self, theta0: np.ndarray) -> None:
        """Score the shared initial network, which a method reports when it never improves on it."""
        val = self.problem.val_loss(theta0)
        if val < self.best_val:
            self.best_val, self.best_theta, self.best_units = val, theta0.copy(), 0.0

    def start_refinement(self) -> None:
        """Record the hand-off from a search phase to gradient refinement."""
        self.handoff_units = self.units

    def _spend(self, units: float) -> None:
        self.units += units
        if self.stop_reason is None and self.units >= self.next_check - 1e-9:
            self._check()
            self.next_check = (np.floor((self.units + 1e-9) / self.check_step) + 1.0) * self.check_step
        if self.stop_reason is not None:
            raise BudgetExhausted(self.stop_reason)

    def _check(self) -> None:
        theta = self._incumbent()
        val = self.problem.val_loss(theta)
        self.n_checks += 1
        if val < self.best_val:
            self.best_val, self.best_theta, self.best_units = val, theta.copy(), self.units
        if self.n_checks % self.log_every == 0:
            self.history.append({"units": round(self.units, 3),
                                 "seconds": round(time.perf_counter() - self._start, 4),
                                 "train_loss": self.problem.train_loss(theta),
                                 "val_loss": val, "best_val_loss": self.best_val})
        if self.units >= self.budget - 1e-9:
            self.stop_reason = "budget"

    def finish(self) -> None:
        """Score the final incumbent when a method ends before its budget, for example at convergence."""
        if self.stop_reason is None:
            self._check()
            if self.stop_reason is None:
                self.stop_reason = "converged"

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self._start

    def loss(self, theta: np.ndarray) -> float:
        """Training loss of one candidate network; costs one unit."""
        value = self.problem.train_loss(theta)
        self._spend(1.0)
        return value

    def loss_grad_batch(self, theta: np.ndarray, idx: np.ndarray):
        """Mini-batch loss and gradient; costs 3 b / n units."""
        p = self.problem
        value, grad = p.model.loss_grad(theta, p.X_train[idx], p.y_train[idx], p.task)
        self._spend(3.0 * len(idx) / p.n_train)
        return value, grad

    def loss_grad_full(self, theta: np.ndarray):
        """Full-batch loss and gradient; costs 3 units."""
        p = self.problem
        value, grad = p.model.loss_grad(theta, p.X_train, p.y_train, p.task)
        self._spend(3.0)
        return value, grad
