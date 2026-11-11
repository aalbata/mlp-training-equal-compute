"""Fully connected network with ReLU hidden layers whose parameters live in one flat vector.

Every optimizer works on the same flat vector, so gradient methods and population
methods train exactly the same model. Classification uses a softmax output with one
unit per class and mean cross-entropy; regression uses one linear output and mean
squared error.
"""
from __future__ import annotations

import numpy as np

LARGE_LOSS = 1e10


class MLP:
    def __init__(self, n_inputs: int, hidden: tuple, n_outputs: int):
        self.sizes = [int(n_inputs), *[int(h) for h in hidden], int(n_outputs)]
        self.slices = []
        offset = 0
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            w = slice(offset, offset + a * b)
            offset += a * b
            bias = slice(offset, offset + b)
            offset += b
            self.slices.append((w, (a, b), bias))
        self.n_params = offset

    def init(self, rng: np.random.Generator) -> np.ndarray:
        """Glorot-uniform weights and zero biases."""
        theta = np.zeros(self.n_params)
        for w, (a, b), _ in self.slices:
            limit = np.sqrt(6.0 / (a + b))
            theta[w] = rng.uniform(-limit, limit, a * b)
        return theta

    def _layers(self, theta: np.ndarray) -> list:
        return [(theta[w].reshape(shape), theta[b]) for w, shape, b in self.slices]

    def forward(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        layers = self._layers(theta)
        A = X
        with np.errstate(over="ignore", invalid="ignore"):
            for W, b in layers[:-1]:
                A = np.maximum(A @ W + b, 0.0)
            W, b = layers[-1]
            return A @ W + b

    def loss_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray, task: str):
        """Mean loss and its gradient with respect to the flat parameter vector."""
        layers = self._layers(theta)
        acts = [X]
        A = X
        n = X.shape[0]
        with np.errstate(over="ignore", invalid="ignore"):
            for W, b in layers[:-1]:
                A = np.maximum(A @ W + b, 0.0)
                acts.append(A)
            W, b = layers[-1]
            Z = A @ W + b
            if task == "classification":
                Zs = Z - Z.max(axis=1, keepdims=True)
                E = np.exp(Zs)
                S = E.sum(axis=1, keepdims=True)
                loss = float(np.mean(np.log(S[:, 0]) - Zs[np.arange(n), y]))
                dZ = E / S
                dZ[np.arange(n), y] -= 1.0
                dZ /= n
            else:
                r = Z[:, 0] - y
                loss = float(np.mean(r * r))
                dZ = (2.0 / n) * r[:, None]
        if not np.isfinite(loss):
            return LARGE_LOSS, np.zeros_like(theta)
        grad = np.empty_like(theta)
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(len(layers) - 1, -1, -1):
                w_sl, _, b_sl = self.slices[i]
                grad[w_sl] = (acts[i].T @ dZ).ravel()
                grad[b_sl] = dZ.sum(axis=0)
                if i > 0:
                    dZ = (dZ @ layers[i][0].T) * (acts[i] > 0.0)
        if not np.all(np.isfinite(grad)):
            return LARGE_LOSS, np.zeros_like(theta)
        return loss, grad


def mean_loss(Z: np.ndarray, y: np.ndarray, task: str) -> float:
    """Mean cross-entropy (classification) or mean squared error (regression) of outputs ``Z``."""
    with np.errstate(over="ignore", invalid="ignore"):
        if task == "classification":
            Zs = Z - Z.max(axis=1, keepdims=True)
            value = float(np.mean(np.log(np.exp(Zs).sum(axis=1)) - Zs[np.arange(len(y)), y]))
        else:
            r = Z[:, 0] - y
            value = float(np.mean(r * r))
    return value if np.isfinite(value) else LARGE_LOSS
