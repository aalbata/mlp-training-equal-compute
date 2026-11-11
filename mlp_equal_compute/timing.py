"""Measured cost of a forward and backward pass relative to a forward pass, for the benchmark networks.

For every evaluation data set the script times, on the whole training split, the
loss of the initial network (one forward pass) and its loss and gradient (a forward
and a backward pass), and reports the ratio of the median times.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_prepared
from .experiments import BENCHMARK_HIDDEN
from .problem import Problem


def _median_time(fn, min_seconds: float = 0.2, repeats: int = 7) -> float:
    fn()
    n = 1
    while True:
        t = time.perf_counter()
        for _ in range(n):
            fn()
        if time.perf_counter() - t >= min_seconds / repeats:
            break
        n *= 2
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        times.append((time.perf_counter() - t) / n)
    return float(np.median(times))


def measure(table: pd.DataFrame, prepared_dir: Path) -> pd.DataFrame:
    rows = []
    for name in table[table.role == "evaluation"].name:
        arrays, meta = load_prepared(prepared_dir, name)
        p = Problem(arrays, meta, BENCHMARK_HIDDEN)
        theta = p.model.init(np.random.default_rng([0, 1]))
        fwd = _median_time(lambda: p.train_loss(theta))
        both = _median_time(lambda: p.model.loss_grad(theta, p.X_train, p.y_train, p.task))
        rows.append({"dataset": name, "n_train": p.n_train, "n_params": p.model.n_params,
                     "forward_seconds": fwd, "forward_backward_seconds": both, "ratio": both / fwd})
    return pd.DataFrame(rows)
