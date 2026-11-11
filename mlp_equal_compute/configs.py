"""Search spaces for tuning. Every method gets the same number of candidate settings.

The first candidate of each method is a standard setting from the literature; the
others are drawn at random from the method's search space with a fixed seed. A
method with fewer distinct settings than the tuning allowance evaluates all of them.
"""
from __future__ import annotations

import math

import numpy as np

from .optimizers import METHODS

N_CONFIGS = 10
CONFIG_SEED = 2026
PLUGIN_SPACES: dict = {}  # method key -> space(rng=None) of a method loaded from a plug-in file


def _lu(rng, lo, hi):
    return float(10.0 ** rng.uniform(math.log10(lo), math.log10(hi)))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _adam(rng=None):
    if rng is None:
        return {"lr": 1e-3, "batch_size": 64}
    return {"lr": _lu(rng, 1e-4, 3e-2), "batch_size": _pick(rng, [32, 64, 128, 256])}


def _pso(rng=None):
    if rng is None:
        return {"pop_size": 40, "inertia": 0.7298, "c1": 1.49618, "c2": 1.49618, "init_scale": 0.1, "vmax": 0.2}
    return {"pop_size": _pick(rng, [20, 40, 80]), "inertia": float(rng.uniform(0.4, 0.9)),
            "c1": float(rng.uniform(1.0, 2.0)), "c2": float(rng.uniform(1.0, 2.0)),
            "init_scale": _lu(rng, 0.01, 1.0), "vmax": _pick(rng, [0.05, 0.2, 1.0])}


def _gwo(rng=None):
    if rng is None:
        return {"pop_size": 20, "init_scale": 0.1}
    return {"pop_size": _pick(rng, [10, 20, 40, 80]), "init_scale": _lu(rng, 0.01, 1.0)}


def _fixed(rng=None):
    if rng is None:
        return {"explore_fraction": 0.1}
    return {"explore_fraction": _pick(rng, [0.05, 0.1, 0.2, 0.4])}


def _stagnation(rng=None):
    if rng is None:
        return {"stall_iterations": 10, "tol": 0.0, "cap_fraction": 0.5}
    return {"stall_iterations": _pick(rng, [5, 10, 20]), "tol": _pick(rng, [0.0, 1e-4, 1e-3]),
            "cap_fraction": _pick(rng, [0.25, 0.5, 0.75])}


def _space(method, rng=None):
    if method in PLUGIN_SPACES:
        return PLUGIN_SPACES[method](rng)
    if method == "adam":
        return _adam(rng)
    if method == "rmsprop":
        if rng is None:
            return {"lr": 1e-3, "batch_size": 64}
        return {"lr": _lu(rng, 1e-4, 1e-2), "batch_size": _pick(rng, [32, 64, 128, 256])}
    if method == "sgd_momentum":
        if rng is None:
            return {"lr": 1e-2, "momentum": 0.9, "batch_size": 64}
        return {"lr": _lu(rng, 1e-3, 3e-1), "momentum": _pick(rng, [0.5, 0.9, 0.95]),
                "batch_size": _pick(rng, [32, 64, 128, 256])}
    if method == "lbfgs":
        if rng is None:
            return {"memory": 10}
        return {"memory": _pick(rng, [5, 10, 20, 50])}
    if method == "ga":
        if rng is None:
            return {"pop_size": 50, "mutation_rate": 0.05, "mutation_scale": 0.1, "init_scale": 0.1}
        return {"pop_size": _pick(rng, [20, 50, 100]), "mutation_rate": _lu(rng, 1e-3, 0.2),
                "mutation_scale": _lu(rng, 0.01, 0.5), "init_scale": _lu(rng, 0.01, 1.0)}
    if method == "pso":
        return _pso(rng)
    if method == "gwo":
        return _gwo(rng)
    if method == "de":
        if rng is None:
            return {"pop_size": 50, "F": 0.5, "CR": 0.9, "init_scale": 0.1}
        return {"pop_size": _pick(rng, [20, 50, 100]), "F": float(rng.uniform(0.3, 0.9)),
                "CR": float(rng.uniform(0.1, 0.99)), "init_scale": _lu(rng, 0.01, 1.0)}
    if method == "cmaes":
        if rng is None:
            return {"sigma0": 0.1, "popsize_factor": 1}
        return {"sigma0": _lu(rng, 0.01, 1.0), "popsize_factor": _pick(rng, [1, 2, 4])}
    if method == "lshade":
        if rng is None:
            return {"pop_size": 100, "init_scale": 0.1}
        return {"pop_size": _pick(rng, [20, 50, 100, 200]), "init_scale": _lu(rng, 0.01, 1.0)}
    if method == "cso":
        if rng is None:
            return {"pop_size": 100, "phi": 0.0, "init_scale": 0.1}
        return {"pop_size": _pick(rng, [20, 50, 100, 200]), "phi": _pick(rng, [0.0, 0.05, 0.1, 0.2]),
                "init_scale": _lu(rng, 0.01, 1.0)}
    if method in ("pso_adam_fixed", "gwo_adam_fixed"):
        explorer = _pso if method.startswith("pso") else _gwo
        return {**explorer(rng), **_adam(rng), **_fixed(rng)}
    if method in ("pso_adam_stagnation", "gwo_adam_stagnation"):
        explorer = _pso if method.startswith("pso") else _gwo
        return {**explorer(rng), **_adam(rng), **_stagnation(rng)}
    if method == "adam_multistart":
        if rng is None:
            return {**_adam(), "starts": 8, "explore_fraction": 0.1}
        return {**_adam(rng), "starts": _pick(rng, [4, 8, 16]), "explore_fraction": _pick(rng, [0.05, 0.1, 0.2, 0.4])}
    raise KeyError(method)


# The search spaces of _space, stated as data for the table in the paper; tests/test_methods.py checks
# that the sampler above draws only from them. ("log", lo, hi) is log-uniform, ("uniform", lo, hi) is
# uniform, and ("choice", values) draws one of the values with equal probability.
_ADAM = {"lr": ("log", 1e-4, 3e-2), "batch_size": ("choice", (32, 64, 128, 256))}
_PSO = {"pop_size": ("choice", (20, 40, 80)), "inertia": ("uniform", 0.4, 0.9), "c1": ("uniform", 1.0, 2.0),
        "c2": ("uniform", 1.0, 2.0), "init_scale": ("log", 0.01, 1.0), "vmax": ("choice", (0.05, 0.2, 1.0))}
_GWO = {"pop_size": ("choice", (10, 20, 40, 80)), "init_scale": ("log", 0.01, 1.0)}
_FIXED = {"explore_fraction": ("choice", (0.05, 0.1, 0.2, 0.4))}
_STAGNATION = {"stall_iterations": ("choice", (5, 10, 20)), "tol": ("choice", (0.0, 1e-4, 1e-3)),
               "cap_fraction": ("choice", (0.25, 0.5, 0.75))}
SEARCH_SPACES = {
    "adam": _ADAM,
    "rmsprop": {"lr": ("log", 1e-4, 1e-2), "batch_size": ("choice", (32, 64, 128, 256))},
    "sgd_momentum": {"lr": ("log", 1e-3, 3e-1), "momentum": ("choice", (0.5, 0.9, 0.95)),
                     "batch_size": ("choice", (32, 64, 128, 256))},
    "lbfgs": {"memory": ("choice", (5, 10, 20, 50))},
    "ga": {"pop_size": ("choice", (20, 50, 100)), "mutation_rate": ("log", 1e-3, 0.2),
           "mutation_scale": ("log", 0.01, 0.5), "init_scale": ("log", 0.01, 1.0)},
    "pso": _PSO,
    "gwo": _GWO,
    "de": {"pop_size": ("choice", (20, 50, 100)), "F": ("uniform", 0.3, 0.9), "CR": ("uniform", 0.1, 0.99),
           "init_scale": ("log", 0.01, 1.0)},
    "cmaes": {"sigma0": ("log", 0.01, 1.0), "popsize_factor": ("choice", (1, 2, 4))},
    "lshade": {"pop_size": ("choice", (20, 50, 100, 200)), "init_scale": ("log", 0.01, 1.0)},
    "cso": {"pop_size": ("choice", (20, 50, 100, 200)), "phi": ("choice", (0.0, 0.05, 0.1, 0.2)),
            "init_scale": ("log", 0.01, 1.0)},
    "pso_adam_fixed": {**_PSO, **_ADAM, **_FIXED},
    "gwo_adam_fixed": {**_GWO, **_ADAM, **_FIXED},
    "pso_adam_stagnation": {**_PSO, **_ADAM, **_STAGNATION},
    "gwo_adam_stagnation": {**_GWO, **_ADAM, **_STAGNATION},
    "adam_multistart": {**_ADAM, "starts": ("choice", (4, 8, 16)),
                        "explore_fraction": ("choice", (0.05, 0.1, 0.2, 0.4))},
}


def candidate_configs(method: str, n: int = N_CONFIGS, seed: int = CONFIG_SEED) -> list:
    """The default setting followed by random settings, without duplicates, at most ``n`` in total."""
    rng = np.random.default_rng([seed, list(METHODS).index(method)])
    out, seen = [], set()
    for k in range(50 * n):
        cfg = _space(method) if k == 0 else _space(method, rng)
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
        if len(out) == n:
            break
    return out
