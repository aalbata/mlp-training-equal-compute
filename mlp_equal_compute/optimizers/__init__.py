"""Registry of the sixteen training methods, their display names and their families.

Methods loaded from plug-in files (see ``mlp_equal_compute.plugins``) are added to these
dictionaries; ``STUDY_METHODS`` keeps the sixteen methods of the study.
"""
from __future__ import annotations

from functools import partial

from . import evolution, gradient, hybrid, population


def _search_only(fn):
    def run(ev, theta0, cfg, rng):
        fn(ev, theta0, cfg, rng)
    return run


METHODS = {
    "adam": gradient.run_adam,
    "rmsprop": gradient.run_rmsprop,
    "sgd_momentum": gradient.run_sgd_momentum,
    "lbfgs": gradient.run_lbfgs,
    "ga": _search_only(population.ga),
    "pso": _search_only(population.pso),
    "gwo": _search_only(population.gwo),
    "de": _search_only(population.de),
    "cmaes": _search_only(evolution.cmaes),
    "lshade": _search_only(evolution.lshade),
    "cso": _search_only(population.cso),
    "pso_adam_fixed": partial(hybrid.run_fixed, explorer="pso"),
    "gwo_adam_fixed": partial(hybrid.run_fixed, explorer="gwo"),
    "pso_adam_stagnation": partial(hybrid.run_stagnation, explorer="pso"),
    "gwo_adam_stagnation": partial(hybrid.run_stagnation, explorer="gwo"),
    "adam_multistart": hybrid.run_adam_multistart,
}
STUDY_METHODS = tuple(METHODS)

LABELS = {
    "adam": "Adam", "rmsprop": "RMSprop", "sgd_momentum": "SGD with momentum", "lbfgs": "L-BFGS",
    "ga": "GA", "pso": "PSO", "gwo": "GWO", "de": "DE",
    "cmaes": "CMA-ES", "lshade": "L-SHADE", "cso": "CSO",
    "pso_adam_fixed": "PSO then Adam, fixed split", "gwo_adam_fixed": "GWO then Adam, fixed split",
    "pso_adam_stagnation": "PSO then Adam, stagnation trigger",
    "gwo_adam_stagnation": "GWO then Adam, stagnation trigger",
    "adam_multistart": "Multi-start Adam",
}

FAMILIES = {
    "adam": "gradient", "rmsprop": "gradient", "sgd_momentum": "gradient", "lbfgs": "gradient",
    "ga": "population", "pso": "population", "gwo": "population", "de": "population", "cso": "population",
    "cmaes": "adaptive population", "lshade": "adaptive population",
    "pso_adam_fixed": "hybrid", "gwo_adam_fixed": "hybrid",
    "pso_adam_stagnation": "hybrid", "gwo_adam_stagnation": "hybrid",
    "adam_multistart": "gradient",
}
