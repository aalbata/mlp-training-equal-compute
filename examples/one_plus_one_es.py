"""Example plug-in: a (1+1) evolution strategy with a one-fifth success rule.

Tune it, run it on the evaluation data sets and compare it with the sixteen methods of the study:

    python reproduce.py --plugin examples/one_plus_one_es.py --stage prepare --stage tune --stage benchmark --stage compare

The plug-in interface is described in mlp_equal_compute/plugins.py.
"""
import numpy as np

NAME = "one_plus_one_es"
LABEL = "(1+1)-ES"
FAMILY = "adaptive population"

UP, DOWN = np.exp(1.0 / 3.0), np.exp(-1.0 / 12.0)  # the step size is stable on average at one success in five


def space(rng=None):
    """Initial mutation step size: 0.1 by default, otherwise log-uniform between 0.001 and 1."""
    if rng is None:
        return {"sigma0": 0.1}
    return {"sigma0": float(10.0 ** rng.uniform(-3.0, 0.0))}


def run(ev, theta0, cfg, rng):
    """Mutate the parent with isotropic Gaussian noise and keep the child when its training loss is not higher.

    Each candidate costs one forward pass over the training split. The step size is multiplied by
    exp(1/3) after a success and by exp(-1/12) after a failure, so it stays constant on average when
    one mutation in five succeeds.
    """
    parent = {"x": theta0.copy(), "f": None}
    ev.set_incumbent(lambda: parent["x"])
    parent["f"] = ev.loss(parent["x"])
    sigma = float(cfg["sigma0"])
    while True:
        child = parent["x"] + sigma * rng.standard_normal(theta0.size)
        f = ev.loss(child)
        if f <= parent["f"]:
            parent["x"], parent["f"] = child, f
            sigma *= UP
        else:
            sigma *= DOWN
