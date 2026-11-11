"""Search-then-refine methods: a population search or several short Adam runs, then Adam on the best network.

Fixed split: the search phase ends after a preset fraction of the budget.
Stagnation trigger: the search phase ends when the best training loss has not
improved by more than a relative tolerance for a set number of consecutive
iterations, or when a cap on its share of the budget is reached. With tolerance 0
and a stall count of 10 this is the rule of Zhang et al. (2007), who hand the best particle
to back-propagation once it has been unchanged for ten generations.
The first iteration of the search sets the reference loss, so the earliest hand-off
follows the eleventh iteration. In GWO the control parameter falls from 2 to 0 over the
search share of a fixed split, or over the cap of the stagnation trigger.
Multi-start Adam spends the search share on several short Adam runs from different
initial networks and continues the one with the lowest training loss.
"""
from __future__ import annotations

import numpy as np

from .gradient import Adam, train_minibatch
from .population import gwo, pso

EXPLORERS = {"pso": pso, "gwo": gwo}


class StagnationRule:
    def __init__(self, ev, stall_iterations: int, tol: float, cap_units: float):
        self.ev, self.stall_iterations, self.tol, self.cap_units = ev, int(stall_iterations), float(tol), cap_units
        self.best = np.inf
        self.count = 0

    def __call__(self, it: int, f: float) -> bool:
        if not np.isfinite(self.best) or f < self.best - self.tol * abs(self.best):
            self.best, self.count = f, 0
        else:
            self.count += 1
        return self.count >= self.stall_iterations or self.ev.units >= self.cap_units


def _refine(ev, theta, cfg, rng) -> None:
    ev.start_refinement()
    opt = Adam(theta.copy(), cfg["lr"])
    ev.set_incumbent(lambda: opt.theta)
    train_minibatch(ev, opt, cfg["batch_size"], rng)


def run_fixed(ev, theta0, cfg, rng, explorer: str) -> None:
    cap = cfg["explore_fraction"] * ev.budget
    found = EXPLORERS[explorer](ev, theta0, cfg, rng, stop=lambda it, f: ev.units >= cap, horizon_units=cap)
    _refine(ev, found.x, cfg, rng)


def run_stagnation(ev, theta0, cfg, rng, explorer: str) -> None:
    cap = cfg["cap_fraction"] * ev.budget
    rule = StagnationRule(ev, cfg["stall_iterations"], cfg["tol"], cap)
    found = EXPLORERS[explorer](ev, theta0, cfg, rng, stop=rule, horizon_units=cap)
    _refine(ev, found.x, cfg, rng)


def run_adam_multistart(ev, theta0, cfg, rng) -> None:
    starts = int(cfg["starts"])
    share = cfg["explore_fraction"] * ev.budget / starts
    holder = {"best": None, "best_loss": np.inf, "current": None}
    ev.set_incumbent(lambda: (holder["best"] or holder["current"]).theta)
    for k in range(starts):
        theta = theta0.copy() if k == 0 else ev.problem.model.init(rng)
        opt = Adam(theta, cfg["lr"])
        holder["current"] = opt
        train_minibatch(ev, opt, cfg["batch_size"], rng, until=(k + 1) * share)
        loss = ev.loss(opt.theta)
        if loss < holder["best_loss"]:
            holder["best"], holder["best_loss"] = opt, loss
    ev.start_refinement()
    chosen = holder["best"]
    ev.set_incumbent(lambda: chosen.theta)
    train_minibatch(ev, chosen, cfg["batch_size"], rng)
