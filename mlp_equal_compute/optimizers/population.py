"""Population-based training on the flat weight vector: GA, PSO, GWO, DE and CSO.

Every population starts around the shared initial network: the first member is that
network and the others add Gaussian noise with standard deviation ``init_scale``.
Each candidate scored on the training split costs one unit. The incumbent is the best
candidate found so far. The optional ``stop`` callback receives the iteration count
and the best training loss after every iteration and ends the search when it returns
True; the hybrid methods use it.
"""
from __future__ import annotations

import numpy as np


class Best:
    """Best candidate seen so far and its training loss."""

    def __init__(self, theta0: np.ndarray):
        self.x = theta0.copy()
        self.f = np.inf

    def offer(self, x: np.ndarray, f: float) -> None:
        if f < self.f:
            self.f, self.x = f, x.copy()


def init_population(theta0: np.ndarray, size: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    P = theta0 + scale * rng.standard_normal((int(size), theta0.size))
    P[0] = theta0
    return P


def evaluate(ev, P: np.ndarray, best: Best) -> np.ndarray:
    fit = np.empty(len(P))
    for i in range(len(P)):
        fit[i] = ev.loss(P[i])
        best.offer(P[i], fit[i])
    return fit


def ga(ev, theta0, cfg, rng, stop=None, horizon_units=None) -> Best:
    """Real-coded genetic algorithm: 10 percent elitism, binary tournaments, uniform crossover, Gaussian mutation."""
    N = int(cfg["pop_size"])
    n_elite = max(1, int(round(0.1 * N)))
    best = Best(theta0)
    ev.set_incumbent(lambda: best.x)
    P = init_population(theta0, N, cfg["init_scale"], rng)
    fit = evaluate(ev, P, best)
    D = theta0.size
    it = 0
    while True:
        elite = np.argsort(fit)[:n_elite]
        n_child = N - n_elite
        t1 = rng.integers(N, size=(n_child, 2))
        t2 = rng.integers(N, size=(n_child, 2))
        p1 = np.where(fit[t1[:, 0]] <= fit[t1[:, 1]], t1[:, 0], t1[:, 1])
        p2 = np.where(fit[t2[:, 0]] <= fit[t2[:, 1]], t2[:, 0], t2[:, 1])
        C = np.where(rng.random((n_child, D)) < 0.5, P[p1], P[p2])
        C += (rng.random((n_child, D)) < cfg["mutation_rate"]) * rng.normal(0.0, cfg["mutation_scale"], (n_child, D))
        cfit = evaluate(ev, C, best)
        P = np.concatenate([P[elite], C])
        fit = np.concatenate([fit[elite], cfit])
        it += 1
        if stop is not None and stop(it, best.f):
            return best


def pso(ev, theta0, cfg, rng, stop=None, horizon_units=None) -> Best:
    """Global-best particle swarm with constant inertia and velocity clamping."""
    N = int(cfg["pop_size"])
    w, c1, c2, vmax = cfg["inertia"], cfg["c1"], cfg["c2"], cfg["vmax"]
    best = Best(theta0)
    ev.set_incumbent(lambda: best.x)
    X = init_population(theta0, N, cfg["init_scale"], rng)
    V = np.zeros_like(X)
    pbest = X.copy()
    pfit = evaluate(ev, X, best)
    it = 0
    while True:
        g = int(np.argmin(pfit))
        r1 = rng.random(X.shape)
        r2 = rng.random(X.shape)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (pbest[g] - X)
        np.clip(V, -vmax, vmax, out=V)
        X = X + V
        for i in range(N):
            f = ev.loss(X[i])
            best.offer(X[i], f)
            if f < pfit[i]:
                pfit[i] = f
                pbest[i] = X[i]
        it += 1
        if stop is not None and stop(it, best.f):
            return best


def gwo(ev, theta0, cfg, rng, stop=None, horizon_units=None) -> Best:
    """Grey wolf optimizer; the control parameter a falls linearly from 2 to 0 over the available units."""
    N = int(cfg["pop_size"])
    horizon = ev.remaining() if horizon_units is None else horizon_units
    T = max(1, int(horizon // N))
    best = Best(theta0)
    ev.set_incumbent(lambda: best.x)
    X = init_population(theta0, N, cfg["init_scale"], rng)
    leaders = [[theta0.copy(), np.inf] for _ in range(3)]
    it = 0
    while True:
        for i in range(N):
            f = ev.loss(X[i])
            best.offer(X[i], f)
            if f < leaders[0][1]:
                leaders = [[X[i].copy(), f], leaders[0], leaders[1]]
            elif f < leaders[1][1]:
                leaders = [leaders[0], [X[i].copy(), f], leaders[1]]
            elif f < leaders[2][1]:
                leaders[2] = [X[i].copy(), f]
        a = 2.0 * max(0.0, 1.0 - it / T)
        new = np.zeros_like(X)
        for L, _ in leaders:
            A = 2.0 * a * rng.random(X.shape) - a
            C = 2.0 * rng.random(X.shape)
            new += L - A * np.abs(C * L - X)
        X = new / 3.0
        it += 1
        if stop is not None and stop(it, best.f):
            return best


def de(ev, theta0, cfg, rng, stop=None, horizon_units=None) -> Best:
    """Differential evolution, DE/rand/1/bin, with generational selection."""
    N = int(cfg["pop_size"])
    F, CR = cfg["F"], cfg["CR"]
    best = Best(theta0)
    ev.set_incumbent(lambda: best.x)
    P = init_population(theta0, N, cfg["init_scale"], rng)
    fit = evaluate(ev, P, best)
    D = theta0.size
    it = 0
    while True:
        U = np.empty_like(P)
        for i in range(N):
            r = rng.choice(N - 1, 3, replace=False)
            r[r >= i] += 1
            v = P[r[0]] + F * (P[r[1]] - P[r[2]])
            mask = rng.random(D) < CR
            mask[rng.integers(D)] = True
            U[i] = np.where(mask, v, P[i])
        fu = evaluate(ev, U, best)
        keep = fu <= fit
        P[keep] = U[keep]
        fit[keep] = fu[keep]
        it += 1
        if stop is not None and stop(it, best.f):
            return best


def cso(ev, theta0, cfg, rng, stop=None, horizon_units=None) -> Best:
    """Competitive swarm optimizer (Cheng and Jin): in random pairs the loser learns from the winner."""
    m = int(cfg["pop_size"]) // 2 * 2
    phi = cfg["phi"]
    best = Best(theta0)
    ev.set_incumbent(lambda: best.x)
    X = init_population(theta0, m, cfg["init_scale"], rng)
    V = np.zeros_like(X)
    fit = evaluate(ev, X, best)
    D = theta0.size
    half = m // 2
    it = 0
    while True:
        perm = rng.permutation(m)
        a, b = perm[:half], perm[half:]
        a_wins = fit[a] <= fit[b]
        win = np.where(a_wins, a, b)
        lose = np.where(a_wins, b, a)
        xbar = X.mean(axis=0)
        R1, R2, R3 = rng.random((half, D)), rng.random((half, D)), rng.random((half, D))
        V[lose] = R1 * V[lose] + R2 * (X[win] - X[lose]) + phi * R3 * (xbar - X[lose])
        X[lose] = X[lose] + V[lose]
        for j in lose:
            fit[j] = ev.loss(X[j])
            best.offer(X[j], fit[j])
        it += 1
        if stop is not None and stop(it, best.f):
            return best
