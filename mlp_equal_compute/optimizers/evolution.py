"""Evolution strategies and adaptive differential evolution: CMA-ES (pycma) and L-SHADE."""
from __future__ import annotations

import warnings

import numpy as np

from .population import Best, evaluate, init_population

FULL_COVARIANCE_LIMIT = 1000


def cmaes(ev, theta0, cfg, rng, stop=None, horizon_units=None) -> Best:
    """CMA-ES from the shared initial network; separable (diagonal) above 1,000 parameters.

    When pycma's own termination criteria fire before the budget is spent, a new run
    starts from the best network found so far with the initial step size.
    """
    import cma

    D = theta0.size
    best = Best(theta0)
    ev.set_incumbent(lambda: best.x)
    lam = int(cfg["popsize_factor"] * (4 + int(3 * np.log(D))))
    start = theta0.copy()
    idle_restarts = 0
    while idle_restarts < 3:
        opts = {"seed": int(rng.integers(1, 2 ** 31 - 1)), "popsize": lam,
                "CMA_diagonal": D > FULL_COVARIANCE_LIMIT, "verbose": -9, "verb_disp": 0,
                "verb_log": 0, "maxfevals": 10 ** 12, "tolfun": 1e-12, "tolx": 1e-12}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es = cma.CMAEvolutionStrategy(start, cfg["sigma0"], opts)
            evaluated = 0
            while not es.stop():
                xs = es.ask()
                values = []
                for x in xs:
                    f = ev.loss(x)
                    best.offer(np.asarray(x), f)
                    values.append(f)
                es.tell(xs, values)
                evaluated += len(xs)
        idle_restarts = idle_restarts + 1 if evaluated == 0 else 0
        start = best.x.copy()
    return best


def lshade(ev, theta0, cfg, rng, stop=None, horizon_units=None) -> Best:
    """L-SHADE (Tanabe and Fukunaga, 2014): SHADE with linear population size reduction.

    Fixed settings follow the original description: memory size H = 6, p = 0.11,
    archive rate 2.6 and final population size 4. The initial population size is a
    tuned setting: the original size of 18 times the dimension would exceed the budget
    for networks with more than 555 weights and leave few generations for smaller ones.
    """
    N_init = int(cfg["pop_size"])
    N_min, H, p_best, arc_rate = 4, 6, 0.11, 2.6
    D = theta0.size
    max_nfe = ev.remaining()
    best = Best(theta0)
    ev.set_incumbent(lambda: best.x)
    P = init_population(theta0, N_init, cfg["init_scale"], rng)
    fit = evaluate(ev, P, best)
    nfe = N_init
    M_CR, M_F = np.full(H, 0.5), np.full(H, 0.5)
    terminal = np.zeros(H, dtype=bool)
    k = 0
    archive = np.empty((0, D))
    N = N_init
    it = 0
    while True:
        idx = np.arange(N)
        r = rng.integers(H, size=N)
        CR = np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0)
        CR[terminal[r]] = 0.0
        F = M_F[r] + 0.1 * rng.standard_cauchy(N)
        bad = F <= 0.0
        while bad.any():
            F[bad] = M_F[r[bad]] + 0.1 * rng.standard_cauchy(int(bad.sum()))
            bad = F <= 0.0
        F = np.minimum(F, 1.0)
        n_p = max(2, int(round(p_best * N)))
        pb = np.argsort(fit)[rng.integers(n_p, size=N)]
        r1 = rng.integers(N - 1, size=N)
        r1 += r1 >= idx
        pool = np.vstack([P, archive]) if len(archive) else P
        r2 = rng.integers(len(pool), size=N)
        clash = (r2 == idx) | (r2 == r1)
        while clash.any():
            r2[clash] = rng.integers(len(pool), size=int(clash.sum()))
            clash = (r2 == idx) | (r2 == r1)
        V = P + F[:, None] * (P[pb] - P) + F[:, None] * (P[r1] - pool[r2])
        mask = rng.random((N, D)) < CR[:, None]
        mask[idx, rng.integers(D, size=N)] = True
        U = np.where(mask, V, P)
        fu = evaluate(ev, U, best)
        nfe += N
        improved = fu < fit
        if improved.any():
            archive = np.vstack([archive, P[improved]])
            w = fit[improved] - fu[improved]
            w = w / w.sum()
            sF, sCR = F[improved], CR[improved]
            M_F[k] = np.sum(w * sF ** 2) / np.sum(w * sF)
            if terminal[k] or np.max(sCR) == 0.0:
                terminal[k] = True
            else:
                M_CR[k] = np.sum(w * sCR ** 2) / np.sum(w * sCR)
            k = (k + 1) % H
        replace = fu <= fit
        P[replace] = U[replace]
        fit[replace] = fu[replace]
        N_next = max(N_min, int(round((N_min - N_init) / max_nfe * nfe + N_init)))
        if N_next < N:
            keep = np.argsort(fit)[:N_next]
            P, fit, N = P[keep], fit[keep], N_next
        cap = int(round(arc_rate * N))
        if len(archive) > cap:
            archive = archive[rng.choice(len(archive), cap, replace=False)]
        it += 1
        if stop is not None and stop(it, best.f):
            return best
