import numpy as np
import pytest

from mlp_equal_compute.configs import candidate_configs
from mlp_equal_compute.experiments import run_one
from mlp_equal_compute.optimizers import METHODS
from mlp_equal_compute.optimizers.hybrid import StagnationRule

from conftest import make_problem


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_method_trains_within_budget(method, task):
    problem = make_problem(task, hidden=(6,))
    cfg = candidate_configs(method)[0]
    rec, hist = run_one(problem, method, cfg, budget=800, seed=0)
    assert rec["units_used"] <= 800 + 3.0 + 1e-6
    assert rec["stop_reason"] in ("budget", "converged")
    if method != "lbfgs":
        assert rec["units_used"] >= 800 - 1e-6
    assert np.isfinite(rec["best_val_loss"])
    if task == "classification":
        assert 0.0 <= rec["test_accuracy"] <= 1.0
    else:
        assert np.isfinite(rec["test_rmse"])
    if "adam_" in method or method == "adam_multistart":
        assert rec["handoff_units"] is not None


def test_candidate_configs_are_distinct_and_start_with_default():
    for method in METHODS:
        configs = candidate_configs(method)
        keys = {tuple(sorted(c.items())) for c in configs}
        assert len(keys) == len(configs)
        assert 1 <= len(configs) <= 10
    assert len(candidate_configs("lbfgs")) == 4


def test_documented_search_spaces_match_the_sampler():
    from mlp_equal_compute import configs
    from mlp_equal_compute.optimizers import STUDY_METHODS

    assert set(configs.SEARCH_SPACES) == set(STUDY_METHODS)
    rng = np.random.default_rng(0)

    def inside(value, spec):
        if spec[0] == "choice":
            return value in spec[1]
        return spec[1] <= value <= spec[2]

    for method in STUDY_METHODS:
        space = configs.SEARCH_SPACES[method]
        default = configs._space(method)
        assert set(default) == set(space) and all(inside(v, space[k]) for k, v in default.items())
        seen = {k: set() for k, s in space.items() if s[0] == "choice"}
        for _ in range(400):
            cfg = configs._space(method, rng)
            assert set(cfg) == set(space) and all(inside(v, space[k]) for k, v in cfg.items())
            for k in seen:
                seen[k].add(cfg[k])
        assert all(seen[k] == set(space[k][1]) for k in seen)


def test_stagnation_rule_follows_zhang_et_al():
    class Ev:
        units = 0.0
    rule = StagnationRule(Ev(), stall_iterations=10, tol=0.0, cap_units=1e9)
    assert not rule(1, 5.0)
    assert not rule(2, 4.0)
    fired = [rule(3 + i, 4.0) for i in range(10)]
    assert fired == [False] * 9 + [True]


def test_population_methods_gain_over_initial_network():
    problem = make_problem("classification", hidden=(6,))
    theta0 = problem.model.init(np.random.default_rng([0, 1]))
    start = problem.val_loss(theta0)
    for method in ("pso", "de", "lshade", "cmaes", "cso"):
        rec, _ = run_one(problem, method, candidate_configs(method)[0], budget=2000, seed=0)
        assert rec["best_val_loss"] < start
