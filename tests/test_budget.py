import numpy as np
import pytest

from mlp_equal_compute.experiments import run_one
from mlp_equal_compute.problem import BudgetExhausted, Evaluator


def test_unit_costs(cls_problem):
    ev = Evaluator(cls_problem, budget=1000)
    theta = cls_problem.model.init(np.random.default_rng(0))
    ev.set_incumbent(lambda: theta)
    ev.loss(theta)
    assert ev.units == pytest.approx(1.0)
    ev.loss_grad_full(theta)
    assert ev.units == pytest.approx(4.0)
    idx = np.arange(cls_problem.n_train // 4)
    ev.loss_grad_batch(theta, idx)
    assert ev.units == pytest.approx(4.0 + 3.0 * len(idx) / cls_problem.n_train)


def test_budget_stops_run(cls_problem):
    ev = Evaluator(cls_problem, budget=50)
    theta = cls_problem.model.init(np.random.default_rng(0))
    ev.set_incumbent(lambda: theta)
    with pytest.raises(BudgetExhausted):
        for _ in range(100):
            ev.loss(theta)
    assert ev.stop_reason == "budget"
    assert ev.units == pytest.approx(50.0)


def test_run_continues_without_improvement_until_budget(cls_problem):
    ev = Evaluator(cls_problem, budget=2_000, checks=1000)
    theta = cls_problem.model.init(np.random.default_rng(0))
    ev.set_incumbent(lambda: theta)
    with pytest.raises(BudgetExhausted):
        while True:
            ev.loss(theta)
    assert ev.stop_reason == "budget"
    assert ev.units == pytest.approx(2_000.0)
    assert ev.n_checks == 1000


@pytest.mark.parametrize("method,cfg", [
    ("adam", {"lr": 1e-2, "batch_size": 32}),
    ("pso", {"pop_size": 10, "inertia": 0.7, "c1": 1.5, "c2": 1.5, "init_scale": 0.3, "vmax": 0.2}),
])
def test_runs_are_deterministic(cls_problem, method, cfg):
    a, _ = run_one(cls_problem, method, cfg, 600, seed=3)
    b, _ = run_one(cls_problem, method, cfg, 600, seed=3)
    for key in ("units_used", "best_val_loss", "test_accuracy", "stop_reason"):
        assert a[key] == b[key]


def test_diverging_run_falls_back_to_initial_network(reg_problem):
    theta0 = reg_problem.model.init(np.random.default_rng([0, 1]))
    rec, _ = run_one(reg_problem, "sgd_momentum", {"lr": 1e6, "momentum": 0.99, "batch_size": 16}, 600, seed=0)
    assert rec["best_val_loss"] == pytest.approx(reg_problem.val_loss(theta0))
    assert rec["best_at_units"] == 0.0
    assert np.isfinite(rec["test_rmse"])


def test_selected_network_has_best_validation_loss(cls_problem):
    rec, hist = run_one(cls_problem, "adam", {"lr": 3e-2, "batch_size": 16}, 3000, seed=0)
    assert rec["val_loss"] == pytest.approx(rec["best_val_loss"])
    assert rec["best_val_loss"] <= hist["val_loss"].min() + 1e-12
