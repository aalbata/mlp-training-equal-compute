"""Experiment definitions and the parallel runner.

Three experiments share one runner:
* tuning: every candidate setting of every method on the eight tuning datasets;
* benchmark: every method with its tuned setting on the evaluation datasets, one hidden layer of 16 units;
* scaling: seven methods on eight evaluation datasets at five network sizes.
Each run writes one JSON record and one CSV training history. Existing records are
skipped, so an interrupted experiment resumes where it stopped.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import plugins
from .configs import candidate_configs
from .data import load_prepared
from .optimizers import METHODS
from .problem import BudgetExhausted, Evaluator, Problem

BUDGET = 10_000
BENCHMARK_HIDDEN = (16,)
SCALING_DATASETS = ["optdigits", "letter", "pendigits", "satimage", "phoneme", "spambase",
                    "california_housing", "superconductivity"]
SCALING_HIDDEN = [(4,), (16,), (64,), (256,), (64, 64)]
SCALING_METHODS = ["adam", "pso", "gwo", "cmaes", "lshade", "cso", "pso_adam_stagnation"]


def hidden_tag(hidden) -> str:
    return "-".join(str(h) for h in hidden)


def run_one(problem: Problem, method: str, cfg: dict, budget: float, seed: int):
    """Train once and return the record and the training history."""
    theta0 = problem.model.init(np.random.default_rng([seed, 1]))
    rng = np.random.default_rng([seed, 2])
    ev = Evaluator(problem, budget)
    ev.consider_initial(theta0)
    try:
        METHODS[method](ev, theta0, dict(cfg), rng)
    except BudgetExhausted:
        pass
    ev.finish()
    theta = ev.best_theta if ev.best_theta is not None else theta0
    record = {"method": method, "config": cfg, "hidden": hidden_tag(problem.model.sizes[1:-1]),
              "n_params": problem.model.n_params, "budget": budget, "units_used": round(ev.units, 3),
              "stop_reason": ev.stop_reason, "seconds": round(ev.seconds, 3),
              "best_val_loss": ev.best_val, "best_at_units": round(ev.best_units, 3),
              "handoff_units": None if ev.handoff_units is None else round(ev.handoff_units, 3)}
    for split in ("train", "val", "test"):
        for key, value in problem.metrics(theta, split).items():
            record[f"{split}_{key}"] = value
    return record, pd.DataFrame(ev.history)


_CACHE: dict = {}


def _load(prepared_dir: Path, name: str):
    if name not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.clear()
        _CACHE[name] = load_prepared(prepared_dir, name)
    return _CACHE[name]


def task_path(out_dir: Path, task: dict) -> Path:
    stem = f"{task['dataset']}__{task['method']}"
    if task["experiment"] == "tuning":
        stem += f"__cfg{task['config_index']}"
    if task["experiment"] == "scaling":
        stem += f"__h{hidden_tag(task['hidden'])}"
    return out_dir / task["experiment"] / f"{stem}.json"


def execute(task: dict, prepared_dir: Path, out_dir: Path, seed: int) -> dict:
    path = task_path(out_dir, task)
    if path.exists():
        return json.loads(path.read_text())
    if task.get("plugin"):
        plugins.load(task["plugin"])
    arrays, meta = _load(prepared_dir, task["dataset"])
    problem = Problem(arrays, meta, tuple(task["hidden"]))
    try:
        record, history = run_one(problem, task["method"], task["config"], task["budget"], seed)
    except Exception:
        record, history = {"method": task["method"], "error": traceback.format_exc()}, pd.DataFrame()
    record.update({"experiment": task["experiment"], "dataset": task["dataset"], "task": meta["task"],
                   "data_id": meta["data_id"], "config_index": task.get("config_index"), "seed": seed})
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path.with_suffix(".csv"), index=False)
    path.write_text(json.dumps(record, indent=1, default=float))
    return record


def predicted_cost(task: dict, sizes: dict) -> float:
    n_train, n_inputs, n_out = sizes[task["dataset"]]
    dims = [n_inputs, *task["hidden"], n_out]
    n_params = sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
    return task["budget"] * (5e-10 * n_train * n_params + 1.5e-7 * n_train + 3e-5)


def _with_plugin(task: dict) -> dict:
    """Name the plug-in file of a method loaded from one, so that worker processes can load it too."""
    if task["method"] in plugins.LOADED:
        task["plugin"] = str(plugins.LOADED[task["method"]])
    return task


def tuning_tasks(datasets: pd.DataFrame, methods=None) -> list:
    tasks = []
    for name in datasets[datasets.role == "tuning"].name:
        for method in methods or METHODS:
            for k, cfg in enumerate(candidate_configs(method)):
                tasks.append(_with_plugin({"experiment": "tuning", "dataset": name, "method": method,
                                           "config_index": k, "config": cfg, "hidden": list(BENCHMARK_HIDDEN),
                                           "budget": BUDGET}))
    return tasks


def select_configs(out_dir: Path) -> dict:
    """Setting with the best mean rank of validation loss across the tuning datasets, per method."""
    rows = [json.loads(p.read_text()) for p in (out_dir / "tuning").glob("*.json")]
    df = pd.DataFrame([r for r in rows if "error" not in r])
    chosen = {}
    for method, g in df.groupby("method"):
        g = g.assign(rank=g.groupby("dataset")["best_val_loss"].rank(method="average"))
        mean_rank = g.groupby("config_index")["rank"].mean().sort_values(kind="stable")
        k = int(mean_rank.index[0])
        chosen[method] = {"config_index": k, "mean_rank": float(mean_rank.iloc[0]),
                          "config": candidate_configs(method)[k]}
    return chosen


def benchmark_tasks(datasets: pd.DataFrame, chosen: dict, methods=None) -> list:
    tasks = []
    for name in datasets[datasets.role == "evaluation"].name:
        for method in methods or METHODS:
            tasks.append(_with_plugin({"experiment": "benchmark", "dataset": name, "method": method,
                                       "config": chosen[method]["config"], "hidden": list(BENCHMARK_HIDDEN),
                                       "budget": BUDGET}))
    return tasks


def scaling_tasks(chosen: dict, methods=None) -> list:
    tasks = []
    for name in SCALING_DATASETS:
        for hidden in SCALING_HIDDEN:
            for method in (m for m in SCALING_METHODS if methods is None or m in methods):
                tasks.append({"experiment": "scaling", "dataset": name, "method": method,
                              "config": chosen[method]["config"], "hidden": list(hidden), "budget": BUDGET})
    return tasks


def run_tasks(tasks: list, prepared_dir: Path, out_dir: Path, seed: int, jobs: int, log=print) -> list:
    """Run tasks in parallel, longest first, skipping tasks that already have a record."""
    from joblib import Parallel, delayed

    sizes = {}
    for task in tasks:
        if task["dataset"] not in sizes:
            meta = json.loads((prepared_dir / f"{task['dataset']}.json").read_text())
            n_out = meta["n_classes"] if meta["task"] == "classification" else 1
            sizes[task["dataset"]] = (meta["n_train"], meta["n_inputs"], n_out)
    pending = [t for t in tasks if not task_path(out_dir, t).exists()]
    pending.sort(key=lambda t: -predicted_cost(t, sizes))
    log(f"{len(tasks) - len(pending)} of {len(tasks)} runs already done; running {len(pending)}")
    start = time.time()
    records = []
    results = Parallel(n_jobs=jobs, backend="loky", batch_size=1, return_as="generator_unordered")(
        delayed(execute)(t, prepared_dir, out_dir, seed) for t in pending)
    for i, rec in enumerate(results, 1):
        records.append(rec)
        if "error" in rec:
            log(f"error in {rec['dataset']} {rec['method']}: {rec['error'].splitlines()[-1]}")
        if i % 25 == 0 or i == len(pending):
            log(f"{i}/{len(pending)} runs finished, {time.time() - start:.0f} s elapsed")
    return records
