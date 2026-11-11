import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlp_equal_compute import compare, configs, experiments, plugins
from mlp_equal_compute.optimizers import LABELS, METHODS, STUDY_METHODS

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "one_plus_one_es.py"


def test_example_plugin_trains_within_budget(cls_problem):
    name = plugins.load(EXAMPLE)
    assert name == "one_plus_one_es" and name in METHODS and name not in STUDY_METHODS
    assert LABELS[name] == "(1+1)-ES" and plugins.load(EXAMPLE) == name
    candidates = configs.candidate_configs(name)
    assert candidates[0] == {"sigma0": 0.1} and len(candidates) == configs.N_CONFIGS
    record, _ = experiments.run_one(cls_problem, name, candidates[0], budget=300, seed=0)
    theta0 = cls_problem.model.init(np.random.default_rng([0, 1]))
    assert record["stop_reason"] == "budget" and 300 <= record["units_used"] <= 303
    assert record["best_val_loss"] <= cls_problem.val_loss(theta0)


def test_plugin_tasks_name_the_plugin_file():
    name = plugins.load(EXAMPLE)
    table = pd.DataFrame({"name": ["a", "b"], "role": ["tuning", "evaluation"]})
    tasks = experiments.tuning_tasks(table, [name])
    assert len(tasks) == configs.N_CONFIGS and {t["plugin"] for t in tasks} == {str(EXAMPLE)}
    assert "plugin" not in experiments.tuning_tasks(table, ["adam"])[0]


def test_invalid_plugins_are_rejected(tmp_path):
    taken = tmp_path / "taken.py"
    taken.write_text("NAME = 'adam'\nLABEL = 'A'\nFAMILY = 'gradient'\n"
                     "def space(rng=None):\n    return {}\n\ndef run(ev, theta0, cfg, rng):\n    pass\n")
    with pytest.raises(ValueError, match="already a training method"):
        plugins.load(taken)
    incomplete = tmp_path / "incomplete.py"
    incomplete.write_text("NAME = 'incomplete'\n")
    with pytest.raises(ValueError, match="does not define"):
        plugins.load(incomplete)
    with pytest.raises(FileNotFoundError):
        plugins.load(tmp_path / "absent.py")


def _write_records(folder, method, accuracies, seed=0):
    (folder / "benchmark").mkdir(parents=True, exist_ok=True)
    for i, acc in enumerate(accuracies):
        record = {"experiment": "benchmark", "dataset": f"d{i}", "task": "classification", "method": method,
                  "test_accuracy": acc, "seconds": 1.0, "seed": seed}
        (folder / "benchmark" / f"d{i}__{method}.json").write_text(json.dumps(record))


def test_compare_ranks_and_tests_a_new_method(tmp_path):
    name = plugins.load(EXAMPLE)
    reference, new = tmp_path / "reference", tmp_path / "new"
    base = np.linspace(0.6, 0.9, 12)
    _write_records(reference, "adam", base)
    _write_records(reference, "adam_multistart", base + 0.001)
    _write_records(new, name, base - 0.02)
    result = compare.compare(new, reference, [name], tmp_path / "out")
    table = result["classification"]["comparison"].set_index("reference")
    assert table.loc["adam", "losses"] == 12 and table.loc["adam", "median_gain"] == pytest.approx(-0.02)
    assert (tmp_path / "out" / "comparison_classification.csv").exists()
    _write_records(new, name, base - 0.02, seed=7)
    with pytest.raises(ValueError, match="different seeds"):
        compare.compare(new, reference, [name], tmp_path / "out")
