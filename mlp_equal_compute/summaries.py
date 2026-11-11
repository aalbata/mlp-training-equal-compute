"""Summary tables computed from the run records of the benchmark and scaling experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis
from .optimizers import FAMILIES, LABELS, STUDY_METHODS

REFERENCE = "adam"


def method_summary(df: pd.DataFrame, task: str) -> pd.DataFrame:
    scores = analysis.score_matrix(df, task)
    ranks = analysis.rank_matrix(scores, task)
    sub = df[df.task == task]
    secs = sub.groupby("method")["seconds"].median()
    out = pd.DataFrame({
        "method": scores.columns,
        "label": [LABELS[m] for m in scores.columns],
        "family": [FAMILIES[m] for m in scores.columns],
        "mean_score": scores.mean().to_numpy(),
        "median_score": scores.median().to_numpy(),
        "mean_rank": ranks.mean().to_numpy(),
        "n_best": (ranks == ranks.min(axis=1).to_numpy()[:, None]).sum().to_numpy(),
        "median_seconds": secs.reindex(scores.columns).to_numpy(),
    })
    out["task"] = task
    out["n_datasets"] = len(scores)
    return out.sort_values("mean_rank").reset_index(drop=True)


def planned_pairs() -> list:
    """The fixed family of paired comparisons: every method against Adam, plus the hybrid contrasts."""
    pairs = [(m, REFERENCE) for m in STUDY_METHODS if m != REFERENCE]
    pairs += [("pso_adam_stagnation", "pso_adam_fixed"), ("gwo_adam_stagnation", "gwo_adam_fixed")]
    pairs += [(m, "adam_multistart") for m in
              ("pso_adam_fixed", "gwo_adam_fixed", "pso_adam_stagnation", "gwo_adam_stagnation")]
    return pairs


def paired_table(df: pd.DataFrame, task: str) -> pd.DataFrame:
    scores = analysis.score_matrix(df, task)
    pairs = [(a, b) for a, b in planned_pairs() if a in scores and b in scores]
    out = analysis.paired_comparisons(scores, task, pairs)
    out["task"] = task
    return out


def handoff_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df.method.isin(["pso_adam_fixed", "gwo_adam_fixed", "pso_adam_stagnation",
                             "gwo_adam_stagnation", "adam_multistart"])].copy()
    sub["handoff_fraction"] = sub["handoff_units"] / sub["budget"]
    return sub[["dataset", "task", "method", "handoff_fraction"]]


def scaling_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, hidden), g in df.groupby(["dataset", "hidden"]):
        task = g.task.iloc[0]
        metric, higher = analysis.SCORE[task]
        ref = g.loc[g.method == REFERENCE, metric]
        ref = float(ref.iloc[0]) if len(ref) else np.nan
        for r in g.itertuples():
            score = getattr(r, metric)
            gain = (score - ref) if higher else (ref - score)
            rows.append({"dataset": dataset, "task": task, "hidden": hidden, "n_params": r.n_params,
                         "method": r.method, "score": score, "gain_over_adam": gain, "seconds": r.seconds})
    return pd.DataFrame(rows).sort_values(["dataset", "n_params", "method"]).reset_index(drop=True)
