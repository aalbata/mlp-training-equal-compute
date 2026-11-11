"""Statistics over datasets: ranks, Friedman test, Holm-corrected Wilcoxon tests, win/tie/loss counts.

Each dataset contributes one result per method, so the dataset is the unit of every
test (Demsar, 2006). Classification is scored by test accuracy (higher is better)
and regression by test RMSE on the standardized target (lower is better). Ranks are
computed within each dataset, rank 1 being the best method.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .optimizers import METHODS

SCORE = {"classification": ("test_accuracy", True), "regression": ("test_rmse", False)}
TIE_ACCURACY = 0.005
TIE_RMSE = 0.005


def load_records(runs_dir: Path, experiment: str) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted((runs_dir / experiment).glob("*.json"))]
    df = pd.DataFrame(rows)
    if "error" in df:
        bad = df[df["error"].notna()]
        if len(bad):
            raise RuntimeError(f"{len(bad)} runs in {experiment} ended with an error")
    return df


def score_matrix(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Datasets by methods, holding the test score of each run."""
    metric, _ = SCORE[task]
    sub = df[df.task == task]
    return sub.pivot(index="dataset", columns="method", values=metric)[[m for m in METHODS if m in set(sub.method)]]


def rank_matrix(scores: pd.DataFrame, task: str) -> pd.DataFrame:
    _, higher = SCORE[task]
    return (-scores if higher else scores).rank(axis=1, method="average")


def friedman(ranks: pd.DataFrame) -> dict:
    stat, p = stats.friedmanchisquare(*[ranks[c].to_numpy() for c in ranks.columns])
    return {"chi2": float(stat), "p": float(p), "n_datasets": int(len(ranks)), "n_methods": int(ranks.shape[1])}


def holm(pvalues: dict) -> dict:
    """Holm step-down adjustment of a family of p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def paired_comparisons(scores: pd.DataFrame, task: str, pairs: list) -> pd.DataFrame:
    """Wilcoxon signed-rank test, median difference and win/tie/loss for each (method, reference) pair."""
    metric, higher = SCORE[task]
    tie = TIE_ACCURACY if task == "classification" else TIE_RMSE
    rows, raw = [], {}
    for a, b in pairs:
        diff = (scores[a] - scores[b]) * (1.0 if higher else -1.0)
        exact = diff.round(12)
        wins, losses = int((exact >= tie).sum()), int((exact <= -tie).sum())
        nonzero = diff[diff != 0.0]
        p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 1 else 1.0
        raw[(a, b)] = p
        boot = np.random.default_rng(0).choice(diff.to_numpy(), size=(5000, len(diff)), replace=True)
        lo, hi = np.percentile(np.median(boot, axis=1), [2.5, 97.5])
        rows.append({"method": a, "reference": b, "median_gain": float(np.median(diff)),
                     "ci_low": float(lo), "ci_high": float(hi), "wins": wins,
                     "ties": int(len(diff) - wins - losses), "losses": losses, "p": p})
    adjusted = holm(raw)
    out = pd.DataFrame(rows)
    out["p_holm"] = [adjusted[(r.method, r.reference)] for r in out.itertuples()]
    return out
