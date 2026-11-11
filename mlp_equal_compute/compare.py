"""Compare training methods loaded from plug-in files with the sixteen methods of the study.

The new methods need benchmark records on every evaluation data set, made with the same seed as the
reference records, so that every comparison is paired: same data split and same initial network.
Each new method is ranked among all methods and tested against Adam and multi-start Adam with the
statistics of the study (median difference with a 95% bootstrap interval, wins, ties and losses,
and the Wilcoxon signed-rank test with the Holm correction over the comparisons of each task).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import analysis, summaries
from .optimizers import LABELS, STUDY_METHODS

REFERENCES = ("adam", "adam_multistart")


def compare(new_runs: Path, reference_runs: Path, methods: list, out_dir: Path) -> dict:
    """Write ranking_<task>.csv and comparison_<task>.csv to out_dir and return both tables for each task."""
    new = analysis.load_records(new_runs, "benchmark")
    absent = sorted(set(methods) - (set(new["method"]) if "method" in new else set()))
    if absent:
        raise ValueError(f"no benchmark records for {', '.join(absent)} in {new_runs}; run the benchmark stage first")
    new = new[new.method.isin(methods)]
    ref = analysis.load_records(reference_runs, "benchmark")
    ref = ref[ref.method.isin(STUDY_METHODS)]
    for r in REFERENCES:
        if r not in set(ref.method):
            raise ValueError(f"no benchmark records for the reference method {r} in {reference_runs}")
    seeds = sorted(set(new.seed) | set(ref.seed))
    if len(seeds) != 1:
        raise ValueError(f"new and reference records come from different seeds {seeds}; comparisons must be paired")
    df = pd.concat([ref, new], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for task in ("classification", "regression"):
        if not (df.task == task).any():
            continue
        scores = analysis.score_matrix(df, task)
        missing = {m: scores.index[scores[m].isna()].tolist() for m in methods if scores[m].isna().any()}
        if missing:
            raise ValueError(f"missing {task} records: {missing}")
        ranking = summaries.method_summary(df, task)
        pairs = [(m, r) for m in methods for r in REFERENCES]
        table = analysis.paired_comparisons(scores, task, pairs)
        table["task"] = task
        ranking.to_csv(out_dir / f"ranking_{task}.csv", index=False)
        table.to_csv(out_dir / f"comparison_{task}.csv", index=False)
        result[task] = {"ranking": ranking, "comparison": table}
    return result


def report_lines(result: dict, methods: list) -> list:
    """Plain-text summary: mean rank of each new method and its paired comparisons."""
    lines = []
    for task, tables in result.items():
        ranking, comp = tables["ranking"].set_index("method"), tables["comparison"]
        scale, unit, digits = (100.0, "percentage points of accuracy", 2) if task == "classification" else \
            (1.0, "target standard deviations of RMSE reduction", 3)
        lines.append(f"{task}: {int(ranking.n_datasets.iloc[0])} data sets, {len(ranking)} methods "
                     f"(differences in {unit}; positive favours the new method)")
        for m in methods:
            r = ranking.loc[m]
            lines.append(f"  {LABELS[m]}: mean score {r.mean_score:.3f}, mean rank {r.mean_rank:.2f} "
                         f"(best mean rank {ranking.mean_rank.min():.2f})")
            for c in comp[comp.method == m].itertuples():
                lines.append(f"    against {LABELS[c.reference]}: median {scale * c.median_gain:+.{digits}f} "
                             f"[{scale * c.ci_low:+.{digits}f}, {scale * c.ci_high:+.{digits}f}], "
                             f"wins/ties/losses {c.wins}/{c.ties}/{c.losses}, Holm-corrected p {c.p_holm:.3f}")
    return lines
