"""Tables, figures and the numbers quoted in the text, all computed from the run records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import analysis, diagram, summaries
from .experiments import BUDGET, SCALING_DATASETS, SCALING_HIDDEN, SCALING_METHODS
from .optimizers import FAMILIES, LABELS, METHODS

FAMILY_COLORS = {"gradient": "#2a78d6", "population": "#eb6834",
                 "adaptive population": "#1baf7a", "hybrid": "#eda100"}
FAMILY_ORDER = list(FAMILY_COLORS)
WIDTH = 13.4 / 2.54  # text width of the journal page, in inches
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
MINUS = "−"
POPULATION = ["ga", "pso", "gwo", "de", "cmaes", "lshade", "cso"]
HYBRIDS = ["pso_adam_fixed", "gwo_adam_fixed", "pso_adam_stagnation", "gwo_adam_stagnation"]
TASK_TITLE = {"classification": "Classification", "regression": "Regression"}


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "standard",
        "pdf.fonttype": 42, "axes.grid": False,
    })


def _count_ticks(ax, low: float, high: float) -> None:
    """Major ticks at powers of ten between low and high, labelled 100, 1,000, 10,000; no minor labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ticks = [10.0 ** k for k in range(0, 7) if low <= 10.0 ** k <= high]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Creator": "Matplotlib", "Producer": "Matplotlib", "Author": None, "Title": None,
                                "Subject": None, "Keywords": None, "CreationDate": None})
    plt.close(fig)


def gains_vs_reference(scores: pd.DataFrame, task: str, reference: str = "adam") -> pd.DataFrame:
    """Per data set and method: accuracy gain in percentage points, or RMSE reduction, relative to Adam."""
    diff = scores.sub(scores[reference], axis=0)
    diff = 100.0 * diff if task == "classification" else -diff
    return diff.drop(columns=[reference])


def _symlog_ticks(ax, task):
    from matplotlib.ticker import FixedLocator, FuncFormatter

    if task == "classification":
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
        ticks = [-30, -10, -3, -1, 0, 1, 3, 10]
    else:
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
        ticks = [-0.1, -0.01, 0, 0.01, 0.1]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", MINUS)))
    ax.xaxis.set_minor_locator(FixedLocator([]))


def figure_gains(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    tasks = [t for t in ("classification", "regression") if (df.task == t).any()]
    height = 2.55 * len(tasks) + 0.3
    fig, axes = plt.subplots(len(tasks), 1, figsize=(WIDTH, height), squeeze=False)
    for ax, task in zip(axes[:, 0], tasks):
        gains = gains_vs_reference(analysis.score_matrix(df, task), task)
        order = gains.median().sort_values(kind="stable").index.tolist()
        data = [gains[m].dropna().to_numpy() for m in order]
        bp = ax.boxplot(data, vert=False, widths=0.62, patch_artist=True, showfliers=True, whis=(5, 95),
                        medianprops={"color": INK, "linewidth": 0.9},
                        flierprops={"marker": "o", "markersize": 1.5, "markeredgewidth": 0,
                                    "markerfacecolor": MUTED, "alpha": 0.7},
                        whiskerprops={"linewidth": 0.5}, capprops={"linewidth": 0.5}, boxprops={"linewidth": 0.5})
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(FAMILY_COLORS[FAMILIES[m]])
            patch.set_edgecolor(INK)
        ax.axvline(0.0, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
        ax.set_yticks(range(1, len(order) + 1), [LABELS[m] for m in order])
        _symlog_ticks(ax, task)
        ax.set_xlabel("accuracy gain over Adam\n(percentage points)" if task == "classification"
                      else "RMSE reduction relative to Adam\n(target standard deviations)")
        ax.set_title(f"{TASK_TITLE[task]}, {gains.shape[0]} data sets", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLORS[f], edgecolor=INK, linewidth=0.5)
               for f in FAMILY_ORDER]
    fig.legend(handles, FAMILY_ORDER, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.22 / height, 1, 1), pad=0.3, h_pad=1.0)
    _save(fig, path)


def anytime_curves(runs_dir: Path, df: pd.DataFrame, task: str, methods: list) -> pd.DataFrame:
    """Median over data sets of the best validation loss so far, divided by the lowest final value on that data set.

    Each recorded validation check is placed at the multiple of B/1000 units that triggered it.
    """
    sub = df[df.task == task]
    floor = sub.groupby("dataset")["best_val_loss"].min()
    grid = np.unique(np.round(np.geomspace(100, BUDGET, 40), 3))
    step = BUDGET / 1000
    rows = []
    for m in methods:
        curves = []
        for r in sub[sub.method == m].itertuples():
            h = pd.read_csv(runs_dir / r.experiment / f"{r.dataset}__{r.method}.csv")
            if h.empty:
                continue
            ratio = h.best_val_loss.to_numpy() / max(floor[r.dataset], 1e-12)
            at = np.floor(h.units.to_numpy() / step + 1e-9) * step
            curves.append(np.interp(grid, at, ratio, left=np.nan))
        if curves:
            stack = np.vstack(curves)
            med = np.array([np.nanmedian(col) if np.isfinite(col).any() else np.nan for col in stack.T])
            rows += [{"method": m, "units": u, "median_ratio": v} for u, v in zip(grid, med)]
    return pd.DataFrame(rows)


ANYTIME_METHODS = ["adam", "lbfgs", "adam_multistart", "pso", "cmaes", "lshade", "pso_adam_stagnation"]
ANYTIME_STYLE = {"adam": "-", "lbfgs": (0, (4, 1.5)), "adam_multistart": (0, (1, 1)), "pso": "-",
                 "cmaes": "-", "lshade": (0, (4, 1.5)), "pso_adam_stagnation": (0, (5, 1.5, 1, 1.5))}


def figure_anytime(curves: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    tasks = [t for t in ("classification", "regression") if t in curves and not curves[t].empty]
    height = 2.9
    fig, axes = plt.subplots(1, len(tasks), figsize=(WIDTH, height), squeeze=False)
    handles = {}
    for ax, task in zip(axes[0], tasks):
        c = curves[task]
        for m in ANYTIME_METHODS:
            s = c[c.method == m].dropna()
            if s.empty:
                continue
            (line,) = ax.plot(s.units, s.median_ratio, color=FAMILY_COLORS[FAMILIES[m]],
                              linestyle=ANYTIME_STYLE[m], linewidth=1.1)
            handles[m] = line
        ax.set_xscale("log")
        ax.set_yscale("log")
        top = float(np.nanmax(c.median_ratio))
        ticks = [v for v in (1, 1.5, 2, 3, 5, 10, 20) if v <= top * 1.05]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(90, BUDGET * 1.05)
        _count_ticks(ax, 90, BUDGET * 1.05)
        ax.set_xlabel("compute spent\n(forward-pass equivalents)")
        ax.set_ylabel("best validation loss / lowest final value")
        ax.set_title(TASK_TITLE[task], loc="left")
    order = [m for m in ANYTIME_METHODS if m in handles]
    fig.legend([handles[m] for m in order], [LABELS[m] for m in order], loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=3.0, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.48 / height, 1, 1), pad=0.3)
    _save(fig, path)


SCALING_MARKERS = {"adam": "o", "pso": "s", "gwo": "D", "cmaes": "^", "lshade": "v", "cso": "<",
                   "pso_adam_stagnation": "P"}
SCALING_STYLE = {"adam": "-", "pso": "-", "gwo": (0, (3, 1.5)), "cmaes": "-", "lshade": (0, (3, 1.5)),
                 "cso": (0, (1, 1)), "pso_adam_stagnation": (0, (3, 1.5))}


def figure_scaling(table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [d for d in SCALING_DATASETS if d in set(table.dataset)]
    ncol = 3
    nrow = int(np.ceil((len(names) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, 1.68 * nrow), squeeze=False)
    low, high = 0.8 * table.n_params.min(), 1.25 * table.n_params.max()
    for ax, name in zip(axes.ravel(), names):
        g = table[table.dataset == name]
        for m in SCALING_METHODS:
            s = g[g.method == m].sort_values("n_params")
            if s.empty:
                continue
            color = FAMILY_COLORS[FAMILIES[m]]
            ax.plot(s.n_params, s.score, color=color, marker=SCALING_MARKERS[m],
                    linestyle=SCALING_STYLE[m], markersize=3.0, markeredgewidth=0.4, markeredgecolor=SURFACE,
                    linewidth=0.9)
            two = s[s.hidden.astype(str) == "64-64"]
            ax.plot(two.n_params, two.score, linestyle="none", marker=SCALING_MARKERS[m], markersize=3.4,
                    markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(low, high)
        _count_ticks(ax, low, high)
        ax.set_title(name, loc="left")
        ax.set_ylabel("test accuracy" if g.task.iloc[0] == "classification" else "test RMSE (target s.d.)")
        ax.set_xlabel("trainable parameters")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=FAMILY_COLORS[FAMILIES[m]], marker=SCALING_MARKERS[m], markersize=3.5,
                          linestyle=SCALING_STYLE[m], markeredgewidth=0.4, markeredgecolor=SURFACE, linewidth=0.9)
               for m in SCALING_METHODS]
    labels = [LABELS[m].replace(", ", ",\n") for m in SCALING_METHODS]
    axes.ravel()[-1].legend(handles, labels, loc="center", frameon=False, handlelength=2.6)
    fig.tight_layout(pad=0.3)
    _save(fig, path)


def _f(x, digits):
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def _signed(x, digits):
    """Signed number in LaTeX math with a fixed number of decimals; values that round to zero carry no sign."""
    text = f"{x:+.{digits}f}"
    if float(text) == 0.0:
        text = text[1:]
    return f"${text}$"


def _p(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


SETTING_NAMES = {"lr": "learning rate", "batch_size": "batch", "momentum": "momentum", "memory": "memory",
                 "pop_size": "population", "mutation_rate": "mutation rate", "mutation_scale": "mutation s.d.",
                 "init_scale": "initial s.d.", "inertia": "inertia", "c1": "$c_1$", "c2": "$c_2$",
                 "vmax": "$v_{\\max}$", "F": "$F$", "CR": "CR", "sigma0": "$\\sigma_0$",
                 "popsize_factor": "population factor", "phi": "$\\varphi$", "explore_fraction": "search share",
                 "stall_iterations": "stall iterations", "tol": "tolerance", "cap_fraction": "search cap",
                 "starts": "starts"}


def _value(v) -> str:
    if isinstance(v, float):
        return f"{v:g}" if float(f"{v:.6g}") == v else f"{v:#.3g}"
    return str(v)


def _setting_text(method: str, cfg: dict) -> str:
    return ", ".join(f"{SETTING_NAMES.get(k, k)} {_value(v)}" for k, v in cfg.items())


def _space_text(spec: tuple) -> str:
    if spec[0] == "choice":
        return ", ".join(_value(v) for v in spec[1])
    kind = "log-uniform" if spec[0] == "log" else "uniform"
    return f"{kind} {_value(spec[1])} to {_value(spec[2])}"


def table_search_spaces(path: Path) -> None:
    """Default setting of every method of the study and, in parentheses, the space tuning draws from."""
    from .configs import SEARCH_SPACES, _space
    from .optimizers import STUDY_METHODS

    left = r">{\raggedright\arraybackslash}"
    lines = [rf"\begin{{tabular}}{{{left}p{{0.22\textwidth}}{left}p{{0.72\textwidth}}}}", r"\toprule",
             r"Method & Default setting (search space) \\", r"\midrule"]
    for m in STUDY_METHODS:
        default = _space(m)
        cells = [f"{SETTING_NAMES.get(k, k)} {_value(default[k])} ({_space_text(s)})"
                 for k, s in SEARCH_SPACES[m].items()]
        lines.append(f"{LABELS[m]} & {'; '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_methods(chosen: dict, path: Path) -> None:
    left = r">{\raggedright\arraybackslash}"
    lines = [rf"\begin{{tabular}}{{{left}p{{0.22\textwidth}}{left}p{{0.12\textwidth}}{left}p{{0.58\textwidth}}}}",
             r"\toprule", r"Method & Family & Tuned setting \\", r"\midrule"]
    for m in (m for m in METHODS if m in chosen):
        family = "gradient (control)" if m == "adam_multistart" else FAMILIES[m]
        lines.append(f"{LABELS[m]} & {family} & {_setting_text(m, chosen[m]['config'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def table_summary(summ: dict, path: Path) -> None:
    cls, reg = summ["classification"].set_index("method"), summ["regression"].set_index("method")
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Classification} & \multicolumn{3}{c}{Regression} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & Accuracy & Rank & Best & RMSE & Rank & Best \\", r"\midrule"]
    for m in summ["classification"].method:
        c, r = cls.loc[m], reg.loc[m]
        lines.append(f"{LABELS[m]} & {_f(c.mean_score, 3)} & {c.mean_rank:.2f} & {int(c.n_best)} & "
                     f"{_f(r.mean_score, 3)} & {r.mean_rank:.2f} & {int(r.n_best)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


COMPARISON_GROUPS = {"adam": "Against Adam", "pso_adam_fixed": "Stagnation trigger against fixed split",
                     "gwo_adam_fixed": "Stagnation trigger against fixed split",
                     "adam_multistart": "Against multi-start Adam"}


def table_comparisons(pairs: dict, path: Path) -> None:
    """Planned comparisons, one block of rows per task so that the table fits a single text column."""
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & Median [95\% CI] & W/T/L & $p$ \\"]
    tabs = {t: p.set_index(["method", "reference"]) for t, p in pairs.items()}
    for t, scale, digits in (("classification", 100.0, 2), ("regression", 1.0, 3)):
        group = None
        for (a, b) in summaries.planned_pairs():
            if COMPARISON_GROUPS[b] != group:
                group = COMPARISON_GROUPS[b]
                lines += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{{TASK_TITLE[t]}, {group[0].lower()}{group[1:]}}}}} \\"]
            r = tabs[t].loc[(a, b)]
            ci = f"[{_signed(scale * r.ci_low, digits)}, {_signed(scale * r.ci_high, digits)}]"
            lines.append(" & ".join([LABELS[a], f"{_signed(scale * r.median_gain, digits)} {ci}",
                                     f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}", _p(r.p_holm)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_classes(gap_table: pd.DataFrame, path: Path) -> None:
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & Binary & Multi-class & $\rho$ (classes) & $\rho$ (parameters) \\", r"\midrule"]
    for m in POPULATION:
        r = gap_table.loc[m]
        lines.append(f"{LABELS[m]} & {_signed(r.binary, 2)} & {_signed(r.multiclass, 2)} & "
                     f"{_signed(r.rho_classes, 2)} & {_signed(r.rho_params, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def class_gap_table(df: pd.DataFrame, n_classes: pd.Series) -> pd.DataFrame:
    scores = analysis.score_matrix(df, "classification")
    gaps = 100.0 * scores[POPULATION].sub(scores["adam"], axis=0)
    classes = n_classes.reindex(gaps.index)
    params = df[(df.task == "classification") & (df.method == "adam")].set_index("dataset").n_params.reindex(gaps.index)
    binary = classes.reindex(gaps.index) == 2
    rows = {}
    for m in POPULATION:
        rows[m] = {"binary": float(gaps.loc[binary, m].median()), "multiclass": float(gaps.loc[~binary, m].median()),
                   "rho_classes": float(stats.spearmanr(classes.reindex(gaps.index), gaps[m])[0]),
                   "rho_params": float(stats.spearmanr(params, gaps[m])[0]),
                   "n_binary": int(binary.sum()), "n_multiclass": int((~binary).sum())}
    return pd.DataFrame(rows).T


def class_gap_tests(df: pd.DataFrame, n_classes: pd.Series) -> dict:
    """Two-sided Mann-Whitney test of the accuracy difference from Adam, binary against multi-class data sets."""
    scores = analysis.score_matrix(df, "classification")
    gaps = scores[POPULATION].sub(scores["adam"], axis=0)
    binary = n_classes.reindex(gaps.index) == 2
    return {m: float(stats.mannwhitneyu(gaps.loc[binary, m], gaps.loc[~binary, m], alternative="two-sided").pvalue)
            for m in POPULATION}


def convention_check(runs_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Share of the budget, and of a population run's wall-clock time, that Adam needed to reach its final validation loss."""
    rows = []
    for ds in df.dataset.unique():
        a = pd.read_csv(runs_dir / "benchmark" / f"{ds}__adam.csv")
        for m in POPULATION:
            r = df[(df.dataset == ds) & (df.method == m)].iloc[0]
            hit = a[a.best_val_loss <= r.best_val_loss]
            rows.append({"dataset": ds, "method": m,
                         "budget_fraction": float(hit.units.iloc[0] / BUDGET) if len(hit) else np.nan,
                         "time_fraction": float(hit.seconds.iloc[0] / r.seconds) if len(hit) else np.nan})
    return pd.DataFrame(rows)


def build(results_dir: Path, out_dir: Path, datasets_csv: Path) -> dict:
    """Write summary CSV files and numbers.json to results_dir, and LaTeX tables and PDF figures to out_dir."""
    _style()
    runs = results_dir / "runs"
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures, tables = out_dir / "figures", out_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    df = analysis.load_records(runs, "benchmark")
    chosen = json.loads((results_dir / "tuned_configs.json").read_text())
    n = {"budget": BUDGET}
    summ, pairs, curves = {}, {}, {}
    for task in ("classification", "regression"):
        scores = analysis.score_matrix(df, task)
        scores.to_csv(summary_dir / f"{task}_scores.csv")
        summ[task] = summaries.method_summary(df, task)
        summ[task].to_csv(summary_dir / f"{task}_method_summary.csv", index=False)
        pairs[task] = summaries.paired_table(df, task)
        pairs[task].to_csv(summary_dir / f"{task}_comparisons.csv", index=False)
        fr = analysis.friedman(analysis.rank_matrix(scores, task))
        n[f"{task}_n"] = fr["n_datasets"]
        n[f"{task}_friedman_chi2"] = fr["chi2"]
        n[f"{task}_friedman_p"] = fr["p"]
        curves[task] = anytime_curves(runs, df, task, ANYTIME_METHODS)
        curves[task].to_csv(summary_dir / f"{task}_anytime.csv", index=False)
        for r in summ[task].itertuples():
            n[f"{task}_{r.method}_mean_score"] = r.mean_score
            n[f"{task}_{r.method}_mean_rank"] = r.mean_rank
            n[f"{task}_{r.method}_n_best"] = int(r.n_best)
            n[f"{task}_{r.method}_median_seconds"] = r.median_seconds
        for r in pairs[task].itertuples():
            key = f"{task}_{r.method}_vs_{r.reference}"
            scale = 100.0 if task == "classification" else 1.0
            n[key + "_median"] = scale * r.median_gain
            n[key + "_ci_low"] = scale * r.ci_low
            n[key + "_ci_high"] = scale * r.ci_high
            n[key + "_wtl"] = f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}"
            n[key + "_p_holm"] = r.p_holm
    handoff = summaries.handoff_table(df)
    handoff.to_csv(summary_dir / "handoff.csv", index=False)
    for m in ("pso_adam_stagnation", "gwo_adam_stagnation"):
        h = handoff[handoff.method == m].handoff_fraction
        cap = chosen[m]["config"]["cap_fraction"]
        n[f"{m}_handoff_median"] = float(h.median())
        n[f"{m}_handoff_before_cap_share"] = float((h < cap - 1e-6).mean())
        n[f"{m}_handoff_before_cap_runs"] = int((h < cap - 1e-6).sum())
        n[f"{m}_handoff_median_before_cap"] = float(h[h < cap - 1e-6].median())
        n[f"{m}_handoff_below_half_cap_runs"] = int((h < cap / 2).sum())
    nets = df.drop_duplicates("dataset")
    n["bench_params_min"] = int(df.n_params.min())
    n["bench_params_max"] = int(df.n_params.max())
    n["bench_params_median"] = float(nets.n_params.median())
    n["bench_separable_cmaes_nets"] = int((nets.n_params > 1000).sum())
    n["lshade_original_size_over_budget"] = int((18 * nets.n_params > BUDGET).sum())
    n["lshade_original_size_generations_max"] = float((BUDGET / (18 * nets.n_params)).max())
    n["lbfgs_converged_runs"] = int(((df.method == "lbfgs") & (df.stop_reason == "converged")).sum())
    n["total_benchmark_runs"] = int(len(df))
    datasets = pd.read_csv(datasets_csv).set_index("name")
    cg = class_gap_table(df, datasets.n_classes)
    cg.to_csv(summary_dir / "class_gaps.csv")
    table_classes(cg, tables / "class_gaps.tex")
    for m, r in cg.iterrows():
        for k in ("binary", "multiclass", "rho_classes", "rho_params"):
            n[f"classgap_{m}_{k}"] = float(r[k])
    n["classgap_n_binary"] = int(cg.n_binary.iloc[0])
    n["classgap_n_multiclass"] = int(cg.n_multiclass.iloc[0])
    adam_cls = df[(df.task == "classification") & (df.method == "adam")].set_index("dataset")
    n["rho_classes_params"] = float(stats.spearmanr(datasets.n_classes.reindex(adam_cls.index), adam_cls.n_params)[0])
    mw = class_gap_tests(df, datasets.n_classes)
    for m, p in analysis.holm(mw).items():
        n[f"classgap_{m}_mannwhitney_p"] = mw[m]
        n[f"classgap_{m}_mannwhitney_p_holm"] = p
    t = pd.read_csv(results_dir / "timing.csv")
    n["ratio_min"], n["ratio_median"], n["ratio_max"] = float(t.ratio.min()), float(t.ratio.median()), float(t.ratio.max())
    n["ratio_share_above_3"] = float((t.ratio > 3.0).mean())
    threshold = 3.0 / n["ratio_max"]
    n["conv_threshold"] = threshold
    conv = convention_check(runs, df)
    conv.to_csv(summary_dir / "convention_check.csv", index=False)
    for m, g in conv.groupby("method"):
        n[f"conv_{m}_share_within_threshold"] = float((g.budget_fraction <= threshold).sum() / len(g))
        n[f"conv_{m}_share_first_check"] = float((g.budget_fraction <= 0.0101).sum() / len(g))
        n[f"conv_{m}_median_time_fraction"] = float(g.time_fraction.median())
    n["conv_share_within_own_time"] = float((conv.time_fraction <= 1.0).sum() / len(conv))
    _save(diagram.figure_protocol(datasets_csv), figures / "protocol.pdf")
    figure_gains(df, figures / "benchmark_gains.pdf")
    figure_anytime(curves, figures / "anytime.pdf")
    table_methods(chosen, tables / "methods.tex")
    table_search_spaces(tables / "search_spaces.tex")
    table_summary(summ, tables / "benchmark_summary.tex")
    table_comparisons(pairs, tables / "comparisons.tex")
    sc = summaries.scaling_table(analysis.load_records(runs, "scaling"))
    sc.to_csv(summary_dir / "scaling.csv", index=False)
    figure_scaling(sc, figures / "scaling.pdf")
    n["scal_params_min"] = int(sc.n_params.min())
    n["scal_params_max"] = int(sc.n_params.max())
    for task in ("classification", "regression"):
        s = sc[(sc.task == task) & (sc.method != "adam")]
        scale = 100.0 if task == "classification" else 1.0
        tie = analysis.TIE_ACCURACY if task == "classification" else analysis.TIE_RMSE
        for m, g in s.groupby("method"):
            exact = g.gain_over_adam.round(12)
            wins, losses = int((exact >= tie).sum()), int((exact <= -tie).sum())
            n[f"scal_{task}_{m}_wtl"] = f"{wins}/{len(g) - wins - losses}/{losses}"
            n[f"scal_{task}_{m}_median_gain"] = scale * float(g.gain_over_adam.median())
            n[f"scal_{task}_{m}_share_behind"] = float((g.gain_over_adam < 0).mean())
            by_h = g.groupby("hidden").gain_over_adam.median() * scale
            for h, v in by_h.items():
                n[f"scal_{task}_{m}_h{h}_median_gain"] = float(v)
    n["scal_runs"] = int(len(sc))
    (summary_dir / "numbers.json").write_text(json.dumps(n, indent=1, default=float))
    return n
