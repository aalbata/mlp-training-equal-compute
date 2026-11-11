"""Flowchart of the study protocol, drawn from the data set table and the constants of the code.

Shapes follow the classic flowchart set, each with a light-to-medium vertical gradient: documents
for data sets, a diamond for the selection rule, rectangles for processing steps, a parallelogram for
the budget setting, clipped-corner rectangles for the start and end of the loop over data sets, a
stadium for the secondary scaling study and the display shape for the reported results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import configs, data, experiments
from .optimizers import STUDY_METHODS

TEAL = ("#B8E3DA", "#5FB3A3", "#3F8F80")
STYLE = {  # light top colour, medium bottom colour, border colour
    "process": TEAL, "loop_start": TEAL, "loop_end": TEAL,
    "alternative": ("#C6E0FF", "#66AAFF", "#4A86D6"),
    "decision": ("#C3E3A8", "#7DB84E", "#5E9A35"),
    "display": ("#FFD9B0", "#FF9F40", "#E07B1A"),
    "io": ("#FFF2B3", "#FFD54F", "#D6B03A"),
    "document": ("#E6D5EE", "#A780BF", "#8A63A3"),
}
ARROW, LINE_WIDTH, FONT = "#262626", 0.6, 6.5
WIDTH = 3.45  # inches
K = 0.5523  # cubic Bezier constant for a quarter ellipse


def _outline(kind: str, cx: float, cy: float, w: float, h: float):
    from matplotlib.path import Path as MPath

    M, L, C, Z = MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CLOSEPOLY

    def poly(points):
        return MPath(list(points) + [points[0]], [M] + [L] * (len(points) - 1) + [Z])

    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    if kind == "process":
        return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    if kind == "decision":
        return poly([(cx, y1), (x1, cy), (cx, y0), (x0, cy)])
    if kind == "io":
        s = 0.12
        return poly([(x0 + s, y1), (x1, y1), (x1 - s, y0), (x0, y0)])
    if kind == "loop_start":
        c = 0.07
        return poly([(x0, y0), (x1, y0), (x1, y1 - c), (x1 - c, y1), (x0 + c, y1), (x0, y1 - c)])
    if kind == "loop_end":
        c = 0.07
        return poly([(x0 + c, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1), (x0, y1), (x0, y0 + c)])
    if kind == "alternative":
        r = h / 2
        return MPath([(x0 + r, y1), (x1 - r, y1), (x1 - r + K * r, y1), (x1, cy + K * r), (x1, cy),
                      (x1, cy - K * r), (x1 - r + K * r, y0), (x1 - r, y0), (x0 + r, y0), (x0 + r - K * r, y0),
                      (x0, cy - K * r), (x0, cy), (x0, cy + K * r), (x0 + r - K * r, y1), (x0 + r, y1), (x0 + r, y1)],
                     [M, L, C, C, C, C, C, C, L, C, C, C, C, C, C, Z])
    if kind == "display":
        p, r = 0.13, 0.16
        xr = x1 - r
        return MPath([(x0 + p, y1), (xr, y1), (xr + K * r, y1), (x1, cy + K * h / 2), (x1, cy),
                      (x1, cy - K * h / 2), (xr + K * r, y0), (xr, y0), (x0 + p, y0), (x0, cy), (x0 + p, y1)],
                     [M, L, C, C, C, C, C, C, L, L, Z])
    if kind == "document":
        a = 0.045
        yb = y0 + a
        return MPath([(x0, y1), (x1, y1), (x1, yb), (x1 - 0.4 * w, yb - 2.2 * a), (x0 + 0.4 * w, yb + 2.2 * a),
                      (x0, yb), (x0, y1)], [M, L, L, C, C, C, Z])
    raise ValueError(kind)


def _shape(ax, kind, cx, cy, w, h, text, stack=False):
    """One shape with a vertical gradient from a light top to a medium bottom, a thin border and black text."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import PathPatch, Rectangle

    top, bottom, edge = (np.array(to_rgb(c)) for c in STYLE[kind])
    for dx, dy in ([(0.06, 0.05), (0.03, 0.025), (0.0, 0.0)] if stack else [(0.0, 0.0)]):
        path = _outline(kind, cx + dx, cy + dy, w, h)
        patch = PathPatch(path, facecolor="none", edgecolor=edge, linewidth=LINE_WIDTH, zorder=3)
        ax.add_patch(patch)
        ext = path.get_extents()
        bands = 48
        ys = np.linspace(ext.y1, ext.y0, bands + 1)
        for i in range(bands):
            f = i / (bands - 1)
            band = Rectangle((ext.x0 - 0.01, ys[i + 1]), ext.width + 0.02, (ys[i] - ys[i + 1]) * 1.6,
                             facecolor=tuple((1 - f) * top + f * bottom), edgecolor="none", linewidth=0, zorder=2)
            band.set_clip_path(patch)
            ax.add_patch(band)
    ax.text(cx, cy + (0.015 if kind == "document" else 0.0), text, ha="center", va="center", fontsize=FONT,
            color="black", linespacing=1.15, zorder=4)


def _arrow(ax, points, label=None, label_at=None):
    """Thin polyline through the points with an arrowhead on its last segment."""
    from matplotlib.patches import FancyArrowPatch

    for (xa, ya), (xb, yb) in zip(points[:-2], points[1:-1]):
        ax.plot([xa, xb], [ya, yb], color=ARROW, linewidth=LINE_WIDTH, solid_capstyle="butt", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=6, linewidth=LINE_WIDTH,
                                 color=ARROW, shrinkA=0, shrinkB=0, zorder=1))
    if label:
        ax.text(*label_at, label, fontsize=FONT - 0.5, ha="center", va="center", color="black", zorder=4)


def figure_protocol(datasets_csv: Path):
    """Return the flowchart of data selection, tuning, the benchmark runs and the statistical analysis."""
    import matplotlib.pyplot as plt

    t = pd.read_csv(datasets_csv)
    n_total, n_excl = len(t), int((t.role == "excluded").sum())
    n_kept, n_tune = n_total - n_excl, int((t.role == "tuning").sum())
    ev = t[t.role == "evaluation"]
    n_eval, n_cls = len(ev), int((ev.task == "classification").sum())
    n_reg = n_eval - n_cls
    assert n_tune == data.N_TUNING and n_tune + n_eval == n_kept
    split = ":".join(str(round(100 * f)) for f in (data.TRAIN_FRACTION, data.VALID_FRACTION,
                                                    1 - data.TRAIN_FRACTION - data.VALID_FRACTION))

    gap, margin = 0.15, 0.08
    xc, wc = 1.21, 2.30  # main column
    xs, ws = 2.92, 0.98  # side column
    heights = {"suites": 0.36, "rule": 0.56, "split": 0.36, "budget": 0.34, "tune": 0.36, "loop": 0.32,
               "train": 0.48, "end": 0.24, "tests": 0.36, "out": 0.30}
    height = sum(heights.values()) + gap * (len(heights) - 1) + 2 * margin + 0.05
    y, row = height - margin - 0.05, {}
    for name, h in heights.items():
        row[name] = y - h / 2
        y -= h + gap
    half = {k: v / 2 for k, v in heights.items()}

    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.set_axis_off()

    _shape(ax, "document", xc, row["suites"], wc - 0.1, heights["suites"],
           f"{n_total} data sets in OpenML-CC18\nand OpenML-CTR23", stack=True)
    _shape(ax, "decision", xc, row["rule"], 1.86, heights["rule"],
           f"At most {data.MAX_ROWS:,} rows\nand {data.MAX_INPUTS} inputs?")
    _shape(ax, "document", xs, row["rule"], ws, 0.30, f"{n_excl} excluded")
    _shape(ax, "process", xc, row["split"], wc, heights["split"],
           f"Split the {n_kept} kept data sets {split}\n(training ≤ {data.MAX_TRAIN:,} rows, "
           f"validation ≤ {data.MAX_VALID:,})")
    _shape(ax, "document", xs, row["budget"], ws, 0.36, f"{n_tune} tuning data sets\ndrawn at random")
    _shape(ax, "io", xc - 0.06, row["budget"], wc - 0.12, heights["budget"],
           f"Budget per run of {experiments.BUDGET:,} forward-pass\nequivalents, a gradient pass counted as 3")
    _shape(ax, "process", xc, row["tune"], wc, heights["tune"],
           f"Tune each of the {len(STUDY_METHODS)} methods over up to {configs.N_CONFIGS}\n"
           "settings by mean rank of validation loss")
    _shape(ax, "loop_start", xc, row["loop"], wc, heights["loop"],
           f"For each of the {n_eval} evaluation data sets\n({n_cls} classification, {n_reg} regression)")
    _shape(ax, "process", xc, row["train"], wc, heights["train"],
           "Train each method once from the shared initial\nnetwork and score the network with the lowest\n"
           "validation loss on the test split")
    _shape(ax, "loop_end", xc, row["end"], wc, heights["end"], "Next evaluation data set")
    _shape(ax, "process", xc, row["tests"], wc, heights["tests"],
           "Friedman test and planned Wilcoxon\nsigned-rank tests with Holm correction")
    _shape(ax, "display", xc, row["out"], wc - 0.3, heights["out"], "Scores, ranks and paired comparisons")
    y_scaling = (row["train"] + row["end"]) / 2
    _shape(ax, "alternative", xs, y_scaling, ws, 0.50,
           f"Scaling study on\n{len(experiments.SCALING_DATASETS)} data sets with\n"
           f"{len(experiments.SCALING_HIDDEN)} network sizes")

    g, right = 0.012, xc + wc / 2
    _arrow(ax, [(xc, row["suites"] - half["suites"] + 0.01), (xc, row["rule"] + half["rule"] + g)])
    _arrow(ax, [(xc + 0.93, row["rule"]), (xs - ws / 2 - g, row["rule"])], "no", (xc + 1.06, row["rule"] + 0.07))
    _arrow(ax, [(xc, row["rule"] - half["rule"]), (xc, row["split"] + half["split"] + g)], "yes",
           (xc + 0.13, row["rule"] - half["rule"] - 0.06))
    _arrow(ax, [(right, row["split"]), (xs, row["split"]), (xs, row["budget"] + 0.18 + g)])
    _arrow(ax, [(xc, row["split"] - half["split"]), (xc, row["budget"] + half["budget"] + g)])
    _arrow(ax, [(xs, row["budget"] - 0.16), (xs, row["tune"] + 0.08), (right + g, row["tune"] + 0.08)])
    _arrow(ax, [(xc, row["budget"] - half["budget"]), (xc, row["tune"] + half["tune"] + g)])
    _arrow(ax, [(xc, row["tune"] - half["tune"]), (xc, row["loop"] + half["loop"] + g)])
    _arrow(ax, [(right, row["tune"] - 0.08), (xs + 0.25, row["tune"] - 0.08), (xs + 0.25, y_scaling + 0.25 + g)])
    _arrow(ax, [(xc, row["loop"] - half["loop"]), (xc, row["train"] + half["train"] + g)])
    _arrow(ax, [(xc, row["train"] - half["train"]), (xc, row["end"] + half["end"] + g)])
    _arrow(ax, [(xc, row["end"] - half["end"]), (xc, row["tests"] + half["tests"] + g)])
    _arrow(ax, [(xc, row["tests"] - half["tests"]), (xc, row["out"] + half["out"] + g)])
    _arrow(ax, [(xs, y_scaling - 0.25), (xs, row["out"]), (xc + (wc - 0.3) / 2 + g, row["out"])])
    return fig
