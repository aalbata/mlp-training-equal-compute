"""Reproduce the experiments, tables and figures.

    python reproduce.py                        rebuild the tables and figures from the shipped results
    python reproduce.py --full                 download the data, tune, run both experiments, rebuild
    python reproduce.py --full --seed 7        the same with another seed for the split, initialization and training
    python reproduce.py --stage benchmark      run one stage only (prepare, tune, benchmark, scaling, timing, report)

Test a training method of your own under the same protocol (see examples/one_plus_one_es.py):

    python reproduce.py --plugin examples/one_plus_one_es.py --stage prepare --stage tune --stage benchmark --stage compare

With --plugin, the tune and benchmark stages run only the plug-in methods unless --methods names others,
and the compare stage ranks them among the sixteen methods and tests them against Adam and multi-start
Adam on the same splits and initial networks.

The full path needs internet access once, to download the datasets from OpenML.
The shipped run records in results/runs are only read; the report stage on its own rewrites
results/summary, figures/ and tables/ from them, byte for byte. A run that computes anything writes
its records, tables and figures to rerun_seed<k>/, so it never mixes with the shipped results; an
interrupted run resumes in the same folder.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mlp_equal_compute import data, experiments, plugins  # noqa: E402
from mlp_equal_compute.optimizers import METHODS  # noqa: E402

STAGES = ["prepare", "tune", "benchmark", "scaling", "timing", "report", "compare"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def stage_prepare(args, table: pd.DataFrame, prepared: Path) -> None:
    pending = table[table.role != "excluded"]
    for i, row in enumerate(pending.itertuples(), 1):
        if (prepared / f"{row.name}.npz").exists():
            continue
        arrays, meta = data.prepare_dataset(row.data_id, row.task, seed=args.seed, cache_dir=args.data_dir / "openml")
        meta["name"] = row.name
        data.save_prepared(arrays, meta, prepared, row.name)
        log(f"prepared {i}/{len(pending)} {row.name}: {meta['n_train']} training rows, {meta['n_inputs']} inputs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="run every stage from the raw downloads")
    ap.add_argument("--stage", action="append", choices=STAGES, help="run only the named stage (repeatable)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the data split, the initial network and the training randomness")
    ap.add_argument("--jobs", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--reselect", action="store_true",
                    help="apply the selection rule to the OpenML suites again instead of reading datasets.csv")
    ap.add_argument("--plugin", action="append", type=Path, default=[],
                    help="Python file that defines a training method (repeatable); see mlp_equal_compute/plugins.py")
    ap.add_argument("--methods", default=None,
                    help="comma-separated methods for the tune, benchmark and scaling stages (default: all, "
                         "or only the plug-in methods when --plugin is given)")
    args = ap.parse_args()
    added = [plugins.load(p) for p in args.plugin]
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else (added or None)
    unknown = [m for m in methods or [] if m not in METHODS]
    if unknown:
        ap.error(f"unknown methods {', '.join(unknown)}; available: {', '.join(METHODS)}")
    if args.full and added:
        ap.error("--full rebuilds the study; with --plugin, name the stages, for example "
                 "--stage prepare --stage tune --stage benchmark --stage compare")
    stages = [s for s in STAGES if s != "compare"] if args.full else (args.stage or ["report"])
    if "compare" in stages and not added:
        ap.error("the compare stage needs at least one --plugin")
    shipped = HERE / "results"
    if args.results_dir is None:
        computing = any(s != "report" for s in stages)
        args.results_dir = shipped if (not computing and args.seed == 0) else HERE / f"rerun_seed{args.seed}"
    out_dir = HERE if args.results_dir.resolve() == shipped.resolve() else args.results_dir

    if args.reselect:
        data.set_api_cache(args.data_dir / "openml_api")
        data.select_datasets(args.data_dir / "openml").to_csv(HERE / "datasets.csv", index=False)
    table = pd.read_csv(HERE / "datasets.csv")
    prepared = args.data_dir / f"prepared_seed{args.seed}"
    out = args.results_dir / "runs"
    tuned_path = args.results_dir / "tuned_configs.json"

    if "prepare" in stages:
        log("preparing datasets")
        stage_prepare(args, table, prepared)
    if "tune" in stages:
        log("tuning")
        experiments.run_tasks(experiments.tuning_tasks(table, methods), prepared, out, args.seed, args.jobs, log)
        tuned_path.write_text(json.dumps(experiments.select_configs(out), indent=1))
    if "benchmark" in stages:
        log("benchmark")
        chosen = json.loads(tuned_path.read_text())
        experiments.run_tasks(experiments.benchmark_tasks(table, chosen, methods), prepared, out, args.seed,
                              args.jobs, log)
    if "scaling" in stages:
        log("scaling")
        chosen = json.loads(tuned_path.read_text())
        experiments.run_tasks(experiments.scaling_tasks(chosen, methods), prepared, out, args.seed, args.jobs, log)
    if "timing" in stages:
        from mlp_equal_compute import timing

        log("timing forward and backward passes")
        timing.measure(table, prepared).to_csv(args.results_dir / "timing.csv", index=False)
    if "report" in stages:
        from mlp_equal_compute import report

        log("tables and figures")
        report.build(args.results_dir, out_dir, HERE / "datasets.csv")
    if "compare" in stages:
        from mlp_equal_compute import compare

        has_reference = any((out / "benchmark").glob("*__adam.json"))
        if not has_reference and args.seed != 0:
            ap.error(f"no Adam records for seed {args.seed} in {out}; run --stage benchmark --methods adam,adam_multistart "
                     "with this seed first")
        reference = out if has_reference else shipped / "runs"
        log(f"comparing {', '.join(added)} with the methods in {reference}")
        result = compare.compare(out, reference, added, args.results_dir)
        for line in compare.report_lines(result, added):
            print(line)
        log(f"tables written to {args.results_dir}")
    log("done")


if __name__ == "__main__":
    main()
