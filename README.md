# mlp-training-equal-compute

Code, data selection and run records for the paper *A Comparison of Population-Based and Gradient Training of Multilayer Perceptrons* by Ali Al Bataineh (Higher Colleges of Technology) and Ibrahim Alkore Alshalabi (Al-Hussein Bin Talal University).

The study trains the same multilayer perceptron with sixteen methods: four gradient methods (Adam, RMSprop, stochastic gradient descent (SGD) with momentum, and the limited-memory Broyden-Fletcher-Goldfarb-Shanno method (L-BFGS)), seven population-based methods (a genetic algorithm (GA), particle swarm optimization (PSO), the grey wolf optimizer (GWO), differential evolution (DE), the covariance matrix adaptation evolution strategy (CMA-ES), success-history based adaptive differential evolution with linear population size reduction (L-SHADE) and the competitive swarm optimizer (CSO)), four hybrids that run PSO or GWO before Adam with a fixed split or a stagnation trigger, and multi-start Adam. Every run receives the same budget of 10,000 forward-pass equivalents over the training split, with every candidate network of a population-based method scored on the whole training split, and every method receives the same tuning allowance on eight tuning data sets. The evaluation uses 57 classification data sets from OpenML-CC18 and 31 regression data sets from OpenML-CTR23, and a scaling study trains networks with 34 to 21,249 parameters on eight of them.

## Results

Mean test score, mean rank among the sixteen methods (1 is best) and number of data sets with the best score, from `results/summary/*_method_summary.csv`. Accuracy is a fraction; RMSE is the root mean squared error of the standardized target, in standard deviations of the target.

| Method | Accuracy | Rank | Best | RMSE | Rank | Best |
|---|---|---|---|---|---|---|
| Multi-start Adam | 0.851 | 5.32 | 6 | 0.434 | 5.61 | 4 |
| Adam | 0.851 | 5.39 | 14 | 0.423 | 4.32 | 6 |
| RMSprop | 0.849 | 6.26 | 7 | 0.433 | 5.68 | 1 |
| GWO then Adam, fixed split | 0.843 | 6.54 | 9 | 0.436 | 5.77 | 3 |
| SGD with momentum | 0.845 | 6.59 | 6 | 0.463 | 5.94 | 7 |
| L-BFGS | 0.844 | 7.30 | 6 | 0.419 | 5.35 | 3 |
| PSO then Adam, stagnation trigger | 0.843 | 7.34 | 6 | 0.430 | 5.32 | 2 |
| GWO then Adam, stagnation trigger | 0.844 | 7.36 | 6 | 0.423 | 6.19 | 1 |
| PSO then Adam, fixed split | 0.844 | 7.50 | 8 | 0.449 | 8.23 | 1 |
| CMA-ES | 0.840 | 7.71 | 8 | 0.497 | 8.81 | 0 |
| GA | 0.813 | 10.11 | 2 | 0.554 | 12.61 | 1 |
| L-SHADE | 0.806 | 10.26 | 4 | 0.528 | 10.84 | 0 |
| PSO | 0.810 | 10.77 | 2 | 0.539 | 11.61 | 1 |
| CSO | 0.785 | 11.97 | 4 | 0.582 | 14.03 | 0 |
| DE | 0.779 | 12.47 | 3 | 0.572 | 13.94 | 0 |
| GWO | 0.778 | 13.11 | 1 | 0.557 | 11.74 | 2 |

All 21 planned comparisons with Holm-corrected Wilcoxon signed-rank tests are in `results/summary/classification_comparisons.csv` and `results/summary/regression_comparisons.csv`.

## Reproduce

Python 3.12.10 and the pinned packages in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

Rebuild every table and figure, and every number quoted in the paper, from the shipped run records (under three minutes on the machine described below):

```
python reproduce.py
```

Run everything from the raw data (downloads the data sets from OpenML once, then tunes, runs the benchmark, the scaling study and the timing measurement, and rebuilds the report):

```
python reproduce.py --full --jobs 12
```

On the machine used for the paper, a laptop with an Intel Core Ultra 7 155U processor under Windows 11 with 12 worker processes, the full path took about 3.5 hours after 41 minutes of downloading and preparing the data: tuning 51 minutes, benchmark 67 minutes and scaling 90 minutes. Interrupted runs resume where they stopped, because existing run records are skipped.

Each method is trained once on each data set. Accuracy on an individual data set can change under a different random split or initialization; any experiment can be repeated with another seed, for example

```
python reproduce.py --full --seed 7
```

Single stages run with `--stage prepare|tune|benchmark|scaling|timing|report|compare`. The shipped run records in `results/runs/` are only read. Any run that computes something writes its records, tables and figures to `rerun_seed<k>/` (`rerun_seed0/` for the full path above, `rerun_seed7/` here), and `python reproduce.py` alone rewrites `results/summary/`, `figures/` and `tables/` from the shipped records, byte for byte.

The tests check the gradients against finite differences, the compute accounting, the selection rule, determinism, every method and the plug-in interface:

```
python -m pytest -q
```

## Test your own training method

A training method written in a separate Python file runs under the protocol of the study: the same data splits and initial networks, the same budget and accounting of compute, the same tuning allowance of ten settings on the eight tuning data sets, and the same statistics. The file defines `NAME`, `LABEL`, `FAMILY`, a function `space(rng=None)` that returns the default setting or a random one, and a function `run(ev, theta0, cfg, rng)` that trains from the shared initial network and spends compute only through the evaluator `ev` (`ev.loss`, `ev.loss_grad_batch` and `ev.loss_grad_full`). `mlp_equal_compute/plugins.py` documents the interface, and `examples/one_plus_one_es.py` is a complete example, a (1+1) evolution strategy with a one-fifth success rule:

```
python reproduce.py --plugin examples/one_plus_one_es.py --stage prepare --stage tune --stage benchmark --stage compare
```

The prepare stage downloads and prepares the data sets once. On the machine described above, tuning the example and running it on all evaluation data sets then took about 7 minutes with 12 worker processes. The compare stage ranks the new method among the sixteen methods and tests it against Adam and multi-start Adam with the Wilcoxon signed-rank test and the Holm correction, using the shipped run records of seed 0 as the reference, so every comparison is paired. It prints a summary and writes `ranking_<task>.csv` and `comparison_<task>.csv` to `rerun_seed0/`. With another `--seed`, run the reference methods with that seed as well, for example with `--methods one_plus_one_es,adam,adam_multistart`.

## Layout

| Path | Content |
|---|---|
| `reproduce.py` | command-line entry point for all stages |
| `mlp_equal_compute/data.py` | suite selection rule, download, splitting and encoding |
| `mlp_equal_compute/mlp.py` | network, losses and gradients on one flat parameter vector |
| `mlp_equal_compute/problem.py` | compute accounting in forward-pass equivalents and the selection rule |
| `mlp_equal_compute/optimizers/` | the sixteen training methods |
| `mlp_equal_compute/configs.py` | tuning search spaces and default settings |
| `mlp_equal_compute/experiments.py` | tuning, benchmark and scaling runs, in parallel and resumable |
| `mlp_equal_compute/analysis.py`, `summaries.py`, `report.py` | statistics, tables, figures and numbers |
| `mlp_equal_compute/timing.py` | measured cost of a forward and backward pass relative to a forward pass |
| `mlp_equal_compute/diagram.py` | flowchart of the study protocol, `figures/protocol.pdf` |
| `mlp_equal_compute/plugins.py` | interface and loader for training methods in separate files |
| `mlp_equal_compute/compare.py` | ranking and paired comparisons of plug-in methods |
| `examples/one_plus_one_es.py` | example plug-in |
| `datasets.csv` | the 107 suite data sets with their role (evaluation, tuning or excluded) |
| `results/runs/` | one JSON record and one CSV training history per run |
| `results/tuned_configs.json` | setting chosen by tuning for each method |
| `results/timing.csv` | forward and backward timing per evaluation data set |
| `results/summary/` | summary tables and `numbers.json`, written by `python reproduce.py` |
| `figures/`, `tables/` | figures (PDF) and LaTeX tables, written by `python reproduce.py` |
| `tests/` | unit tests |

## Column definitions

`datasets.csv`: `task` (classification or regression), `data_id` (OpenML data set id), `name`, `n_rows`, `n_inputs` (input columns counted by the selection rule from the levels declared on OpenML; the encoder fitted on a training split can give fewer), `n_classes` (0 for regression), `role` (evaluation, tuning or excluded).

Run records `results/runs/<experiment>/<data set>__<method>[__cfg<k>][__h<sizes>].json`:

| Field | Meaning |
|---|---|
| `experiment` | tuning, benchmark or scaling |
| `dataset`, `data_id`, `task` | data set name, OpenML id and task |
| `method`, `config`, `config_index` | training method, its setting and, for tuning, the index of the candidate setting |
| `hidden`, `n_params` | hidden layer sizes and number of trainable parameters |
| `budget`, `units_used` | budget and compute spent, in forward-pass equivalents over the training split |
| `stop_reason` | budget, or converged for L-BFGS when it could no longer reduce the training loss |
| `seconds` | wall-clock time of the run, measured with 12 runs sharing the processor |
| `best_val_loss`, `best_at_units` | validation loss of the reported network and the compute at which it was found |
| `handoff_units` | compute at which a hybrid or multi-start Adam switched to its final Adam phase |
| `train_*`, `val_*`, `test_*` | loss, and accuracy or RMSE and R squared, of the reported network on each split |
| `seed` | seed of the data split, the initial network and the training randomness |

Training histories (`.csv` next to each record) log every tenth validation check: `units`, `seconds`, `train_loss`, `val_loss` and `best_val_loss`.

`results/summary/*_method_summary.csv`: `mean_score` and `median_score` (test accuracy or RMSE over data sets), `mean_rank`, `n_best` (data sets with the best score, ties included), `median_seconds`. `*_comparisons.csv`: `median_gain` (median over data sets of the difference from the reference, positive values favour the method; accuracy as a fraction, RMSE in target standard deviations), `ci_low` and `ci_high` (95% bootstrap interval of the median), `wins`, `ties` and `losses` (a difference below 0.005, accuracy as a fraction or RMSE in target standard deviations, counts as a tie), `p` and `p_holm` (Wilcoxon signed-rank p-value before and after the Holm correction over the 21 comparisons of the task).

## Data

The data sets are downloaded from OpenML. A data set from OpenML-CC18 (study 99) or OpenML-CTR23 (study 353) is kept when it has at most 100,000 rows and at most 300 input columns after encoding; 96 are kept, eight of them are reserved for tuning, and 11 are excluded. Each kept data set is split 60:20:20 into training, validation and test parts, with the training part capped at 5,000 rows and the validation part at 2,500 rows.

## License

MIT License, see `LICENSE`.
