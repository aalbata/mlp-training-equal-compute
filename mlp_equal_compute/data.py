"""Benchmark suites, the dataset selection rule, download and preprocessing.

Datasets come from two curated OpenML suites, OpenML-CC18 (classification) and
OpenML-CTR23 (regression). A dataset is kept when it has at most ``MAX_ROWS`` rows
and at most ``MAX_INPUTS`` input columns after encoding, where a numeric feature
counts as one column, a two-level categorical feature as one column, and a
categorical feature with k levels as k columns, k being the number of levels
declared in the OpenML feature description (distinct values for string features).
The one-hot encoder fitted on a training split can give fewer columns, because it
encodes only the levels that occur in that split. Eight kept datasets, drawn at
random with a fixed seed in proportion to the two tasks, are reserved for tuning.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

OPENML_API = "https://www.openml.org/api/v1/json"
SUITES = {"classification": 99, "regression": 353}
MAX_ROWS = 100_000
MAX_INPUTS = 300
N_TUNING = 8
SELECTION_SEED = 2026
TRAIN_FRACTION, VALID_FRACTION = 0.6, 0.2
MAX_TRAIN, MAX_VALID = 5_000, 2_500
USER_AGENT = "mlp-equal-compute"
_API_CACHE: Path | None = None


def set_api_cache(path: Path | None) -> None:
    """Store OpenML API responses under ``path`` so that repeated selections need no network."""
    global _API_CACHE
    _API_CACHE = Path(path) if path is not None else None


def _get_json(url: str, retries: int = 4, timeout: int = 120) -> dict:
    cached = None
    if _API_CACHE is not None:
        key = url.replace(OPENML_API, "").strip("/").replace("/", "_")
        if len(key) > 48 or "," in key:
            key = key.split("_")[0] + "_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        cached = _API_CACHE / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8")
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(payload, encoding="utf-8")
            return json.loads(payload)
        except Exception as exc:  # network errors are retried with a growing pause
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"OpenML request failed: {url}") from last


def _declared_width(features: list) -> tuple[int, list]:
    """Input columns implied by the declared feature types; string features are returned separately."""
    width, strings = 0, []
    for f in features:
        if "true" in (f.get("is_target"), f.get("is_ignore"), f.get("is_row_identifier")):
            continue
        kind = f["data_type"]
        if kind == "numeric":
            width += 1
        elif kind == "nominal":
            levels = len(f.get("nominal_value", []) or [])
            width += 1 if levels <= 2 else levels
        else:
            strings.append(f["name"])
    return width, strings


def _fetch_frame(data_id: int, cache_dir: Path):
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=str(cache_dir))
    return bunch.data, bunch.target


def select_datasets(cache_dir: Path) -> pd.DataFrame:
    """Apply the selection rule to both suites and draw the tuning datasets."""
    rows = []
    for task, study in SUITES.items():
        ids = _get_json(f"{OPENML_API}/study/{study}")["study"]["data"]["data_id"]
        listing = _get_json(f"{OPENML_API}/data/list/data_id/{','.join(map(str, ids))}/limit/500")
        for ds in listing["data"]["dataset"]:
            q = {x["name"]: x["value"] for x in ds.get("quality", [])}
            did = int(ds["did"])
            feats = _get_json(f"{OPENML_API}/data/features/{did}")["data_features"]["feature"]
            width, strings = _declared_width(feats)
            n_rows = int(float(q["NumberOfInstances"]))
            if strings and width <= MAX_INPUTS and n_rows <= MAX_ROWS:
                X, _ = _fetch_frame(did, cache_dir)
                for col in strings:
                    levels = X[col].dropna().nunique()
                    width += 1 if levels <= 2 else int(levels)
            n_classes = int(float(q.get("NumberOfClasses", 0) or 0)) if task == "classification" else 0
            rows.append(dict(task=task, data_id=did, name=ds["name"], n_rows=n_rows,
                             n_inputs=width, n_classes=n_classes,
                             kept=bool(width <= MAX_INPUTS and n_rows <= MAX_ROWS)))
    table = pd.DataFrame(rows).sort_values(["task", "data_id"]).reset_index(drop=True)
    kept = table[table.kept]
    n_cls = int((kept.task == "classification").sum())
    n_tune_cls = round(N_TUNING * n_cls / len(kept))
    rng = random.Random(SELECTION_SEED)
    tuning = set(rng.sample(list(kept[kept.task == "classification"].name), n_tune_cls))
    tuning |= set(rng.sample(list(kept[kept.task == "regression"].name), N_TUNING - n_tune_cls))
    table["role"] = np.where(~table.kept, "excluded",
                             np.where(table.name.isin(tuning), "tuning", "evaluation"))
    return table.drop(columns="kept")


def _subsample(idx: np.ndarray, size: int, labels, seed: int) -> np.ndarray:
    from sklearn.model_selection import train_test_split

    if len(idx) <= size:
        return idx
    strat = labels[idx] if labels is not None else None
    keep, _ = train_test_split(idx, train_size=size, random_state=seed, stratify=strat)
    return np.sort(keep)


def prepare_dataset(data_id: int, task: str, seed: int, cache_dir: Path) -> tuple[dict, dict]:
    """Download one dataset, split it 60/20/20 and encode it with training-split statistics only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X, y = _fetch_frame(data_id, cache_dir)
    keep = pd.notna(y).to_numpy()
    X, y = X.loc[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    if task == "classification":
        classes = sorted(pd.unique(y.astype(str)))
        labels = y.astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)
        strat = labels
    else:
        classes = []
        labels = y.astype(float).to_numpy()
        strat = None
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=TRAIN_FRACTION, random_state=seed, stratify=strat)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed,
                              stratify=strat[rest] if strat is not None else None)
    tr = _subsample(np.sort(tr), MAX_TRAIN, strat, seed)
    va = _subsample(np.sort(va), MAX_VALID, strat, seed)
    te = np.sort(te)

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    Xo = X.copy()
    for c in cat_cols:
        Xo[c] = X[c].astype(object).where(pd.notna(X[c]), np.nan)
    for c in num_cols:
        Xo[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    encoder = ColumnTransformer(
        [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
         ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
          cat_cols)],
        remainder="drop", sparse_threshold=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A_tr = encoder.fit_transform(Xo.iloc[tr])
        A_va = encoder.transform(Xo.iloc[va])
        A_te = encoder.transform(Xo.iloc[te])
    arrays = {"X_train": A_tr, "X_val": A_va, "X_test": A_te}
    meta = dict(data_id=int(data_id), task=task, n_classes=len(classes), n_inputs=int(A_tr.shape[1]),
                n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)))
    if task == "classification":
        arrays.update(y_train=labels[tr], y_val=labels[va], y_test=labels[te])
    else:
        mu, sd = float(labels[tr].mean()), float(labels[tr].std()) or 1.0
        arrays.update(y_train=(labels[tr] - mu) / sd, y_val=(labels[va] - mu) / sd, y_test=(labels[te] - mu) / sd)
        meta.update(target_mean=mu, target_std=sd)
    for k in list(arrays):
        arrays[k] = np.ascontiguousarray(arrays[k], dtype=np.int64 if (k.startswith("y_") and task == "classification") else np.float64)
    return arrays, meta


def save_prepared(arrays: dict, meta: dict, prepared_dir: Path, name: str) -> None:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prepared_dir / f"{name}.npz", **arrays)
    (prepared_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))


def load_prepared(prepared_dir: Path, name: str) -> tuple[dict, dict]:
    with np.load(prepared_dir / f"{name}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads((prepared_dir / f"{name}.json").read_text())
    return arrays, meta
