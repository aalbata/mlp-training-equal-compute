"""Training methods defined in separate Python files (plug-ins), run under the protocol of this study.

A plug-in file defines five names:

``NAME``
    Method key: lower-case letters, digits and underscores, different from the sixteen methods.
``LABEL``
    Name shown in tables.
``FAMILY``
    One of "gradient", "population", "adaptive population" or "hybrid".
``space(rng=None)``
    Returns the default setting (a dict) when ``rng`` is None and a random setting drawn with the
    NumPy generator ``rng`` otherwise. Tuning evaluates the default and up to nine random settings,
    the same allowance as every method of the study.
``run(ev, theta0, cfg, rng)``
    Trains from the shared initial network ``theta0`` (a flat NumPy vector of weights and biases)
    with the setting ``cfg`` and the generator ``rng``. Compute is spent only through the evaluator
    ``ev``: ``ev.loss(theta)`` scores a network on the whole training split for one unit,
    ``ev.loss_grad_batch(theta, idx)`` returns the loss and gradient on the training rows ``idx``
    for 3 b / n units, and ``ev.loss_grad_full(theta)`` returns them on the whole training split
    for 3 units. Before spending, ``run`` calls ``ev.set_incumbent(f)``, where ``f()`` returns the
    network the method would report at that moment; the evaluator scores it on the validation split
    after every thousandth of the budget. When the budget is spent the evaluator raises
    ``BudgetExhausted``, which ends the run; a method that stops earlier simply returns.

``examples/one_plus_one_es.py`` is a complete plug-in.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from . import configs
from .optimizers import FAMILIES, LABELS, METHODS

FAMILY_NAMES = ("gradient", "population", "adaptive population", "hybrid")
REQUIRED = ("NAME", "LABEL", "FAMILY", "space", "run")
LOADED: dict = {}  # method key -> resolved path of the plug-in file that defines it


def load(path) -> str:
    """Register the method defined in a plug-in file and return its key; loading the same file again has no effect."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"plug-in file not found: {path}")
    for name, source in LOADED.items():
        if source == path:
            return name
    spec = importlib.util.spec_from_file_location(f"mlp_equal_compute_plugin_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [attr for attr in REQUIRED if not hasattr(module, attr)]
    if missing:
        raise ValueError(f"{path.name} does not define {', '.join(missing)}; see mlp_equal_compute/plugins.py")
    name = module.NAME
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"{path.name}: NAME must use lower-case letters, digits and underscores, not {name!r}")
    if name in METHODS:
        raise ValueError(f"{path.name}: {name!r} is already a training method; choose another NAME")
    if module.FAMILY not in FAMILY_NAMES:
        raise ValueError(f"{path.name}: FAMILY must be one of {', '.join(FAMILY_NAMES)}, not {module.FAMILY!r}")
    if not isinstance(module.space(), dict):
        raise ValueError(f"{path.name}: space() must return the default setting as a dict")
    METHODS[name] = module.run
    LABELS[name] = str(module.LABEL)
    FAMILIES[name] = module.FAMILY
    configs.PLUGIN_SPACES[name] = module.space
    LOADED[name] = path
    return name
