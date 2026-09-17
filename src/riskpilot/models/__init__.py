"""Model training, challengers and probabilistic evaluation.

``riskpilot.models.train`` and ``riskpilot.models.challengers`` double as
command-line entry points (``python -m ...``). They - and ``comparison``, which
imports ``train`` - are therefore imported lazily here: importing them eagerly
from the package ``__init__`` makes ``runpy`` load the executed module twice and
emit a ``RuntimeWarning``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from riskpilot.models.evaluate import calibration_table, compute_metrics

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from riskpilot.models.challengers import ChallengerSpec, fit_challenger, run_selection_stage
    from riskpilot.models.comparison import load_frozen_split, paired_bootstrap
    from riskpilot.models.train import BaselineConfig, make_split, run_baseline

__all__ = [
    "BaselineConfig",
    "ChallengerSpec",
    "calibration_table",
    "compute_metrics",
    "fit_challenger",
    "load_frozen_split",
    "make_split",
    "paired_bootstrap",
    "run_baseline",
    "run_selection_stage",
]

_LAZY_EXPORTS: dict[str, str] = {
    "BaselineConfig": "riskpilot.models.train",
    "make_split": "riskpilot.models.train",
    "run_baseline": "riskpilot.models.train",
    "ChallengerSpec": "riskpilot.models.challengers",
    "fit_challenger": "riskpilot.models.challengers",
    "run_selection_stage": "riskpilot.models.challengers",
    "load_frozen_split": "riskpilot.models.comparison",
    "paired_bootstrap": "riskpilot.models.comparison",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
