"""Model training and probabilistic evaluation.

``riskpilot.models.train`` doubles as the command-line entry point
(``python -m riskpilot.models.train``). It is therefore imported lazily here:
importing it eagerly from the package ``__init__`` makes ``runpy`` load the
module twice when it is executed with ``-m`` and emit a ``RuntimeWarning``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from riskpilot.models.evaluate import calibration_table, compute_metrics

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from riskpilot.models.train import BaselineConfig, make_split, run_baseline

__all__ = ["BaselineConfig", "calibration_table", "compute_metrics", "make_split", "run_baseline"]

_TRAIN_EXPORTS = frozenset({"BaselineConfig", "make_split", "run_baseline"})


def __getattr__(name: str) -> Any:
    if name in _TRAIN_EXPORTS:
        from riskpilot.models import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
