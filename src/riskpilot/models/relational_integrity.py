"""Training-only stage provenance and content-verified experiment checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import lightgbm
import pandas as pd
import sklearn

from riskpilot.features.relational.provenance import (
    file_hash,
    save_checkpoint,
    validate_checkpoint,
    validated_cache,
)
from riskpilot.models import challengers


def training_context(X, y, ids, paths, cfg) -> dict:
    """No test labels, predictions or metrics are inputs to a selection checkpoint."""
    for source in cfg.sources:
        validated_cache(source, paths.relational_dir / f"{source}.parquet")
    digest = hashlib.sha256()
    for obj in (X, y, ids):
        digest.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
    directory = Path(__file__).parent
    return {
        "training_sha256": digest.hexdigest(),
        "columns": list(X),
        "dtypes": [str(d) for d in X.dtypes],
        "config": cfg.to_dict(),
        "ms2_selection": file_hash(paths.challenger_selection_path),
        "tables": {s: file_hash(paths.relational_dir / f"{s}.manifest.json") for s in cfg.sources},
        "code": {
            str(p.relative_to(directory.parent)): file_hash(p)
            for folder in (directory, directory.parent / "features")
            for p in sorted(folder.rglob("*.py"))
        },
        "versions": {
            "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }


class CheckpointTrialLog(challengers.TrialLog):
    """Reuse a trial only when its training/config context and CSV match."""

    def __init__(self, path: Path, context: dict, *, resume: bool = False):
        self.context = context
        self.manifest = path.with_suffix(".checkpoint.json")
        if resume and path.exists():
            validate_checkpoint(self.manifest, context, [path])
        super().__init__(path, resume=resume)

    def append(self, record):
        result = super().append(record)
        save_checkpoint(self.manifest, self.context, [self.path])
        return result
