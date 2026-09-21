"""Small, fail-closed manifests for reproducible relational artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from riskpilot import config
from riskpilot.features.relational.temporal import POLICY_VERSION

RAW_FILES = {
    "bureau": ("bureau", "bureau_balance"),
    "previous": ("previous",),
    "installments": ("installments",),
    "credit_card": ("credit_card",),
    "pos": ("pos",),
}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Publish a manifest only after its artifacts have been fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def raw_signature(source: str, raw_dir: Path) -> dict:
    result = {}
    for table in RAW_FILES[source]:
        path = raw_dir / config.RELATIONAL_TABLES[table]["filename"]
        result[table] = {"file": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)}
    return result


def builder_signature(source: str) -> dict:
    directory = Path(__file__).parent
    module = "pos_cash" if source == "pos" else source
    return {
        "format": 2,
        "temporal_policy": POLICY_VERSION,
        "code": {
            name: file_hash(directory / name)
            for name in (f"{module}.py", "common.py", "temporal.py", "provenance.py")
        },
        "pandas": pd.__version__,
    }


def table_schema(frame: pd.DataFrame) -> dict:
    if "SK_ID_CURR" not in frame or frame["SK_ID_CURR"].isna().any():
        raise ValueError("Cache customer key is missing.")
    if not frame["SK_ID_CURR"].is_unique:
        raise ValueError("Cache violates one row per customer.")
    return {
        "rows": len(frame),
        "customers": int(frame["SK_ID_CURR"].nunique()),
        "columns": list(frame),
        "dtypes": [str(d) for d in frame.dtypes],
    }


def register_cache(
    source: str, path: Path, frame: pd.DataFrame, *, raw_dir: Path, nrows: int | None = None
) -> dict:
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "raw_dir": str(raw_dir.resolve()),
        "raw_files": raw_signature(source, raw_dir),
        "builder": builder_signature(source),
        "nrows": nrows,
        "schema": table_schema(frame),
        "sha256": file_hash(path),
    }
    write_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def validated_cache(
    source: str, path: Path, *, raw_dir: Path | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Reject legacy, stale, partial, changed-schema or tampered caches."""
    from riskpilot.features.relational.assemble import SOURCE_MODULES

    try:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        directory = raw_dir or Path(manifest["raw_dir"])
        if (
            manifest["source"] != source
            or manifest["nrows"] != nrows
            or manifest["builder"] != builder_signature(source)
            or manifest["raw_files"] != raw_signature(source, directory)
            or manifest["sha256"] != file_hash(path)
        ):
            raise ValueError("Cache provenance mismatch")
        frame = pd.read_parquet(path)
        module = SOURCE_MODULES[source]
        expected = ["SK_ID_CURR", *[f"{module.PREFIX}__{s.name}" for s in module.SPECS]]
        if table_schema(frame) != manifest["schema"] or list(frame) != expected:
            raise ValueError("Cache schema mismatch")
        return frame
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid {source} cache; rebuild relational features: {exc}") from exc


def save_checkpoint(path: Path, context: dict, files: list[Path]) -> None:
    write_json(path, {"context": context, "files": {str(p): file_hash(p) for p in files}})


def validate_checkpoint(path: Path, context: dict, files: list[Path]) -> None:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != {"context": context, "files": {str(p): file_hash(p) for p in files}}:
            raise ValueError("provenance or artifact checksum mismatch")
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Cannot resume {path.name}: {exc}. Run the stage fresh.") from exc
