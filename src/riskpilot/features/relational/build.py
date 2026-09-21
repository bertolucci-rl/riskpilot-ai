"""Build (and cache) the customer-level relational tables, one source at a time.

Run from the command line::

    python -m riskpilot.features.relational.build                 # all sources, cached
    python -m riskpilot.features.relational.build --sources installments --force

Each source is read with an explicit column subset and compact dtypes,
aggregated to one row per ``SK_ID_CURR``, verified, and written to
``data/processed/relational/<source>.parquet`` (git-ignored, reproducible).
A build log (``relational_build_log.json``, kept in ``data/processed/relational``
and copied to ``artifacts/metrics``) records for every source the raw-file
signature, input rows, output customers, feature count, run time and output
size; a source is rebuilt only when its raw files changed, when ``--force`` is
given, or when the cached table is missing. The data-quality audit of every
source is written to ``artifacts/metrics/relational_data_audit.json`` and the
feature catalog to ``artifacts/metrics/relational_feature_catalog.csv``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from riskpilot import __version__, config
from riskpilot.features.relational.assemble import (
    SOURCE_MODULES,
    SOURCES,
    feature_catalog,
    table_path,
)
from riskpilot.features.relational.common import ID

logger = logging.getLogger(__name__)

RAW_FILES: dict[str, tuple[str, ...]] = {
    "bureau": ("bureau", "bureau_balance"),
    "previous": ("previous",),
    "installments": ("installments",),
    "credit_card": ("credit_card",),
    "pos": ("pos",),
}


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def raw_signature(source: str, raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Size and modification time of every raw file a source depends on."""
    out = {}
    for table in RAW_FILES[source]:
        path = raw_dir / config.RELATIONAL_TABLES[table]["filename"]
        stat = path.stat() if path.is_file() else None
        out[table] = {
            "file": path.name,
            "size_bytes": stat.st_size if stat else None,
            "mtime": stat.st_mtime if stat else None,
        }
    return out


def load_log(processed_dir: Path) -> dict[str, Any]:
    path = processed_dir / "relational_build_log.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"sources": {}}


def save_log(log: dict[str, Any], processed_dir: Path, metrics_dir: Path | None) -> None:
    text = json.dumps(log, indent=2)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "relational_build_log.json").write_text(text, encoding="utf-8")
    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "relational_build_log.json").write_text(text, encoding="utf-8")


def is_cached(
    source: str, log: dict[str, Any], *, raw_dir: Path, processed_dir: Path, nrows: int | None
) -> bool:
    entry = log["sources"].get(source)
    if entry is None or not table_path(source, processed_dir).is_file():
        return False
    return entry.get("raw_files") == raw_signature(source, raw_dir) and entry.get("nrows") == nrows


def build_source(
    source: str,
    *,
    raw_dir: Path = config.RAW_DATA_DIR,
    processed_dir: Path = config.RELATIONAL_PROCESSED_DIR,
    nrows: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one source, write its parquet table, return (log entry, audit)."""
    module = SOURCE_MODULES[source]
    started = time.perf_counter()
    features, audit, sizes = module.load_and_build(raw_dir=raw_dir, nrows=nrows)
    if not features[ID].is_unique:
        raise AssertionError(f"{source}: built table is not one row per {ID}.")
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = table_path(source, processed_dir)
    features.to_parquet(path, index=False)
    seconds = time.perf_counter() - started
    entry = {
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "riskpilot_version": __version__,
        "raw_files": raw_signature(source, raw_dir),
        "nrows": nrows,
        "input_rows": sizes,
        "output_customers": int(len(features)),
        "n_features": int(features.shape[1] - 1),
        "seconds": round(seconds, 1),
        "output_memory_mb": round(float(features.memory_usage(deep=True).sum() / 1e6), 1),
        "output_file_mb": round(path.stat().st_size / 1e6, 1),
        "path": _display(path),
    }
    logger.info(
        "%s: %s -> %s customers x %d features in %.0fs (%.0f MB in memory)",
        source,
        sizes,
        f"{len(features):,}",
        entry["n_features"],
        seconds,
        entry["output_memory_mb"],
    )
    del features
    gc.collect()
    return entry, audit


def build_all(
    sources: list[str] | None = None,
    *,
    raw_dir: Path = config.RAW_DATA_DIR,
    processed_dir: Path = config.RELATIONAL_PROCESSED_DIR,
    metrics_dir: Path | None = config.METRICS_DIR,
    force: bool = False,
    nrows: int | None = None,
) -> dict[str, Any]:
    """Build every requested source (skipping valid caches) and write the artifacts."""
    sources = list(sources or SOURCES)
    log = load_log(processed_dir)
    audit_path = metrics_dir / "relational_data_audit.json" if metrics_dir else None
    audits: dict[str, Any] = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path and audit_path.is_file()
        else {}
    )
    status: dict[str, str] = {}
    for source in sources:
        if not force and is_cached(
            source, log, raw_dir=raw_dir, processed_dir=processed_dir, nrows=nrows
        ):
            logger.info(
                "%s: cached table is up to date (%s)",
                source,
                _display(table_path(source, processed_dir)),
            )
            status[source] = "cached"
            continue
        entry, audit = build_source(
            source, raw_dir=raw_dir, processed_dir=processed_dir, nrows=nrows
        )
        log["sources"][source] = entry
        audits[source] = audit
        status[source] = "built"
        save_log(log, processed_dir, metrics_dir)  # after every source: restartable
        if audit_path is not None:
            audit_path.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    log["last_run"] = {
        "at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "raw_dir": _display(raw_dir),
        "processed_dir": _display(processed_dir),
    }
    save_log(log, processed_dir, metrics_dir)
    if metrics_dir is not None:
        feature_catalog().to_csv(metrics_dir / "relational_feature_catalog.csv", index=False)
    return log


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riskpilot.features.relational.build",
        description="Aggregate the Home Credit relational tables to one row per customer.",
    )
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the cache is valid.")
    parser.add_argument(
        "--nrows", type=int, default=None, help="Row limit per raw table (smoke runs)."
    )
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=config.RELATIONAL_PROCESSED_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    try:
        log = build_all(
            args.sources,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            metrics_dir=args.metrics_dir,
            force=args.force,
            nrows=args.nrows,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    rows = [
        {
            "source": name,
            "status": log["last_run"]["status"].get(name, "cached"),
            "customers": entry["output_customers"],
            "features": entry["n_features"],
            "seconds": entry["seconds"],
            "memory_mb": entry["output_memory_mb"],
        }
        for name, entry in log["sources"].items()
    ]
    print("\nRelational build\n" + pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
