"""Fetch the Home Credit competition files through the official Kaggle CLI and verify them.

Usage::

    python -m riskpilot.data.download                      # application_train.csv (Milestone 1)
    python -m riskpilot.data.download --files relational   # the six historical tables (Milestone 3)
    python -m riskpilot.data.download --files all --force  # everything, re-downloaded

Kaggle credentials are read by the Kaggle CLI itself (``kaggle auth login`` or
the ``KAGGLE_API_TOKEN`` environment variable). This module never reads,
stores or prints them. Raw files land in ``data/raw`` (git-ignored).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from riskpilot import config

logger = logging.getLogger(__name__)

RELATIONAL_FILENAMES: dict[str, str] = {
    name: spec["filename"] for name, spec in config.RELATIONAL_TABLES.items()
}


class KaggleDownloadError(RuntimeError):
    """Raised when the Kaggle CLI is unavailable or the download fails."""


def kaggle_executable() -> Path:
    """Locate the ``kaggle`` CLI, preferring the one installed next to this interpreter."""
    scripts_dir = Path(sys.executable).parent
    for candidate in ("kaggle.exe", "kaggle"):
        local = scripts_dir / candidate
        if local.is_file():
            return local
    found = shutil.which("kaggle")
    if found:
        return Path(found)
    raise KaggleDownloadError(
        "The 'kaggle' CLI was not found. Install it into this environment with "
        "'pip install kaggle' (it is part of the [dev] extras)."
    )


def extract_if_zipped(dest_dir: Path, filename: str) -> Path:
    """If Kaggle delivered ``<filename>.zip``, extract it and remove the archive."""
    target = dest_dir / filename
    archive = dest_dir / f"{filename}.zip"
    if archive.is_file():
        logger.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            members = zf.namelist()
            if filename not in members:
                raise KaggleDownloadError(
                    f"Archive {archive.name} does not contain {filename}; members: {members}"
                )
            zf.extract(filename, path=dest_dir)
        archive.unlink()
    if not target.is_file():
        raise KaggleDownloadError(f"Expected {target} after download, but it does not exist.")
    return target


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def count_rows(path: Path, *, key_column: str, chunksize: int = 2_000_000) -> int:
    """Row count of a large CSV by streaming a single column (memory-safe)."""
    total = 0
    for chunk in pd.read_csv(path, usecols=[key_column], chunksize=chunksize):
        total += len(chunk)
    return total


def verify_table(
    path: Path,
    *,
    required_columns: Iterable[str],
    count: bool = True,
) -> dict[str, Any]:
    """Check that a CSV exists, is non-empty, parses and carries the required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.")
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"{path} is empty.")
    header = pd.read_csv(path, nrows=5)
    required = list(required_columns)
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s) {missing}.")
    if header.empty:
        raise ValueError(f"{path} has a header but no rows.")
    summary: dict[str, Any] = {
        "path": path.as_posix(),
        "size_mb": round(size_bytes / 1e6, 1),
        "n_columns": int(header.shape[1]),
        "columns": list(header.columns),
    }
    if count:
        summary["n_rows"] = count_rows(path, key_column=required[0])
    logger.info(
        "Verified %s: %.1f MB, %d columns", path.name, summary["size_mb"], summary["n_columns"]
    )
    return summary


def verify_application_train(path: Path = config.APPLICATION_TRAIN_PATH) -> dict[str, Any]:
    """Check the CSV exists, is non-empty, parses, and carries the target column."""
    summary = verify_table(path, required_columns=config.REQUIRED_COLUMNS, count=False)
    target = pd.read_csv(path, usecols=[config.TARGET_COL])[config.TARGET_COL]
    summary.update({"n_rows": int(len(target)), "target_prevalence": float(target.mean())})
    del summary["columns"]
    logger.info("Verified %s: %s", path.name, summary)
    return summary


def verify_relational_tables(
    raw_dir: Path = config.RAW_DATA_DIR,
    *,
    sources: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify every relational table: existence, key columns, row counts."""
    out: dict[str, dict[str, Any]] = {}
    for name in sources or list(config.RELATIONAL_TABLES):
        spec = config.RELATIONAL_TABLES[name]
        out[name] = verify_table(raw_dir / spec["filename"], required_columns=spec["keys"])
    return out


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_competition_file(
    dest_dir: Path = config.RAW_DATA_DIR,
    *,
    filename: str,
    competition: str = config.KAGGLE_COMPETITION,
    force: bool = False,
) -> Path:
    """Download a single competition file with the Kaggle CLI and unpack it.

    Returns the path of the CSV. Raises :class:`KaggleDownloadError` with the
    CLI's own message when authentication or rule acceptance is missing.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.is_file() and not force:
        logger.info("%s already present; skipping download (use --force to refresh).", target)
        return target

    cmd = [
        str(kaggle_executable()),
        "competitions",
        "download",
        competition,
        "-f",
        filename,
        "-p",
        str(dest_dir),
    ]
    if force:
        cmd.append("--force")
    logger.info("Running: %s", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise KaggleDownloadError(
            f"Kaggle CLI exited with code {completed.returncode}.\n{output.strip()}"
        )
    logger.info("Kaggle CLI output:\n%s", output.strip())
    return extract_if_zipped(dest_dir, filename)


def download_application_train(
    dest_dir: Path = config.RAW_DATA_DIR,
    *,
    competition: str = config.KAGGLE_COMPETITION,
    filename: str = config.APPLICATION_TRAIN_FILENAME,
    force: bool = False,
) -> Path:
    """Milestone 1 entry point: download ``application_train.csv`` (or any one file)."""
    return download_competition_file(
        dest_dir, filename=filename, competition=competition, force=force
    )


def download_relational_tables(
    dest_dir: Path = config.RAW_DATA_DIR,
    *,
    force: bool = False,
    include_dictionary: bool = True,
) -> dict[str, Path]:
    """Download the six historical tables (and the column dictionary), one file at a time."""
    paths: dict[str, Path] = {}
    if include_dictionary:
        paths["columns_description"] = download_competition_file(
            dest_dir, filename=config.COLUMNS_DESCRIPTION_FILENAME, force=force
        )
    for name, filename in RELATIONAL_FILENAMES.items():
        paths[name] = download_competition_file(dest_dir, filename=filename, force=force)
    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m riskpilot.data.download")
    parser.add_argument("--dest-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument(
        "--files",
        choices=["application", "relational", "all"],
        default="application",
        help="Which competition files to fetch and verify.",
    )
    args = parser.parse_args(argv)
    try:
        if args.files in ("application", "all"):
            path = download_application_train(args.dest_dir, force=args.force)
            summary = verify_application_train(path)
            print(
                f"OK: {summary['path']} ({summary['size_mb']} MB, {summary['n_rows']:,} rows, "
                f"{summary['n_columns']} columns, target prevalence "
                f"{summary['target_prevalence']:.4f})"
            )
        if args.files in ("relational", "all"):
            download_relational_tables(args.dest_dir, force=args.force)
            for name, summary in verify_relational_tables(args.dest_dir).items():
                print(
                    f"OK: {name}: {summary['path']} ({summary['size_mb']} MB, "
                    f"{summary['n_rows']:,} rows, {summary['n_columns']} columns)"
                )
    except (KaggleDownloadError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
