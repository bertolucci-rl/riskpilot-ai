"""Fetch ``application_train.csv`` through the official Kaggle CLI and verify it.

Usage::

    python -m riskpilot.data.download            # download if absent, then verify
    python -m riskpilot.data.download --force    # re-download

Kaggle credentials are read by the Kaggle CLI itself (``kaggle auth login`` or
the ``KAGGLE_API_TOKEN`` environment variable). This module never reads,
stores or prints them.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from riskpilot import config

logger = logging.getLogger(__name__)


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


def verify_application_train(path: Path = config.APPLICATION_TRAIN_PATH) -> dict[str, Any]:
    """Check the CSV exists, is non-empty, parses, and carries the target column."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.")
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"{path} is empty.")
    header = pd.read_csv(path, nrows=5)
    missing = [c for c in config.REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s) {missing}.")
    target = pd.read_csv(path, usecols=[config.TARGET_COL])[config.TARGET_COL]
    summary = {
        "path": path.as_posix(),
        "size_mb": round(size_bytes / 1e6, 1),
        "n_rows": int(len(target)),
        "n_columns": int(header.shape[1]),
        "target_prevalence": float(target.mean()),
    }
    logger.info("Verified %s: %s", path.name, summary)
    return summary


def download_application_train(
    dest_dir: Path = config.RAW_DATA_DIR,
    *,
    competition: str = config.KAGGLE_COMPETITION,
    filename: str = config.APPLICATION_TRAIN_FILENAME,
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m riskpilot.data.download")
    parser.add_argument("--dest-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args(argv)
    try:
        path = download_application_train(args.dest_dir, force=args.force)
        summary = verify_application_train(path)
    except (KaggleDownloadError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(
        f"OK: {summary['path']} ({summary['size_mb']} MB, {summary['n_rows']:,} rows, "
        f"{summary['n_columns']} columns, target prevalence {summary['target_prevalence']:.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
