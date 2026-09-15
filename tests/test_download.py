import zipfile
from pathlib import Path

import pytest

from riskpilot.data import download as dl


def test_extract_if_zipped_unpacks_and_removes_archive(tmp_path: Path, synthetic_csv: Path):
    archive = tmp_path / "raw" / "application_train.csv.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(synthetic_csv, arcname="application_train.csv")
    out = dl.extract_if_zipped(archive.parent, "application_train.csv")
    assert out.is_file() and out.name == "application_train.csv"
    assert not archive.exists()
    assert out.read_bytes() == synthetic_csv.read_bytes()


def test_extract_if_zipped_rejects_wrong_archive(tmp_path: Path):
    archive = tmp_path / "application_train.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("other.csv", "a,b\n1,2\n")
    with pytest.raises(dl.KaggleDownloadError, match="does not contain"):
        dl.extract_if_zipped(tmp_path, "application_train.csv")


def test_extract_if_zipped_requires_file(tmp_path: Path):
    with pytest.raises(dl.KaggleDownloadError, match="does not exist"):
        dl.extract_if_zipped(tmp_path, "application_train.csv")


def test_verify_application_train(synthetic_csv: Path, synthetic_df):
    summary = dl.verify_application_train(synthetic_csv)
    assert summary["n_rows"] == len(synthetic_df)
    assert summary["n_columns"] == synthetic_df.shape[1]
    assert summary["target_prevalence"] == pytest.approx(synthetic_df["TARGET"].mean())


def test_verify_rejects_missing_target(tmp_path: Path, synthetic_df):
    path = tmp_path / "bad.csv"
    synthetic_df.drop(columns=["TARGET"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="TARGET"):
        dl.verify_application_train(path)


def test_download_skips_when_present(tmp_path: Path, synthetic_csv: Path, monkeypatch):
    def boom(*args, **kwargs):  # the Kaggle CLI must not be invoked
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(dl.subprocess, "run", boom)
    out = dl.download_application_train(synthetic_csv.parent, filename=synthetic_csv.name)
    assert out == synthetic_csv


def test_download_reports_cli_failure(tmp_path: Path, monkeypatch):
    class Completed:
        returncode = 1
        stdout = "Authentication required to call the Kaggle API."
        stderr = ""

    monkeypatch.setattr(dl, "kaggle_executable", lambda: Path("kaggle"))
    monkeypatch.setattr(dl.subprocess, "run", lambda *a, **k: Completed())
    with pytest.raises(dl.KaggleDownloadError, match="Authentication required"):
        dl.download_application_train(tmp_path)


def test_main_returns_error_code_on_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dl, "kaggle_executable", lambda: Path("kaggle"))

    class Completed:
        returncode = 1
        stdout = "nope"
        stderr = ""

    monkeypatch.setattr(dl.subprocess, "run", lambda *a, **k: Completed())
    assert dl.main(["--dest-dir", str(tmp_path)]) == 1
