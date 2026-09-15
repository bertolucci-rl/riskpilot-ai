# RiskPilot AI

Credit Risk Decision Intelligence Platform, built incrementally from a rigorous
probabilistic baseline toward a full decision engine with explainability,
monitoring, a FastAPI service and an LLM-based AI Risk Analyst.

## Problem

A lender must decide whether to grant a loan before observing whether the
applicant will repay. The statistical object of interest is the **probability of
default**

```
P(Y = 1 | X = x)
```

where `Y = 1` means the client had payment difficulties (the Home Credit
`TARGET` definition) and `x` is what is known at application time. Everything
downstream (pricing, approval thresholds, expected-loss estimates, portfolio
monitoring) consumes that probability, so the model is judged on **probability
quality** (log loss, Brier score, calibration) and **ranking quality** (ROC-AUC,
PR-AUC), never on accuracy. Accuracy is meaningless at an ~8% default rate: a
model that rejects nobody is 92% "accurate".

## Current milestone: probabilistic baseline

This repository currently implements **Milestone 1**: a leakage-safe, reproducible
Logistic Regression baseline on the `application_train` table, with
probabilistic evaluation and a calibration analysis.

Status of this milestone:

| Component | State |
|---|---|
| Package (`riskpilot`): config, loader, validation, preprocessing, training, evaluation | implemented and unit-tested |
| Kaggle download helper (`python -m riskpilot.data.download`) | implemented |
| Baseline metrics, figures, notebooks and technical report | **pending: waiting for the Kaggle dataset** (requires Kaggle authentication and competition-rule acceptance on the developer's machine) |

The metrics section below is populated from `artifacts/metrics/baseline_metrics.json`
once the baseline has actually been run on the real data. No number in this
README is typed by hand.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
(Kaggle). Milestone 1 uses **only** `application_train.csv` (one row per loan
application, ~307k rows, 122 columns, binary `TARGET`). The relational tables
(bureau, previous applications, installments, ...) are reserved for the
feature-engineering milestone.

Raw data is **never committed**: `data/raw/` is git-ignored, and the file must be
obtained with the official Kaggle CLI after accepting the competition rules:

```bash
python -m riskpilot.data.download
# equivalent to:
# kaggle competitions download home-credit-default-risk -f application_train.csv -p data/raw
```

## Methodology

**Validation.** The application table has no application date (only the
weekday and hour of the appointment), so an honest temporal split is not
possible from this table alone. The baseline uses a stratified random holdout
(80% train / 20% test, `random_state=42`). The test partition is never touched
by any fitting step. Split membership is written to
`data/processed/baseline_split_membership.csv` so later milestones evaluate on
the identical rows.

**Leakage controls.**
- `SK_ID_CURR` is dropped before modeling; `TARGET` is separated first.
- Every learned transformation (median imputation, scaling, one-hot
  vocabularies) lives inside a scikit-learn `Pipeline` and is fitted on the
  training split only.
- Feature typing (numeric vs. categorical) is inferred from the training split.
- `riskpilot.data.validation.leakage_screen` flags suspicious column names and
  numeric features with an unusually high univariate AUC; results are reported,
  not silently acted upon.

**Preprocessing** (`riskpilot.features.preprocessing`).
- A stateless cleaning step maps the `DAYS_EMPLOYED == 365243` sentinel to
  missing so it is not treated as a magnitude of about +1000 years.
- Numeric: median imputation with missing-value indicators, then standardization.
- Categorical: explicit `"missing"` level, then `OneHotEncoder(handle_unknown="ignore")`
  (sparse output). Cardinality is inspected before encoding; the largest column
  in this table has a few dozen levels, so one-hot encoding is memory-safe.

**Model.** `LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)` with L2
regularization and **no class re-weighting**: re-weighting would shift the
predicted probabilities away from the true base rate, which is exactly what we
are trying to estimate. Convergence is checked and recorded, never suppressed.

**Evaluation** (`riskpilot.models.evaluate`).
- Ranking: ROC-AUC, average precision (PR-AUC) with the prevalence as reference.
- Probability quality: log loss and Brier score, each next to the score of a
  constant "predict the prevalence" model; Brier skill score.
- Calibration: quantile-binned reliability table, expected calibration error,
  reliability diagram and predicted-probability distribution by class.

## Metrics

Not yet available: the experiment has not been run because the dataset could
not be downloaded without Kaggle credentials. This section will be filled with
the contents of `artifacts/metrics/baseline_metrics.json` after the first real
run.

## Architecture

```text
riskpilot-ai/
├── data/
│   ├── raw/                  # application_train.csv (git-ignored)
│   └── processed/            # split membership etc. (git-ignored)
├── notebooks/                # 01_data_understanding, 02_baseline (added with the first run)
├── src/riskpilot/
│   ├── config.py             # pathlib-based paths, RANDOM_STATE, column names, sentinels
│   ├── data/
│   │   ├── download.py       # Kaggle CLI wrapper + verification
│   │   ├── load.py           # loader with required-column checks
│   │   └── validation.py     # overview, missingness, cardinality, anomalies, leakage screen
│   ├── features/
│   │   └── preprocessing.py  # cleaning step, ColumnTransformer, baseline Pipeline
│   └── models/
│       ├── train.py          # CLI: split -> fit -> evaluate -> persist artifacts
│       └── evaluate.py       # metrics, calibration table, figures
├── tests/                    # pytest suite on small synthetic fixtures
├── artifacts/
│   ├── figures/              # ROC, PR, reliability, probability distribution (PNG)
│   ├── metrics/              # baseline_metrics.json
│   └── models/               # baseline pipeline (joblib, git-ignored)
├── reports/                  # baseline_report.md
├── pyproject.toml
└── README.md
```

## Reproduction (Windows-friendly)

```powershell
git clone https://github.com/bertolucci-rl/riskpilot-ai.git
cd riskpilot-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Kaggle: log in once (OAuth) or set KAGGLE_API_TOKEN, and accept the competition rules at
# https://www.kaggle.com/competitions/home-credit-default-risk/rules
kaggle auth login
python -m riskpilot.data.download

# Train + evaluate the baseline, write metrics/figures/model
python -m riskpilot.models.train

# Quality checks
pytest
ruff check src tests
```

On macOS/Linux replace the activation line with `source .venv/bin/activate`.

## Roadmap

1. Challenger models (gradient boosting) against the same split and metrics.
2. Feature engineering across the relational tables (bureau, previous applications, installments).
3. Probability calibration (Platt / isotonic) with proper held-out comparison.
4. Cost-sensitive Decision Engine (expected-loss thresholds, approve/review/decline).
5. Explainability (global and per-decision).
6. FastAPI scoring service.
7. Model monitoring (drift in inputs and in probability distributions).
8. AI Risk Analyst with LLM tool calling over the scoring and explanation APIs.
9. RAG over credit policies.
10. AI evaluation harness for the analyst.

None of items 1-10 is implemented yet.

## License

MIT
