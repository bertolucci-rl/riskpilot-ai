# MS3 repair and acceptance report

## Problem and root cause

The recovered experiment clipped 17 future bureau update timestamps to zero while
retaining their reported snapshot values, and then described all history as safe.
Those 16 car loans and one mortgage originated in the past, but their last updates
are dated 10-372 days after application. The official metadata does not establish that
these are harmless anomalies. Three per-model plots inherited a Logistic Regression
legend. The report overstated monotone calibration improvement, misstated relative
retuning gain and bootstrap precision, and described insufficient cache checks.

## Repair

A centralized, tested application-time policy masks unavailable bureau snapshots
while retaining historical origination/type and independently dated monthly facts.
Positive planned maturity remains allowed for eligible snapshots. Exactly 213
aggregate values across 23 features and 17 customers changed; rows and feature
count stayed constant. The four other customer tables are exactly unchanged.
The full field-by-field audit and limitations are in the
[technical report](relational_features_report.md#temporal-policy-and-data-audit).
No additional future-observation issue was found in included fields.

## Cache integrity and reproducibility

The recovered cache checked raw size/mtime and row limit; ablation resume checked
only prediction length. Repaired per-source sidecars record raw SHA-256/size,
builder/common/policy code hashes, policy version, pandas version, row limit,
creation time, output hash, row/customer count, ordered columns and dtypes. Loading
also validates expected source columns and unique non-null customer keys. Legacy,
partial, corrupted, source-changed, policy-changed or schema-changed caches fail
closed or are rebuilt by the feature-build CLI. All five caches were rebuilt once;
a second build reused all five without changing their content.

Experiment checkpoints add hashes of the ordered training features/labels/IDs,
source manifests, selection configuration, relevant code and library versions.
Ablation requires exact validation order and verified prediction/metric hashes;
retuning verifies the trial CSV; final resume verifies the locked context and
saved model, predictions, metrics, comparison and bootstrap hashes. Completion
manifests are published atomically after their files; an interrupted/mismatched
checkpoint is rejected with a fresh-stage instruction. Successful prior sources
and trial fits can be reused. No workflow engine or new model family was added.

```bash
python -m riskpilot.features.relational.build
python -m riskpilot.models.relational_experiment --stage ablation
python -m riskpilot.models.relational_experiment --stage retune
python -m riskpilot.models.relational_experiment --stage final
# Reuse verified work; mismatched/legacy checkpoints fail closed.
python -m riskpilot.features.relational.build
python -m riskpilot.models.relational_experiment --resume
jupyter nbconvert --to notebook --execute --inplace notebooks/04_relational_features.ipynb
pytest
ruff check src tests
ruff format --check src tests
```

The notebook calls the same final CLI orchestration with resume validation, then
reads the corrected artifacts. It does not repeat fitting or bootstrap simply to
render the analytical narrative. Raw files, split memberships, feature caches,
model binaries and predictions remain ignored. Small metrics, chosen figures,
source/tests and the executed notebook are versioned.


## Re-execution and source selection

All five source caches were rebuilt. The original 14 internal ablation configurations,
500-resample source-retention comparisons, small coordinate retune, locked final fit,
and 1,000-resample test bootstrap were rerun. No new model family, aggregation search,
hyperparameter grid or test-driven iteration was added. Original and corrected
source sets: ['bureau', 'previous', 'installments', 'credit_card', 'pos'] / ['bureau', 'previous', 'installments', 'credit_card', 'pos'].

| Configuration | Features | ROC-AUC | PR-AUC | Log loss | Brier | ECE | Fit s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0_application | 120 | 0.754684 | 0.237950 | 0.247175 | 0.068084 | 0.003499 | 30.0 |
| M1_+bureau | 169 | 0.764772 | 0.251048 | 0.244282 | 0.067472 | 0.004452 | 83.2 |
| M2_+previous | 211 | 0.774202 | 0.264509 | 0.241473 | 0.066897 | 0.006247 | 67.7 |
| M3_+installments | 237 | 0.782313 | 0.273197 | 0.238958 | 0.066389 | 0.005538 | 76.4 |
| M4_+credit_card | 271 | 0.783550 | 0.273151 | 0.238692 | 0.066371 | 0.006055 | 79.7 |
| M5_+pos | 291 | 0.785573 | 0.276026 | 0.237947 | 0.066187 | 0.005094 | 75.6 |
| S_previous | 162 | 0.766758 | 0.252396 | 0.243765 | 0.067402 | 0.004626 | 47.3 |
| S_installments | 146 | 0.767205 | 0.251352 | 0.243772 | 0.067444 | 0.004446 | 36.4 |
| S_credit_card | 154 | 0.759767 | 0.243443 | 0.245833 | 0.067816 | 0.003865 | 42.8 |
| S_pos | 140 | 0.763692 | 0.251205 | 0.244467 | 0.067500 | 0.004744 | 54.0 |
| L-bureau | 242 | 0.779659 | 0.264616 | 0.240073 | 0.066705 | 0.005191 | 87.0 |
| L-previous | 249 | 0.781790 | 0.271116 | 0.239186 | 0.066461 | 0.005924 | 60.3 |
| L-installments | 265 | 0.779687 | 0.270776 | 0.239711 | 0.066505 | 0.006526 | 68.2 |
| L-credit_card | 257 | 0.783529 | 0.272882 | 0.238684 | 0.066357 | 0.005287 | 55.4 |

| Removed source | Validation log-loss cost | 95% interval |
| --- | --- | --- |
| bureau | 0.002126 | [0.001539, 0.002685] |
| previous | 0.001240 | [0.000753, 0.001739] |
| installments | 0.001765 | [0.001247, 0.002330] |
| credit_card | 0.000737 | [0.000274, 0.001221] |
| pos | 0.000745 | [0.000345, 0.001112] |

The machine-readable lock predates corrected test scoring:
`2026-09-21T18:03:32+00:00`. The application-only reference matches accepted MS2
with maximum metric error 0. Both frozen/internal membership hashes are unchanged.

## Results: recovered versus corrected

| Metric | Recovered (not accepted) | Corrected | Corrected - recovered |
| --- | --- | --- | --- |
| roc_auc | 0.790003725 | 0.789981040 | -0.000022685 |
| average_precision | 0.294307227 | 0.294834549 | +0.000527322 |
| log_loss | 0.235351818 | 0.235324695 | -0.000027123 |
| brier_score | 0.065315965 | 0.065306211 | -0.000009754 |
| expected_calibration_error | 0.002321789 | 0.002095607 | -0.000226182 |

Old run metadata, spec, metrics and exact deltas are preserved in
`artifacts/metrics/ms3_repair_comparison.json`; full old features, metrics,
predictions and model are retained under ignored `data/processed/ms3_pre_repair`.
The original public recovered commit remains in Git history. No old result is
silently represented as a corrected result.

## Test comparison and uncertainty

| Model | ROC-AUC | PR-AUC | Log loss | Brier | Brier skill | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.750412 | 0.235318 | 0.248205 | 0.068216 | 0.080782 | 0.002090 |
| LightGBM | 0.762200 | 0.253806 | 0.244556 | 0.067391 | 0.091900 | 0.002197 |
| LightGBM + history | 0.789981 | 0.294835 | 0.235325 | 0.065306 | 0.119990 | 0.002096 |

| Metric | Corrected - application-only | 95% paired percentile interval |
| --- | --- | --- |
| roc_auc | +0.027780882 | [+0.023872021, +0.031515033] |
| average_precision | +0.041028841 | [+0.033737474, +0.048690103] |
| log_loss | -0.009231770 | [-0.010474664, -0.008037377] |
| brier_score | -0.002084603 | [-0.002432806, -0.001752964] |

The correction does not remove the material improvement over application-only
LightGBM. The small numerical repair effect does not retrospectively validate the
unsafe original treatment. Confidence intervals are conditional paired test-row
intervals, not proof of causal effects or future deployment performance.

## Calibration

Corrected ECE **0.002096**, slope **1.001652**, intercept **0.011741**; mean prediction **0.080163**, observed prevalence **0.080728**. The reliability curve uses ten equal-frequency bins. These diagnostics do not presently justify formal recalibration. No Platt/isotonic comparison was performed, so they do not establish that recalibration could never help.

## Reporting and plot corrections

Individual ROC, PR and calibration figures now receive the relational model label
from the same key as their prediction arrays. Tests inspect actual figure legends.
Comparison figures use the LR, application-only LightGBM and corrected relational
prediction columns. Validation ablation axes remain explicitly validation axes.
The report/notebook are rewritten around corrected artifacts; ECE non-monotonicity,
actual fit time and actual confidence bounds are reported without inflated claims.

## Resources

| Source | Raw rows | Customers in raw source | Features (+ history flag) | Training coverage | Build s | Output MB |
| --- | --- | --- | --- | --- | --- | --- |
| bureau | 1,716,428 + 27,299,925 | 305,811 | 48 + 1 | 85.7% | 42.0 | 59.9 |
| previous | 1,670,214 | 338,857 | 41 + 1 | 94.6% | 19.6 | 56.9 |
| installments | 13,605,401 | 339,587 | 25 + 1 | 94.8% | 35.0 | 35.3 |
| credit_card | 3,840,312 | 103,558 | 33 + 1 | 28.3% | 14.3 | 14.1 |
| pos | 10,001,358 | 337,252 | 19 + 1 | 94.1% | 29.0 | 27.0 |

Fresh ablation including validation bootstrap took 1,053.2s; the five-fit retune
took 657.0s. Their original timings are preserved in the repair-comparison artifact;
selection metadata from the later resume describes that verification run.
The full training feature matrix is 323 MB. Sources are read/aggregated sequentially;
no full-table Cartesian join is built. Raw temporal re-audit uses 250,000-row chunks.
Observed build-process peak working set reached at least 1,682 MiB during sampling;
this is not a complete system-wide peak measurement. Per-fit times above are actual
wall times and depend on contention on the 8 GB machine. Final model fit was
154.04s, six threads. Valid-cache checks read hashes/schema without aggregation;
normal experiment resume reuses recorded fits rather than retraining models.

## Acceptance verification

**Milestone 3 ACCEPTED.** 154 tests pass; Ruff lint and formatting pass.
The notebook executes all 10 code cells without errors.
Fresh CLI feature/experiment stages completed; valid-cache reuse and full CLI
`--resume` were verified with no repeated model fits. The persisted final model
reproduces saved test predictions within 1e-16;
all 14 validation vectors reproduce their recorded metrics. Configurations without
bureau exactly reproduce their recovered predictions. The 1,000-resample bootstrap
estimates match saved prediction differences. Per-model and comparison plot legends
were checked, and all 32 accepted MS1/MS2 artifact files retain their Git contents.

Machine-readable evidence: `artifacts/metrics/ms3_repair_verification.json`.
The 19 requested acceptance items are covered by the temporal/grain/target audit,
validated rebuild and resume checks, unchanged memberships and exact MS2 reference,
validation-only ablation/retuning and pre-test lock, corrected metrics/bootstrap/
calibration, rewritten reports/README, executed notebook, and passing quality checks.
No test-set outcomes informed the temporal policy or feature/source selection.
MS1/MS2 accepted artifacts are preserved. The holdout was previously observed in
earlier milestones and recovery; this repair does not claim a newly untouched cohort.

## Conclusion

**Milestone 3 ACCEPTED** under the documented Home Credit snapshot semantics and
limitations. The future-update treatment, stale-cache behavior, legends and report
claims are corrected. Repaired results retain the substantive improvement over
application-only LightGBM. No later milestone was started.

Recommended next milestone: **cost-sensitive decision-policy evaluation**. Current
calibration diagnostics do not warrant a separate recalibration implementation;
the next useful question is how verified probabilities support explicit lending
costs and review/approval decisions. That work is not implemented here.
