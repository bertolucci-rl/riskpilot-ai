# RiskPilot AI — Milestone 1 technical report: the Logistic Regression baseline

Run of 2026-09-16 on `application_train.csv` (Home Credit Default Risk, Kaggle).
Every number below is copied from `artifacts/metrics/baseline_metrics.json`,
`artifacts/metrics/data_validation.json` or the executed notebooks
`notebooks/01_data_understanding.ipynb` and `notebooks/02_baseline.ipynb`.

## 1. Summary

The milestone establishes an honest, uncalibrated Logistic Regression baseline for
the probability of default `P(TARGET = 1 | application features)`, together with
the evaluation protocol that every later model must be measured against. On the
untouched 20 % holdout the model reaches ROC-AUC 0.750 and average precision 0.235
against a prevalence of 0.081, beats the constant "predict the prevalence" model
on log loss (0.248 vs 0.281) and Brier score (0.0682 vs 0.0742, skill score 0.081),
and produces probabilities that are calibrated in the large and in shape (slope
1.006, intercept 0.015, expected calibration error 0.002). The optimizer converged;
no warning was suppressed. The two warnings that did appear were engineering
defects, not modeling ones, and were fixed.

| Metric (test set, n = 61,503) | Baseline | Null model / ideal |
|---|---|---|
| ROC-AUC | **0.7504** | 0.5 |
| Average precision (PR-AUC) | **0.2353** | 0.0807 (prevalence) |
| Log loss | **0.2482** | 0.2805 |
| Brier score | **0.0682** | 0.0742 |
| Brier skill score | **0.0808** | 0 |
| Expected calibration error (10 quantile bins) | **0.0021** | 0 |
| Calibration slope / intercept (diagnostic) | **1.006 / 0.015** | 1 / 0 |

Nothing in this milestone chooses a decision threshold, re-weights the classes or
recalibrates the probabilities. Those are deliberate omissions, explained in §9.

## 2. Data

**Source and shape.** `application_train.csv`, downloaded with the official Kaggle
CLI through `python -m riskpilot.data.download` (166.1 MB, git-ignored). 307,511
rows × 122 columns: `SK_ID_CURR`, `TARGET`, and 120 raw features (104 numeric,
16 string). 529.5 MB in pandas.

**Target.** `TARGET = 1` for 24,825 applications: prevalence **8.073 %**, imbalance
1 : 11.4, no missing values. Accuracy is therefore not a metric of this project; the
null model that predicts 0.0807 for everyone has Brier score 0.0742 and log loss
0.2805, and those are the references every probabilistic metric is reported against.

**Findings on the real data** (all from `riskpilot.data.validation`, executed in
notebook 01):

| Topic | What the data shows | Verdict | Decision |
|---|---|---|---|
| Duplicates | 0 duplicate rows, 0 duplicate ids; `SK_ID_CURR` unique and monotonic | clean | — |
| Identifier | univariate AUC of `SK_ID_CURR` vs `TARGET` = 0.498; default rate flat across id deciles (0.079–0.083) | uninformative | dropped before modeling, kept only to persist the split |
| Missingness | 67 columns with gaps, 50 above 20 %, 41 above 50 %; 24.4 % of all cells missing. The building block (`*_AVG`, `*_MODE`, `*_MEDI`, 47 columns) is 47–70 % missing with a shared pattern; `OWN_CAR_AGE` (66 %) is missing exactly when `FLAG_OWN_CAR = N`; `EXT_SOURCE_1` 56 %, `OCCUPATION_TYPE` 31 %, `EXT_SOURCE_3` 20 % | structural, informative | median + missing indicator (numeric), explicit `"missing"` level (categorical); nothing dropped |
| `DAYS_EMPLOYED == 365243` | 55,374 rows (18.01 %); the only positive value in any `DAYS_*` column; coincides exactly with `NAME_INCOME_TYPE ∈ {Pensioner, Unemployed}` and with `ORGANIZATION_TYPE == "XNA"`; default rate 5.40 % vs 8.66 % for the rest | documented convention, not corruption | mapped to NaN + indicator by the stateless `clean_features` step; the information survives through the indicator and the two categorical levels |
| Other `DAYS_*` values | `DAYS_BIRTH` 20.5–69.1 years; tenures 0–49 years; `DAYS_LAST_PHONE_CHANGE == 0` on 12.3 % of rows (legal value) | plausible | untouched |
| `XNA` / `Unknown` | `ORGANIZATION_TYPE == "XNA"` 55,374 (the sentinel population); `CODE_GENDER == "XNA"` 4 rows; `NAME_FAMILY_STATUS == "Unknown"` 2 rows | one real level, six defective rows | kept as one-hot levels; rare levels are shrunk by L2 and `handle_unknown="ignore"` covers test-only levels |
| Cardinality | max 58 levels (`ORGANIZATION_TYPE`), then 18, 8, 7…; 146 one-hot columns in total | modest | plain one-hot, memory-safe |
| Numeric columns that are codes | 32 binary flags (`FLAG_*`, `REG_*`, `LIVE_*`) and 7 small integer codes are typed numeric | acceptable for a linear baseline | standardized with the rest (consequence quantified in §7) |
| Near-constant columns | 18 columns ≥ 99 % constant (`FLAG_MOBIL` 99.9997 %, `FLAG_DOCUMENT_12` 99.9993 %, …) | genuine but uninformative | kept; see §7 |
| Extreme values | `AMT_INCOME_TOTAL` max 117,000,000 (247 × q99, one applicant); `AMT_REQ_CREDIT_BUREAU_QRT` max 261 (q99 = 2); `OBS_30_CNT_SOCIAL_CIRCLE` max 348 (q99 = 10); no negative amounts or counts | likely data-entry error, unverifiable | kept for the baseline; effect on `StandardScaler` quantified in §7; revision candidate |
| Leakage | no column name matches a post-outcome pattern; highest univariate AUCs `EXT_SOURCE_3` 0.679, `EXT_SOURCE_1` 0.666, `EXT_SOURCE_2` 0.656, `DAYS_BIRTH` 0.583; nothing near the 0.80 alarm | no candidates | all 120 features kept |

## 3. Validation methodology

**Split.** Stratified random holdout, 80 % / 20 %, `random_state = 42`
(`riskpilot.models.train.make_split`). Train: 246,008 rows, prevalence 0.080729.
Test: 61,503 rows (4,965 defaults), prevalence 0.080728. The membership of every
`SK_ID_CURR` is written to `data/processed/baseline_split_membership.csv`
(git-ignored, regenerated deterministically) so that challenger models are scored
on identical rows; notebook 02 re-derives the split from the seed and confirms it
matches the file.

**Why not a temporal split.** The table has no application date, only the weekday
and hour of the appointment (`WEEKDAY_APPR_PROCESS_START`, `HOUR_APPR_PROCESS_START`),
and the identifier carries no time signal (§2). A chronological holdout cannot be
built from this table; the caveat is that a random split measures interpolation
within the same period, not performance on future applicants. This is recorded as a
known limitation rather than hidden behind a pseudo-temporal split.

**Leakage controls.**

* `TARGET` is separated and `SK_ID_CURR` dropped before anything is fitted.
* Feature typing (numeric vs. categorical) is inferred on the training split.
* All learned transformations (medians, scaling statistics, one-hot vocabularies)
  live inside one scikit-learn `Pipeline` fitted on the training split only; the
  test partition is touched once, by `predict_proba`.
* The name-based and univariate-AUC screens above found no leakage candidate.
* The persisted pipeline reproduces the in-memory test probabilities exactly
  (`np.allclose`) and the metrics file agrees with the in-memory payload.

## 4. Preprocessing

`riskpilot.features.preprocessing.build_baseline_pipeline` =
`clean → ColumnTransformer(num, cat) → LogisticRegression`.

* **clean** (stateless `FunctionTransformer`): `DAYS_EMPLOYED == 365243 → NaN`;
  `None → np.nan` in string columns.
* **num** (104 columns): `SimpleImputer(strategy="median", add_indicator=True)` then
  `StandardScaler`. 62 missing indicators are created (61 columns with gaps in the
  training split plus the sentinel-induced one for `DAYS_EMPLOYED`).
* **cat** (16 columns): `SimpleImputer(strategy="constant", fill_value="missing")`
  then `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`: 146 columns.
* **Width:** 104 + 62 + 146 = **312** features. The one-hot block is emitted sparse
  but the `ColumnTransformer` densifies the result (the 166 dense numeric and
  indicator columns push the overall density above its 0.3 threshold): the training
  design matrix is a dense float64 array of 246,008 × 312 ≈ **614 MB**. Acceptable on
  an 8 GB laptop, and the reason the fit takes about 30 s.
* **Redundancy:** 38 transformed columns are exact duplicates of another column
  (17 groups), almost all missing indicators of the `_AVG/_MODE/_MEDI` triples. The
  L2 penalty makes the problem strictly convex, so this is harmless for the
  baseline; it is worth de-duplicating in the feature-engineering milestone.

## 5. Model

`LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, class_weight=None,
random_state=42)` with the L2 penalty (`l1_ratio = 0`, scikit-learn 1.9.1).

* **Convergence:** lbfgs stopped after **123 iterations** (budget 1000); no
  `ConvergenceWarning` was raised. The fit is wrapped in a recording
  `warnings.catch_warnings` block that would write `converged: false` to the
  metrics file rather than hide the warning. Fit time about 30 s (29.5 s in the
  recorded run), whole run under a minute.
* **No class weights,** on purpose: re-weighting would move the predicted
  probabilities away from the 8.07 % base rate that the model is supposed to
  estimate. Ranking metrics would barely change; log loss, Brier score and
  calibration would degrade.
* 312 coefficients, intercept −0.574; median |coefficient| 0.042; 56 coefficients
  below 0.01 in absolute value.

## 6. Results

| Metric | Test | Train | Null model / ideal |
|---|---|---|---|
| n / positives | 61,503 / 4,965 | 246,008 / 19,860 | — |
| ROC-AUC | 0.7504 | 0.7512 | 0.5 |
| Average precision | 0.2353 | 0.2292 | 0.0807 |
| Log loss | 0.2482 | 0.2484 | 0.2805 |
| Brier score | 0.06822 | 0.06840 | 0.07421 |
| Brier skill score | 0.0808 | 0.0784 | 0 |
| ECE (10 quantile bins) | 0.0021 | 0.0010 | 0 |
| Mean / median / max predicted probability | 0.0805 / 0.0554 / 0.744 | 0.0807 / 0.0556 / 0.953 | 0.0807 |

* **Ranking.** The ROC curve rises steeply at low false-positive rates and the
  precision–recall curve stays above the prevalence line everywhere; at 20 %
  recall precision is about 0.33, at 60 % recall about 0.17. This is the expected
  level for a linear model on the raw application table; published gradient-boosting
  solutions with engineered relational features reach ROC-AUC ≈ 0.78–0.80.
* **Probability quality.** Both proper scores improve on the null model; the Brier
  skill score of 0.081 states plainly that the model removes 8 % of the null
  model's squared error. That is a modest but real amount of information.
* **Generalization.** Train and test agree to the third decimal on every metric:
  with 246k rows, 312 coefficients and L2 regularization there is no overfitting.
* **Figures.** `artifacts/figures/baseline_roc_curve.png`,
  `baseline_precision_recall_curve.png`, `baseline_calibration_curve.png`,
  `baseline_probability_distribution.png`. The class-conditional probability
  distributions overlap heavily: only 108 of 61,503 test applicants (0.18 %) are
  scored above 0.5 and 99 % are scored below 0.374.

## 7. Calibration

Reliability table on the test set, ten quantile bins (≈ 6,150 applicants each):

| Bin | Range | Mean predicted | Observed rate | Gap |
|---|---|---|---|---|
| 0 | 0.000–0.017 | 0.0119 | 0.0150 | +0.0031 |
| 1 | 0.017–0.026 | 0.0214 | 0.0215 | +0.0001 |
| 2 | 0.026–0.034 | 0.0299 | 0.0291 | −0.0008 |
| 3 | 0.034–0.044 | 0.0390 | 0.0358 | −0.0032 |
| 4 | 0.044–0.055 | 0.0495 | 0.0491 | −0.0004 |
| 5 | 0.055–0.070 | 0.0624 | 0.0613 | −0.0011 |
| 6 | 0.070–0.090 | 0.0797 | 0.0800 | +0.0003 |
| 7 | 0.090–0.121 | 0.1042 | 0.1033 | −0.0010 |
| 8 | 0.121–0.177 | 0.1449 | 0.1420 | −0.0030 |
| 9 | 0.177–0.744 | 0.2623 | 0.2704 | +0.0081 |

* **Calibration in the large:** mean predicted 0.0805 vs observed 0.0807. This is a
  property of maximum-likelihood logistic regression with an intercept and no
  re-weighting (the score equations force the average prediction to equal the
  training base rate), and the stratified split carries it over to the test set.
* **Calibration in shape:** a logistic regression of the outcome on the predicted
  log-odds gives slope **1.006** and intercept **0.015** (1 and 0 are ideal; a slope
  below 1 would indicate over-confident probabilities). ECE is 0.0021 on ten
  quantile bins, 0.0045 on twenty, 0.0012 on ten uniform bins; the largest decile
  gap is +0.008 in the top decile, where the observed rate 0.270 ± 0.011 (95 %)
  contains the predicted 0.262.
* **The tail is not estimable.** On uniform bins, 45,342 applicants fall in
  [0, 0.1) and only 108 above 0.5 (94 in [0.5, 0.6), 12 in [0.6, 0.7), 2 above).
  The wobble above 0.5 (0.534 predicted vs 0.606 observed on 94 rows) is sampling
  noise and says nothing about miscalibration.
* **What this indicates.** The baseline's probabilities can be taken at face value;
  the only discernible pattern is a slight under-prediction of risk at both extremes
  of the ranking, inside sampling error. There is no evidence that Platt or isotonic
  recalibration would improve this model, so the recalibration milestone must
  demonstrate a gain on a proper held-out comparison rather than assume one. The
  limitation of the model is discrimination (a narrow, timid probability
  distribution), not calibration.

## 8. What the model learned

Coefficients are per standardized unit (numeric) or per level (one-hot), all
conditional on the other 311 columns.

* Strongest protective effects: `AMT_GOODS_PRICE` −1.00 together with `AMT_CREDIT`
  +0.92 (the pair is 0.99-correlated; the model uses their difference: credit
  granted beyond the price of the goods raises risk), `EXT_SOURCE_3` −0.49,
  `EXT_SOURCE_2` −0.38, `EXT_SOURCE_1` −0.19, `CODE_GENDER_F` −0.41,
  `ORGANIZATION_TYPE` Military −0.55, Police −0.34, Security Ministries −0.33,
  `NAME_CONTRACT_TYPE` Revolving loans −0.29, `NAME_INCOME_TYPE` Pensioner −0.26.
* Strongest risk-raising effects: `ORGANIZATION_TYPE` Transport: type 3 +0.62,
  Realtor +0.42, Legal Services +0.27, Construction +0.27, Self-employed +0.17;
  `OCCUPATION_TYPE` Low-skill Laborers +0.20; `REGION_RATING_CLIENT_W_CITY` +0.15;
  `FLAG_DOCUMENT_3` +0.15.
* The sentinel population is handled as designed: the `Pensioner` and
  `ORGANIZATION_TYPE_XNA` levels carry the (negative) effect and the `DAYS_EMPLOYED`
  missing indicator is close to zero (−0.016) because the levels already explain it.
* `AMT_INCOME_TOTAL` (+0.007 per SD) and the near-constant flags (`FLAG_MOBIL`
  +0.009, `FLAG_DOCUMENT_12` −0.012) are effectively unused; §9 explains why.

## 9. Methodological review of the pre-data assumptions

| Assumption coded before the data was available | Verified? | Comment |
|---|---|---|
| Prevalence ≈ 8 % | yes, 8.073 % | null-model references computed from it |
| No duplicates, id is a pure key | yes | — |
| Missingness handled by median + indicator | yes, and it is structural | indicators are informative; 38 of them are duplicates of each other |
| `DAYS_EMPLOYED == 365243` is a sentinel | yes, exactly the pensioner/unemployed rows | the only positive `DAYS_*` value; `config.SENTINEL_VALUES` is complete |
| `XNA` tokens can stay as levels | yes | `ORGANIZATION_TYPE_XNA` is meaningful; 6 defective rows are negligible |
| One-hot is memory-safe (max "a few dozen" levels) | yes, max 58 | 146 columns; the *dense* matrix (614 MB) is the actual memory cost, because the numeric block dominates |
| Identifier handling | yes | dropped; uninformative |
| No leakage candidates | yes | `EXT_SOURCE_*` are application-time scores |
| Extreme values are harmless under standardization | **partly** | the 117M income inflates the SD of `AMT_INCOME_TOTAL` 2.4 × (259k instead of 108k) and sits at z = 451; 18 standardized columns hold a training row at |z| > 50 (three at ≈ 496). The fitted coefficients on those columns are a few thousandths, so the extreme rows receive a private log-odds shift and no other applicant is distorted. The cost is lost signal: income is effectively discarded. Nothing was changed for the baseline; robust/log scaling of amounts and unscaled 0/1 flags are candidate revisions to be *compared* on the same split in the next milestone. |
| lbfgs converges within 1000 iterations | yes, 123 | despite the ill-conditioning above |
| No class weights, no threshold, no recalibration | held | see §5 and §7 |

**Runtime warnings inspected.** Two appeared during the first real run, none of
them from the optimizer:

1. `RuntimeWarning: 'riskpilot.models.train' found in sys.modules after import of
   package 'riskpilot.models'` — the package `__init__` imported the module that
   `python -m` was executing, so the module was loaded twice. Fixed by importing the
   training symbols lazily in `riskpilot/models/__init__.py`; a test now runs the
   entry point with `-W error::RuntimeWarning`.
2. The metrics file recorded `"penalty": "deprecated"`: scikit-learn 1.8 deprecated
   `LogisticRegression.penalty` in favour of `l1_ratio`. The model is genuinely L2
   (`l1_ratio = 0`); `train.py` now records the penalty by name from the parameters
   that are actually in force, and the end-to-end test asserts `"l2"`.

No `ConvergenceWarning`, `FutureWarning` or pandas warning was emitted by the
pipeline, the validation module or the notebooks (both executed with warnings
visible).

## 10. Limitations and deferred decisions

* **Random, not temporal, validation** (§3): the reported metrics describe
  interpolation within the same period.
* **Only the application table** is used; the bureau, previous-application and
  installment tables are where most of the remaining signal lives.
* **Linear model on raw features:** interactions and non-linearities (age, income,
  credit-to-income ratios) are not captured; the probability distribution is narrow.
* **Scaling of heavy-tailed amounts and binary flags** (§9) costs signal.
* Deferred on purpose, to be evaluated against this baseline on the persisted split:
  decision thresholds and cost-sensitive policies, class weights, probability
  recalibration, robust scaling, feature engineering, challenger models.

## 11. Reproduction and artifacts

```powershell
python -m riskpilot.data.download        # requires Kaggle login + accepted competition rules
python -m riskpilot.models.train         # split -> fit -> evaluate -> artifacts (about 50 s)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_understanding.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline.ipynb
pytest && ruff check src tests
```

| Artifact | Path | Tracked |
|---|---|---|
| Metrics, config, split summary, calibration table | `artifacts/metrics/baseline_metrics.json` | yes |
| Data validation report | `artifacts/metrics/data_validation.json` | yes |
| ROC, PR, reliability, probability-distribution figures | `artifacts/figures/baseline_*.png` | yes |
| Fitted pipeline | `artifacts/models/baseline_logistic_regression.joblib` | no (git-ignored) |
| Split membership | `data/processed/baseline_split_membership.csv` | no (git-ignored, deterministic) |
| Raw data | `data/raw/application_train.csv` | no (git-ignored) |
