# RiskPilot AI — Milestone 2 technical report: gradient-boosting challengers

Run of 2026-09-16 on `application_train.csv` (Home Credit Default Risk, Kaggle),
scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1. Every number below is copied
from `artifacts/metrics/challenger_trials.csv`, `challenger_selection.json`,
`lightgbm_metrics.json`, `xgboost_metrics.json`, `model_comparison.csv`,
`bootstrap_comparison.json` or the executed `notebooks/03_gradient_boosting.ipynb`.

## 1. Summary

**Question.** Milestone 1 established that the Logistic Regression baseline is
well calibrated and that its limitation is discrimination (ROC-AUC 0.750). Gradient
boosting was tested because it is the model class that extracts interactions and
non-linearities from tabular data without hand-crafted features, and the question
was whether it can *materially* improve discrimination on the same 120 application
features while keeping probability quality acceptable.

**Answer.** Yes, and without paying for it. On the frozen Milestone 1 holdout
(61,503 applications, never used for any decision), both challengers beat the
baseline on every metric, with 95 % paired-bootstrap intervals well clear of zero:

| Model (frozen test split) | ROC-AUC | PR-AUC | Log loss | Brier | Brier skill | ECE | Slope / intercept |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (Milestone 1, unchanged) | 0.7504 | 0.2353 | 0.2482 | 0.06822 | 0.0808 | 0.0021 | 1.006 / 0.015 |
| **LightGBM** (1,380 trees, 15 leaves) | **0.7622** | **0.2538** | **0.2446** | **0.06739** | **0.0919** | 0.0022 | 1.022 / 0.049 |
| XGBoost (1,140 trees, depth 3) | 0.7617 | 0.2522 | 0.2447 | 0.06743 | 0.0914 | 0.0023 | 1.018 / 0.042 |

Δ vs. the baseline: LightGBM +0.0118 ROC-AUC [+0.0092, +0.0145], +0.0185 PR-AUC
[+0.0130, +0.0236], −0.0037 log loss [−0.0044, −0.0030], −0.0008 Brier [−0.0010,
−0.0006]; XGBoost +0.0113, +0.0169, −0.0035, −0.0008 with intervals of the same
width. The two challengers are statistically indistinguishable from each other.
Calibration is preserved (ECE 0.0022 vs 0.0021; slope 1.02; mean prediction equal
to the prevalence). **LightGBM advances as the primary model**, XGBoost as a
validated alternate. No class re-weighting, threshold, or recalibration was applied.

## 2. How the frozen holdout was preserved

* The Milestone 1 partition is reloaded from
  `data/processed/baseline_split_membership.csv` by
  `riskpilot.models.comparison.load_frozen_split`, which (i) requires the
  identifiers in the data and in the file to be the same set, (ii) re-derives the
  split from the seed (`train_test_split(..., random_state=42, stratify=y)`) and
  raises unless it reproduces the file. Train: 246,008 rows; test: 61,503 rows
  (4,965 defaults, prevalence 0.08073), exactly as in Milestone 1.
* The baseline is not refitted. `load_baseline_reference` reloads
  `artifacts/models/baseline_logistic_regression.joblib`, scores the test rows and
  raises unless ROC-AUC, PR-AUC, log loss and Brier match
  `artifacts/metrics/baseline_metrics.json` within 1e-6. Observed maximum
  difference: 2.8e-17. The baseline row of every table in this report is therefore
  the Milestone 1 model, bit for bit.
* The string columns are stored as pandas categoricals for the whole experiment
  (`riskpilot.data.load.categorize_strings`), which halves the table's memory
  (530 MB → 266 MB); the integrity check above shows the linear pipeline is
  unaffected by the representation.
* The test partition is touched by exactly three functions (§11).

## 3. Internal model selection

`riskpilot.models.challengers.run_selection_stage(X_train, y_train, ...)`
receives the training portion only; the signature has no test argument.

* **Validation split.** One stratified random holdout inside the training
  portion, `VALIDATION_SIZE = 0.2`, seed 42: fit subset 196,806 rows, validation
  subset 49,202 rows, prevalence 0.0807 in both. A single split rather than
  cross-validation because every fit needs an early-stopping monitor, the
  validation subset holds about 3,970 defaults (enough for stable log loss and
  ROC-AUC at the 1e-3 level that separates configurations), and the machine has
  8 GB of RAM; the paired bootstrap on the test split (§8) is the uncertainty
  analysis that matters for the conclusions.
* **Reference.** The Milestone 1 pipeline was refitted on the fit subset and scored
  on the validation subset (log loss 0.2505, ROC-AUC 0.7443, PR-AUC 0.2217,
  ECE 0.0026), so the challengers have a like-for-like reference before the test
  split is involved.
* **Selection criterion.** Validation **log loss**, the proper scoring rule that
  rewards ranking and probability quality together. ROC-AUC, PR-AUC, Brier,
  ECE, slope and intercept are recorded for every trial
  (`challenger_trials.csv`, 33 rows) and were inspected; nothing was chosen on
  ROC-AUC alone, and no threshold or accuracy was involved.
* **Early stopping.** Every trial monitors the validation log loss with patience
  100 and a cap of 5,000 rounds; the number of rounds at the best iteration is part
  of the locked specification, and the final refit on the full training portion
  runs that fixed number of rounds with **no** monitor at all. Predictions after
  early stopping and after a fixed-round refit were verified to coincide
  (`tests/test_challengers.py`).

## 4. Tree preprocessing (`riskpilot.features.tree_preprocessing`)

The linear pipeline imputes, adds 62 missing indicators, standardizes and one-hot
encodes 16 string columns into 146 columns: 312 dense `float64` features, 614 MB
for the training split. Trees need none of it: they are invariant to monotone
rescaling, learn a default direction for missing values at every split, and both
libraries split natively on categorical columns. `TreePreprocessor` therefore:

| Step | Linear pipeline (M1) | Tree pipeline (M2) |
|---|---|---|
| Sentinel `DAYS_EMPLOYED == 365243` | → NaN (stateless) | → NaN (same function) |
| Numeric (104) | median imputation + 62 indicators + standardization | cast to `float32`, **nothing else**; NaN kept |
| Categorical (16) | `"missing"` level + one-hot (146 columns) | `pandas.Categorical` with the training vocabulary; unseen level → missing |
| Width / memory (training split) | 312 / 614 MB dense | 120 / 108 MB |
| Learned on | training split | training split (vocabularies; ablation bounds) |

Categorical handling is deliberately native in both libraries but not identical
under the hood, and that is documented rather than hidden: LightGBM applies its
gradient-sorted partition splits (`max_cat_to_onehot = 4`, `cat_smooth = 10`,
`min_data_per_group = 100`); XGBoost (`tree_method = "hist"`,
`enable_categorical = True`) one-hot-splits categories with ≤ 4 levels and uses
partition-based splits above that. Both receive the same frame with the same
category codes, fixed at fit time, so codes cannot drift between training and
scoring. The transformer refuses to fit if `TARGET` or `SK_ID_CURR` is present.

## 5. Configurations and tuning

**Starting points** (sensible, untuned, no class weights, `random_state = 42`,
`n_jobs = 6`):

* LightGBM: `objective=binary, learning_rate=0.05, num_leaves=31, min_child_samples=100,
  subsample=0.8 (freq 1), colsample_bytree=0.8, reg_lambda=1.0`.
* XGBoost: `binary:logistic, eval_metric=logloss, tree_method=hist, learning_rate=0.05,
  max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0`.

**Search.** A compact coordinate descent: one axis at a time in a fixed order,
two or three values per axis, the winner on validation log loss carried forward.
LightGBM axes: `num_leaves {15, 31, 63}`, `min_child_samples {20, 100, 500}`,
`colsample_bytree {0.3, 0.5, 0.8}`, `subsample {0.6, 0.8, 1.0}`, `reg_lambda
{0, 1, 10}`, `learning_rate {0.02, 0.05}`. XGBoost axes: `max_depth {3, 4, 6}`,
`min_child_weight {1, 10, 100}`, `colsample_bytree`, `subsample`, `reg_lambda
{1, 10}`, `reg_alpha {0, 1}`, `learning_rate {0.02, 0.05}`. Twelve fits per library
plus four ablation fits; 33 trials in total including the linear reference.

| LightGBM trial | change | trees | val log loss | val ROC-AUC | val PR-AUC |
|---|---|---:|---:|---:|---:|
| 01 | starting configuration | 370 | 0.24782 | 0.7529 | 0.2344 |
| 02 | num_leaves = 15 ✓ | 441 | 0.24762 | 0.7535 | 0.2358 |
| 03 | num_leaves = 63 | 210 | 0.24821 | 0.7521 | 0.2329 |
| 04 | min_child_samples = 20 | 452 | 0.24790 | 0.7523 | 0.2354 |
| 05 | min_child_samples = 500 ✓ | 575 | 0.24747 | 0.7538 | 0.2368 |
| 06 | colsample_bytree = 0.3 | 611 | 0.24732 | 0.7547 | 0.2371 |
| 07 | colsample_bytree = 0.5 ✓ | 570 | 0.24729 | 0.7544 | 0.2377 |
| 08 | subsample = 0.6 | 558 | 0.24741 | 0.7540 | 0.2377 |
| 09 | subsample = 1.0 | 583 | 0.24740 | 0.7538 | 0.2375 |
| 10 | reg_lambda = 0 | 587 | 0.24738 | 0.7542 | 0.2374 |
| 11 | reg_lambda = 10 | 564 | 0.24735 | 0.7540 | 0.2382 |
| **12** | **learning_rate = 0.02 ✓ (locked)** | **1,380** | **0.24717** | **0.7547** | **0.2380** |

| XGBoost trial | change | trees | val log loss | val ROC-AUC | val PR-AUC |
|---|---|---:|---:|---:|---:|
| 01 | starting configuration | 219 | 0.24845 | 0.7503 | 0.2318 |
| 02 | max_depth = 3 ✓ | 768 | 0.24743 | 0.7542 | 0.2360 |
| 03 | max_depth = 4 | 634 | 0.24778 | 0.7527 | 0.2362 |
| 04 | min_child_weight = 1 ✓ | 766 | 0.24742 | 0.7539 | 0.2365 |
| 05 | min_child_weight = 100 | 922 | 0.24747 | 0.7538 | 0.2366 |
| 06 | colsample_bytree = 0.3 ✓ | 1,088 | 0.24725 | 0.7550 | 0.2370 |
| 07 | colsample_bytree = 0.5 | 644 | 0.24729 | 0.7546 | 0.2372 |
| 08 | subsample = 0.6 | 1,062 | 0.24748 | 0.7545 | 0.2357 |
| 09 | subsample = 1.0 | 1,003 | 0.24731 | 0.7550 | 0.2362 |
| 10 | reg_lambda = 10 ✓ | 1,123 | 0.24706 | 0.7556 | 0.2378 |
| **11** | **reg_alpha = 1 ✓ (locked)** | **1,140** | **0.24689** | **0.7562** | **0.2382** |
| 12 | learning_rate = 0.02 | 2,622 | 0.24706 | 0.7556 | 0.2376 |

**Locked LightGBM:** `learning_rate=0.02, n_estimators=1380, num_leaves=15,
min_child_samples=500, subsample=0.8 (freq 1), colsample_bytree=0.5, reg_alpha=0,
reg_lambda=1.0`; heavy_tail = none, missing_indicators = none.
**Locked XGBoost:** `learning_rate=0.05, n_estimators=1140, max_depth=3,
min_child_weight=1, subsample=0.8, colsample_bytree=0.3, reg_alpha=1.0,
reg_lambda=10.0`; heavy_tail = none, missing_indicators = none.

**What tuning showed.** The whole search moves validation log loss by less than
0.001 (LightGBM 0.2478 → 0.2472, XGBoost 0.2485 → 0.2469), a fifth of the gap to
the linear reference (0.2505). Both libraries converged on the same recipe: many
small, heavily regularized, feature-subsampled trees. The model class matters; the
hyper-parameters barely do. The 2,511 s recorded for XGBoost trial 06 is wall time
on a laptop that was paging memory at that moment (the identical configuration
refits in about 80 s); the other 32 fits took 8–170 s each.

## 6. Ablations (validation data, best searched configuration as incumbent)

A variant is adopted only if it lowers validation log loss by at least 0.0005
(parsimony: added preprocessing must earn its place).

**Heavy tails.** The rule "non-negative, unbounded (max > 1), ≥ 20 distinct values,
skewness > 1 on the training split" selects nine columns: `AMT_INCOME_TOTAL`,
`AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `OBS_30/60_CNT_SOCIAL_CIRCLE`,
`AMT_REQ_CREDIT_BUREAU_MON/YEAR`, `OWN_CAR_AGE`; the bounded building-description
scores are excluded on purpose. Treatments: `log1p`; winsorization at the training
0.1 / 99.9 percentiles (bounds learned on the fit subset only, e.g. income clipped
to [31,500; 900,000]).

| Variant | LightGBM log loss (Δ) | XGBoost log loss (Δ) |
|---|---:|---:|
| incumbent | 0.24717 | 0.24689 |
| + log1p | 0.24717 (0.00000, identical model) | 0.24689 (0.00000, identical model) |
| + winsorize | 0.24731 (+0.00014) | 0.24701 (+0.00012) |

`log1p` reproduces the incumbent exactly, tree for tree: histogram boosting bins
each feature by quantiles and a monotone transform does not change the order, so
the model is literally the same. Winsorization merges the tails into one bin and
is marginally worse. **Not retained.** The heavy-tail question that deserved a
section for the standardized linear model is a no-op for trees, which is itself
the answer to "does sensible treatment help stability": there was nothing to
stabilise.

**Missing indicators.** Milestone 1 reported 38 duplicated transformed columns.
Verified here on the training portion: of the 62 missing indicators the linear
pipeline builds, only 26 missingness patterns are distinct (36 exact duplicates in
16 groups, the `_AVG/_MODE/_MEDI` building triples sharing one pattern each).

| Variant | features | LightGBM log loss (Δ) / fit s | XGBoost log loss (Δ) / fit s |
|---|---:|---:|---:|
| none (native NaN handling) | 120 | 0.24717 / 27 | 0.24689 / 82 |
| all 62 indicators | 182 | 0.24721 (+0.00004) / 36 | 0.24714 (+0.00025) / 131 |
| de-duplicated 26 | 146 | 0.24698 (−0.00019) / 20 | 0.24713 (+0.00024) / 104 |

Differences are inside noise and below the adoption threshold; the full set is
worse for both, the de-duplicated set helps one library and hurts the other by the
same amount. A learned default direction per split already carries the
information. **Not retained.** For the linear model, where the duplicates are a
conditioning issue, the de-duplication remains a candidate for the
feature-engineering milestone.

## 7. Final comparison on the frozen test split

Locked specifications refitted on the whole training portion (246,008 rows) with
the locked number of rounds and no monitor; one `predict_proba` on the test split.

| | Logistic Regression | LightGBM | XGBoost |
|---|---:|---:|---:|
| ROC-AUC | 0.7504 | **0.7622** | 0.7617 |
| PR-AUC (average precision) | 0.2353 | **0.2538** | 0.2522 |
| Log loss (null model 0.2805) | 0.2482 | **0.2446** | 0.2447 |
| Brier score (null model 0.0742) | 0.06822 | **0.06739** | 0.06743 |
| Brier skill score | 0.0808 | **0.0919** | 0.0914 |
| ECE (10 quantile bins) | 0.0021 | 0.0022 | 0.0023 |
| Calibration slope / intercept | 1.006 / 0.015 | 1.022 / 0.049 | 1.018 / 0.042 |
| Mean / max predicted probability | 0.0805 / 0.744 | 0.0805 / 0.784 | 0.0804 / 0.802 |
| Train ROC-AUC / log loss | 0.751 / 0.248 | 0.803 / 0.233 | 0.788 / 0.237 |
| Features after preprocessing | 312 | 120 | 120 |
| Trees | — | 1,380 | 1,140 |
| Fit time (6 threads) | 32 s | 37 s | 35 s |
| Inference, 61,503 rows | 0.56 s | 1.86 s | 0.84 s |
| Persisted pipeline | 27 KB | 2.6 MB | 1.5 MB |

**Deltas vs. the baseline** (absolute; relative in parentheses):

| | Δ ROC-AUC | Δ PR-AUC | Δ Log loss | Δ Brier |
|---|---:|---:|---:|---:|
| LightGBM | +0.0118 (+1.6 %) | +0.0185 (+7.9 %) | −0.0037 (−1.5 %) | −0.0008 (−1.2 %) |
| XGBoost | +0.0113 (+1.5 %) | +0.0169 (+7.2 %) | −0.0035 (−1.4 %) | −0.0008 (−1.2 %) |

The trade-off is favourable on every axis: discrimination improved (PR-AUC most,
because the extra non-linear signal lands at the top of the ranking) and both
proper scores improved with it. The train–test gap is where the model classes
differ (linear 0.751 vs 0.750; LightGBM 0.803 vs 0.762; XGBoost 0.788 vs 0.762):
the ordinary optimism of boosted trees, contained by small leaves, feature
subsampling and locked round counts, and §9 shows it does not become test-set
over-confidence. The notebook refit of both locked specifications reproduces the
command-line test metrics to the last digit (same data, seed, thread count).

## 8. Uncertainty: paired bootstrap

1,000 resamples of the 61,503 test rows with replacement (seed 42), each resample
expressed as multinomial counts used as sample weights so that every model is
scored on the *same* rows; percentile intervals of the replicate-wise
differences. `fraction improving` is the share of resamples in which the
challenger beats the reference: a stability summary, not a significance test.

| Difference | Δ ROC-AUC | Δ PR-AUC | Δ Log loss | Δ Brier |
|---|---:|---:|---:|---:|
| LightGBM − LR | +0.0118 [+0.0092, +0.0145], 100 % | +0.0185 [+0.0130, +0.0236], 100 % | −0.0037 [−0.0044, −0.0030], 100 % | −0.0008 [−0.0010, −0.0006], 100 % |
| XGBoost − LR | +0.0113 [+0.0087, +0.0140], 100 % | +0.0169 [+0.0116, +0.0215], 100 % | −0.0035 [−0.0042, −0.0028], 100 % | −0.0008 [−0.0010, −0.0006], 100 % |
| XGBoost − LightGBM | −0.0005 [−0.0015, +0.0005], 18 % | −0.0016 [−0.0036, +0.0004], 6 % | +0.0002 [−0.0001, +0.0004], 15 % | +0.0000 [−0.0000, +0.0001], 15 % |

Bootstrap standard deviations: 0.0013 (ΔROC-AUC), 0.0026 (ΔPR-AUC), 0.0004
(Δlog loss), 0.0001 (ΔBrier). The lower bounds of the challenger-vs-baseline
intervals sit roughly seven standard deviations from zero: the improvement is
stable across resamples of this holdout, not the luck of one split. Between the
challengers nothing is resolved: XGBoost is nominally behind on every metric, every
interval includes zero, and the two prediction vectors correlate at 0.99. These are
confidence intervals for the observed differences on this holdout; they are not
presented as hypothesis tests, and a random (not temporal) holdout still measures
interpolation within the same period (Milestone 1, §3). Re-running the bootstrap
with the same seed reproduces every interval exactly (checked in the notebook).

Per-model 95 % intervals: ROC-AUC 0.750 [0.743, 0.758] (LR), 0.762 [0.755, 0.769]
(LightGBM), 0.762 [0.754, 0.769] (XGBoost); the *paired* intervals above are
narrower than these overlap would suggest because the models are scored on the
same rows.

## 9. Calibration behaviour

Ten-quantile reliability tables (≈ 6,150 applicants per bin) on the test split:

| Decile | LR predicted / observed | LightGBM predicted / observed | XGBoost predicted / observed |
|---|---:|---:|---:|
| 0 | 0.012 / 0.015 | 0.014 / 0.012 | 0.012 / 0.013 |
| 1 | 0.021 / 0.021 | 0.022 / 0.020 | 0.021 / 0.018 |
| 2 | 0.030 / 0.029 | 0.029 / 0.029 | 0.029 / 0.031 |
| 3 | 0.039 / 0.036 | 0.037 / 0.035 | 0.037 / 0.034 |
| 4 | 0.049 / 0.049 | 0.047 / 0.043 | 0.047 / 0.044 |
| 5 | 0.062 / 0.061 | 0.059 / 0.059 | 0.060 / 0.060 |
| 6 | 0.080 / 0.080 | 0.076 / 0.080 | 0.077 / 0.080 |
| 7 | 0.104 / 0.103 | 0.101 / 0.100 | 0.101 / 0.102 |
| 8 | 0.145 / 0.142 | 0.145 / 0.150 | 0.144 / 0.147 |
| 9 | 0.262 / 0.270 | 0.277 / 0.280 | 0.276 / 0.278 |

* **Calibration in the large:** mean predicted 0.0805 (LightGBM) and 0.0804
  (XGBoost) vs 0.0807 observed; no drift of the base rate, because no class
  re-weighting was applied.
* **Calibration in shape:** slope 1.022 / 1.018, intercept 0.049 / 0.042 (1 / 0
  ideal). A slope slightly above 1 means the probabilities are, if anything,
  marginally *under*-confident (the log-odds could be stretched a little further from
  the base rate): the mild, safe direction, and with 61,503 rows the slope's
  sampling error is about ± 0.03, so 1.02 is not distinguishable from 1. ECE is
  0.0022 / 0.0023 against 0.0021 for the baseline; the largest decile gap is +0.5
  percentage points (LightGBM, decile 8: 0.145 predicted vs 0.150 observed).
* **Tail:** discrimination widened the distribution. 197 (LightGBM) and 213
  (XGBoost) test applicants are scored above 0.5 against 108 for the baseline; the
  99th percentile of predictions moved from 0.374 to 0.408; the top decile
  (0.277 vs 0.280) is calibrated. On training rows the slopes are 1.22 / 1.15, i.e.
  the boosters are *more* timid than the training outcomes would justify: the
  regularization that contains overfitting also keeps the training predictions
  away from over-confidence.
* **Verdict.** Neither challenger is over-confident, under-confident beyond noise,
  or in need of formal calibration. The calibration milestone should treat these
  probabilities as a strong null and demonstrate any Platt / isotonic gain on a
  proper held-out design rather than assume one.

Figures: `artifacts/figures/challenger_calibration_comparison.png`,
`challenger_probability_distributions.png`, and per-model
`{lightgbm,xgboost}_calibration_curve.png` / `_probability_distribution.png`.

## 10. Computational trade-offs

| | Logistic Regression | LightGBM | XGBoost |
|---|---:|---:|---:|
| Design matrix (training split) | 312 dense float64, 614 MB | 120 columns, 108 MB | 120 columns, 108 MB |
| Selection cost | 1 fit (20–36 s) | 16 fits, 247 s total | 16 fits, ≈ 1,200 s total excluding the paging-affected trial (3,709 s recorded) |
| Final fit | 32 s | 37 s | 35 s |
| Inference, 61,503 rows | 0.56 s | 1.86 s | 0.84 s |
| Persisted pipeline | 27 KB | 2.6 MB | 1.5 MB |
| Threads | 1 | 6 | 6 |

Both challengers fit in about the same wall time as the linear baseline and score
tens of thousands of applications per second; the model artifacts are small.
LightGBM's search was five times cheaper than XGBoost's on this machine. Memory
was the binding constraint of the milestone (an 8 GB laptop with other
applications resident): the experiment was made to run in well under 1 GB by
storing strings as categoricals and `float32` numerics, and the trial log doubles
as a resumable cache (`--resume`) so that an interrupted selection stage continues
without refitting. The one trial recorded at 2,511 s is a measurement of paging,
not of the model.

## 11. Test-set leakage audit

Performed in code (notebook 03, section 9) and by inspection:

* `run_selection_stage(X_train, y_train, cfg, *, log)` has no test argument
  (asserted by `tests/test_challengers.py::test_selection_stage_never_receives_test_data`).
* Every function in `riskpilot.models.challengers` and `riskpilot.models.comparison`
  whose source references `X_test`: `run_final_stage` (scores the locked models once),
  `load_baseline_reference` (scores the frozen baseline once), `load_frozen_split`
  (partitions the data). Nothing else.
* Early stopping: only on the validation subset, only during selection; the final
  refit has `early_stopping = "none (rounds locked during selection)"` in both
  metrics files.
* Hyper-parameters, ablations and the adoption decisions: chosen on validation
  log loss (`challenger_selection.json` records every candidate and decision).
* Preprocessing fit: `TreePreprocessor.fit` is called inside `fit_challenger` on the
  frame being fitted (fit subset during selection, training portion in the final
  stage); category vocabularies, clipping bounds and indicator groups come from
  those rows only. Feature selection: none. Calibration fitting: none (diagnostics
  only).
* The frozen split itself: identifiers verified against the membership file and
  against a re-derivation from the seed; the baseline reproduces its recorded
  metrics to 2.8e-17.

## 12. Recommendation

**Advance LightGBM as the primary model; keep XGBoost as a validated alternate.**
Both discriminate better than the baseline by a stable margin and both keep the
baseline's calibration, so the choice between them is not statistical (all
pairwise intervals include zero). LightGBM is nominally best on every metric
(ROC-AUC 0.7622, PR-AUC 0.2538, log loss 0.2446, Brier skill 0.092), was five
times cheaper to tune, and its native categorical handling is the more mature of
the two. XGBoost's protocol, artifacts and tests are identical, so it can be
swapped in at any time.

What the next milestones should take from this one:

* the probabilities need no correction today; recalibration must prove itself
  against these numbers on a held-out design;
* the ceiling on `application_train` alone is close (published solutions reach
  0.78–0.80 ROC-AUC only with the relational tables); feature engineering, not
  further tuning, is the lever;
* the deferred linear-model revisions (robust scaling, indicator de-duplication)
  remain candidates for the linear model only; they are irrelevant to the trees.

## 13. Limitations

* Random, not temporal, validation (inherited from Milestone 1): the metrics
  describe interpolation within the same period.
* One validation split rather than repeated cross-validation for selection; the
  selection differences are tiny, so a different split could pick a neighbouring
  configuration, but §8 shows the conclusions do not depend on it.
* Application table only; relational tables are the next milestone.
* Persisted challenger pipelines are joblib pickles tied to the recorded library
  versions (LightGBM 4.7.0, XGBoost 3.4.1); the metrics files record them.

## 14. Reproduction and artifacts

```powershell
python -m riskpilot.models.train              # Milestone 1 (frozen split + baseline), about 50 s
python -m riskpilot.models.challengers        # selection (33 fits, about 35 min) + frozen-test stage; --resume reuses recorded trials
python -m riskpilot.models.challengers --stage final
jupyter nbconvert --to notebook --execute --inplace notebooks/03_gradient_boosting.ipynb
pytest && ruff check src tests
```

| Artifact | Path | Tracked |
|---|---|---|
| Trial log (33 fits, validation metrics, params) | `artifacts/metrics/challenger_trials.csv` | yes |
| Selection record (split, reference, ablation decisions, locked specs) | `artifacts/metrics/challenger_selection.json` | yes |
| Final metrics, calibration tables, config per challenger | `artifacts/metrics/lightgbm_metrics.json`, `xgboost_metrics.json` | yes |
| Comparison table with deltas and cost figures | `artifacts/metrics/model_comparison.csv` | yes |
| Paired bootstrap | `artifacts/metrics/bootstrap_comparison.json` | yes |
| Comparison figures (ROC, PR, reliability, distributions, deltas) | `artifacts/figures/challenger_*.png` | yes |
| Per-model figures | `artifacts/figures/{lightgbm,xgboost}_*.png` | yes |
| Fitted pipelines | `artifacts/models/{lightgbm,xgboost}_challenger.joblib` | no (git-ignored) |
| Test predictions of the three models | `data/processed/challenger_test_predictions.csv` | no (git-ignored, regenerated) |
| Notebook | `notebooks/03_gradient_boosting.ipynb` | yes (executed) |
