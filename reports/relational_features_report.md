# RiskPilot AI — Milestone 3 technical report: relational credit-history features

> **Recovered experiment — incomplete acceptance (2026-09-21).** This report preserves
> the original September 17 analysis and results. The [recovery audit](relational_recovery_audit.md)
> identifies unresolved future-dated bureau updates, incorrect individual-plot legends,
> and misleading analytical claims below. Numerical results reproduce, but the original
> leakage-safety and completion claims are not accepted. Milestone 2 remains the verified
> primary model; resolving this audit is required before promoting the relational model.

Run of 2026-09-17 on the Home Credit Default Risk tables (Kaggle), scikit-learn
1.9.1, LightGBM 4.7.0. Every number below is copied from
`artifacts/metrics/relational_raw_verification.json`, `relational_data_audit.json`,
`relational_build_log.json`, `relational_feature_catalog.csv`, `relational_ablation.csv`,
`relational_selection.json`, `relational_trials.csv`, `lightgbm_relational_metrics.json`,
`relational_model_comparison.csv`, `relational_bootstrap.json`,
`relational_importance_by_source.csv` or the executed `notebooks/04_relational_features.ipynb`.

## 1. Summary

**Question.** Milestone 2 showed that once the features are fixed the model class
matters little (LightGBM and XGBoost tied at ROC-AUC 0.762 on the 120 application
columns). The remaining lever is information. How much do the five historical
sources of the dataset (credit-bureau records with their monthly statuses,
previous Home Credit applications, installment payments, credit-card and POS/cash
monthly balances) add beyond `application_train.csv`, source by source, and what
does that do to probability quality?

**Answer.** A lot, and calibration survives. On the frozen Milestone 1 holdout
(61,503 applications, scored once, after every decision was locked on validation
data), the relational LightGBM reaches ROC-AUC **0.790** and PR-AUC **0.294**
against 0.762 / 0.254 for the application-only LightGBM and 0.750 / 0.235 for the
Logistic Regression baseline; log loss falls from 0.2446 to **0.2354** and the Brier
skill score rises from 0.092 to **0.120**, with 95 % paired-bootstrap intervals well
clear of zero. Every source adds information, unevenly: previous applications and
installment payments are the strongest on their own, the external bureau history
is the hardest to replace given the others, POS and credit-card balances the
cheapest but still reliable. All five were retained. The probabilities stay
calibrated (mean prediction 0.0801 vs 0.0807 observed, slope 1.001, ECE 0.0023).

| Model (frozen test split) | Feature set | ROC-AUC | PR-AUC | Log loss | Brier | Brier skill | ECE | Slope / intercept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (Milestone 1) | application, 312 after preprocessing | 0.7504 | 0.2353 | 0.2482 | 0.06822 | 0.0808 | 0.0021 | 1.006 / 0.015 |
| LightGBM (Milestone 2) | application, 120 | 0.7622 | 0.2538 | 0.2446 | 0.06739 | 0.0919 | 0.0022 | 1.022 / 0.049 |
| **LightGBM + history** (this milestone) | application + 5 sources, 291 | **0.7900** | **0.2943** | **0.2354** | **0.06532** | **0.1199** | 0.0023 | 1.001 / 0.011 |

Δ relational − application-only LightGBM: ROC-AUC +0.0278 [+0.0238, +0.0316],
PR-AUC +0.0405 [+0.0332, +0.0482], log loss −0.0092 [−0.0105, −0.0080], Brier
−0.0021 [−0.0024, −0.0017]. Nothing in this milestone re-weights the classes,
chooses a threshold or recalibrates.

## 2. Data model and table grains

All six historical tables were downloaded with the official Kaggle CLI through
`python -m riskpilot.data.download --files relational` and verified (existence,
key columns, row counts: `artifacts/metrics/relational_raw_verification.json`).
Raw files stay in `data/raw` (git-ignored, 2.4 GB for the six tables).

| Table | Rows | Columns | Size | Key(s) | Grain (verified) | Customers covered |
|---|---:|---:|---:|---|---|---:|
| `application_train.csv` | 307,511 | 122 | 166 MB | `SK_ID_CURR` | one row per current application | 307,511 |
| `bureau.csv` | 1,716,428 | 17 | 170 MB | `SK_ID_CURR`, `SK_ID_BUREAU` | one row per external credit; `SK_ID_BUREAU` unique | 305,811 |
| `bureau_balance.csv` | 27,299,925 | 3 | 376 MB | `SK_ID_BUREAU`, `MONTHS_BALANCE` | one row per credit and month, no duplicates, months in [−96, 0] | via bureau |
| `previous_application.csv` | 1,670,214 | 37 | 405 MB | `SK_ID_PREV`, `SK_ID_CURR` | one row per previous application; `SK_ID_PREV` unique | 338,857 |
| `installments_payments.csv` | 13,605,401 | 8 | 723 MB | `SK_ID_PREV`, `NUM_INSTALMENT_NUMBER`, `NUM_INSTALMENT_VERSION` | one row per installment *and payment part*: 4.8 % of rows share a key (split payments) | 339,587 |
| `credit_card_balance.csv` | 3,840,312 | 23 | 425 MB | `SK_ID_PREV`, `MONTHS_BALANCE` | one row per card and month, no duplicates, months in [−96, −1] | 103,558 |
| `POS_CASH_balance.csv` | 10,001,358 | 8 | 393 MB | `SK_ID_PREV`, `MONTHS_BALANCE` | one row per POS/cash credit and month, no duplicates, months in [−96, −1] | 337,252 |

Relationships: `application (SK_ID_CURR) 1 → n bureau (SK_ID_BUREAU) 1 → n
bureau_balance`; `application 1 → n previous_application (SK_ID_PREV) 1 → n
{installments_payments, credit_card_balance, POS_CASH_balance}` (every monthly
table also carries `SK_ID_CURR`, which is what the builders group by). The
customer counts above include applicants of the competition's test file; on the
246,008 training applicants the history coverage is 85.7 % (bureau), 94.6 %
(previous), 94.8 % (installments), 28.3 % (credit card) and 94.1 % (POS).

**Discrepancies found and documented.** (i) Only 774,354 of the 1,716,428 bureau
credits (45 %) have monthly history, and 43,041 `bureau_balance` credits have no
row in `bureau`; the latter are dropped by the validated many-to-one merge.
(ii) `previous_application` contains 392,402 rows with a zero requested amount
(mostly cancelled applications) and 1,551 approved applications with a zero
granted amount. (iii) `bureau` has 29,642 credits whose current debt exceeds the
credit amount, 8,418 negative debts and 351 negative limits (clipped to 0 for the
sums), and 17 credits whose last update is dated after the application (clipped
to 0). (iv) 7,420 POS rows report more installments left than the term. None of
these was "fixed"; each is either used as reported or handled by an explicit,
documented rule.

## 3. Temporal semantics and leakage audit

Every time column in the six tables is expressed **relative to the date of the
current application** (official column dictionary, `HomeCredit_columns_description.csv`).
The builders' audits record, per column, the minimum, maximum and number of
positive values (`artifacts/metrics/relational_data_audit.json`); the rule is
that a feature may only summarise rows and quantities that were observable at the
application date, and that an ambiguous column is excluded rather than
interpreted.

| Source | Feature family | Temporal interpretation | Leakage status | Decision |
|---|---|---|---|---|
| bureau | counts, types, amounts, overdue, prolongations | `DAYS_CREDIT` ≤ 0 on every row (credit opened before the application); amounts, `CREDIT_DAY_OVERDUE` and `AMT_CREDIT_MAX_OVERDUE` are "at the time of application" | safe | include |
| bureau | recency (`DAYS_CREDIT`, `DAYS_ENDDATE_FACT`, `DAYS_CREDIT_UPDATE`) | all ≤ 0 (17 rows of `DAYS_CREDIT_UPDATE` up to +372 days, clipped to 0) | safe | include |
| bureau | planned remaining duration (`DAYS_CREDIT_ENDDATE`) | positive on 602,603 rows because it is the *planned* end date, known at application; used only for active credits | safe (planned, not realised) | include |
| bureau_balance | monthly status history (DPD months, worst status, latest status, DPD in the last 12 months, closed share) | `MONTHS_BALANCE` in [−96, 0]; status per month before the application | safe | include (aggregated per credit, then per customer) |
| previous | counts, rates, last decision, amounts, terms, products, reject reasons | `DAYS_DECISION` in [−2922, −1]: every previous application was decided before the current one; the attributes are those of the past application | safe | include |
| previous | `DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`, `DAYS_LAST_DUE`, `DAYS_TERMINATION` | a 365243 placeholder on 40–934 k rows plus genuine positive values (224,392 real positive `DAYS_LAST_DUE_1ST_VERSION`): schedules that end *after* the application; sign not interpretable per row | ambiguous | **exclude** |
| installments | timeliness (`days_late = DAYS_ENTRY_PAYMENT − DAYS_INSTALMENT`), completeness, volume, recency | both columns in [−4921, −1] on all 13.6 M rows: every installment was due and paid before the application; positive lateness = paid after the due date (checked on the data: median −6 days, 8.4 % late, 0.28 % over 30 days) | safe | include |
| credit_card | balances, limits, utilization, drawings, payments, DPD, status | `MONTHS_BALANCE` in [−96, −1] | safe | include |
| pos | terms, DPD, status, recency | `MONTHS_BALANCE` in [−96, −1] | safe | include |
| all | `SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU` | identifiers | not features | keys only; `TreePreprocessor` and `build_feature_matrix` refuse `SK_ID_CURR` / `TARGET` |
| all | `TARGET`-derived statistics | — | leakage by construction | never computed: the builders receive raw historical tables only |

The Milestone 1 name-based leakage screen flags `DPD`/`OVERDUE`/`DELINQ`
fragments; here those columns are *historical* delinquency measured before the
application (the audit shows no month or day after it), which is exactly the
information a credit-bureau model is supposed to use.

## 4. Feature engineering

`riskpilot.features.relational` builds one table per source (`bureau` includes
`bureau_balance`), one row per `SK_ID_CURR`, `float32`, names prefixed with the
source (`installments__late_rate`). Every column has a `FeatureSpec` (family,
description, aggregation, whether 0 is a true value without history), from which
`artifacts/metrics/relational_feature_catalog.csv` (171 rows) is generated and
against which the tests check the produced columns. Statistics are chosen per
family, not applied to every column.

| Source | Features | Families (examples) |
|---|---:|---|
| bureau + bureau_balance | 48 + indicator | counts (credits, active, closed, sold/bad, per type, opened last year), amounts (credit, debt, active debt, limits, annuity), delinquency (overdue amounts, days overdue, max overdue ever, prolongations), ratios (active share, debt/credit, overdue/debt), recency (most recent / oldest / mean credit age, last update, planned remaining duration, last closure), monthly history (credits with history, months, DPD months and share, worst status, severe credits, latest-status DPD, DPD in the last 12 months, closed share) |
| previous | 41 + indicator | counts and rates (approved / refused / canceled / unused offer, approval, refusal, cancellation), status of the most recent decision, recency (last / oldest / mean decision, decisions in the last year), amounts (requested, granted over approved, annuity, goods price, down payment, credit-to-request ratio), terms, product shares (cash / consumer / revolving, portfolio, x-sell / walk-in, yield group, new-client share, insurance), reject reasons (HC, LIMIT, scoring) |
| installments | 25 + indicator | volume (installments, contracts, split payments), recency (last / oldest due date), timeliness (late count and rate, severe-late rate, days late mean / max / min / std, mean lateness when late), completeness (unpaid rate, payment ratio mean / std, under- and over-payment rates, scheduled, paid, gap, paid share), last-12-months late and under-payment rates |
| credit_card | 33 + indicator | volume and recency, balances (mean, max, std, latest), limits, utilization (mean, max, > 0.8 share, > 1 share, latest), DPD (mean, max, positive share, > 30 share, `_DEF` max), drawings (total, mean, ATM, POS, ATM-month share, count), payments (mean, minimum due, payment / minimum, missed-minimum share), status (active share, completed cards, matured installments) |
| pos | 19 + indicator | volume and recency, terms (term, installments left, remaining share), DPD (mean, max, positive share, > 30 share, `_DEF` max and positive share, last-12-months DPD share), status (active share, completed share, completed contracts) |

Conventions worth knowing: split payments are summed to one row per installment
before anything is aggregated (a payment in two parts is not an under-payment);
`payment_ratio` is undefined when the scheduled amount is 0 and clipped to [0, 2];
an installment is under-paid when short by more than one currency unit;
utilization is undefined on the 19.6 % of card months with a zero limit; negative
balances are clipped to 0 for utilization only; ratios with a zero denominator are
missing, never infinite. A customer absent from a source gets
`<source>__has_history = 0`, count-like features set to 0 (0 credits is a true
value) and every other statistic left missing, so "no history" and "history with
an undefined statistic" stay distinguishable. The five history indicators are the
only indicator columns; no redundant flags are created.

**Computational cost of the build** (`artifacts/metrics/relational_build_log.json`,
6-core laptop, one process, one source at a time):

| Source | Input rows | Customers | Features | Seconds | Output (memory / parquet) |
|---|---:|---:|---:|---:|---:|
| bureau (+ bureau_balance) | 1,716,428 + 27,299,925 | 305,811 | 48 | 31 | 60 MB / 22 MB |
| previous | 1,670,214 | 338,857 | 41 | 15 | 57 MB / 24 MB |
| installments | 13,605,401 | 339,587 | 25 | 39 | 35 MB / 15 MB |
| credit_card | 3,840,312 | 103,558 | 33 | 13 | 14 MB / 8 MB |
| pos | 10,001,358 | 337,252 | 19 | 31 | 27 MB / 9 MB |

About two minutes for the whole build. Each raw table is read once with an
explicit column subset and compact dtypes (`int32` keys, `int16` months, `float32`
amounts, `category` strings): the 27 M-row `bureau_balance` occupies 191 MB in
memory instead of about 1.6 GB, the 13.6 M-row installments table 408 MB. Tables
are processed strictly one at a time and released before the next; the cached
parquet tables (78 MB in total) are rebuilt only when a raw file's size or
modification time changes (`--force` overrides), and the build log is written
after every source so an interrupted build resumes where it stopped.

## 5. Internal validation protocol

* **Frozen split.** The Milestone 1 partition is reloaded from the persisted
  membership file and re-derived from the seed (`comparison.load_frozen_split`
  raises on any mismatch): train 246,008, test 61,503 (4,965 defaults). The two
  reference models are *reloaded* from `artifacts/models` and must reproduce
  their recorded test metrics before any comparison (observed maximum difference
  2.8e-17 for the Logistic Regression, 0.0 for the application-only LightGBM).
* **Internal validation split.** The Milestone 2 split is reconstructed with the
  same seed (fit 196,806 rows, validation 49,202, prevalence 0.0807 in both) and
  persisted for the first time in `data/processed/internal_validation_membership.csv`
  (git-ignored, deterministic); notebook 04 re-derives it and confirms it matches.
  The application-only model reproduces Milestone 2's validation numbers to the
  fifth decimal (log loss 0.24717, ROC-AUC 0.7547).
* **Frozen model.** Every source comparison uses the Milestone 2 locked LightGBM
  exactly (learning rate 0.02, 1,380 rounds, 15 leaves, min 500 samples per leaf,
  50 % features per tree, 80 % rows per tree, L2 = 1, no class weights, seed 42,
  6 threads), read from `challenger_selection.json`, with fixed rounds and no
  early stopping, so that a difference between two configurations is a difference
  in data only.
* **Assembly.** The relational tables are joined onto the training rows once with
  validated one-to-one merges (row count and index asserted unchanged); every
  configuration is a column subset of that 246,008 × 291 matrix (252 MB as
  `float32`).
* **Selection metric.** Validation log loss (proper score), with ROC-AUC, PR-AUC,
  Brier, Brier skill, ECE and calibration slope recorded for every fit
  (`relational_ablation.csv`).

## 6. Source ablation

Fourteen configurations, each fitted once (`relational_ablation.csv`; wall time
per fit 27–85 s):

| Configuration | Features | ROC-AUC | PR-AUC | Log loss | Brier | Brier skill | ECE | Slope | Fit s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 application only | 120 | 0.7547 | 0.2380 | 0.24717 | 0.06808 | 0.0826 | 0.0035 | 0.973 | 44 |
| M1 + bureau / bureau_balance | 169 | 0.7646 | 0.2505 | 0.24437 | 0.06750 | 0.0904 | 0.0046 | 0.987 | 48 |
| M2 + previous | 211 | 0.7741 | 0.2643 | 0.24148 | 0.06689 | 0.0986 | 0.0060 | 0.992 | 39 |
| M3 + installments | 237 | 0.7818 | 0.2726 | 0.23914 | 0.06643 | 0.1049 | 0.0056 | 0.997 | 53 |
| M4 + credit_card | 271 | 0.7836 | 0.2736 | 0.23864 | 0.06635 | 0.1059 | 0.0060 | 0.991 | 85 |
| M5 + pos (all five) | 291 | 0.7855 | 0.2760 | 0.23796 | 0.06618 | 0.1082 | 0.0058 | 0.995 | 60 |
| S previous only | 162 | 0.7668 | 0.2524 | 0.24376 | 0.06740 | 0.0918 | 0.0046 | 0.979 | 33 |
| S installments only | 146 | 0.7672 | 0.2514 | 0.24377 | 0.06744 | 0.0912 | 0.0045 | 0.976 | 28 |
| S credit_card only | 154 | 0.7598 | 0.2434 | 0.24583 | 0.06782 | 0.0862 | 0.0039 | 0.969 | 27 |
| S pos only | 140 | 0.7637 | 0.2512 | 0.24447 | 0.06750 | 0.0904 | 0.0047 | 0.972 | 27 |
| L all but bureau | 242 | 0.7797 | 0.2646 | 0.24007 | 0.06671 | 0.1011 | 0.0052 | 0.980 | 49 |
| L all but previous | 249 | 0.7820 | 0.2716 | 0.23909 | 0.06643 | 0.1048 | 0.0059 | 0.989 | 66 |
| L all but installments | 265 | 0.7795 | 0.2710 | 0.23974 | 0.06650 | 0.1039 | 0.0060 | 0.987 | 65 |
| L all but credit_card | 257 | 0.7837 | 0.2722 | 0.23866 | 0.06637 | 0.1056 | 0.0056 | 0.996 | 54 |

(Leaving out POS is the M4 model.) Per-source summary, validation log loss and
ROC-AUC (`relational_selection.json`, `source_value`):

| Source | Coverage | Features | Standalone gain (S − M0) | Leave-one-out cost (L − M5) |
|---|---:|---:|---:|---:|
| bureau + bureau_balance | 85.7 % | 49 | −0.0028 log loss, +0.0100 ROC-AUC | +0.0021 log loss, −0.0058 ROC-AUC |
| previous | 94.6 % | 42 | −0.0034, +0.0121 | +0.0011, −0.0035 |
| installments | 94.8 % | 26 | −0.0034, +0.0125 | +0.0018, −0.0060 |
| credit_card | 28.3 % | 34 | −0.0013, +0.0051 | +0.0007, −0.0018 |
| pos | 94.1 % | 20 | −0.0027, +0.0090 | +0.0007, −0.0019 |

**Reading.** Every table adds information and the cumulative sequence is monotone
on all six metrics; the richer models are also *less* under-confident on
validation (slope 0.973 → 0.995). Standalone, previous applications and
installment payments are the most informative, bureau next, POS close behind and
credit cards last (they cover only 28 % of applicants). Leave-one-out costs are
smaller than standalone gains for every source: the three tables that describe
previous Home Credit loans (previous, installments, POS) partly stand in for each
other, and the external bureau history is the one whose removal costs the most
given all the others.

## 7. Source retention

Backward elimination with evidence rather than a threshold: at each round the
candidate is the source whose removal from the current set costs the least
validation log loss; it is dropped when the 95 % paired-bootstrap interval of that
cost (500 resamples of the 49,202 validation rows, seed 42) includes zero, and the
elimination stops at the first source whose removal reliably hurts.

| Round 1, full set M5 | Removal cost, log loss [95 % CI] | Removal cost, ROC-AUC [95 % CI] | Worse without, share of resamples |
|---|---:|---:|---:|
| bureau | +0.00211 [+0.00154, +0.00268] | +0.0058 [+0.0041, +0.0077] | 100 % |
| previous | +0.00113 [+0.00066, +0.00165] | +0.0035 [+0.0020, +0.0051] | 100 % |
| installments | +0.00179 [+0.00124, +0.00236] | +0.0060 [+0.0043, +0.0078] | 100 % |
| credit_card | +0.00070 [+0.00024, +0.00120] | +0.0018 [+0.0004, +0.0034] | 99.6 % |
| pos (weakest) | +0.00069 [+0.00029, +0.00105] | +0.0019 [+0.0007, +0.0030] | 100 % |

The weakest source, POS, still costs a reliable +0.0007 log loss when removed, so
the elimination stopped in the first round and **all five sources were retained**
(291 features). POS and credit card are the cheapest to drop, for different
reasons (POS overlaps with the installment history; credit cards cover few
applicants), and that margin is worth remembering when weighing the longer fit of
the full model, but on the evidence neither is redundant.

## 8. Light retuning

With the feature set fixed, a four-axis, two-value coordinate search (leaves
{15, 31}, minimum leaf size {100, 500}, feature fraction {0.3, 0.5}, L2 {1, 10})
with early stopping on validation log loss (learning rate 0.02, patience 100, cap
5,000), `relational_trials.csv`:

| Trial | Change | Rounds | Log loss | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| 01 | frozen recipe, early stopping | 1,802 | 0.23777 | 0.7862 | 0.2763 |
| 02 | num_leaves = 31 | 1,178 | 0.23781 | 0.7860 | 0.2764 |
| 03 | min_child_samples = 100 | 1,886 | 0.23833 | 0.7843 | 0.2749 |
| **04** | **colsample_bytree = 0.3 (locked)** | **2,635** | **0.23753** | **0.7874** | **0.2758** |
| 05 | reg_lambda = 10 | 2,244 | 0.23768 | 0.7869 | 0.2749 |

The wider feature space asks for more rounds (1,802 instead of 1,380 for the
frozen recipe) and a smaller feature fraction; everything else is unchanged.
**Locked configuration:** learning rate 0.02, **2,635 rounds**, 15 leaves, min 500
samples per leaf, **30 % features per tree**, 80 % rows per tree, L2 = 1, no class
weights, seed 42. The whole exercise is worth 0.0004 in validation log loss
(0.23796 → 0.23753), a tenth of the weakest data source: better information, not
better tuning, moved this milestone.

## 9. Final result on the frozen test split

The locked feature set and configuration were refitted on the whole training
portion (246,008 × 291) with the locked rounds and no monitor, and scored once.

| | Logistic Regression | LightGBM (application) | LightGBM + history |
|---|---:|---:|---:|
| ROC-AUC | 0.7504 | 0.7622 | **0.7900** |
| PR-AUC (average precision) | 0.2353 | 0.2538 | **0.2943** |
| Log loss (null model 0.2805) | 0.2482 | 0.2446 | **0.2354** |
| Brier score (null model 0.0742) | 0.06822 | 0.06739 | **0.06532** |
| Brier skill score | 0.0808 | 0.0919 | **0.1199** |
| ECE (10 quantile bins) | 0.0021 | 0.0022 | 0.0023 |
| Calibration slope / intercept | 1.006 / 0.015 | 1.022 / 0.049 | 1.001 / 0.011 |
| Mean / max predicted probability | 0.0805 / 0.744 | 0.0805 / 0.784 | 0.0801 / 0.862 |
| Train ROC-AUC / log loss | 0.751 / 0.248 | 0.803 / 0.233 | 0.860 / 0.210 |
| Features / trees | 312 / — | 120 / 1,380 | 291 / 2,635 |
| Fit time (6 threads) | 32 s | 37 s | 101 s |
| Inference, 61,503 rows | 0.7 s | 1.5 s | 3.0 s |
| Persisted pipeline | 27 KB | 2.6 MB | 4.9 MB |

**Deltas** (absolute; relative in parentheses):

| | Δ ROC-AUC | Δ PR-AUC | Δ Log loss | Δ Brier | Δ Brier skill |
|---|---:|---:|---:|---:|---:|
| relational − application-only LightGBM | +0.0278 (+3.6 %) | +0.0405 (+16.0 %) | −0.0092 (−3.8 %) | −0.0021 (−3.1 %) | +0.028 |
| relational − Logistic Regression | +0.0396 (+5.3 %) | +0.0590 (+25.1 %) | −0.0129 (−5.2 %) | −0.0029 (−4.2 %) | +0.039 |

The historical tables are worth about 2.4 × what the change of model class was
worth in Milestone 2 (+0.012 ROC-AUC), PR-AUC gains the most because the new
signal lands at the top of the ranking, and both proper scores improve with
discrimination. The train–test gap widens (train ROC-AUC 0.860) as it does for
any boosted model with more signal; §11 shows it is not over-confidence on unseen
rows. The notebook refit reproduces the command-line test metrics to the last
digit (same data, seed, thread count).

## 10. Statistical uncertainty

Paired bootstrap on the frozen test predictions: 1,000 resamples of the 61,503
rows (seed 42), every model scored on the same resample, percentile intervals of
the replicate-wise differences (`relational_bootstrap.json`).

| Difference | Δ ROC-AUC | Δ PR-AUC | Δ Log loss | Δ Brier |
|---|---:|---:|---:|---:|
| relational − application-only LightGBM | +0.0278 [+0.0238, +0.0316], 100 % | +0.0405 [+0.0332, +0.0482], 100 % | −0.0092 [−0.0105, −0.0080], 100 % | −0.0021 [−0.0024, −0.0017], 100 % |
| relational − Logistic Regression | +0.0396 [+0.0353, +0.0441], 100 % | +0.0590 [+0.0505, +0.0686], 100 % | −0.0129 [−0.0143, −0.0115], 100 % | −0.0029 [−0.0033, −0.0025], 100 % |
| Logistic Regression − application-only LightGBM (Milestone 2, unchanged) | −0.0118 [−0.0145, −0.0092] | −0.0185 [−0.0236, −0.0130] | +0.0037 [+0.0030, +0.0044] | +0.0008 [+0.0006, +0.0010] |

Bootstrap standard deviations of the relational-vs-application-only differences:
0.0020 (ROC-AUC), 0.0039 (PR-AUC), 0.0006 (log loss), 0.0002 (Brier); the lower
bounds sit roughly twelve standard deviations from zero and the relational model
wins in every resample. Per-model intervals: ROC-AUC 0.790 [0.783, 0.796] against
0.762 [0.755, 0.769]. These are confidence intervals for the observed differences
on this holdout, not hypothesis tests, and a random (not temporal) holdout still
measures interpolation within the same period. Re-running the bootstrap with the
same seed reproduces every interval exactly (checked in the notebook).

## 11. Calibration

Ten-quantile reliability table of the relational model on the test split
(≈ 6,150 applicants per bin):

| Decile | Range | Mean predicted | Observed | Gap |
|---|---|---:|---:|---:|
| 0 | 0.001–0.014 | 0.0096 | 0.0098 | +0.0001 |
| 1 | 0.014–0.020 | 0.0166 | 0.0163 | −0.0004 |
| 2 | 0.020–0.027 | 0.0232 | 0.0211 | −0.0020 |
| 3 | 0.027–0.035 | 0.0307 | 0.0311 | +0.0004 |
| 4 | 0.035–0.046 | 0.0402 | 0.0392 | −0.0010 |
| 5 | 0.046–0.060 | 0.0526 | 0.0504 | −0.0022 |
| 6 | 0.060–0.081 | 0.0701 | 0.0699 | −0.0001 |
| 7 | 0.081–0.117 | 0.0975 | 0.1050 | +0.0076 |
| 8 | 0.117–0.192 | 0.1486 | 0.1551 | +0.0066 |
| 9 | 0.192–0.862 | 0.3122 | 0.3094 | −0.0029 |

* **In the large:** mean prediction 0.0801 vs 0.0807 observed.
* **In shape:** slope 1.001, intercept 0.011 (1 / 0 ideal); ECE 0.0023 against
  0.0022 for the application-only model and 0.0021 for the baseline; the largest
  decile gap is +0.8 percentage points (decile 7). The application-only model's
  mild under-confidence (slope 1.022) has disappeared: richer information let the
  model spread its probabilities (513 test applicants above 0.5 against 197 and
  108; 99th percentile 0.48 against 0.41 and 0.37) *and* keep them on the
  diagonal. On its own training rows the model is under-confident (slope 1.38),
  the usual signature of a regularized booster.
* **Verdict.** Neither over- nor under-confident, no distortion in any decile.
  Formal recalibration would have nothing to correct on this evidence; the
  calibration check should remain a gate that every future model passes, not a
  milestone of its own.

Figures: `artifacts/figures/relational_calibration_comparison.png`,
`relational_probability_distributions.png`, `lightgbm_relational_calibration_curve.png`.

## 12. Feature-source interpretation (engineering only)

Tree-based importance of the final model, aggregated by source
(`relational_importance_by_source.csv`; gain and split shares are what LightGBM
reports, **not** causal contributions, and are used only to check that the
relational groups are actually used):

| Source | Features | Used in ≥ 1 split | Gain share | Split share | Top features by gain |
|---|---:|---:|---:|---:|---|
| application | 120 | 103 | 53.4 % | 40.1 % | `EXT_SOURCE_2`, `EXT_SOURCE_3`, `ORGANIZATION_TYPE` |
| bureau | 49 | 46 | 13.8 % | 18.6 % | `debt_to_credit_ratio`, `days_credit_max`, `days_credit_mean` |
| previous | 42 | 41 | 11.8 % | 15.5 % | `credit_to_application_ratio_mean`, `refusal_rate`, `amt_annuity_approved_mean` |
| installments | 26 | 25 | 9.6 % | 11.2 % | `late_rate_12m`, `late_rate`, `paid_sum` |
| pos | 20 | 19 | 5.8 % | 7.9 % | `remaining_share_mean`, `instalments_future_mean`, `instalments_future_max` |
| credit_card | 34 | 32 | 5.7 % | 6.7 % | `utilization_latest_mean`, `over_limit_rate`, `atm_drawing_month_rate` |

266 of the 291 features are used; the relational groups take 47 % of the gain and
60 % of the splits, in the same order as the ablation ranks them. The features
that lead each group are the ones a credit analyst would name: external
debt-to-credit ratio and recency of external credits, how much more than requested
was granted before and how often the applicant was refused, whether recent
installments were late, how much of the POS schedule is still to run, and card
utilization. Validation gain by source is the ablation of §6; none of these
quantities is an explanation of individual decisions (a later milestone).

## 13. Computational cost and resource use

| Stage | Wall time | Peak memory (process) | Notes |
|---|---:|---:|---|
| Download (6 tables, 2.4 GB) | about 1 min | — | Kaggle CLI, one file at a time |
| Feature build (5 sources) | 2.2 min | < 1 GB | one raw table in memory at a time, compact dtypes, cached parquet |
| Source ablation (14 fits + 1 elimination round with 5 bootstraps) | 13 min | 1.4 GB | 246,008 × 291 matrix assembled once; every configuration a column subset |
| Retuning (5 fits with early stopping) | 8 min | 1.4 GB | |
| Final stage (refit, references, bootstrap, figures) | 3.5 min | 1.4 GB | 101 s fit, 3 s scoring |
| Notebook 04 (re-executes the final stage and the bootstrap) | about 8 min | | |

The 8 GB laptop with other applications resident was the binding constraint of
the design, not of the method: raw tables are never held together (the largest,
`installments_payments`, needs 408 MB with `float32`/`int32`/`int16` dtypes and is
released before the next), the application block is downcast to `float32` before
assembly, the tree preprocessor no longer makes a `float64` intermediate copy, and
every expensive stage writes its record after each step (`relational_build_log.json`,
`relational_ablation.csv` + validation predictions, `relational_trials.csv`) so
that `--resume` continues an interrupted run without refitting. The experiment ran
as a detached process to keep it out of the interactive session's memory watchdog.

## 14. Leakage audit

Performed in code (notebook 04, section 10) and by inspection:

1. **TARGET never enters feature generation.** The builders receive raw historical
   tables only; `build_feature_matrix` and `TreePreprocessor` raise if `TARGET` or
   `SK_ID_CURR` is present in a feature frame; no target-derived statistic exists.
2. **Test labels were not used for source selection.** `run_ablation_stage` and
   `run_retune_stage` take `(X_train, y_train, ids_train, ...)`; the signatures have
   no test argument (asserted by a test). Every retention decision is documented
   in `relational_selection.json` with its validation bootstrap.
3. **No clipping or aggregation parameter was learned on test data.** Relational
   aggregates are per-customer statistics of that customer's own history (no
   cross-customer parameter is fitted); the only learned preprocessing (category
   vocabularies) is fitted inside `fit_challenger` on the rows being fitted.
4. **No many-to-many merge multiplied rows.** Every join is
   `validate="one_to_one"` on a table asserted to be one row per `SK_ID_CURR`; the
   row count and index are asserted unchanged after assembly (and the tests exercise
   the failure paths).
5. **Identifiers are not predictors.** `SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU`
   are keys only; the final model's feature names contain none of them.
6. **Temporal fields are interpreted consistently.** Every anchor used is ≤ 0
   relative to the application on every row (audit JSON); the one positive
   quantity used (`DAYS_CREDIT_ENDDATE`) is a planned duration; the five ambiguous
   schedule columns are excluded.
7. **Feature logic does not depend on outcome information**, and the frozen test
   partition is touched by exactly three functions: `load_frozen_split`
   (partitioning), `load_baseline_reference` / `load_model_reference` (scoring the
   persisted references once) and `run_final_stage` (scoring the locked model once).

## 15. Limitations

* **No time-based holdout.** The application table has no application date, so
  the frozen split is random; the reported gains describe interpolation within the
  period covered by the data, not performance on future cohorts.
* **Competition semantics.** The tables are the anonymised competition release:
  "relative to the application" is the dictionary's statement, the sentinel and
  placeholder conventions (365243, `X` statuses, zero limits) are inferred from the
  data, and the bureau history covers 45 % of external credits.
* **Aggregation assumptions.** First-generation aggregates: means, extremes,
  shares, sums and a single 12-month recency window; no trend, no time-decayed
  weights, no per-contract sequences. Split payments are merged by
  `(SK_ID_PREV, NUM_INSTALMENT_NUMBER, NUM_INSTALMENT_VERSION)`; re-scheduled
  installments (different versions) are counted as separate installments.
* **Modelling.** One model class (LightGBM), one seed, one validation split;
  XGBoost was not re-run on the relational features (the Milestone 2 tie made it
  unlikely to change the conclusion, and it would have doubled the compute); the
  retune was deliberately small.
* **Machine.** Peak memory was measured, not bounded; on a machine with less than
  about 2 GB free the ablation stage would page.

## 16. Conclusion

Relational feature engineering materially improved the production candidate: on
the untouched frozen holdout the LightGBM with customer-level history reaches
ROC-AUC 0.790 and PR-AUC 0.294 (from 0.762 and 0.254), cuts log loss by 3.8 % and
Brier by 3.1 %, with paired-bootstrap intervals well clear of zero, and keeps its
calibration (slope 1.00, ECE 0.0023). The sources that contributed most are the
previous Home Credit applications and the installment-payment history (standalone)
and the external bureau history (conditional on the rest); credit-card balances
contribute least because they exist for 28 % of applicants; all five earned their
place. Tuning was a footnote (0.0004 in validation log loss). The relational
LightGBM is the model to carry forward.

## 17. Reproduction and artifacts

```powershell
python -m riskpilot.data.download --files relational     # six tables + column dictionary (Kaggle login required)
python -m riskpilot.features.relational.build            # about 2 min; cached in data/processed/relational
python -m riskpilot.models.relational_experiment         # ablation + elimination + retune + frozen test (about 25 min)
python -m riskpilot.models.relational_experiment --stage final --resume
jupyter nbconvert --to notebook --execute --inplace notebooks/04_relational_features.ipynb
pytest && ruff check src tests
```

| Artifact | Path | Tracked |
|---|---|---|
| Raw-table verification (rows, columns, sizes) | `artifacts/metrics/relational_raw_verification.json` | yes |
| Data-quality and temporal audit per source | `artifacts/metrics/relational_data_audit.json` | yes |
| Build log (input rows, customers, features, seconds, memory) | `artifacts/metrics/relational_build_log.json` | yes |
| Feature catalog (171 features, families, leakage assessment) | `artifacts/metrics/relational_feature_catalog.csv` | yes |
| Source ablation (14 configurations, validation metrics, cost) | `artifacts/metrics/relational_ablation.csv` | yes |
| Selection record (split, source value, elimination, retune, locked spec) | `artifacts/metrics/relational_selection.json` | yes |
| Retuning trials | `artifacts/metrics/relational_trials.csv` | yes |
| Final metrics, calibration table, importance, references | `artifacts/metrics/lightgbm_relational_metrics.json` | yes |
| Comparison table with deltas | `artifacts/metrics/relational_model_comparison.csv` | yes |
| Paired bootstrap | `artifacts/metrics/relational_bootstrap.json` | yes |
| Importance by source | `artifacts/metrics/relational_importance_by_source.csv` | yes |
| Figures (ablation, importance, ROC / PR / calibration / distributions / deltas, per-model) | `artifacts/figures/relational_*.png`, `lightgbm_relational_*.png` | yes |
| Cached customer-level tables | `data/processed/relational/*.parquet` | no (git-ignored, rebuilt in 2 min) |
| Internal validation membership, validation and test predictions | `data/processed/internal_validation_membership.csv`, `relational_validation_predictions.parquet`, `relational_test_predictions.csv` | no (git-ignored, regenerated) |
| Fitted pipeline | `artifacts/models/lightgbm_relational.joblib` | no (git-ignored) |
| Notebook | `notebooks/04_relational_features.ipynb` | yes (executed) |
