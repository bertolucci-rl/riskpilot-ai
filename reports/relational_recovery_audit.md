# Recovered relational-feature experiment: acceptance review

> Historical record of the incomplete recovered experiment. The subsequent
> [MS3 repair report](ms3_repair_report.md) resolves these findings and records
> acceptance of the corrected rerun. The original findings and recovery evidence
> below are retained for provenance; they do not describe the current implementation.

This is existing, uncommitted work produced on 2026-09-17, not a new experiment
started during the Git-authorship cleanup. The original Milestone 3 request is
present in the local project session log at 2026-09-17 19:43:59 (America/Sao_Paulo).
The user confirmed the session exhausted its credits before committing/pushing.
File timestamps, build/run metadata, and notebook execution timestamps agree.

Review performed on 2026-09-21 against the original 50-section MS3 specification.
The owner requested preservation as an incomplete recovered experiment. No new
feature configuration was selected and no project model was retrained.

## Acceptance decision

The work is substantial and reproducible, but Milestone 3 must not yet be called
complete or independently certified leakage-safe. Preserve its implementation and
recorded results; resolve the findings below before promoting its final model.
The numerical results describe the existing implementation and data treatment.

## Findings requiring resolution

1. **Future-dated bureau updates (temporal acceptance blocker).** The official
   local column dictionary describes DAYS_CREDIT_UPDATE as days before the current
   application. Seventeen bureau records instead contain positive values (10 to
   372 days), affecting 17 application_train customers: 15 in the frozen training
   population and 2 in the frozen test population. build_bureau_features clips
   these values to zero and includes them in bureau__days_update_max. Clipping is
   not evidence of decision-time availability. The report and feature catalog
   must not classify every bureau feature as unconditionally safe. A separate
   repair must either substantiate the anomaly's semantics or conservatively
   exclude ambiguous information, invalidate affected caches, and verify the
   existing experiment protocol again. Do not use test labels to choose a repair.
2. **Incorrect plot legends.** The individual relational ROC, precision-recall,
   and calibration plots say Logistic Regression because make_evaluation_figures
   does not forward a model label. The plotted values are relational predictions;
   the comparison plots correctly identify the models. Correct rendering can use
   persisted predictions; no model training is needed.
3. **Misleading analytical claims.** The report/notebook claim all six validation
   metrics improve monotonically. Validation ECE actually moves from 0.003499 to
   0.005748 and is not monotone. The retuning improvement in validation log loss is
   0.000426903, about 62% of the weakest source's leave-one-out cost (0.000686661),
   not one tenth. The nearest confidence bounds are about 8.6 to 12.8 bootstrap
   standard deviations from zero, rather than uniformly twelve. Fit-time prose
   should match the saved 102.28-second final fit.
   Calibration diagnostics do not establish that recalibration could never help;
   retain the stated limitation that no recalibration comparison was performed.
4. **Recovery robustness.** Feature-cache validity currently checks raw-file size,
   modification time and row limit, not builder-code changes. Model-ablation
   resume checks prediction row count without verifying membership/order and model
   configuration. The current cached artifacts passed the independent checks
   below, but changed-input/code resumes should fail closed in a later repair.

## Independent verification during recovery

- All six raw-table row counts agree with the saved verification records.
- Raw-file signatures, cache schemas, unique customer keys and absence of infinite
  feature values agree with all five build records.
- The current feature builders reproduce cached features for 64 sampled customers
  per source, using the original raw records. No cache was overwritten.
- The 171 feature-catalog entries match the current source specifications.
- Frozen test and internal validation membership reproduce from the configured
  seed; validation excludes the frozen test customers.
- All 14 stored validation prediction vectors reproduce their ablation metrics.
- All 61,503 stored test predictions align with original customer IDs and targets.
- Persisted Logistic Regression, application-only LightGBM, and relational LightGBM
  reproduce the saved test predictions (maximum absolute error about 1e-16) and
  reported metrics, without fitting or changing any model.
- All statistics in the saved 1,000-resample paired bootstrap reproduce from saved
  predictions with seed 42 (maximum absolute difference 5.6e-17); no model was fit.
- The final pipeline contains 291 features and no TARGET/customer/contract ID.
- The notebook contains 13 executed code cells and no recorded error outputs;
  execution timestamps span 2026-09-17 23:43-23:48 UTC. It was not re-executed during
  recovery, because its final-stage cell refits models and overwrites artifacts.
- The session log records one ablation-process observation: peak working set
  1,428 MiB and private memory 2,116 MiB. This supports the reported approximate
  resource scale, but does not establish peak memory for every stage.
- Existing checks: 132 tests passed; Ruff lint passed; 33 source/test files passed
  Ruff format validation. The first sandboxed pytest attempt failed on temporary
  directory permissions; the unrestricted rerun passed.
- Published Milestone 1/2 metrics, figures, reports and notebooks match the
  pre-recovery HEAD. No raw data, split membership, predictions, cache or model
  binary is intended for this preservation commit.

## Original dirty-tree inventory

All 42 files are meaningful recovered work; none is an accidental modification.

| Category | Files | Count |
|---|---|---:|
| A: source/configuration/documentation | README.md; pyproject.toml; src/riskpilot/config.py; src/riskpilot/data/download.py; src/riskpilot/features/tree_preprocessing.py; src/riskpilot/models/comparison.py | 6 |
| A: source | src/riskpilot/features/relational/{__init__,assemble,build,bureau,common,credit_card,installments,pos_cash,previous}.py | 9 |
| A: source | src/riskpilot/models/relational_experiment.py | 1 |
| A: tests | tests/test_relational_features.py; tests/test_relational_experiment.py | 2 |
| B: intentionally versioned figures | artifacts/figures/lightgbm_relational_*.png (4); artifacts/figures/relational_*.png (7) | 11 |
| B: intentionally versioned metrics | artifacts/metrics/lightgbm_relational_metrics.json; relational_{ablation,feature_catalog,importance_by_source,model_comparison,trials}.csv; relational_{bootstrap,build_log,data_audit,raw_verification,selection}.json | 11 |
| B: executed notebook/report | notebooks/04_relational_features.ipynb; reports/relational_features_report.md | 2 |

No category C/D files occur among the original 42. Existing .gitignore rules
already cover raw data, processed tables, memberships, predictions, model binaries,
virtual environments and test/lint caches. Keep those useful local files in place.

## Preservation scope

The original 42 files were archived locally before changing documentation. The
preservation commit keeps implementation, numerical artifacts, figures, and notebook
code/output cells unchanged. The README and opening report/notebook notices clearly
mark the work incomplete; this audit documents unresolved findings rather than
silently repairing the experiment or asserting that it passed full acceptance.
No .gitignore change, file restoration, or deletion was necessary. The next step for
this experiment is a separate MS3 acceptance repair; no later milestone was started.
