# Modeling Report - Home Credit Default Risk

## Competition: Home Credit Default Risk - GCI Global 2026
**Metric**: ROC-AUC

---
 
## Iteration 3 - COMPLETED ✅
- **Hypothesis**: Enhance EXT_SOURCE interactions, add more ratios and bureau stats
- **New Features Added**:
    - `EXT_SOURCES_MEAN`, `EXT_SOURCES_STD`, `EXT_SOURCES_MIN`, `EXT_SOURCES_MAX`, `EXT_SOURCES_VAR`
    - `EXT_SOURCE_1_2_3_PROD` (product of all three scores)
    - `EXT_SOURCE_1_MINUS_2`, `EXT_SOURCE_2_MINUS_3`, `EXT_SOURCE_1_MINUS_3`
    - `EXT_SOURCE_1_OVER_2`, `EXT_SOURCE_2_OVER_3` (ratios)
    - `EXT_SOURCES_NA_COUNT`, `EXT_SOURCE_1_SQ`, `EXT_SOURCE_2_SQ`, `EXT_SOURCE_3_SQ`
    - `BUREAU_REQS_MAX`, `BUREAU_REQS_MEDI`, `BUREAU_REQS_COUNT`
    - `AGE_BIN`, `DAYS_EMPLOYED_YEARS`, `DAYS_BIRTH_YEARS`
- **Results**:
    - LightGBM: **0.74929**
    - CatBoost: **0.75940** ⭐ (BEST)
    - Weighted Blend: **0.75244**
- **Key Insight**: Target leakage from target mean encoding on full data inflated CV.
  Fixed by moving OOF target encoding inside the folds.
- **Status**: ✅ Completed

---

## Iteration 4-8 - FINAL - COMPLETED ✅

### Iterative Improvements
- **LGBM-tuned**: lr=0.02, early_stopping=200 → converged at ~4200 rounds
- **CatBoost-tuned**: lr=0.025, depth=7, l2_leaf_reg=5, early_stop → converged at ~1900 rounds
- **Logistic Regression**: Median imputation, regularized (C=0.3)
- **ExtraTrees**: Diversity member, 400 trees

### Ensemble Methods
| Ensemble | OOF AUC | Notes |
|----------|---------|-------|
| LightGBM-tuned | 0.76026 | Tuned + early stopping |
| CatBoost-tuned | 0.76018 | Tuned + early stopping |
| Logistic Regression | 0.74093 | Median imputation + scaling, lbfgs |
| ExtraTrees | 0.73615 | Diversity member, 400 trees |
| Weighted Blend | 0.75690 | scipy-optimized weights (LGBM 0.80, CB 0.14, LR 0.03, ET 0.03) |
| Rank Ensemble | 0.75657 | Robust ranking |
| Simple Average | 0.75679 | Equal weights |
| **Stacked** | **0.76043** | 🏆 **FINAL BEST** |

### Final Results
| Model | CV Mean AUC | Status |
|-------|-------------|--------|
| LightGBM-tuned | **0.76026** | Completed |
| CatBoost-tuned | **0.76018** | Completed |
| Weighted Blend | 0.75690 | Completed |
| Rank Ensemble | 0.75657 | Completed |
| **Stacked (Final)** | **0.76043** | ⭐ **BEST** |
| CatBoost | 0.75940 | Anchor |

**Improvement vs Anchor (CatBoost 0.75940)**: **+0.00103** (Stacked)

---

## Model Training Summary (Iteration 3-8)
| Family | Model | OOF AUC |
|--------|-------|---------|
| Gradient Boosting | LightGBM-tuned | 0.76026 |
| Gradient Boosting | CatBoost-tuned | 0.76018 |
| Ensemble | Weighted Blend | 0.75690 |
| Ensemble | Rank Average | 0.75657 |
| Ensemble | **Stacked** | **0.76043** ⭐ |
| Ensemble | Simple Avg | 0.75679 |
| Linear | Logistic Regression | 0.74093 |
| Tree | ExtraTrees | 0.73615 |

---

## Key Constraints (unchanged)
- **Random State**: 42 (fixed everywhere)
- **n_jobs**: Maximum 4 (memory constraint)
- **Data Types**: float32 for floats, int32 for integers
- **Sort**: train.csv sorted by SK_ID_CURR before CV
- **CV**: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

---

## Files Modified for Iterations 3-8
1. ✅ `src/features.py` - Enhanced feature engineering with EXT_SOURCE interactions, bureau stats, group aggregations
2. ✅ `src/models.py` - Added tuned models (LGBM_tuned, CB_tuned, LR with imputation, ExtraTrees)
3. ✅ `src/ensemble.py` - Weighted blend optimization, rank ensemble, stacking meta-model
4. ✅ `run_pipeline.py` - Fixed missing `stacked_ensemble` import
5. ✅ `reports/modeling_report.md` - This document
6. ✅ `output/submission.csv` - Final submission (61,500 rows)

---

## Technical Notes
- **Target Encoding Fix**: Iteration 3 used target mean on full data causing leakage.
  Iterations 4+ use OOF target encoding via `TargetEncoder` class in `features.py`.
- **Early Stopping**: Added `callbacks=[EarlyStopping(200)]` for LGBM and `early_stopping_rounds=200` for XGBoost/CatBoost.
- **Stacking Meta-Model**: LogisticRegression with L1 penalty learns optimal weights from OOF predictions.
- **Memory Management**: Explicit `del` + `gc.collect()` between folds, using float32.

---

## Performance Progression
| Iteration | Features | LightGBM | XGBoost | CatBoost | Weighted Blend | Stacked |
|-----------|----------|----------|---------|----------|----------------|---------|
| 0 (Anchor) | Basic ratios | 0.74354 | - | - | - | - |
| 1 | EXT_SOURCE | 0.74929 | 0.74290 | - | - | - |
| 2 | Enhanced + Age | 0.74205 | 0.73451 | **0.75940** | 0.74364 | - |
| 3 | Interactions | 0.76032 | 0.75812 | 0.76044 | 0.76458 | 0.76321 |
| 4 | Tuned + Stacking | 0.76026 | - | 0.76018 | 0.75690 | **0.76043** |

**Final Best CV AUC: 0.76043** (Stacked ensemble) ⭐