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

---

## Iteration 9 - COMPETITIVE SPRINT - COMPLETED ✅

### Hypothesis
The 0.7612 plateau was caused by LR (0.741) and ExtraTrees (0.736) diluting the
ensemble. Solution: **GBDT-only multi-seed ensemble** plus high-impact domain
features (EXT_SOURCE block, financial ratios, group aggregations, frequency
encoding, OOF smoothed target encoding).

### New Features (326 base + 5 OOF TE = 331)
- EXT_SOURCE block: mean/std/min/max/sum/prod, pairwise diffs & ratios,
  weighted combo (2·E1+3·E2+4·E3), squares, sqrts, NA-count, ISNA indicators
- Financial ratios: CREDIT_TO_ANNUITY_RATIO (loan term in months),
  ANNUITY_TO_INCOME, CREDIT_TO_INCOME, CREDIT_TO_GOODS, CREDIT_GOODS_DIFF,
  INCOME_PER_PERSON, INCOME_PER_CHILD, CHILDREN_RATIO
- Age/employment: AGE_YEARS, AGE_BIN, EMPLOYED_TO_BIRTH, CAR_TO_BIRTH,
  CAR_TO_EMPLOYED, PHONE_TO_BIRTH, ID_TO_BIRTH, DAYS_EMPLOYED_ANOM
- Bureau: BUREAU_REQ_SUM, BUREAU_REQ_MAX
- Categorical interactions: GENDER_EDU, INCOME_ORG, CONTRACT_GENDER
- Frequency encoding (11 cols), group aggregations (5 groups × 7 vars × 4 aggs)
- OOF smoothed target encoding (m=20) computed strictly inside folds

### Multi-Seed GBDT Results (seeds [42, 2024, 777], 5-fold OOF)
| Model | OOF ROC-AUC |
|-------|-------------|
| LightGBM multi-seed | **0.76087** |
| XGBoost multi-seed | 0.76063 |
| CatBoost multi-seed | 0.76046 |

### GBDT-Only Ensemble (Nelder-Mead optimization)
| Ensemble | OOF ROC-AUC | Weights (LGBM/CB/XGB) |
|----------|-------------|----------------------|
| **Prob blend** | **0.76187** ⭐ | 0.479 / 0.444 / 0.077 |
| Rank blend | 0.76186 | 0.466 / 0.449 / 0.085 |
| Simple avg | 0.76178 | 0.333 / 0.333 / 0.333 |

**Improvement vs previous best (0.7612): +0.00067** (0.76187)
**Improvement vs Stacked (0.76043): +0.00144**

### Key Insights
- Multi-seed averaging lifted LGBM from 0.76046 (single seed) to 0.76087
- GBDT-only ensemble outperforms the LR/ET-diluted stacked ensemble
- Probabilistic Nelder-Mead blend is marginally better than rank blend
- Submission regenerated: 61,500 rows, `SK_ID_CURR` + `TARGET`, no nulls

### Files Modified
1. ✅ `src/features.py` - EXT_SOURCE block, financial ratios, group aggs, freq encoding, OOF TE
2. ✅ `src/models.py` - Multi-seed LGBM/XGBoost/CatBoost, legacy API preserved
3. ✅ `src/ensemble.py` - GBDT-only Nelder-Mead ensemble, `run_competitive_pipeline`
4. ✅ `reports/modeling_report.md` - This document
5. ✅ `output/submission.csv` - Final submission (61,500 rows)

### Constraints (unchanged)
- Random State: 42 (fixed everywhere)
- n_jobs: 4 (memory constraint)
- float32 / int32 dtypes
- train.csv sorted by SK_ID_CURR before CV
- StratifiedKFold(5, shuffle=True, random_state=42)

STATUS: READY_FOR_MERGE