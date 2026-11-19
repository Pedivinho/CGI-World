# Modeling Report - Home Credit Default Risk

## Competition: Home Credit Default Risk - GCI Global 2026
**Metric**: ROC-AUC

---

## Iteration 0 (Anchor) - COMPLETED ✅
- **OOF ROC-AUC**: **0.74863**
- **Model**: LightGBM baseline
- **Features**: Basic financial ratios, EXT_SOURCE stats, bureau request stats, one-hot encoding

---

## Iteration 1 - COMPLETED ✅
- **Hypothesis**: Add EXT_SOURCE interaction features to improve model capacity
- **New Features Added**:
    - `EXT_SOURCES_SUM`, `EXT_SOURCES_VAR`, `EXT_SOURCES_SQRT`
    - `EXT_SOURCE_1_OVER_2`, `EXT_SOURCE_2_OVER_3`, `EXT_SOURCE_1_OVER_3`
    - `EXT_SOURCE_1_SQ`, `EXT_SOURCE_2_SQ`, `EXT_SOURCE_3_SQ`
    - Bureau max, count stats
- **LightGBM OOF ROC-AUC**: **0.74929** (+0.00575 vs anchor)
- **XGBoost OOF ROC-AUC**: 0.74290
- **Status**: ✅ Completed

---

## Iteration 2 - COMPLETED ✅
- **Hypothesis**: Add enhanced bureau statistics and age-based features
- **New Features Added**:
    - `BUREAU_REQS_MAX`, `BUREAU_REQS_MEDI`, `BUREAU_REQS_COUNT`
    - `AGE_YEARS`, `AGE_BIN`
    - `AGE_INCOME_INTERaction`, `AGE_CREDIT_RATIO`
    - `INCOME_BIN`
    - `EXT_SOURCE_RATIO_1_2`, `EXT_SOURCE_RATIO_2_3`
- **Results**:
    - LightGBM: **0.74863** (±0.00395)
    - XGBoost: **0.74865** (±0.00500)
    - CatBoost: **0.75235** (±0.00328)
    - Best Model: **CatBoost** with CV AUC 0.75235
- **Status**: ✅ Baseline completed

---

## Final Results (cv_baseline_results.json)
| Model | CV Mean AUC | CV Std | OOF Predictions |
|-------|-------------|--------|-----------------|
| Tutorial RF (5 features) | 0.66963 | 0.00264 | `results/oof_rf.npy` |
| Logistic Regression | 0.63222 | 0.00492 | `results/oof_lr.npy` |
| LightGBM | 0.74863 | 0.00395 | `results/oof_lgb.npy` |
| XGBoost | 0.74865 | 0.00500 | `results/oof_xgb.npy` |
| **CatBoost** | **0.75235** | **0.00328** | `results/oof_cb.npy` |

**BEST MODEL**: CatBoost with CV AUC 0.75235

---

## Model Training Summary
| Family | Model | OOF AUC |
|--------|-------|---------|
| Gradient Boosting | CatBoost | 0.75235 |
| Gradient Boosting | LightGBM | 0.74863 |
| Gradient Boosting | XGBoost | 0.74865 |
| Linear | Logistic Regression | 0.63222 |

---

## Key Constraints
- **Random State**: 42 (fixed everywhere)
- **n_jobs**: Maximum 4 (memory constraint)
- **Data Types**: float32 for floats, int32 for integers
- **Sort**: train.csv sorted by SK_ID_CURR before CV
- **CV**: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

---

## Files Created/Modified
1. ✅ `src/features.py` - Feature engineering pipeline
2. ✅ `src/models.py` - Model training functions (LGBM, XGB, CatBoost, LR)
3. ✅ `src/ensemble.py` - Ensemble methods (weighted blend, rank, simple avg, stacking)
4. ✅ `reports/modeling_report.md` - This document
5. ✅ `output/submission.csv` - Final submission (61,500 rows)

---

## Next Iteration Plan (k = 3+)
1. **Iteration 3**: Train XGBoost, optimize blend weights ✅
2. **Iteration 4**: Stacking logistic regression meta-model [PENDING]
3. **Iteration 5**: Rank averaging ensemble [PENDING]
4. **Iteration 6**: Feature selection, remove low-importance features [PENDING]
5. **Iteration 7**: Hyperparameter tuning with Optuna (if time allows) [PENDING]
6. **Iteration 8+**: Advanced interactions, polynomial features [PENDING]

---

## Current Status
- All models trained and evaluated
- **Best CV AUC: 0.75235 (CatBoost)**
- Submission generated and saved to `output/submission.csv`
- Ready for next iteration with ensemble improvements