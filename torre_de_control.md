# Torre de Control — Home Credit Default Risk · GCI Global 2026

## 0. Contrato compartido (CV)
- **Datos**: `Competition/input/train.csv`, `Competition/input/test.csv`, `Competition/input/sample_submission.csv`.
- **Orden**: train/test leídos con pandas y ordenados ascendentemente por `SK_ID_CURR` (train `0..171201`, test `171202..232701`). `sample_submission.SK_ID_CURR == test.SK_ID_CURR`.
- **CV**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` aplicada sobre `train.csv` ordenado por `SK_ID_CURR`.
- **Métrica oficial**: `roc_auc_score(TARGET_real, TARGET_predicho)`.
- **Semilla global**: `seed=42` en todos los `RandomState`/`random_state`.
- **Entregable**: `output/submission.csv` con esquema exacto `['SK_ID_CURR','TARGET']`, 61,500 filas, mismo orden de `SK_ID_CURR` que `sample_submission.csv`.

## 1. Propiedad de ficheros (evitar colisiones en git)
- **OpenCode (tú, rama `main`)**: `torre_de_control.md`, `src/eda.py`, `src/baseline.py`, `src/validate_submission.py`, `run_pipeline.py`, `reports/eda_summary.md`, `output/submission.csv`.
- **OpenCode 2 (rama/worktree `Pedivinho/*`)**: `src/features.py`, `src/models.py`, `src/ensemble.py`, `reports/modeling_report.md`.
- **Solo lectura (compartido)**: `Competition/input/*`, `Competition/HomeCredit_columns_description.xlsx`.

## 2. Restricciones técnicas (entorno verificado)
- HW: 12 CPUs, 8.34 GB RAM total (~0.8 GB libre), **sin GPU**.
- Python 3.13, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.9.0, torch 2.13.0+cpu.
- Instalados: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, optuna 5.1.0, imbalanced-learn 0.14.2.
- **Reglas de memoria**: convertir features a `float32`/`int32`; `n_jobs=4` máximo en todos los modelos; `gc.collect()` tras cada fold/modelo; evitar OHE de `ORGANIZATION_TYPE` (58 categorías) sin necessity.

## 3. Fases de mi frente (ejecución secuencial)
- **Fase 1 — EDA**: `src/eda.py` → `reports/eda_summary.md` (missing %, distribuciones, outliers en `DAYS_*`/`AMT_*`, target-rate por categórica, correlaciones, feature importance rápida).
- **Fase 2 — Baseline**: `src/baseline.py` → 5-fold CV ROC-AUC (tutorial RF, LogisticRegression, LightGBM, XGBoost, CatBoost) → genera `output/submission.csv` inicial.
- **Fase 3 — Validador**: `src/validate_submission.py` → verifica esquema, filas, orden, rango, cero NaN/Inf. Exit code 0 = válido.
- **Fase 4 — Integración con OpenCode 2**: monitoreo de ramas `Pedivinho/*` hasta `STATUS: READY_FOR_MERGE` en `reports/modeling_report.md`; `git merge` a `main`; `run_pipeline.py` conecta `features.py` → `models.py` → `ensemble.py`.
- **Fase 5 — GitOps**: commits atómicos por fase, conventional commits, sin Co-Authored-By, push continuo a `origin/main`.

## 4. Hallazgos EDA (pendiente de ejecución)
- [pendiente]

## 5. Tablero de métricas CV ROC-AUC
| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Media ± Std |
| --- | --- | --- | --- | --- | --- | --- |
| [pendiente] | | | | | | |

## 6. Estado del pipeline
- [ ] Fase 1 EDA
- [ ] Fase 2 Baseline
- [ ] Fase 3 Validador
- [ ] Fase 4 Merge OpenCode 2
- [ ] Fase 5 Push final