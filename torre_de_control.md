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

## 4. Hallazgos EDA (completado)
- Se cargaron 171,202 filas de train y 61,500 filas de test.
- Valores nulos más altos: `EXT_SOURCE_1` (69.5%), `OWN_CAR_AGE` (66%), `EXT_SOURCE_3` (31.9%).
- Importancia rápida de características (LightGBM): `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`, `AMT_CREDIT`.
- Se documentaron decisiones de preprocesamiento (mantener `EXT_SOURCE_1`, tratar OWN_CAR_AGE con decenas, target-encoding para ORGANIZATION_TYPE, etc.).

## 5. Tablero de métricas CV ROC-AUC
| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Media ± Std |
| --- | --- | --- | --- | --- | --- | --- |
| rf_tutorial_5feat | 0.6731 | 0.6654 | 0.6695 | 0.6685 | 0.6716 | **0.6696 ± 0.0026** |
| logreg_full | 0.6331 | 0.6242 | 0.6388 | 0.6299 | 0.6351 | **0.6322 ± 0.0049** |
| lightgbm | 0.7430 | 0.7500 | 0.7536 | 0.7452 | 0.7513 | **0.7486 ± 0.0039** |
| xgboost | 0.7420 | 0.7509 | 0.7534 | 0.7424 | 0.7506 | **0.7479 ± 0.0047** |
| **catboost (MEJOR)** | 0.7494 | 0.7516 | 0.7584 | 0.7495 | 0.7528 | **0.7524 ± 0.0033** |

## 6. Estado del pipeline
- [x] Fase 1 EDA → `reports/eda_summary.md` generada
- [x] Fase 2 Baseline → `output/submission.csv` generado, CV AUCs guardados en `reports/cv_baseline_results.json`
- [x] Fase 3 Validador → `src/validate_submission.py` validó exitosamente `output/submission.csv`
- [ ] Fase 4 Merge OpenCode 2 (pendiente):
    - Espere hasta `STATUS: READY_FOR_MERGE` en `reports/modeling_report.md` de `Pedivinho/jormungandr` o `Pedivinho/nuckelavee`
    - Usar `run_pipeline.py` para integrar `features.py` → `models.py` → `ensemble.py`
- [ ] Fase 5 Push final (pendiente):
    - `git push` solo después de que OpenCode 2 haya entregado un submission validado

## 7. Subdirectorio del pipeline
- `output/submission.csv` — submission validado listo para `Omnicampus` (base CatBoost)
- `reports/cv_baseline_results.json` — detalles completos de CV para torre_de_control.md