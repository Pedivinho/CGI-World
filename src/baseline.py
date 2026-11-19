#!/usr/bin/env python
"""
Baseline 5-fold CV para Home Credit Default Risk.
- Contrato CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) sobre train ordenado por SK_ID_CURR.
- Metrica: roc_auc_score.
- Memoria: float32, n_jobs=4, gc.collect() tras cada fold.
Genera output/submission.csv con el mejor modelo y actualiza los prints de CV.
"""
import os
import gc
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
os.environ['OMP_NUM_THREADS'] = '4'

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / 'Competition' / 'input'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_JOBS = 4
N_SPLITS = 5
RANDOM_STATE = 42

CAT_COLS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
]

NUM_COLS = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DAYS_LAST_PHONE_CHANGE',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
]


def load_data():
    train = pd.read_csv(INPUT_DIR / 'train.csv')
    test = pd.read_csv(INPUT_DIR / 'test.csv')
    train = train.sort_values('SK_ID_CURR').reset_index(drop=True)
    test = test.sort_values('SK_ID_CURR').reset_index(drop=True)
    return train, test


def build_features(train, test, fit=False):
    """Feature engineering comun a train y test (fit solo con train).
    Devuelve (X_train, y, X_test, feature_names)."""
    y = train['TARGET'].astype('float32').values.copy()
    X_train = train.drop(columns=['SK_ID_CURR', 'TARGET']).copy()
    X_test = test.drop(columns=['SK_ID_CURR']).copy()

    # --- Indicadores de missing (antes de imputar) ---
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OWN_CAR_AGE']:
        X_train[col + '_IS_MISSING'] = X_train[col].isna().astype('float32')
        X_test[col + '_IS_MISSING'] = X_test[col].isna().astype('float32')

    # --- EXT_SOURCE_2: imputar con media de train ---
    es2_mean = float(X_train['EXT_SOURCE_2'].mean())
    X_train['EXT_SOURCE_2'] = X_train['EXT_SOURCE_2'].fillna(es2_mean)
    X_test['EXT_SOURCE_2'] = X_test['EXT_SOURCE_2'].fillna(es2_mean)

    # --- OWN_CAR_AGE: >=60 es outlier; decenas ---
    X_train.loc[X_train['OWN_CAR_AGE'] >= 60, 'OWN_CAR_AGE'] = np.nan
    X_test.loc[X_test['OWN_CAR_AGE'] >= 60, 'OWN_CAR_AGE'] = np.nan
    X_train['OWN_CAR_AGE_DECADE'] = (X_train['OWN_CAR_AGE'] // 10).astype('float32')
    X_test['OWN_CAR_AGE_DECADE'] = (X_test['OWN_CAR_AGE'] // 10).astype('float32')
    X_train['OWN_CAR_AGE'] = X_train['OWN_CAR_AGE'].fillna(X_train['OWN_CAR_AGE'].median())
    X_test['OWN_CAR_AGE'] = X_test['OWN_CAR_AGE'].fillna(X_test['OWN_CAR_AGE'].median())

    # --- EXT_SOURCE_1/3: mantener valores (nan OK para GBDT), imputar mediana para LR ---
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_3']:
        med = float(X_train[col].median())
        X_train[col] = X_train[col].fillna(med).astype('float32')
        X_test[col] = X_test[col].fillna(med).astype('float32')

    # --- DAYS_EMPLOYED: 365243 (positivos) es missing codificado ---
    for df in (X_train, X_test):
        df.loc[df['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = np.nan
    de_med = float(X_train['DAYS_EMPLOYED'].median())
    X_train['DAYS_EMPLOYED'] = X_train['DAYS_EMPLOYED'].fillna(de_med).astype('float32')
    X_test['DAYS_EMPLOYED'] = X_test['DAYS_EMPLOYED'].fillna(de_med).astype('float32')
    # Ratio de empleo: dias de empleo actual / dias de vida
    X_train['DAYS_EMPLOYED_RATIO'] = (X_train['DAYS_EMPLOYED'] / X_train['DAYS_BIRTH']).astype('float32')
    X_test['DAYS_EMPLOYED_RATIO'] = (X_test['DAYS_EMPLOYED'] / X_test['DAYS_BIRTH']).astype('float32')

    # --- AMT_REQ_CREDIT_BUREAU_*: 0 = sin consultas ---
    for col in ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']:
        X_train[col] = X_train[col].fillna(0.0).astype('float32')
        X_test[col] = X_test[col].fillna(0.0).astype('float32')
    # Total de consultas
    req_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    X_train['CNT_REQ_CREDIT_BUREAU_TOTAL'] = (
        X_train[req_cols].sum(axis=1)).astype('float32')
    X_test['CNT_REQ_CREDIT_BUREAU_TOTAL'] = (
        X_test[req_cols].sum(axis=1)).astype('float32')

    # --- Features derivadas ---
    X_train['AMT_CREDIT_INCOME_RATIO'] = (
        X_train['AMT_CREDIT'] / X_train['AMT_INCOME_TOTAL']).astype('float32')
    X_test['AMT_CREDIT_INCOME_RATIO'] = (
        X_test['AMT_CREDIT'] / X_test['AMT_INCOME_TOTAL']).astype('float32')
    X_train['AMT_ANNUITY_INCOME_RATIO'] = (
        X_train['AMT_ANNUITY'] / X_train['AMT_INCOME_TOTAL']).astype('float32')
    X_test['AMT_ANNUITY_INCOME_RATIO'] = (
        X_test['AMT_ANNUITY'] / X_test['AMT_INCOME_TOTAL']).astype('float32')
    X_train['AGE_YEARS'] = (-X_train['DAYS_BIRTH'] / 365.25).astype('float32')
    X_test['AGE_YEARS'] = (-X_test['DAYS_BIRTH'] / 365.25).astype('float32')

    # --- Encodado de categoricas: ordinal (XNA = -1) ---
    for col in CAT_COLS:
        if fit:
            cats = sorted(X_train[col].dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(cats)}
        else:
            cats = sorted(X_train[col].dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(cats)}
        X_train[col] = X_train[col].map(mapping).astype('float32')
        X_test[col] = X_test[col].map(mapping).astype('float32')

    feature_names = X_train.columns.tolist()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    return X_train, y, X_test, feature_names


def cv_evaluate(name, X, y, make_model):
    """5-fold CV; devuelve (aucs, oof_predictions)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(y), np.nan, dtype='float64')
    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        model = make_model()
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model.predict_proba(X.iloc[va_idx])
        oof[va_idx] = pred
        auc = roc_auc_score(y[va_idx], pred)
        aucs.append(auc)
        print(f'  [{name}] fold {fold+1}: AUC={auc:.5f} ({time.time()-t0:.0f}s)')
        del model
        gc.collect()
    return aucs, oof


def make_tutorial_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, n_jobs=N_JOBS,
        random_state=SEED, class_weight='balanced')


def make_logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, C=1.0, n_jobs=N_JOBS, random_state=SEED)


def make_lightgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective='binary', n_estimators=600, learning_rate=0.05,
        num_leaves=31, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=40, reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=N_JOBS, random_state=SEED, verbose=-1)


def make_xgboost():
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=600, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=0.1,
        tree_method='hist', n_jobs=N_JOBS, random_state=SEED)


def make_catboost():
    import catboost as cb
    return cb.CatBoostClassifier(
        iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        random_seed=SEED, thread_count=N_JOBS, verbose=200)


def main():
    print('=== Baseline: carga de datos ===')
    train, test = load_data()
    X, y, X_test, feature_names = build_features(train, test, fit=True)
    print(f'Feature matrix: {X.shape}')
    gc.collect()

    # Tutorial RF: 5 features del tutorial
    tutorial_features = ['NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_2',
                         'OWN_CAR_AGE', 'ORGANIZATION_TYPE']
    X_tut = X[tutorial_features]

    results = {}

    print('\n=== [1/5] Tutorial RandomForest (5 features) ===')
    aucs, oof_rf = cv_evaluate('RF-tutorial', X_tut, y, make_tutorial_rf)
    results['rf_tutorial_5feat'] = {'aucs': aucs, 'mean': float(np.mean(aucs)), 'std': float(np.std(aucs)), 'oof': oof_rf}
    gc.collect()

    print('\n=== [2/5] LogisticRegression (features completas) ===')
    aucs, oof_lr = cv_evaluate('LogReg', X, y, make_logistic)
    results['logreg_full'] = {'aucs': aucs, 'mean': float(np.mean(aucs)), 'std': float(np.std(aucs)), 'oof': oof_lr}
    gc.collect()

    print('\n=== [3/5] LightGBM ===')
    aucs, oof_lgb = cv_evaluate('LightGBM', X, y, make_lightgbm)
    results['lightgbm'] = {'aucs': aucs, 'mean': float(np.mean(aucs)), 'std': float(np.std(aucs)), 'oof': oof_lgb}
    gc.collect()

    print('\n=== [4/5] XGBoost ===')
    aucs, oof_xgb = cv_evaluate('XGBoost', X, y, make_xgboost)
    results['xgboost'] = {'aucs': aucs, 'mean': float(np.mean(aucs)), 'std': float(np.std(aucs)), 'oof': oof_xgb}
    gc.collect()

    print('\n=== [5/5] CatBoost ===')
    aucs, oof_cb = cv_evaluate('CatBoost', X, y, make_catboost)
    results['catboost'] = {'aucs': aucs, 'mean': float(np.mean(aucs)), 'std': float(np.std(aucs)), 'oof': oof_cb}
    gc.collect()

    # Tabla resumen
    print('\n=== TABLA DE RESULTADOS CV (ROC-AUC) ===')
    best_name, best_score = None, 0.0
    for name, r in results.items():
        print(f'{name}: {r["mean"]:.5f} +/- {r["std"]:.5f}  (folds: {[round(a,4) for a in r["aucs"]]})')
        if r['mean'] > best_score:
            best_name, best_score = name, r['mean']
    print(f'\nMEJOR MODELO: {best_name} con CV AUC {best_score:.5f}')

    # Guardar resultados para torre_de_control
    summary = {
        name: {k: v for k, v in r.items() if k != 'oof'}
        for name, r in results.items()
    }
    summary['best'] = best_name
    with open(ROOT / 'reports' / 'cv_baseline_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- Entrenar el mejor modelo con TODO el train y predecir test ---
    print(f'\n=== Retrain {best_name} con todo el train ===')
    maker = {
        'rf_tutorial_5feat': make_tutorial_rf,
        'logreg_full': make_logistic,
        'lightgbm': make_lightgbm,
        'xgboost': make_xgboost,
        'catboost': make_catboost,
    }[best_name]
    X_fit = X if best_name != 'rf_tutorial_5feat' else X_tut
    final_model = maker()
    final_model.fit(X_fit, y)
    pred = final_model.predict_proba(X_test)
    del final_model
    gc.collect()

    # Submission en el orden de sample_submission.csv
    sample_sub = pd.read_csv(INPUT_DIR / 'sample_submission.csv')
    assert (sample_sub['SK_ID_CURR'].values == test['SK_ID_CURR'].values).all(), 'Desajuste de SK_ID_CURR'
    sample_sub['TARGET'] = pred.astype('float64')
    sample_sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
    print(f'output/submission.csv escrito: {len(sample_sub)} filas, TARGET [{sample_sub.TARGET.min():.4f}, {sample_sub.TARGET.max():.4f}]')
    print('Fin del baseline.')


if __name__ == '__main__':
    main()
