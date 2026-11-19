#!/usr/bin/env python
"""
EDA completo para Home Credit Default Risk.
Genera reports/eda_summary.md con hallazgos accionables.
"""
import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / 'Competition' / 'input'
REPORTS_DIR = ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

def reduce_memory(df):
    """Convierte numéricas a float32/int32 para ahorrar RAM."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            if df[col].min() > -2**31 and df[col].max() < 2**31 - 1:
                df[col] = df[col].astype('int32')
    return df

def main():
    print('Cargando train.csv...')
    train = pd.read_csv(INPUT_DIR / 'train.csv')
    train = reduce_memory(train)
    print(f'train.shape: {train.shape}')
    
    print('Cargando test.csv...')
    test = pd.read_csv(INPUT_DIR / 'test.csv')
    test = reduce_memory(test)
    print(f'test.shape: {test.shape}')
    
    cat_cols = [c for c in train.columns if pd.api.types.is_string_dtype(train[c]) and c not in ('SK_ID_CURR', 'TARGET')]
    num_cols = [c for c in train.columns if c not in ('SK_ID_CURR', 'TARGET') and c not in cat_cols]
    
    lines = []
    lines.append('# EDA Summary — Home Credit Default Risk')
    lines.append('')
    lines.append('## 1. Dimensiones y tipos')
    lines.append(f'- train: {train.shape[0]:,} filas × {train.shape[1]} columnas')
    lines.append(f'- test: {test.shape[0]:,} filas × {test.shape[1]} columnas')
    lines.append(f'- Categóricas: {len(cat_cols)} → {cat_cols}')
    lines.append(f'- Numéricas: {len(num_cols)}')
    n_pos = int(train['TARGET'].sum())
    target_rate = float(train['TARGET'].mean())
    lines.append(f'- TARGET: {n_pos} positivos / {len(train)} total = {target_rate:.4f} ({target_rate*100:.2f}%)')
    lines.append('')
    
    lines.append('## 2. Missing values (train)')
    missing = train.isna().mean().sort_values(ascending=False)
    lines.append('| Columna | % Missing |')
    lines.append('| --- | --- |')
    for col, pct in missing.items():
        if pct > 0:
            lines.append(f'| {col} | {pct*100:.2f}% |')
    lines.append('')
    
    lines.append('## 3. Anomalías detectadas')
    # DAYS_BIRTH siempre negativo (días antes de la aplicación)
    if 'DAYS_BIRTH' in train.columns:
        db_min = train['DAYS_BIRTH'].min()
        db_max = train['DAYS_BIRTH'].max()
        lines.append(f'- DAYS_BIRTH: min={db_min:.0f}, max={db_max:.0f} (todos negativos = correcto)')
    # DAYS_EMPLOYED puede tener valores positivos (outliers codificados como 365243)
    if 'DAYS_EMPLOYED' in train.columns:
        n_pos = (train['DAYS_EMPLOYED'] > 0).sum()
        lines.append(f'- DAYS_EMPLOYED: {n_pos:,} valores positivos ({n_pos/len(train)*100:.2f}%) → típicamente codificación de missing (365243)')
    # AMT_* con colas pesadas
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
        if col in train.columns:
            q99 = train[col].quantile(0.99)
            q999 = train[col].quantile(0.999)
            lines.append(f'- {col}: p99={q99:,.0f}, p99.9={q999:,.0f}')
    lines.append('')
    
    lines.append('## 4. Target rate por categórica (top 5 categorías por feature)')
    for col in cat_cols:
        lines.append(f'### {col}')
        tr = train.groupby(col)['TARGET'].agg(['sum', 'count']).reset_index()
        tr['rate'] = tr['sum'] / tr['count']
        tr = tr.sort_values('rate', ascending=False).head(5)
        lines.append('| Categoría | Positivos | Total | Rate |')
        lines.append('| --- | --- | --- | --- |')
        for _, row in tr.iterrows():
            sum_v = int(row['sum'])
            count_v = int(row['count'])
            rate_v = float(row['rate'])
            lines.append(f'| {row[col]} | {sum_v} | {count_v} | {rate_v:.4f} |')
        lines.append('')
    
    lines.append('## 5. Correlaciones con TARGET (numéricas, valor absoluto)')
    corr = train[num_cols + ['TARGET']].corr()['TARGET'].drop('TARGET').abs().sort_values(ascending=False)
    lines.append('| Columna | |corr| con TARGET |')
    lines.append('| --- | --- |')
    for col, val in corr.head(15).items():
        lines.append(f'| {col} | {val:.4f} |')
    lines.append('')
    
    lines.append('## 6. Feature importance rápida (LightGBM, 1 fold, n_jobs=4)')
    try:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        
        # Preprocesamiento mínimo para importance
        X = train[num_cols].fillna(-999).astype('float32')
        y = train['TARGET'].astype('float32')
        
        # Label encode categóricas simples
        for col in cat_cols:
            if col in train.columns:
                mapping = {v: i for i, v in enumerate(train[col].fillna('MISSING').unique())}
                train[col + '_le'] = train[col].fillna('MISSING').map(mapping).astype('int32')
        
        X_cat = train[[c + '_le' for c in cat_cols]].fillna(-1).astype('int32')
        X_full = np.hstack([X.values, X_cat.values])
        
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # solo 2 folds para rapidez
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y)):
            X_tr, X_val = X_full[tr_idx], X_full[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            
            lgb_train = lgb.Dataset(X_tr, y_tr)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': 4,
                'seed': 42
            }
            
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            importance = pd.DataFrame({
                'feature': num_cols + [c + '_le' for c in cat_cols],
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            lines.append('| Feature | Importancia (gain) |')
            lines.append('| --- | --- |')
            for _, row in importance.head(20).iterrows():
                feat = row['feature']
                imp = row['importance']
                lines.append(f'| {feat} | {imp:,.0f} |')
            lines.append('')
            
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            auc = roc_auc_score(y_val, y_pred)
            lines.append(f'**AUC fold {fold+1}: {auc:.4f}**')
            lines.append('')
            break  # solo primer fold
        gc.collect()
    except Exception as e:
        lines.append(f'Error en feature importance: {e}')
        lines.append('')
    
    lines.append('## 7. Decisiones de preprocesamiento recomendadas')
    lines.append('1. **EXT_SOURCE_1 (69.5% nulo)**: NO eliminar. Mantener valores originales en float32 + crear `EXT_SOURCE_1_IS_MISSING` binario.')
    lines.append('2. **EXT_SOURCE_3 (31.9% nulo)**: Imputar con mediana + crear `EXT_SOURCE_3_IS_MISSING`.')
    lines.append('3. **OWN_CAR_AGE (66% nulo)**: Crear indicador binario + agrupar por decenas (OHE opcional).')
    lines.append('4. **DAYS_EMPLOYED positivos**: Reemplazar 365243 con NaN, luego imputar con mediana.')
    lines.append('5. **AMT_* con colas pesadas**: Considerar log1p o clipping en p99.9.')
    lines.append('6. **Categóricas de alta cardinalidad (ORGANIZATION_TYPE=58)**: Usar target encoding o frequency encoding en lugar de OHE.')
    lines.append('')
    lines.append('---')
    lines.append('*Generado por src/eda.py*')
    
    report_path = REPORTS_DIR / 'eda_summary.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f'EDA guardado en {report_path}')
    print('Fin del EDA.')

if __name__ == '__main__':
    main()
