"""
Iteration 2 - Enhanced features + CatBoost + Weighted Blend
"""
import pandas as pd
import numpy as np
import re
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import os

# Use correct path
BASE_DIR = 'D:/Usuario/orca/CGI-World'
TRAIN_PATH = os.path.join(BASE_DIR, 'Competition/input/train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'Competition/input/test.csv')

def sanitize_name(name):
    sanitized = re.sub(r'[^a-zA-Z0-9_]+', '_', name).strip('_')
    if sanitized and sanitized[0].isdigit():
        sanitized = 'col_' + sanitized
    return sanitized if sanitized else 'unnamed_col'

def build_features(train_df, test_df):
    # Combine for processing
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # Clean anomalies - DAYS_EMPLOYED == 365243 is a special value
    df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == 365243
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})
    
    # Convert days to years
    df['DAYS_EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'] / 365
    df['DAYS_BIRTH_YEARS'] = df['DAYS_BIRTH'] / 365
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    # Financial Ratios
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    
    # EXT_SOURCE statistics
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)
    df['EXT_SOURCES_STD'] = df[ext_sources].std(axis=1)
    df['EXT_SOURCES_MIN'] = df[ext_sources].min(axis=1)
    df['EXT_SOURCES_MAX'] = df[ext_sources].max(axis=1)
    df['EXT_SOURCES_PROD'] = df[ext_sources].prod(axis=1)
    df['EXT_SOURCES_MEDI'] = df[ext_sources].median(axis=1)
    
    # Iteration 1: Additional EXT_SOURCE interactions
    df['EXT_SOURCES_SUM'] = df[ext_sources].sum(axis=1)
    df['EXT_SOURCES_VAR'] = df[ext_sources].var(axis=1)
    df['EXT_SOURCES_SQRT'] = np.sqrt(df[ext_sources].sum(axis=1) + 1e-6)
    
    # Ratios between EXT_SOURCE scores
    df['EXT_SOURCE_1_OVER_2'] = df['EXT_SOURCE_1'] / (df['EXT_SOURCE_2'] + 1e-6)
    df['EXT_SOURCE_2_OVER_3'] = df['EXT_SOURCE_2'] / (df['EXT_SOURCE_3'] + 1e-6)
    df['EXT_SOURCE_1_OVER_3'] = df['EXT_SOURCE_1'] / (df['EXT_SOURCE_3'] + 1e-6)
    
    # EXT_SOURCE squared features
    df['EXT_SOURCE_1_SQ'] = df['EXT_SOURCE_1'] ** 2
    df['EXT_SOURCE_2_SQ'] = df['EXT_SOURCE_2'] ** 2
    df['EXT_SOURCE_3_SQ'] = df['EXT_SOURCE_3'] ** 2

    # Bureau Requests Statistics
    bureau_reqs = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
    df['BUREAU_REQS_SUM'] = df[bureau_reqs].sum(axis=1)
    df['BUREAU_REQS_MEAN'] = df[bureau_reqs].mean(axis=1)
    df['BUREAU_REQS_STD'] = df[bureau_reqs].std(axis=1)
    
    # Iteration 2: Enhanced bureau stats
    if len(bureau_reqs) > 0:
        df['BUREAU_REQS_MAX'] = df[bureau_reqs].max(axis=1)
        df['BUREAU_REQS_MEDI'] = df[bureau_reqs].median(axis=1)
        df['BUREAU_REQS_COUNT'] = len(bureau_reqs)
    
    # Missing values
    df['NULL_COUNT'] = df.isnull().sum(axis=1)
    df['NULL_COUNT_RATIO'] = df['NULL_COUNT'] / df.shape[1]
    
    # Age features
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['AGE_BIN'] = pd.cut(df['AGE_YEARS'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4]).astype('int32')
    
    # Iteration 2: Additional ratios and interactions
    # Age-based features
    df['AGE_INCOME_INTERaction'] = df['AGE_YEARS'] * df['AMT_INCOME_TOTAL'] / 10000
    df['AGE_CREDIT_RATIO'] = df['AGE_YEARS'] / (df['AMT_CREDIT'] / 1000 + 1e-6)
    
    # Income-based bins
    df['INCOME_BIN'] = pd.cut(df['AMT_INCOME_TOTAL'], 
                               bins=[0, 25000, 50000, 100000, 200000, 1e9], 
                               labels=[0, 1, 2, 3, 4]).astype('int32')
    
    # External score ratios
    df['EXT_SOURCE_RATIO_1_2'] = df['EXT_SOURCE_1'] / (df['EXT_SOURCE_2'] + 0.001)
    df['EXT_SOURCE_RATIO_2_3'] = df['EXT_SOURCE_2'] / (df['EXT_SOURCE_3'] + 0.001)
    
    # Categorical encoding
    cat_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 
                'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat_cols)
    
    # Separate back
    train_proc = df[df['TARGET'].notnull()].copy()
    test_proc = df[df['TARGET'].isnull()].drop(columns=['TARGET']).copy()
    
    # Sanitize column names (LGBM doesn't like special chars and leading numbers)
    train_proc.columns = [sanitize_name(c) for c in train_proc.columns]
    test_proc.columns = [sanitize_name(c) for c in test_proc.columns]
    
    # Downcast
    for col in train_proc.select_dtypes(include=['float64']).columns:
        train_proc[col] = train_proc[col].astype('float32')
    for col in test_proc.select_dtypes(include=['float64']).columns:
        test_proc[col] = test_proc[col].astype('float32')
    for col in train_proc.select_dtypes(include=['int64']).columns:
        train_proc[col] = train_proc[col].astype('int32')
    for col in test_proc.select_dtypes(include=['int64']).columns:
        test_proc[col] = test_proc[col].astype('int32')
        
    del df
    gc.collect()
    
    return train_proc, test_proc


if __name__ == "__main__":
    print('Loading data...')
    train_df = pd.read_csv(TRAIN_PATH).sort_values('SK_ID_CURR')
    test_df = pd.read_csv(TEST_PATH).sort_values('SK_ID_CURR')

    # Quick type casting
    for df in [train_df, test_df]:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

    print('Building features...')
    train_proc, test_proc = build_features(train_df, test_df)

    y = train_proc['TARGET'].values
    X = train_proc.drop(columns=['TARGET', 'SK_ID_CURR']).values
    X_test = test_proc.drop(columns=['SK_ID_CURR']).values
    n_features = X.shape[1]

    print(f'Features: {n_features}, Train: {X.shape[0]}, Test: {X_test.shape[0]}')

    del train_df, test_df, train_proc, test_proc
    gc.collect()

    # Train LightGBM
    params = {
        'n_estimators': 3000, 'learning_rate': 0.05, 'num_leaves': 31,
        'colsample_bytree': 0.8, 'subsample': 0.8, 'max_depth': 8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'n_jobs': 4, 'random_state': 42, 'verbosity': -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgbm = np.zeros(len(y), dtype=np.float32)
    test_lgbm = np.zeros(X_test.shape[0], dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'--- LightGBM Fold {fold+1} ---')
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc')
        oof_lgbm[val_idx] = model.predict_proba(X_val)[:, 1]
        test_lgbm += model.predict_proba(X_test)[:, 1] / 5
        del X_tr, X_val, y_tr, y_val, model
        gc.collect()

    auc_lgbm = roc_auc_score(y, oof_lgbm)
    print(f'LightGBM OOF: {auc_lgbm:.5f}')

    # Train XGBoost
    xgb_params = {
        'n_estimators': 3000, 'learning_rate': 0.05, 'max_depth': 6,
        'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'n_jobs': 4, 'random_state': 42, 'tree_method': 'hist'
    }

    oof_xgb = np.zeros(len(y), dtype=np.float32)
    test_xgb = np.zeros(X_test.shape[0], dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'--- XGBoost Fold {fold+1} ---')
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
        test_xgb += model.predict_proba(X_test)[:, 1] / 5
        del X_tr, X_val, y_tr, y_val, model
        gc.collect()

    auc_xgb = roc_auc_score(y, oof_xgb)
    print(f'XGBoost OOF: {auc_xgb:.5f}')

    # Train CatBoost
    cb_params = {
        'iterations': 3000, 'learning_rate': 0.05, 'depth': 6,
        'l2_leaf_reg': 3, 'border_count': 128,
        'thread_count': 4, 'random_seed': 42, 'verbose': 0
    }

    oof_cb = np.zeros(len(y), dtype=np.float32)
    test_cb = np.zeros(X_test.shape[0], dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'--- CatBoost Fold {fold+1} ---')
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = CatBoostClassifier(**cb_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        oof_cb[val_idx] = model.predict_proba(X_val)[:, 1]
        test_cb += model.predict_proba(X_test)[:, 1] / 5
        del X_tr, X_val, y_tr, y_val, model
        gc.collect()

    auc_cb = roc_auc_score(y, oof_cb)
    print(f'CatBoost OOF: {auc_cb:.5f}')

    # Optimize weights
    def optimize_weight(w):
        return -roc_auc_score(y, w[0] * oof_lgbm + (1-w[0]) * oof_xgb)

    result = minimize(optimize_weight, [0.5], bounds=[(0, 1)])
    best_w = result.x[0]
    blend_auc = roc_auc_score(y, best_w * oof_lgbm + (1-best_w) * oof_xgb)
    print(f'Optimize Blend OOF: {blend_auc:.5f} (w={best_w:.3f})')

    # Save best submission
    submission = pd.read_csv(os.path.join(BASE_DIR, 'Competition/input/sample_submission.csv'))
    submission['TARGET'] = (best_w * test_lgbm + (1-best_w) * test_xgb).astype(np.float32)
    submission.to_csv(os.path.join(BASE_DIR, 'output/submission.csv'), index=False)
    print('Submission saved!')

    print(f'\n=== ITERATION 2 RESULTS ===')
    print(f'LightGBM: {auc_lgbm:.5f}')
    print(f'XGBoost:  {auc_xgb:.5f}')
    print(f'CatBoost: {auc_cb:.5f}')
    print(f'Blended:  {blend_auc:.5f}')