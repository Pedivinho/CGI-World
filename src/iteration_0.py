"""
Iteration 0 - Anchor Model
Baseline LightGBM with basic financial ratios and cleaning.
"""
import pandas as pd
import numpy as np
import gc
import re


def sanitize_name(name):
    """Sanitize column name for LGBM compatibility."""
    sanitized = re.sub(r'[^a-zA-Z0-9_]+', '_', name)
    sanitized = sanitized.strip('_')
    if sanitized and sanitized[0].isdigit():
        sanitized = 'col_' + sanitized
    return sanitized if sanitized else 'unnamed_col'


def load_and_prepare():
    """Load data with proper paths and downcasting."""
    train_df = pd.read_csv('Competition/input/train.csv').sort_values('SK_ID_CURR')
    test_df = pd.read_csv('Competition/input/test.csv').sort_values('SK_ID_CURR')
    
    for df in [train_df, test_df]:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
    
    return train_df, test_df


def build_features(train_df, test_df):
    """Build features for Iteration 0."""
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # Clean anomalies
    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})
    
    # Financial ratios
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    
    # EXT_SOURCE stats
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)
    df['EXT_SOURCES_STD'] = df[ext_sources].std(axis=1)
    df['EXT_SOURCES_MIN'] = df[ext_sources].min(axis=1)
    df['EXT_SOURCES_MAX'] = df[ext_sources].max(axis=1)
    df['EXT_SOURCES_PROD'] = df[ext_sources].prod(axis=1)
    
    # Bureau requests
    bureau_reqs = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
    df['BUREAU_REQS_SUM'] = df[bureau_reqs].sum(axis=1)
    df['BUREAU_REQS_MEAN'] = df[bureau_reqs].mean(axis=1)
    
    # Missing count
    df['NULL_COUNT'] = df.isnull().sum(axis=1)
    
    # Get dummies
    cat_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 
                'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat_cols)
    
    # Separate
    train_proc = df[df['TARGET'].notnull()].copy()
    test_proc = df[df['TARGET'].isnull()].drop(columns=['TARGET']).copy()
    
    # Sanitize
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


def train_lgbm(X, y, X_test, nfold=5):
    """Train LightGBM with k-fold Stratified CV."""
    from lightgbm import LGBMClassifier, early_stopping
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    
    params = {
        'n_estimators': 5000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'subsample_freq': 1,
        'max_depth': 8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_weight': 2,
        'min_child_samples': 20,
        'verbosity': -1,
        'n_jobs': 4,
        'random_state': 42,
        'device': 'cpu'
    }
        
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(100), 'log_evaluation', 0]
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / nfold
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()
        
    auc = roc_auc_score(y, oof_preds)
    print(f"OOF ROC-AUC: {auc:.5f}")
    
    return oof_preds, test_preds.astype(np.float32), auc


if __name__ == "__main__":
    print("=" * 60)
    print("ITERATION 0 - ANCHOR MODEL")
    print("=" * 60)
    
    print("\n--- Loading Data ---")
    train_df, test_df = load_and_prepare()
    
    print("\n--- Building Features ---")
    train_proc, test_proc = build_features(train_df, test_df)
    
    y = train_proc['TARGET'].values
    X = train_proc.drop(columns=['TARGET', 'SK_ID_CURR']).values
    X_test = test_proc.drop(columns=['SK_ID_CURR']).values
    
    del train_df, test_df, train_proc, test_proc
    gc.collect()
    
    print("\n--- Training LightGBM ---")
    oof_preds, test_preds, auc = train_lgbm(X, pd.Series(y), X_test)
    
    # Update report
    with open("reports/modeling_report.md", "r") as f:
        report = f.read()
    report = report.replace("[TO_BE_FILLED]", f"{auc:.5f}")
    report = report.replace("Iteration 0 Completed.", "Iteration 0 Completed.")
    with open("reports/modeling_report.md", "w") as f:
        f.write(report)
    
    # Save submission
    submission = pd.read_csv("Competition/input/sample_submission.csv")
    submission['TARGET'] = test_preds
    submission.to_csv("output/submission.csv", index=False)
    
    print(f"\n=== FINAL OOF ROC-AUC: {auc:.5f} ===")
    print("STATUS: ITERATION 0 COMPLETE")