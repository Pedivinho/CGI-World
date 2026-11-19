"""
Model Training Functions for Home Credit Default Risk
=====================================================

Gradient Boosting models with cross-validation:
- LightGBM: Fast, memory-efficient
- XGBoost: Strong baseline
- CatBoost: Handles categorical well

Linear model:
- Logistic Regression with L1 penalty

All models use StratifiedKFold (5 folds, shuffle=True, random_state=42)
"""
import pandas as pd
import numpy as np
import gc
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_lgbm(X, y, X_test, params=None, nfold=5):
    """Train LightGBM with k-fold Stratified CV."""
    if params is None:
        params = {
            'n_estimators': 3000,
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
        print(f"--- LightGBM Fold {fold+1} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc'
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / nfold
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()
        
    auc = roc_auc_score(y, oof_preds)
    print(f"LightGBM OOF ROC-AUC: {auc:.5f}")
    
    return oof_preds, test_preds.astype(np.float32), auc


def train_xgb(X, y, X_test, nfold=5):
    """Train XGBoost with k-fold Stratified CV."""
    params = {
        'n_estimators': 3000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_weight': 5,
        'n_jobs': 4,
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'auc'
    }
        
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- XGBoost Fold {fold+1} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / nfold
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()
        
    auc = roc_auc_score(y, oof_preds)
    print(f"XGBoost OOF ROC-AUC: {auc:.5f}")
    
    return oof_preds, test_preds.astype(np.float32), auc


def train_catboost(X, y, X_test, nfold=5):
    """Train CatBoost with k-fold Stratified CV."""
    params = {
        'iterations': 3000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'subsample': 0.8,
        'random_seed': 42,
        'task_type': 'CPU',
        'thread_count': 4,
        'verbose': 0
    }
        
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    
    # Identify categorical columns
    cat_features = []
    for i in range(min(20, X.shape[1])):
        if X.iloc[:, i].dtype == 'object' or X.iloc[:, i].dtype.name == 'category':
            cat_features.append(i)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- CatBoost Fold {fold+1} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=0
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / nfold
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()
        
    auc = roc_auc_score(y, oof_preds)
    print(f"CatBoost OOF ROC-AUC: {auc:.5f}")
    
    return oof_preds, test_preds.astype(np.float32), auc


def train_lr(X, y, X_test, nfold=5):
    """Train Logistic Regression with k-fold Stratified CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_scaled), dtype=np.float32)
    test_preds = np.zeros(len(X_test_scaled), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"--- LR Fold {fold+1} ---")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LogisticRegression(
            max_iter=1000,
            C=0.1,
            solver='saga',
            penalty='l1',
            n_jobs=4,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test_scaled)[:, 1] / nfold
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()
        
    auc = roc_auc_score(y, oof_preds)
    print(f"LogisticRegression OOF ROC-AUC: {auc:.5f}")
    
    return oof_preds, test_preds.astype(np.float32), auc