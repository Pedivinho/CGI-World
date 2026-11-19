"""
Model training functions - Home Credit Default Risk
===================================================

All models are trained with the same protocol:

- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- n_jobs <= 4, float32 features
- Early stopping whenever the backend supports it
- Returns (oof_preds, test_preds, auc)

Models
------
- train_lgbm         : LightGBM (fast, strong baseline)
- train_lgbm_tuned   : LightGBM tuned (lr 0.02, early stopping)
- train_xgb          : XGBoost (hist)
- train_catboost     : CatBoost baseline
- train_catboost_tuned: CatBoost tuned (lr 0.025, early stopping)
- train_lr           : Logistic Regression (NaN-safe imputer inside)
- train_et           : ExtraTrees (diversity member)
"""

import gc

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
N_JOBS = 4
N_FOLDS = 5


def _make_cv(nfold):
    return StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED)


def _to_frame(mat, cols, idx):
    """Wrap a (possibly numpy) matrix as a DataFrame with stable columns."""
    if isinstance(mat, pd.DataFrame):
        return mat.loc[idx].copy() if idx is not None else mat
    if cols is None:
        cols = [f"feat_{i}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat[idx] if idx is not None else mat, columns=cols)


def train_lgbm(X, y, X_test, nfold=N_FOLDS):
    """LightGBM baseline with early stopping on each fold's validation."""
    params = {
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbosity": -1,
        "force_col_wise": True,
    }
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- LightGBM fold {fold}/{nfold} ---", flush=True)
        model = LGBMClassifier(**params)
        model.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
            eval_metric="auc",
            callbacks=[early_stopping(200, verbose=False)],
        )
        oof[va] = model.predict_proba(X[va])[:, 1]
        test += model.predict_proba(X_test)[:, 1] / nfold
        del model
        gc.collect()

    auc = roc_auc_score(y, oof)
    print(f"LightGBM OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof, test, auc


def train_lgbm_tuned(X, y, X_test, nfold=N_FOLDS):
    """LightGBM tuned (lower LR, more estimators, early stopping)."""
    params = {
        "n_estimators": 8000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbosity": -1,
        "force_col_wise": True,
    }
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- LightGBM-tuned fold {fold}/{nfold} ---", flush=True)
        model = LGBMClassifier(**params)
        model.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
            eval_metric="auc",
            callbacks=[early_stopping(200, verbose=False)],
        )
        oof[va] = model.predict_proba(X[va])[:, 1]
        test += model.predict_proba(X_test)[:, 1] / nfold
        del model
        gc.collect()

    auc = roc_auc_score(y, oof)
    print(f"LightGBM-tuned OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof, test, auc


def train_xgb(X, y, X_test, nfold=N_FOLDS):
    """XGBoost (hist) with early stopping."""
    params = {
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "tree_method": "hist",
    }
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- XGBoost fold {fold}/{nfold} ---", flush=True)
        model = XGBClassifier(**params)
        model.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
            early_stopping_rounds=200,
            verbose=False,
        )
        oof[va] = model.predict_proba(X[va])[:, 1]
        test += model.predict_proba(X_test)[:, 1] / nfold
        del model
        gc.collect()

    auc = roc_auc_score(y, oof)
    print(f"XGBoost OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof, test, auc


def train_catboost(X, y, X_test, nfold=N_FOLDS):
    """CatBoost baseline with early stopping on validation AUC."""
    params = {
        "iterations": 6000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "subsample": 0.8,
        "random_seed": SEED,
        "thread_count": N_JOBS,
        "task_type": "CPU",
        "verbose": 0,
        "early_stopping_rounds": 200,
        "allow_writing_files": False,
    }
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- CatBoost fold {fold}/{nfold} ---", flush=True)
        model = CatBoostClassifier(**params)
        model.fit(
            X[tr], y[tr],
            eval_set=(X[va], y[va]),
            early_stopping_rounds=200,
            verbose=False,
        )
        oof[va] = model.predict_proba(X[va])[:, 1]
        test += model.predict_proba(X_test)[:, 1] / nfold
        del model
        gc.collect()

    auc = roc_auc_score(y, oof)
    print(f"CatBoost OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof, test, auc


def train_catboost_tuned(X, y, X_test, nfold=N_FOLDS):
    """CatBoost tuned (lower LR, depth 7, early stopping)."""
    params = {
        "iterations": 8000,
        "learning_rate": 0.025,
        "depth": 7,
        "l2_leaf_reg": 5,
        "subsample": 0.85,
        "rsm": 0.85,
        "random_seed": SEED,
        "thread_count": N_JOBS,
        "task_type": "CPU",
        "verbose": 0,
        "early_stopping_rounds": 250,
        "allow_writing_files": False,
    }
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- CatBoost-tuned fold {fold}/{nfold} ---", flush=True)
        model = CatBoostClassifier(**params)
        model.fit(
            X[tr], y[tr],
            eval_set=(X[va], y[va]),
            early_stopping_rounds=250,
            verbose=False,
        )
        oof[va] = model.predict_proba(X[va])[:, 1]
        test += model.predict_proba(X_test)[:, 1] / nfold
        del model
        gc.collect()

    auc = roc_auc_score(y, oof)
    print(f"CatBoost-tuned OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof, test, auc


def train_lr(X, y, X_test, nfold=N_FOLDS):
    """Logistic Regression (lbfgs) with median imputation and scaling."""
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- LogisticRegression fold {fold}/{nfold} ---", flush=True)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X[tr]))
        X_val = scaler.transform(imputer.transform(X[va]))
        X_test_imp = scaler.transform(imputer.transform(X_test))

        model = LogisticRegression(
            C=0.3,
            solver="lbfgs",
            max_iter=2000,
            random_state=SEED,
        )
        model.fit(X_train, y[tr])
        oof[va] = model.predict_proba(X_val)[:, 1]
        test += model.predict_proba(X_test_imp)[:, 1] / nfold
        del imputer, scaler, model, X_train, X_val, X_test_imp
        gc.collect()

    auc = roc_auc_score(y, oof)
    print(f"LogisticRegression OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof, test, auc


def train_et(X, y, X_test, nfold=N_FOLDS):
    """ExtraTrees as a diversity member of the ensemble."""
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- ExtraTrees fold {fold}/{nfold} ---", flush=True)
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_features=0.5,
            min_samples_leaf=10,
            n_jobs=N_JOBS,
            random_state=SEED,
        )
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        test += model.predict_proba(X_test)[:, 1] / nfold
        del model
        gc.collect()

    auc = roc_auc_score(y, oof)
    print(f"ExtraTrees OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof, test, auc


def train_all(X, y, X_test, nfold=N_FOLDS):
    """
    Train the full model lineup and return dicts of predictions.

    Returns
    -------
    (oofs, tests, aucs) : dict[str, np.ndarray] x3
        Keys: lgbm, lgbm_tuned, xgb, catboost, catboost_tuned, lr, et
    """
    oofs, tests, aucs = {}, {}, {}

    for name, fn in [
        ("lgbm", train_lgbm),
        ("lgbm_tuned", train_lgbm_tuned),
        ("xgb", train_xgb),
        ("catboost", train_catboost),
        ("catboost_tuned", train_catboost_tuned),
        ("lr", train_lr),
        ("et", train_et),
    ]:
        print(f"\n[{name}] starting...", flush=True)
        oofs[name], tests[name], aucs[name] = fn(X, y, X_test, nfold=nfold)

    return oofs, tests, aucs


def simple_average(oof_preds, test_preds, y):
    """Simple average of model predictions (helper function for pipelines)."""
    blended_oof = np.mean(oof_preds, axis=0)
    blended_test = np.mean(test_preds, axis=0)
    auc = roc_auc_score(y, blended_oof)
    return auc, blended_test
