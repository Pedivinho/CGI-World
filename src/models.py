"""
Model training functions - Home Credit Default Risk (competitive sprint)
========================================================================

All models are trained with the same protocol:

- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- n_jobs <= 4, float32 features
- Early stopping whenever the backend supports it
- Returns (oof_preds, test_preds, auc)

Competitive lineup (3 GBDT families, multi-seed averaged):
- train_lgbm_competitive : LightGBM (lr 0.02, leaves 31, early stop 150)
- train_xgb_competitive  : XGBoost hist (lr 0.02, depth 5, early stop 150)
- train_catboost_competitive : CatBoost (depth 6, l2 10, early stop 150)
- SEEDS = [42, 2024, 777]; OOF and test predictions are averaged
  per family across the 3 seeds.

Legacy helpers kept for run_pipeline.py:
- train_lgbm, train_xgb, train_catboost, train_lr, train_et
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
SEEDS = [42, 2024, 777]
N_JOBS = 4
N_FOLDS = 5
EARLY_STOP = 150


def _make_cv(nfold):
    return StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED)


# ---------------------------------------------------------------------------
# Competitive GBDT trainers (multi-seed aware)
# ---------------------------------------------------------------------------

def _lgbm_params(seed):
    return {
        "n_estimators": 4000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 5.0,
        "reg_lambda": 10.0,
        "n_jobs": N_JOBS,
        "random_state": seed,
        "verbosity": -1,
        "force_col_wise": True,
    }


def _xgb_params(seed):
    return {
        "n_estimators": 4000,
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 5.0,
        "reg_lambda": 10.0,
        "n_jobs": N_JOBS,
        "random_state": seed,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": EARLY_STOP,
    }


def _cb_params(seed):
    return {
        "iterations": 4500,
        "learning_rate": 0.025,
        "depth": 6,
        "l2_leaf_reg": 10.0,
        "subsample": 0.8,
        "border_count": 254,
        "random_seed": seed,
        "thread_count": N_JOBS,
        "task_type": "CPU",
        "verbose": 0,
        "allow_writing_files": False,
    }


def train_lgbm_competitive(
    X, y, X_test, nfold=N_FOLDS, seeds=None, feature_names=None
):
    """
    LightGBM with multi-seed averaging.

    Each seed runs the full 5-fold OOF; OOF and test predictions are
    averaged across seeds per sample.

    Returns (oof, test, auc, fold_best_iters, feature_importance).
    """
    seeds = seeds if seeds is not None else SEEDS
    skf = _make_cv(nfold)
    splits = list(skf.split(X, y))

    oof_total = np.zeros(len(X), dtype=np.float32)
    test_total = np.zeros(len(X_test), dtype=np.float32)
    total_iters = []
    importances = np.zeros(X.shape[1], dtype=np.float64)

    for seed in seeds:
        oof_s = np.zeros(len(X), dtype=np.float32)
        test_s = np.zeros(len(X_test), dtype=np.float32)
        for fold, (tr, va) in enumerate(splits, 1):
            print(f"--- LightGBM seed={seed} fold {fold}/{nfold} ---", flush=True)
            model = LGBMClassifier(**_lgbm_params(seed))
            model.fit(
                X[tr], y[tr],
                eval_set=[(X[va], y[va])],
                eval_metric="auc",
                callbacks=[early_stopping(EARLY_STOP, verbose=False)],
            )
            oof_s[va] = model.predict_proba(X[va])[:, 1]
            test_s += model.predict_proba(X_test)[:, 1] / nfold
            total_iters.append(model.best_iteration_)
            importances += np.asarray(
                model.feature_importances_, dtype=np.float64
            )
            del model
            gc.collect()
        oof_total += oof_s / len(seeds)
        test_total += test_s / len(seeds)
        del oof_s, test_s
        gc.collect()

    auc = roc_auc_score(y, oof_total)
    print(f"LightGBM multi-seed OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof_total, test_total, auc, total_iters, importances


def train_xgb_competitive(
    X, y, X_test, nfold=N_FOLDS, seeds=None, feature_names=None
):
    """XGBoost (hist) with multi-seed averaging. Returns same as lgbm."""
    seeds = seeds if seeds is not None else SEEDS
    skf = _make_cv(nfold)
    splits = list(skf.split(X, y))

    oof_total = np.zeros(len(X), dtype=np.float32)
    test_total = np.zeros(len(X_test), dtype=np.float32)
    total_iters = []
    importances = np.zeros(X.shape[1], dtype=np.float64)

    for seed in seeds:
        oof_s = np.zeros(len(X), dtype=np.float32)
        test_s = np.zeros(len(X_test), dtype=np.float32)
        for fold, (tr, va) in enumerate(splits, 1):
            print(f"--- XGBoost seed={seed} fold {fold}/{nfold} ---", flush=True)
            model = XGBClassifier(**_xgb_params(seed))
            model.fit(
                X[tr], y[tr],
                eval_set=[(X[va], y[va])],
                verbose=False,
            )
            oof_s[va] = model.predict_proba(X[va])[:, 1]
            test_s += model.predict_proba(X_test)[:, 1] / nfold
            total_iters.append(int(model.best_iteration))
            importances += np.asarray(
                model.feature_importances_, dtype=np.float64
            )
            del model
            gc.collect()
        oof_total += oof_s / len(seeds)
        test_total += test_s / len(seeds)
        del oof_s, test_s
        gc.collect()

    auc = roc_auc_score(y, oof_total)
    print(f"XGBoost multi-seed OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof_total, test_total, auc, total_iters, importances


def train_catboost_competitive(
    X, y, X_test, nfold=N_FOLDS, seeds=None, feature_names=None
):
    """CatBoost with multi-seed averaging. Returns same as lgbm.

    X may be a DataFrame with categorical string columns (passed as
    cat_features) or a numpy matrix.
    """
    seeds = seeds if seeds is not None else SEEDS
    skf = _make_cv(nfold)
    splits = list(skf.split(X, y))

    cat_features = None
    if isinstance(X, pd.DataFrame):
        cat_features = [
            c for c in X.columns if X[c].dtype == object
        ] or None
    n_feat = X.shape[1]

    oof_total = np.zeros(len(X), dtype=np.float32)
    test_total = np.zeros(len(X_test), dtype=np.float32)
    total_iters = []
    importances = np.zeros(n_feat, dtype=np.float64)

    for seed in seeds:
        oof_s = np.zeros(len(X), dtype=np.float32)
        test_s = np.zeros(len(X_test), dtype=np.float32)
        for fold, (tr, va) in enumerate(splits, 1):
            print(f"--- CatBoost seed={seed} fold {fold}/{nfold} ---", flush=True)
            model = CatBoostClassifier(**_cb_params(seed))
            fit_kwargs = {
                "eval_set": (X[va], y[va]),
                "early_stopping_rounds": EARLY_STOP,
                "verbose": False,
            }
            if cat_features is not None:
                fit_kwargs["cat_features"] = cat_features
            model.fit(X[tr], y[tr], **fit_kwargs)
            oof_s[va] = model.predict_proba(X[va])[:, 1]
            test_s += model.predict_proba(X_test)[:, 1] / nfold
            total_iters.append(int(model.get_best_iteration()))
            try:
                importances += np.asarray(
                    model.get_feature_importance(), dtype=np.float64
                )
            except Exception:
                pass
            del model
            gc.collect()
        oof_total += oof_s / len(seeds)
        test_total += test_s / len(seeds)
        del oof_s, test_s
        gc.collect()

    auc = roc_auc_score(y, oof_total)
    print(f"CatBoost multi-seed OOF ROC-AUC: {auc:.5f}", flush=True)
    return oof_total, test_total, auc, total_iters, importances


# ---------------------------------------------------------------------------
# Legacy trainers (kept for run_pipeline.py compatibility)
# ---------------------------------------------------------------------------

def train_lgbm(X, y, X_test, nfold=N_FOLDS):
    """LightGBM baseline with early stopping on each fold's validation."""
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- LightGBM fold {fold}/{nfold} ---", flush=True)
        model = LGBMClassifier(**_lgbm_params(SEED))
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


def train_xgb(X, y, X_test, nfold=N_FOLDS):
    """XGBoost (hist) with early stopping."""
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- XGBoost fold {fold}/{nfold} ---", flush=True)
        model = XGBClassifier(**_xgb_params(SEED))
        model.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
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
    skf = _make_cv(nfold)
    oof = np.zeros(len(X), dtype=np.float32)
    test = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- CatBoost fold {fold}/{nfold} ---", flush=True)
        model = CatBoostClassifier(**_cb_params(SEED))
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
        Keys: lgbm, xgb, catboost, lr, et
    """
    oofs, tests, aucs = {}, {}, {}

    for name, fn in [
        ("lgbm", train_lgbm),
        ("xgb", train_xgb),
        ("catboost", train_catboost),
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
