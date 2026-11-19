"""
Ensemble methods for Home Credit Default Risk competition.
==========================================================

Methods implemented:
1. Weighted blend with optimization (scipy SLSQP)
2. Rank averaging
3. Simple average
4. Stacking with Logistic Regression meta-model

Iterations:
- Iter 7: Weighted blend optimization
- Iter 8: Rank averaging + Stacking
"""

from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


def weighted_blend(oof_preds, test_preds, y):
    """
    Optimize weights for weighted ensemble blend.

    Uses SLSQP optimization with constraints:
    - Sum of weights = 1
    - Each weight in [0, 1]

    Parameters
    ----------
    oof_preds : list of np.ndarray
        Out-of-fold predictions from each model
    test_preds : list of np.ndarray
        Test set predictions from each model
    y : np.ndarray
        Ground truth labels

    Returns
    -------
    best_weights : np.ndarray
        Optimized weights for each model
    best_auc : float
        Best AUC score with optimized weights
    blended_test : np.ndarray
        Blended test predictions
    """
    n_models = len(oof_preds)

    def objective(w):
        blended = np.sum(w * np.column_stack(oof_preds), axis=1)
        return -roc_auc_score(y, blended)

    start_w = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(objective, start_w, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 100})

    best_weights = result.x
    best_auc = -result.fun
    blended_test = np.sum(best_weights * np.column_stack(test_preds), axis=1)

    return best_weights, best_auc, blended_test


def rank_ensemble(oof_preds, test_preds, y):
    """
    Rank-based ensemble: average ranks instead of predictions.

    Rank ensembling is robust to model overconfidence.
    Higher rank means higher prediction value.

    Parameters
    ----------
    oof_preds : list of np.ndarray
        Out-of-fold predictions
    test_preds : list of np.ndarray
        Test set predictions
    y : np.ndarray
        Ground truth labels

    Returns
    -------
    auc : float
        AUC with rank ensemble
    blended_test : np.ndarray
        Rank-blended test predictions
    """
    n_models = len(oof_preds)
    oof_ranks, test_ranks = [], []

    for arr in oof_preds:
        oof_rank = np.argsort(np.argsort(arr)).astype(np.float32)
        oof_ranks.append(oof_rank / len(oof_rank))

    for arr in test_preds:
        rank = np.argsort(np.argsort(arr)).astype(np.float32)
        test_ranks.append(rank / len(rank))

    blended_oof = np.mean(oof_ranks, axis=0)
    blended_test = np.mean(test_ranks, axis=0)
    auc = roc_auc_score(y, blended_oof)

    return auc, blended_test


def simple_average(oof_preds, test_preds, y):
    """Simple average of model predictions."""
    blended_oof = np.mean(oof_preds, axis=0)
    blended_test = np.mean(test_preds, axis=0)
    auc = roc_auc_score(y, blended_oof)
    return auc, blended_test


def stacked_ensemble(oof_preds, test_preds, y):
    """
    Stacking with Logistic Regression meta-model.

    Uses OOF predictions as features for a logistic regression
    that learns the optimal combination.

    Parameters
    ----------
    oof_preds : list of np.ndarray
        Out-of-fold predictions
    test_preds : list of np.ndarray
        Test set predictions
    y : np.ndarray
        Ground truth labels

    Returns
    -------
    auc : float
        AUC with stacked ensemble
    blended_test : np.ndarray
        Stacked test predictions
    meta_model : LogisticRegression
        Trained meta-model
    """
    X_meta = np.column_stack(oof_preds)
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    meta_model = LogisticRegression(
        max_iter=1000, C=0.1, solver="lbfgs",
        random_state=42
    )
    meta_model.fit(X_meta_scaled, y)

    blended_oof = meta_model.predict_proba(X_meta_scaled)[:, 1]

    X_test_meta = np.column_stack(test_preds)
    X_test_meta_scaled = scaler.transform(X_test_meta)
    blended_test = meta_model.predict_proba(X_test_meta_scaled)[:, 1]

    auc = roc_auc_score(y, blended_oof)
    return auc, blended_test, meta_model


def create_submission(test_preds, filename="submission.csv", row_ids=None):
    """
    Create submission file in the expected format.

    Parameters
    ----------
    test_preds : np.ndarray
        Predictions for all rows in test set (must match sample_submission length)
    filename : str
        Output filename
    row_ids : pd.Series or None
        SK_ID_CURR values from test set; if None, uses sample submission index

    Returns
    -------
    pd.DataFrame
        The submission frame
    """
    submission = pd.read_csv("Competition/input/sample_submission.csv")
    submission["TARGET"] = test_preds[: len(submission)].astype("float32")
    out_path = f"output/{filename}"
    submission.to_csv(out_path, index=False)
    return submission