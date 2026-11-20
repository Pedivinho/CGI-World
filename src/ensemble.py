"""
Ensemble methods for Home Credit Default Risk competition.
==========================================================

Competitive sprint methods (GBDT-only):
1. Probabilistic weighted blend - scipy Nelder-Mead over 3 weights
   (sum=1, w>=0) on calibrated OOF probabilities
2. Rank-averaging blend - scipy Nelder-Mead over 3 weights on normalized
   ranks (robust to model overconfidence)
3. Simple average (equal weights) - sanity fallback

The pipeline keeps whichever of (1) or (2) maximizes 5-fold OOF ROC-AUC.
Legacy methods (rank_ensemble, stacked_ensemble) are kept for
run_pipeline.py compatibility.

Public entry point:
    run_competitive_pipeline(train_path, test_path, output_path)
"""

import gc
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

import features as _features_mod
import models as _models_mod


def _nelder_mead_blend(oof_stack, test_stack, y, method="prob"):
    """
    Optimize 3 non-negative weights (sum constrained by normalization)
    with Nelder-Mead. The objective normalizes weights internally so the
    unconstrained optimizer explores the simplex.

    method="prob"  : weights on calibrated probabilities
    method="rank"  : weights on normalized ranks
    """
    n = oof_stack.shape[1]

    def transform(stack):
        if method == "rank":
            out = np.empty_like(stack, dtype=np.float64)
            for j in range(stack.shape[1]):
                out[:, j] = np.argsort(np.argsort(stack[:, j])) / stack.shape[0]
            return out
        return stack

    oof_t = transform(oof_stack).astype(np.float64)
    test_t = transform(test_stack).astype(np.float64)

    def objective(w):
        s = np.abs(w).sum()
        if s <= 0:
            return 1.0
        wn = np.abs(w) / s
        blended = oof_t @ wn
        return -roc_auc_score(y, blended)

    start = np.ones(n) / n
    result = minimize(
        objective, start, method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-9},
    )
    w = np.abs(result.x)
    w = w / w.sum()
    blended_oof = oof_t @ w
    blended_test = test_t @ w
    auc = roc_auc_score(y, blended_oof)
    return w, auc, blended_test


def gbdt_ensemble(oof_preds, test_preds, y, names=("lgbm", "catboost", "xgb")):
    """
    GBDT-only optimal ensemble.

    Tries (a) probabilistic Nelder-Mead blend and (b) rank Nelder-Mead
    blend, plus (c) simple average as a sanity baseline. Returns the
    method with the best OOF ROC-AUC.

    Parameters
    ----------
    oof_preds : list of np.ndarray (3)
    test_preds : list of np.ndarray (3)
    y : np.ndarray

    Returns
    -------
    (best_name, best_auc, weights, best_test, all_results)
    all_results: dict name -> (auc, weights)
    """
    oof_stack = np.column_stack(oof_preds)
    test_stack = np.column_stack(test_preds)

    w_prob, auc_prob, blend_prob = _nelder_mead_blend(
        oof_stack, test_stack, y, method="prob"
    )
    w_rank, auc_rank, blend_rank = _nelder_mead_blend(
        oof_stack, test_stack, y, method="rank"
    )
    simple_test = test_stack.mean(axis=1)
    auc_simple = roc_auc_score(y, oof_stack.mean(axis=1))

    results = {
        "prob_blend": (auc_prob, w_prob),
        "rank_blend": (auc_rank, w_rank),
        "simple_avg": (auc_simple, np.ones(3) / 3),
    }

    best = max(results, key=lambda k: results[k][0])
    if best == "prob_blend":
        return best, auc_prob, w_prob, blend_prob, results
    if best == "rank_blend":
        return best, auc_rank, w_rank, blend_rank, results
    return best, auc_simple, np.ones(3) / 3, simple_test, results


def rank_ensemble(oof_preds, test_preds, y):
    """Legacy equal-weight rank ensemble (kept for run_pipeline.py)."""
    oof_ranks, test_ranks = [], []
    for arr in oof_preds:
        oof_ranks.append(np.argsort(np.argsort(arr)).astype(np.float32) / len(arr))
    for arr in test_preds:
        test_ranks.append(np.argsort(np.argsort(arr)).astype(np.float32) / len(arr))
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


def weighted_blend(oof_preds, test_preds, y):
    """Legacy SLSQP weighted blend (kept for run_pipeline.py)."""
    n_models = len(oof_preds)

    def objective(w):
        blended = np.sum(w * np.column_stack(oof_preds), axis=1)
        return -roc_auc_score(y, blended)

    start_w = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    result = minimize(
        objective, start_w, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 100},
    )
    best_weights = result.x
    best_auc = -result.fun
    blended_test = np.sum(best_weights * np.column_stack(test_preds), axis=1)
    return best_weights, best_auc, blended_test


def stacked_ensemble(oof_preds, test_preds, y):
    """Legacy logistic-regression stacking meta-model (kept for run_pipeline.py)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_meta = np.column_stack(oof_preds)
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    meta_model = LogisticRegression(
        max_iter=1000, C=0.1, solver="lbfgs", random_state=42
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
        Output file name
    row_ids : pd.Series or None
        SK_ID_CURR values from test set; if None, uses sample submission index

    Returns
    -------
    pd.DataFrame
    """
    sample_path = "D:/Usuario/orca/CGI-World/Competition/input/sample_submission.csv"
    if not os.path.exists(sample_path):
        sample_path = "Competition/input/sample_submission.csv"
    submission = pd.read_csv(sample_path)
    submission["TARGET"] = test_preds[: len(submission)].astype("float32")
    out_dir = "D:/Usuario/orca/CGI-World/output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    submission.to_csv(out_path, index=False)
    return submission


# ---------------------------------------------------------------------------
# Competitive pipeline entry point
# ---------------------------------------------------------------------------

def run_competitive_pipeline(
    train_path=None,
    test_path=None,
    output_path=None,
    seeds=None,
    prune=True,
):
    """
    Full competitive sprint pipeline:

    1. Load + sort by SK_ID_CURR
    2. build_features (leak-free)
    3. OOF smoothed target encoding (m=20) computed strictly inside folds
    4. Train 3 GBDT families x multi-seed, average per family
    5. Optional zero-importance pruning (only if it improves CV AUC)
    6. GBDT-only ensemble: Nelder-Mead prob blend vs rank blend
    7. Write submission

    Returns dict with aucs, best ensemble, weights and feature stats.
    """
    base_dir = "D:/Usuario/orca/CGI-World"
    if train_path is None:
        train_path = os.path.join(base_dir, "Competition/input/train.csv")
    if test_path is None:
        test_path = os.path.join(base_dir, "Competition/input/test.csv")
    if output_path is None:
        output_path = os.path.join(base_dir, "output/submission.csv")
    seeds = seeds if seeds is not None else _models_mod.SEEDS
    N_JOBS_OK = True  # noqa: F841 (n_jobs enforced inside model params)

    print("=" * 60, flush=True)
    print("COMPETITIVE PIPELINE - Home Credit Default Risk", flush=True)
    print("=" * 60, flush=True)

    print("[1] Loading data...", flush=True)
    train_df = pd.read_csv(train_path).sort_values("SK_ID_CURR")
    test_df = pd.read_csv(test_path).sort_values("SK_ID_CURR")
    for df in (train_df, test_df):
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype("int32")

    print("[2] Building features...", flush=True)
    train_proc, test_proc = _features_mod.build_features(train_df, test_df)
    n_feat_raw = train_proc.shape[1] - 2  # minus TARGET and SK_ID_CURR
    print(f"Features: {n_feat_raw}, Train: {len(train_proc)}, Test: {len(test_proc)}", flush=True)

    y = train_proc["TARGET"].values
    X = train_proc.drop(columns=["TARGET", "SK_ID_CURR"]).astype(np.float32)
    X_test = test_proc.drop(columns=["SK_ID_CURR"]).astype(np.float32)
    feat_cols = list(X.columns)
    test_ids = test_proc["SK_ID_CURR"].values.astype("int32")
    del train_proc, test_proc
    gc.collect()

    # --- 3. Out-of-fold smoothed target encoding (m=20, strict per fold) ---
    print("[3] Out-of-fold smoothed target encoding (m=20)...", flush=True)
    train_cats, test_cats, cat_cols = _features_mod.get_target_encoding_cats(
        train_df,
        test_df,
        cols=_features_mod.TARGET_ENCODED_COLS,
    )
    # align with X row order
    assert len(train_cats) == len(y) and len(test_cats) == len(X_test)

    skf = _models_mod._make_cv(_models_mod.N_FOLDS)
    oof_te = np.zeros((len(X), len(cat_cols)), dtype=np.float32)
    test_te = np.zeros((len(X_test), len(cat_cols)), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"--- TE fold {fold}/{_models_mod.N_FOLDS} ---", flush=True)
        for j, col in enumerate(cat_cols):
            fit_vals = train_cats[col].to_numpy()[tr]
            val_vals = train_cats[col].to_numpy()[va]
            te_vals = _features_mod.transform_target_encoding(
                *(_features_mod.fit_smoothed_target_encoding(y[tr], fit_vals, smoothing=20)),
                val_vals,
                smoothing=20,
            )
            oof_te[va, j] = te_vals
            te_test = _features_mod.transform_target_encoding(
                *(_features_mod.fit_smoothed_target_encoding(y[tr], fit_vals, smoothing=20)),
                test_cats[col].to_numpy(),
                smoothing=20,
            )
            test_te[:, j] += te_test / _models_mod.N_FOLDS
    del train_cats, test_cats
    gc.collect()

    te_names = [f"TE_{c}" for c in cat_cols]
    X = np.column_stack([X, oof_te]).astype(np.float32)
    X_test = np.column_stack([X_test, test_te]).astype(np.float32)
    feat_cols = feat_cols + te_names
    print(f"Features after TE: {X.shape[1]}", flush=True)

    # --- 4. Train 3 GBDT families, multi-seed -------------------------------
    print("\n[4] Training competitive GBDTs (multi-seed)...", flush=True)
    oof_lgbm, test_lgbm, auc_lgbm, iters_lgbm, imp_lgbm = (
        _models_mod.train_lgbm_competitive(X, y, X_test, seeds=seeds)
    )
    oof_xgb, test_xgb, auc_xgb, iters_xgb, imp_xgb = (
        _models_mod.train_xgb_competitive(X, y, X_test, seeds=seeds)
    )
    oof_cb, test_cb, auc_cb, iters_cb, imp_cb = (
        _models_mod.train_catboost_competitive(X, y, X_test, seeds=seeds)
    )

    print("\n[5] Single-seed (42) baseline family AUCs:", flush=True)
    print(f"  LightGBM: {auc_lgbm:.5f}", flush=True)
    print(f"  XGBoost:  {auc_xgb:.5f}", flush=True)
    print(f"  CatBoost: {auc_cb:.5f}", flush=True)

    # --- 5. Optional zero-importance pruning ---------------------------------
    pruned = False
    if prune:
        print("\n[5b] Zero-importance pruning check...", flush=True)
        total_imp = imp_lgbm + imp_xgb + imp_cb
        keep = total_imp > 0
        n_dropped = int((~keep).sum())
        if 0 < n_dropped < X.shape[1]:
            X_p, X_test_p = X[:, keep], X_test[:, keep]
            kept_names = [c for c, k in zip(feat_cols, keep) if k]
            print(f"  Dropping {n_dropped} zero-importance features -> {X_p.shape[1]}", flush=True)
            # quick single-seed validation on pruned set
            oof_l_p, _, auc_l_p, _, _ = _models_mod.train_lgbm_competitive(
                X_p, y, X_test_p, seeds=[42]
            )
            oof_x_p, _, auc_x_p, _, _ = _models_mod.train_xgb_competitive(
                X_p, y, X_test_p, seeds=[42]
            )
            oof_c_p, _, auc_c_p, _, _ = _models_mod.train_catboost_competitive(
                X_p, y, X_test_p, seeds=[42]
            )
            best_before = max(auc_lgbm, auc_xgb, auc_cb)
            best_after = max(auc_l_p, auc_x_p, auc_c_p)
            if best_after >= best_before:
                X, X_test = X_p, X_test_p
                feat_cols = kept_names
                pruned = True
                print(f"  Pruned set OK (best single-seed {best_after:.5f} >= {best_before:.5f}), keeping pruned.", flush=True)
            else:
                print(f"  Pruned set worse ({best_after:.5f} < {best_before:.5f}), keeping full set.", flush=True)
                del X_p, X_test_p
            del oof_l_p, auc_l_p, oof_x_p, auc_x_p, oof_c_p, auc_c_p
            gc.collect()
        else:
            print("  No zero-importance features to drop.", flush=True)

    # --- 6. GBDT-only ensemble ----------------------------------------------
    print("\n[6] GBDT-only ensemble optimization...", flush=True)
    best_name, best_auc, weights, best_test, all_results = gbdt_ensemble(
        [oof_lgbm, oof_cb, oof_xgb],
        [test_lgbm, test_cb, test_xgb],
        y,
    )
    print(f"  Prob blend : {all_results['prob_blend'][0]:.5f}  w={np.round(all_results['prob_blend'][1], 4)}", flush=True)
    print(f"  Rank blend : {all_results['rank_blend'][0]:.5f}  w={np.round(all_results['rank_blend'][1], 4)}", flush=True)
    print(f"  Simple avg : {all_results['simple_avg'][0]:.5f}", flush=True)
    print(f"  BEST: {best_name} with OOF AUC {best_auc:.5f}", flush=True)

    # --- 7. Submission --------------------------------------------------------
    print("\n[7] Writing submission...", flush=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_path = os.path.join(base_dir, "Competition/input/sample_submission.csv")
    if os.path.exists(sample_path):
        submission = pd.read_csv(sample_path)
        submission["TARGET"] = best_test[: len(X_test)].astype(np.float32)
    else:
        submission = pd.DataFrame(
            {
                "SK_ID_CURR": test_ids,
                "TARGET": best_test[: len(X_test)].astype(np.float32),
            }
        )
    submission.to_csv(output_path, index=False)
    print(f"Saved to: {output_path} ({len(submission)} rows)", flush=True)

    summary = {
        "n_features": int(X.shape[1]),
        "pruned": pruned,
        "lgbm": float(auc_lgbm),
        "xgb": float(auc_xgb),
        "catboost": float(auc_cb),
        "ensemble_method": best_name,
        "ensemble_weights": {
            "lgbm": float(weights[0]),
            "catboost": float(weights[1]),
            "xgb": float(weights[2]),
        },
        "ensemble_auc": float(best_auc),
        "all_ensembles": {k: float(v[0]) for k, v in all_results.items()},
        "seeds": list(seeds),
        "lgbm_iters": iters_lgbm,
        "xgb_iters": iters_xgb,
        "catboost_iters": iters_cb,
    }
    print("\n" + "=" * 60, flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print("=" * 60, flush=True)
    return summary
