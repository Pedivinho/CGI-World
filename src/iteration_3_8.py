#!/usr/bin/env python3
"""
Iterations 3-8 - Home Credit Default Risk Pipeline
===================================================

This script combines all feature engineering and trains all models with
hyperparameter tuning and ensemble methods.
"""

import gc
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Setup path
BASE_DIR = "D:/Usuario/orca/CGI-World"

# Import modules directly from worktree
def local_import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

features = local_import("features", "src/features.py")
build_features = features.build_features

models = local_import("models", "src/models.py")
train_lgbm_tuned = models.train_lgbm_tuned
train_catboost_tuned = models.train_catboost_tuned
train_lr = models.train_lr
train_et = models.train_et

ensemble = local_import("ensemble", "src/ensemble.py")
weighted_blend = ensemble.weighted_blend
rank_ensemble = ensemble.rank_ensemble
stacked_ensemble = ensemble.stacked_ensemble
simple_average = ensemble.simple_average

# Configuration
SEED = 42
N_JOBS = 4
N_FOLDS = 5


def main():
    print("=" * 60)
    print("HOME CREDIT DEFAULT RISK - Iterations 3-8 Pipeline")
    print("=" * 60)

    TRAIN_PATH = os.path.join(BASE_DIR, "Competition/input/train.csv")
    TEST_PATH = os.path.join(BASE_DIR, "Competition/input/test.csv")
    SUBMIT_PATH = os.path.join(BASE_DIR, "output/submission.csv")

    # Load data
    print("\n[1] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH).sort_values("SK_ID_CURR")
    test_df = pd.read_csv(TEST_PATH).sort_values("SK_ID_CURR")

    # Build features
    print("\n[2] Building features (all iterations)...")
    train_proc, test_proc = build_features(train_df, test_df)

    y = train_proc["TARGET"].values
    X = train_proc.drop(columns=["TARGET", "SK_ID_CURR"]).values
    X_test = test_proc.drop(columns=["SK_ID_CURR"]).values

    print(f"Features: {X.shape[1]}, Train: {X.shape[0]}, Test: {X_test.shape[0]}")

    del train_df, test_df, train_proc, test_proc
    gc.collect()

    # ========================================
    # MODEL TRAINING
    # ========================================
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    results = {}

    # LightGBM tuned
    print("\n[3] Training tuned LightGBM...")
    oof_lgbm, test_lgbm, auc_lgbm = train_lgbm_tuned(X, y, X_test)
    results["LightGBM"] = {"auc": auc_lgbm}

    # CatBoost tuned
    print("\n[4] Training tuned CatBoost...")
    oof_cb, test_cb, auc_cb = train_catboost_tuned(X, y, X_test)
    results["CatBoost"] = {"auc": auc_cb}

    # Logistic Regression
    print("\n[5] Training Logistic Regression...")
    oof_lr, test_lr, auc_lr = train_lr(X, y, X_test)
    results["LogisticRegression"] = {"auc": auc_lr}

    # ExtraTrees (diversity)
    print("\n[6] Training ExtraTrees...")
    oof_et, test_et, auc_et = train_et(X, y, X_test)
    results["ExtraTrees"] = {"auc": auc_et}

    # ========================================
    # ENSEMBLE
    # ========================================
    print("\n" + "=" * 60)
    print("ENSEMBLE")
    print("=" * 60)

    # Weighted blend (LGBM + CatBoost + LR + ET)
    print("\n[7] Optimizing weighted blend...")
    weights, blend_auc, blended_test = weighted_blend(
        [oof_lgbm, oof_cb, oof_lr, oof_et],
        [test_lgbm, test_cb, test_lr, test_et],
        y,
    )
    print(f"Weights: LGBM={weights[0]:.3f}, CB={weights[1]:.3f}, LR={weights[2]:.3f}, ET={weights[3]:.3f}")
    print(f"Blended AUC: {blend_auc:.5f}")
    results["Weighted Blend"] = {"auc": blend_auc, "weights": weights.tolist()}

    # Rank ensemble
    print("\n[8] Rank ensemble...")
    rank_auc, rank_test = rank_ensemble(
        [oof_lgbm, oof_cb, oof_lr, oof_et],
        [test_lgbm, test_cb, test_lr, test_et],
        y,
    )
    print(f"Rank AUC: {rank_auc:.5f}")
    results["Rank Ensemble"] = {"auc": rank_auc}

    # Stacking
    print("\n[9] Stacking ensemble...")
    stack_auc, stack_test, _ = stacked_ensemble(
        [oof_lgbm, oof_cb, oof_lr, oof_et],
        [test_lgbm, test_cb, test_lr, test_et],
        y,
    )
    print(f"Stacked AUC: {stack_auc:.5f}")
    results["Stacked"] = {"auc": stack_auc}

    # Simple average
    print("\n[10] Simple average...")
    avg_auc, avg_test = simple_average(
        [oof_lgbm, oof_cb, oof_lr, oof_et],
        [test_lgbm, test_cb, test_lr, test_et],
        y,
    )
    print(f"Simple Average AUC: {avg_auc:.5f}")
    results["Simple Average"] = {"auc": avg_auc}

    # Find best
    models_dict = {
        "LightGBM": auc_lgbm,
        "CatBoost": auc_cb,
        "LogisticRegression": auc_lr,
        "ExtraTrees": auc_et,
        "Weighted Blend": blend_auc,
        "Rank Ensemble": rank_auc,
        "Stacked": stack_auc,
        "Simple Average": avg_auc,
    }
    best_name = max(models_dict, key=models_dict.get)
    best_auc = models_dict[best_name]

    # Select best predictions
    if best_name == "LightGBM":
        best_test = test_lgbm
    elif best_name == "CatBoost":
        best_test = test_cb
    elif best_name == "LogisticRegression":
        best_test = test_lr
    elif best_name == "ExtraTrees":
        best_test = test_et
    elif best_name == "Weighted Blend":
        best_test = blended_test
    elif best_name == "Rank Ensemble":
        best_test = rank_test
    elif best_name == "Stacked":
        best_test = stack_test
    else:
        best_test = avg_test

    # ========================================
    # SUBMISSION
    # ========================================
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)

    sample = pd.read_csv(os.path.join(BASE_DIR, "Competition/input/sample_submission.csv"))
    submission = pd.DataFrame({"TARGET": best_test[: len(sample)].astype(np.float32)})
    submission.to_csv(SUBMIT_PATH, index=False)
    print(f"Saved to: {SUBMIT_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, auc in sorted(models_dict.items(), key=lambda x: -x[1]):
        marker = " BEST" if name == best_name else ""
        print(f"{name}: {auc:.5f}{marker}")

    print(f"\n=== BEST: {best_name} with AUC {best_auc:.5f} ===")

    # Save results
    results["best"] = best_name
    results["best_auc"] = best_auc
    with open(os.path.join(BASE_DIR, "reports/iter_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()