"""
Ensemble methods for Home Credit Default Risk competition.
==========================================================

Methods implemented:
1. Weighted blend with optimization (scipy SLSQP)
2. Rank averaging
3. Simple average
4. Stacking with Logistic Regression meta-model
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import gc
import pandas as pd


def weighted_blend(oof_preds, test_preds, y):
    """
    Optimize weights for weighted ensemble blend.
    
    Uses SLSQP optimization with constraints:
    - Sum of weights = 1
    - Each weight in [0, 1]
    
    Parameters:
    -----------
    oof_preds : list of arrays
        Out-of-fold predictions from each model
    test_preds : list of arrays
        Test set predictions from each model
    y : array
        Ground truth labels
    
    Returns:
    --------
    best_weights : np.array
        Optimized weights for each model
    best_auc : float
        Best AUC score with optimized weights
    blended_test : np.array
        Blended test predictions
    """
    n_models = len(oof_preds)
    
    def objective(w):
        """Negate AUC for minimization."""
        blended = np.sum(w * np.column_stack(oof_preds), axis=1)
        return -roc_auc_score(y, blended)
    
    # Start with equal weights
    start_w = np.ones(n_models) / n_models
    
    # Optimize with bounds [0, 1] for each weight
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Constraint: sum of weights = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    result = minimize(objective, start_w, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = result.x
    best_auc = -result.fun
    
    # Create blended test predictions
    blended_test = np.sum(best_weights * np.column_stack(test_preds), axis=1)
    
    return best_weights, best_auc, blended_test


def rank_ensemble(oof_preds, test_preds, y):
    """
    Rank-based ensemble: average ranks instead of predictions.
    
    Rank ensembling often performs better because it's more robust
    to model overconfidence.
    
    Parameters:
    -----------
    oof_preds : list of arrays
        Out-of-fold predictions from each model
    test_preds : list of arrays
        Test set predictions from each model
    y : array
        Ground truth labels
    
    Returns:
    --------
    auc : float
        AUC score with rank ensemble
    blended_test : np.array
        Rank-blended test predictions
    """
    n_models = len(oof_preds)
    
    # Get ranks for each model (higher rank = better prediction)
    oof_ranks = []
    test_ranks = []
    
    for i in range(n_models):
        # Rank OOF predictions (1 = lowest, n = highest)
        oof_rank = np.argsort(np.argsort(oof_preds[i])).astype(np.float32)
        oof_ranks.append(oof_rank / len(oof_rank))  # Normalize to [0, 1]
        
        # Rank test predictions
        test_rank = np.argsort(np.argsort(test_preds[i])).astype(np.float32)
        test_ranks.append(test_rank / len(test_rank))
    
    # Average ranks
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
    that learns optimal combination.
    
    Parameters:
    -----------
    oof_preds : list of arrays
        Out-of-fold predictions from each model
    test_preds : list of arrays
        Test set predictions from each model
    y : array
        Ground truth labels
    
    Returns:
    --------
    auc : float
        AUC score with stacked ensemble
    blended_test : np.array
        Stacked test predictions
    meta_model : LogisticRegression
        Trained meta-model
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Create meta-features
    X_meta = np.column_stack(oof_preds)
    
    # Scale
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)
    
    # Train meta-model
    meta_model = LogisticRegression(max_iter=1000, C=0.1, solver='saga', 
                                     penalty='l1', n_jobs=4, random_state=42)
    meta_model.fit(X_meta_scaled, y)
    
    # Get meta predictions
    blended_oof = meta_model.predict_proba(X_meta_scaled)[:, 1]
    
    # Scale test predictions
    X_test_meta = np.column_stack(test_preds)
    X_test_meta_scaled = scaler.transform(X_test_meta)
    
    blended_test = meta_model.predict_proba(X_test_meta_scaled)[:, 1]
    
    auc = roc_auc_score(y, blended_oof)
    
    return auc, blended_test, meta_model


def create_submission(test_preds, filename='submission.csv'):
    """Create submission file in the expected format."""
    submission = pd.read_csv('Competition/input/sample_submission.csv')
    submission['TARGET'] = test_preds[:len(submission)]
    submission.to_csv(f'output/{filename}', index=False)
    return submission