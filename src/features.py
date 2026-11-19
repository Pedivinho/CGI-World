"""
Feature Engineering Pipeline for Home Credit Default Risk
==========================================================
Iterative feature engineering for Kaggle competition.
Reference: HomeCredit_columns_description.xlsx
"""
import pandas as pd
import numpy as np
import gc
import re


def sanitize_name(name):
    """Sanitize column name for LGBM compatibility.
    
    Special handling needed because:
    - LGBM cannot handle column names starting with digits
    - Special characters like spaces, brackets cause issues
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_]+', '_', name)
    sanitized = sanitized.strip('_')
    # Don't allow leading digits (JSON parsing issue)
    if sanitized and sanitized[0].isdigit():
        sanitized = 'col_' + sanitized
    return sanitized if sanitized else 'unnamed_col'


def build_features(train_df, test_df):
    """
    Feature engineering pipeline for Home Credit Default Risk.
    
    Features created (Iterations 0-2):
    - Cleaning: DAYS_EMPLOYED anomalies -> NaN + flag
    - Financial ratios: Credit/Income, Annuity/Income, etc.
    - EXT_SOURCE statistics: mean, std, min, max, prod, sum, var
    - Bureau request stats: sum, mean, std, max, median, count
    - Missing count and ratio
    - One-hot encoding for categorical variables
    - Age in years, employed in years
    
    Returns:
    --------
    train_proc : DataFrame
        Processed training data with target
    test_proc : DataFrame
        Processed test data without target
    """
    # Combine for processing
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # 1. Clean anomalies - DAYS_EMPLOYED == 365243 is a special value
    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})
    
    # Convert days to years
    df['DAYS_EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'] / 365
    df['DAYS_BIRTH_YEARS'] = df['DAYS_BIRTH'] / 365
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    # 2. Financial Ratios (from HomeCredit_columns_description)
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace(0, np.nan)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    
    # 3. EXT_SOURCE Statistics (bureau scores)
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
    
    # 4. Bureau Requests Statistics
    bureau_reqs = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
    df['BUREAU_REQS_SUM'] = df[bureau_reqs].sum(axis=1)
    df['BUREAU_REQS_MEAN'] = df[bureau_reqs].mean(axis=1)
    df['BUREAU_REQS_STD'] = df[bureau_reqs].std(axis=1)
    
    # Iteration 2: Enhanced bureau stats
    if len(bureau_reqs) > 0:
        df['BUREAU_REQS_MAX'] = df[bureau_reqs].max(axis=1)
        df['BUREAU_REQS_MEDI'] = df[bureau_reqs].median(axis=1)
        df['BUREAU_REQS_COUNT'] = len(bureau_reqs)
    
    # 5. Missing values
    df['NULL_COUNT'] = df.isnull().sum(axis=1)
    df['NULL_COUNT_RATIO'] = df['NULL_COUNT'] / df.shape[1]
    
    # 6. Age features
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['AGE_BIN'] = pd.cut(df['AGE_YEARS'], bins=[0, 25, 35, 45, 55, 100], 
                           labels=[0, 1, 2, 3, 4]).astype('int32')
    
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