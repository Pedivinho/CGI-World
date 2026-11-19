"""
Feature engineering pipeline - Home Credit Default Risk
=======================================================

Leak-free features shared by train and test (built on the concatenated
frame so statistics are computed identically for both):

- Cleaning: DAYS_EMPLOYED == 365243 -> NaN + anomaly flag
- Financial ratios (credit, annuity, goods price vs income)
- EXT_SOURCE stats: mean/std/min/max/median, sum, var, triple product
- EXT_SOURCE pairwise differences and ratios (guarding against 0)
- EXT_SOURCE squared terms
- Age features: years, bins, employment/income interactions
- Bureau request stats across HOUR/MON/QRT/YEAR
- Null count per row
- Group aggregations (ORGANIZATION_TYPE / NAME_INCOME_TYPE /
  NAME_EDUCATION_TYPE x AMT_CREDIT / AMT_INCOME_TOTAL): mean, std, min, max
- One-hot encoding for the 8 categorical columns

Target encoding (iteration 4) is applied OUT-OF-FOLD inside the training
folds only (see TargetEncoder below) so that neither OOF AUC nor test
predictions leak the target.

All numeric outputs are downcast to float32 / int32 for memory efficiency.
"""

import gc
import re

import numpy as np
import pandas as pd

# Original categorical columns (kept as strings for OOF target encoding)
CAT_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
GROUP_COLS = ["ORGANIZATION_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]
NUM_AGG_COLS = ["AMT_CREDIT", "AMT_INCOME_TOTAL"]
TARGET_ENCODED_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

_MISSING = "__NA__"


def sanitize_name(name):
    """Make a column name safe (no special chars, no leading digit)."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name)).strip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "col_" + sanitized
    return sanitized if sanitized else "unnamed_col"


def _fill_cat(df, cols):
    """Replace NaNs in categorical columns with an explicit missing token."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype("object").fillna(_MISSING)
    return out


def _safe_div(num, den, eps=1e-6):
    return num / (den.abs() + eps)


def build_features(train_df, test_df):
    """
    Build the leak-free feature matrix shared by train and test.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Raw frames (already sorted by SK_ID_CURR upstream).

    Returns
    -------
    train_proc, test_proc : pd.DataFrame
        Aligned feature frames. train_proc keeps the TARGET column;
        test_proc has no TARGET. Column order is identical in both.
    """
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # --- 0. Cleaning -------------------------------------------------------
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("int32")
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    income = df["AMT_INCOME_TOTAL"]
    credit = df["AMT_CREDIT"]
    annuity = df["AMT_ANNUITY"]
    goods = df["AMT_GOODS_PRICE"]

    # --- 1. Financial ratios -----------------------------------------------
    df["CREDIT_INCOME_PERCENT"] = _safe_div(credit, income)
    df["ANNUITY_INCOME_PERCENT"] = _safe_div(annuity, income)
    df["CREDIT_ANNUITY_RATIO"] = _safe_div(credit, annuity)
    df["CREDIT_GOODS_RATIO"] = _safe_div(credit, goods)
    df["INCOME_PER_FAMILY_MEMBER"] = _safe_div(income, df["CNT_FAM_MEMBERS"].replace(0, 1))
    df["CREDIT_PER_FAMILY_MEMBER"] = _safe_div(credit, df["CNT_FAM_MEMBERS"].replace(0, 1))
    df["INCOME_PER_CHILD"] = _safe_div(income, df["CNT_CHILDREN"].replace(0, 1))

    # --- 2. EXT_SOURCE stats (iterations 1-3) ------------------------------
    ext = df[EXT_SOURCES]
    df["EXT_SOURCES_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCES_STD"] = ext.std(axis=1)
    df["EXT_SOURCES_MIN"] = ext.min(axis=1)
    df["EXT_SOURCES_MAX"] = ext.max(axis=1)
    df["EXT_SOURCES_MEDI"] = ext.median(axis=1)
    df["EXT_SOURCES_RANGE"] = ext.max(axis=1) - ext.min(axis=1)
    df["EXT_SOURCES_SUM"] = ext.sum(axis=1, min_count=1)
    df["EXT_SOURCES_VAR"] = ext.var(axis=1)
    df["EXT_SOURCE_1_2_3_PROD"] = (
        df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    )
    df["EXT_SOURCE_1_MINUS_2"] = df["EXT_SOURCE_1"] - df["EXT_SOURCE_2"]
    df["EXT_SOURCE_2_MINUS_3"] = df["EXT_SOURCE_2"] - df["EXT_SOURCE_3"]
    df["EXT_SOURCE_1_MINUS_3"] = df["EXT_SOURCE_1"] - df["EXT_SOURCE_3"]
    df["EXT_SOURCE_1_OVER_2"] = _safe_div(df["EXT_SOURCE_1"], df["EXT_SOURCE_2"])
    df["EXT_SOURCE_2_OVER_3"] = _safe_div(df["EXT_SOURCE_2"], df["EXT_SOURCE_3"])
    df["EXT_SOURCE_1_OVER_3"] = _safe_div(df["EXT_SOURCE_1"], df["EXT_SOURCE_3"])
    df["EXT_SOURCES_NA_COUNT"] = ext.isna().sum(axis=1)
    df["EXT_SOURCE_1_SQ"] = df["EXT_SOURCE_1"] ** 2
    df["EXT_SOURCE_2_SQ"] = df["EXT_SOURCE_2"] ** 2
    df["EXT_SOURCE_3_SQ"] = df["EXT_SOURCE_3"] ** 2

    # --- 3. Age / employment features --------------------------------------
    age = -df["DAYS_BIRTH"] / 365.0
    df["AGE_YEARS"] = age
    df["AGE_BIN"] = pd.cut(
        age, bins=[0, 25, 35, 45, 55, 120], labels=False
    ).astype("float32")
    df["DAYS_BIRTH_YEARS"] = df["DAYS_BIRTH"] / 365.0
    df["DAYS_EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"] / 365.0
    df["DAYS_BIRTH_YEARS"] = df["DAYS_BIRTH"] / 365.0
    df["DAYS_EMPLOYED_PERCENT"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["AGE_INCOME_INTERACTION"] = age * (income / 1e4)
    df["AGE_CREDIT_RATIO"] = _safe_div(age, credit / 1e3 + 1.0)
    df["ID_PUBLISH_YEARS"] = df["DAYS_ID_PUBLISH"] / 365.0
    df["REGISTRATION_YEARS"] = df["DAYS_REGISTRATION"] / 365.0
    df["LAST_PHONE_CHANGE_YEARS"] = df["DAYS_LAST_PHONE_CHANGE"] / 365.0

    # --- 4. Bureau request stats -------------------------------------------
    bureau_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    bureau = df[bureau_cols]
    df["BUREAU_REQS_SUM"] = bureau.sum(axis=1)
    df["BUREAU_REQS_MEAN"] = bureau.mean(axis=1)
    df["BUREAU_REQS_STD"] = bureau.std(axis=1)
    df["BUREAU_REQS_MAX"] = bureau.max(axis=1)
    df["BUREAU_REQS_YEAR_VS_QRT"] = _safe_div(
        df["AMT_REQ_CREDIT_BUREAU_YEAR"], df["AMT_REQ_CREDIT_BUREAU_QRT"]
    )

    # --- 5. Null count (over raw columns, computed before encoding) --------
    df["NULL_COUNT"] = df.isna().sum(axis=1)

    # --- 6. Group aggregations (iteration 4) --------------------------------
    filled = _fill_cat(df, GROUP_COLS)
    for col in GROUP_COLS:
        for num in NUM_AGG_COLS:
            grp = filled.groupby(col)[num]
            df[f"{col}_{num}_MEAN"] = grp.transform("mean")
            df[f"{col}_{num}_STD"] = grp.transform("std")
            df[f"{col}_{num}_MIN"] = grp.transform("min")
            df[f"{col}_{num}_MAX"] = grp.transform("max")
    df["ORG_EMPLOYED_YEARS_MEAN"] = (
        df.groupby("ORGANIZATION_TYPE")["DAYS_EMPLOYED_YEARS"].transform("mean")
    )
    df["ORG_NULL_COUNT_MEAN"] = df.groupby("ORGANIZATION_TYPE")[
        "NULL_COUNT"
    ].transform("mean")

    # --- 7. One-hot encoding -------------------------------------------------
    dummies = pd.get_dummies(df[CAT_COLUMNS], prefix=CAT_COLUMNS)
    dummies.columns = [sanitize_name(c) for c in dummies.columns]
    dummies = dummies.astype("float32")
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=CAT_COLUMNS)
    del dummies

    # --- 8. Split back and downcast ------------------------------------------
    mask = df["TARGET"].notna()
    train_proc = df[mask].copy()
    test_proc = df[~mask].drop(columns=["TARGET"]).copy()
    del df

    # Replace infinities with NaN (they can happen in ratios with 0 denom)
    for frame in (train_proc, test_proc):
        for col in frame.select_dtypes(include=["float64", "float32"]).columns:
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
        for col in frame.select_dtypes(include=["float64"]).columns:
            frame[col] = frame[col].astype("float32")
        for col in frame.select_dtypes(include=["int64"]).columns:
            frame[col] = frame[col].astype("int32")

    gc.collect()
    return train_proc, test_proc


class TargetEncoder:
    """
    Out-of-fold target encoder with Bayesian smoothing.

    Fit on the training part of a fold only, then transform the validation
    part and the test set. This keeps OOF AUC an honest estimate and test
    predictions leakage-free.

    encoding = (n_pos + m * g) / (n + m)

    where n_pos/n are the per-category positives/count in the fit data,
    g is the global positive rate of the fit data, and m is a smoothing
    strength pulling rare categories toward the global rate.
    """

    def __init__(self, smoothing=30):
        self.smoothing = smoothing
        self.stats_ = None
        self.global_rate_ = 0.5

    def fit(self, cat_df, y):
        """cat_df: DataFrame with the categorical columns (NaN-filled)."""
        stats = {}
        global_rate = float(np.asarray(y, dtype=np.float64).mean())
        for col in cat_df.columns:
            s = pd.Series(
                np.asarray(y, dtype=np.float64), index=cat_df[col].to_numpy()
            ).groupby(level=0)
            stats[col] = {
                "n": s.size().to_dict(),
                "npos": s.sum().to_dict(),
            }
        self.stats_ = stats
        self.global_rate_ = global_rate
        return self

    def transform(self, cat_df):
        """Return DataFrame of encoded columns named '{col}_TE'."""
        out = {}
        for col in cat_df.columns:
            table = self.stats_[col]
            cats = cat_df[col].to_numpy()
            n = np.array([table["n"].get(c, 0) for c in cats], dtype=np.float64)
            npos = np.array([table["npos"].get(c, 0) for c in cats], dtype=np.float64)
            enc = (npos + self.smoothing * self.global_rate_) / (
                n + self.smoothing
            )
            # Unknown categories (should not happen) fall back to global rate
            unknown = n == 0
            enc[unknown] = self.global_rate_
            out[f"{col}_TE"] = enc.astype(np.float32)
        return pd.DataFrame(out, index=cat_df.index)


def get_target_encoding_cats(train_df, test_df):
    """
    Build the NaN-filled categorical frames used for OOF target encoding.

    Returns (train_cats, test_cats, col_names) where both frames have the
    same column order and the same index order as the raw inputs.
    """
    train_cats = _fill_cat(train_df, TARGET_ENCODED_COLS)
    test_cats = _fill_cat(test_df, TARGET_ENCODED_COLS)
    return train_cats, test_cats, list(TARGET_ENCODED_COLS)
