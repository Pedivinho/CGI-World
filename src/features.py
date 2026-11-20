"""
Feature engineering pipeline - Home Credit Default Risk (competitive sprint)
============================================================================

Leak-free feature matrix shared by train and test (built on the concatenated
frame so statistics are computed identically for both):

- Cleaning: DAYS_EMPLOYED == 365243 -> NaN + DAYS_EMPLOYED_ANOM flag
- Financial ratios: CREDIT_TO_ANNUITY_RATIO (loan term in months),
  ANNUITY_TO_INCOME_RATIO, CREDIT_TO_INCOME_RATIO, CREDIT_TO_GOODS_RATIO,
  CREDIT_GOODS_DIFF (over-financing), per-person / per-child income
- EXT_SOURCE block: mean/std/min/max/sum/median, product (raw + imputed),
  pairwise diffs and safe ratios, weighted combination, squares, sqrts,
  exact null counts and per-source ISNA indicators
- Age / employment: years, employment-to-birth ratios, car-to-birth/employed,
  phone/id publish to birth ratios
- Bureau request SUM and MAX over AMT_REQ_CREDIT_BUREAU_*
- Categorical interactions: GENDER_EDU, INCOME_ORG, CONTRACT_GENDER
- Frequency encoding (normalized counts fit on train+test) for all
  categorical columns and interactions
- Group aggregations (mean, std, val-mean diff, val/(mean+eps) ratio) of
  EXT_SOURCE_MEAN/2/3, AMT_CREDIT, AMT_INCOME_TOTAL, CREDIT_TO_ANNUITY_RATIO,
  DAYS_BIRTH grouped by ORGANIZATION_TYPE, OCCUPATION_TYPE,
  NAME_EDUCATION_TYPE, NAME_INCOME_TYPE, GENDER_EDU
- One-hot encoding for the 8 base categorical columns
- Null count per row

Smoothed target encoding (m=20) is applied STRICTLY OUT-OF-FOLD inside the
training folds (see fit_smoothed_target_encoding below); it never enters
build_features, so no target information leaks into the static matrix.

All numeric outputs are downcast to float32 / int32.
"""

import gc

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

# Categorical columns used for frequency encoding (base + interactions)
FE_CAT_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "GENDER_EDU",
    "INCOME_ORG",
    "CONTRACT_GENDER",
]

# Categorical columns receiving out-of-fold smoothed target encoding (m=20)
TARGET_ENCODED_COLS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "GENDER_EDU",
]

GROUP_COLS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "GENDER_EDU",
]

GROUP_AGG_VARS = [
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "CREDIT_TO_ANNUITY_RATIO",
    "DAYS_BIRTH",
]

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

_MISSING = "__NA__"
_EPS = 1e-5


def _to_str_col(s):
    """Object column with NaN replaced by the explicit missing token."""
    return s.astype("object").where(s.notna(), _MISSING)


def _fill_cat(df, cols):
    """Replace NaNs in categorical columns with an explicit missing token."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = _to_str_col(out[col])
    return out


def _safe_div(num, den, eps=_EPS):
    return num / (den + eps)


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
    df = df.astype({c: np.float32 for c in df.select_dtypes(include="float64").columns})

    # --- 0. Cleaning --------------------------------------------------------
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("int32")
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    income = df["AMT_INCOME_TOTAL"]
    credit = df["AMT_CREDIT"]
    annuity = df["AMT_ANNUITY"]
    goods = df["AMT_GOODS_PRICE"]

    # --- 1. Financial ratios -------------------------------------------------
    # Loan term in months - the #1 domain feature of Home Credit
    df["CREDIT_TO_ANNUITY_RATIO"] = _safe_div(credit, annuity)
    df["ANNUITY_TO_INCOME_RATIO"] = _safe_div(annuity, income)
    df["CREDIT_TO_INCOME_RATIO"] = _safe_div(credit, income)
    df["CREDIT_TO_GOODS_RATIO"] = _safe_div(credit, goods)
    df["CREDIT_GOODS_DIFF"] = (credit - goods).astype("float32")
    df["INCOME_PER_PERSON"] = _safe_div(income, df["CNT_FAM_MEMBERS"])
    df["INCOME_PER_CHILD"] = income / (1 + df["CNT_CHILDREN"])
    df["CHILDREN_RATIO"] = _safe_div(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])

    # --- 2. EXT_SOURCE block (maximum priority) -------------------------------
    ext = df[EXT_SOURCES]
    ext_imp = ext.fillna(0.0)
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    df["EXT_SOURCE_SUM"] = ext_imp.sum(axis=1)
    df["EXT_SOURCE_PROD"] = (ext_imp[EXT_SOURCES[0]] * ext_imp[EXT_SOURCES[1]] * ext_imp[EXT_SOURCES[2]]).astype("float32")
    df["EXT_SOURCES_RANGE"] = (ext.max(axis=1) - ext.min(axis=1)).astype("float32")
    df["EXT_SOURCE_1_MINUS_2"] = (ext[EXT_SOURCES[0]] - ext[EXT_SOURCES[1]]).astype("float32")
    df["EXT_SOURCE_2_MINUS_3"] = (ext[EXT_SOURCES[1]] - ext[EXT_SOURCES[2]]).astype("float32")
    df["EXT_SOURCE_1_MINUS_3"] = (ext[EXT_SOURCES[0]] - ext[EXT_SOURCES[2]]).astype("float32")
    df["EXT_1_OVER_2"] = _safe_div(ext[EXT_SOURCES[0]], ext[EXT_SOURCES[1]])
    df["EXT_2_OVER_3"] = _safe_div(ext[EXT_SOURCES[1]], ext[EXT_SOURCES[2]])
    df["EXT_SOURCE_WEIGHTED"] = (
        ext[EXT_SOURCES[0]] * 2 + ext[EXT_SOURCES[1]] * 3 + ext[EXT_SOURCES[2]] * 4
    )
    df["EXT_SOURCE_1_SQ"] = ext[EXT_SOURCES[0]] ** 2
    df["EXT_SOURCE_2_SQ"] = ext[EXT_SOURCES[1]] ** 2
    df["EXT_SOURCE_3_SQ"] = ext[EXT_SOURCES[2]] ** 2
    df["EXT_SOURCE_1_SQRT"] = np.sqrt(ext[EXT_SOURCES[0]].fillna(0.0).clip(lower=0.0))
    df["EXT_SOURCE_2_SQRT"] = np.sqrt(ext[EXT_SOURCES[1]].fillna(0.0).clip(lower=0.0))
    df["EXT_SOURCE_3_SQRT"] = np.sqrt(ext[EXT_SOURCES[2]].fillna(0.0).clip(lower=0.0))
    df["EXT_SOURCE_NA_COUNT"] = ext.isna().sum(axis=1).astype("int32")
    df["EXT_1_ISNA"] = ext[EXT_SOURCES[0]].isna().astype("int32")
    df["EXT_2_ISNA"] = ext[EXT_SOURCES[1]].isna().astype("int32")
    df["EXT_3_ISNA"] = ext[EXT_SOURCES[2]].isna().astype("int32")

    # --- 3. Age / employment / car / phone ------------------------------------
    age = (-df["DAYS_BIRTH"] / 365.0).astype("float32")
    df["AGE_YEARS"] = age
    df["AGE_BIN"] = pd.cut(
        age, bins=[0, 25, 35, 45, 55, 120], labels=False
    ).astype("float32")
    emp_years = (df["DAYS_EMPLOYED"] / 365.0).astype("float32")
    df["DAYS_EMPLOYED_YEARS"] = emp_years
    df["EMPLOYED_TO_BIRTH_RATIO"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["DAYS_EMPLOYED_DIFF"] = (df["DAYS_BIRTH"] - df["DAYS_EMPLOYED"]).astype("float32")
    df["CAR_TO_BIRTH_RATIO"] = _safe_div(df["OWN_CAR_AGE"], age)
    df["CAR_TO_EMPLOYED_RATIO"] = _safe_div(df["OWN_CAR_AGE"], emp_years)
    df["PHONE_TO_BIRTH_RATIO"] = _safe_div(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"])
    df["ID_TO_BIRTH_RATIO"] = _safe_div(df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"])

    # --- 4. Bureau request stats ----------------------------------------------
    bureau_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    bureau = df[bureau_cols]
    df["BUREAU_REQ_SUM"] = bureau.sum(axis=1).astype("float32")
    df["BUREAU_REQ_MAX"] = bureau.max(axis=1).astype("float32")

    # --- 5. Null count (over raw columns, before encodings) --------------------
    df["NULL_COUNT"] = df.isna().sum(axis=1).astype("int32")

    # --- 6. Categorical interactions -------------------------------------------
    filled = _fill_cat(df, CAT_COLUMNS)
    df["GENDER_EDU"] = filled["CODE_GENDER"] + "_" + filled["NAME_EDUCATION_TYPE"]
    df["INCOME_ORG"] = filled["NAME_INCOME_TYPE"] + "_" + filled["ORGANIZATION_TYPE"]
    df["CONTRACT_GENDER"] = filled["NAME_CONTRACT_TYPE"] + "_" + filled["CODE_GENDER"]

    # --- 7. Frequency encoding (fit on train+test, non-target statistics) -------
    for col in FE_CAT_COLUMNS:
        vc = df[col].value_counts(normalize=True)
        df[f"FREQ_{col}"] = df[col].map(vc).fillna(0.0).astype("float32")

    # --- 8. Group aggregations (fit on train+test, non-target statistics) -------
    group_cols_dict = {}
    for col in GROUP_COLS:
        grp = df.groupby(col)
        for var in GROUP_AGG_VARS:
            gmean = grp[var].transform("mean")
            gstd = grp[var].transform("std")
            group_cols_dict[f"G_{col}_{var}_MEAN"] = gmean.astype("float32")
            group_cols_dict[f"G_{col}_{var}_STD"] = gstd.astype("float32")
            group_cols_dict[f"G_{col}_{var}_DIFF"] = (df[var] - gmean).astype("float32")
            group_cols_dict[f"G_{col}_{var}_RATIO"] = _safe_div(df[var], gmean)

    df = pd.concat([df, pd.DataFrame(group_cols_dict, index=df.index)], axis=1)
    del group_cols_dict

    # --- 9. One-hot encoding ------------------------------------------------------
    dummies = pd.get_dummies(filled[CAT_COLUMNS], prefix=CAT_COLUMNS)
    dummies.columns = [sanitize_name(c) for c in dummies.columns]
    dummies = dummies.astype("float32")
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=FE_CAT_COLUMNS)
    del dummies, filled

    # --- 10. Split back and downcast -----------------------------------------------
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


def sanitize_name(name):
    """Make a column name safe (no special chars, no leading digit)."""
    import re

    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name)).strip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "col_" + sanitized
    return sanitized if sanitized else "unnamed_col"


def get_target_encoding_cats(train_df, test_df, cols=None):
    """
    Build the NaN-filled categorical frames used for OOF target encoding.

    Returns (train_cats, test_cats, col_names) where both frames have the
    same column order and the same row order as the raw inputs.
    """
    cols = cols if cols is not None else TARGET_ENCODED_COLS

    train_cats = _fill_cat(train_df, CAT_COLUMNS)
    test_cats = _fill_cat(test_df, CAT_COLUMNS)
    train_cats["GENDER_EDU"] = (
        train_cats["CODE_GENDER"] + "_" + train_cats["NAME_EDUCATION_TYPE"]
    )
    test_cats["GENDER_EDU"] = (
        test_cats["CODE_GENDER"] + "_" + test_cats["NAME_EDUCATION_TYPE"]
    )

    order = [c for c in cols if c in train_cats.columns]
    return (
        train_cats[order].reset_index(drop=True),
        test_cats[order].reset_index(drop=True),
        order,
    )


def fit_smoothed_target_encoding(y, cat_values, smoothing=20):
    """
    Compute smoothed target encoding statistics on ONE fold's training part.

    encoding = (n_pos + m * g) / (n + m)

    where n_pos/n are the per-category positives/count in the fit data,
    g is the global positive rate of the fit data, and m is the smoothing
    strength pulling rare categories toward the global rate.

    Parameters
    ----------
    y : array-like, shape (n,)
        Binary targets of the training part of the fold.
    cat_values : array-like, shape (n,)
        Categorical values (string token for missing) of the training part.

    Returns
    -------
    (stats, global_rate) where stats is dict {category: (n, npos)}.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    cats = np.asarray(cat_values)
    global_rate = float(y_arr.mean()) if len(y_arr) else 0.5
    df = pd.DataFrame({"cat": cats, "y": y_arr})
    grouped = df.groupby("cat")["y"]
    counts = grouped.size()
    sums = grouped.sum()
    stats = {c: (int(counts[c]), float(sums[c])) for c in counts.index}
    return stats, global_rate


def transform_target_encoding(stats, global_rate, cat_values, smoothing=20):
    """
    Apply smoothed target encoding using statistics fit on a fold's
    training part (out-of-fold by construction). Unknown categories fall
    back to the global positive rate.
    """
    cats = np.asarray(cat_values, dtype=object)
    # Vectorize lookup via a dict-backed Series
    keys = np.array(list(stats.keys()), dtype=object)
    n_arr = np.array([stats[k][0] for k in keys], dtype=np.float64)
    npos_arr = np.array([stats[k][1] for k in keys], dtype=np.float64)
    lookup = pd.Series(npos_arr, index=pd.Series(keys)).to_dict()
    n_lookup = pd.Series(n_arr, index=pd.Series(keys)).to_dict()

    n = np.array([n_lookup.get(c, 0.0) for c in cats], dtype=np.float64)
    npos = np.array([lookup.get(c, 0.0) for c in cats], dtype=np.float64)
    enc = (npos + smoothing * global_rate) / (n + smoothing)
    unknown = n == 0
    enc[unknown] = global_rate
    return enc.astype(np.float32)


def get_oof_target_encoder(smoothing=20):
    """
    Return a helper that produces out-of-fold smoothed target encodings.

    Usage inside a training fold (skf indices tr/va, raw cat frames):
        oof_te[va] = te(cat_train_va, y[tr], cat_raw_va)
        test_te += te(cat_raw, y[tr], cat_raw_te)/nfold
    """
    def te(cat_fit, y_fit, cat_target):
        stats, g = fit_smoothed_target_encoding(y_fit, cat_fit, smoothing=smoothing)
        return transform_target_encoding(stats, g, cat_target, smoothing=smoothing)

    return te
