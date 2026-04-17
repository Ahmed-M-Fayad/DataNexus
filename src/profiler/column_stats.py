"""
column_stats.py
---------------
Calculates statistical profile for each column in a DataFrame.
Called by profiler_engine.py for every column in the dataset.
"""

import pandas as pd


def basic_column_info(column: pd.Series) -> dict:
    """
    Basic info that applies to every column regardless of type.
    Returns only fields that belong in the required output structure.
    """
    total     = len(column)
    null_count = int(column.isnull().sum())

    return {
        "unique_count": int(column.nunique()),
        "null_count"  : null_count,
        "null_pct"    : round((null_count / total * 100), 2) if total else 0.0,
    }


def numeric_column_stats(column: pd.Series) -> dict:
    """
    Min, max, mean, median for numeric columns.
    Returns None values for non-numeric columns.
    All values are flat — merged directly into the column profile.
    """
    if pd.api.types.is_numeric_dtype(column):
        col = column.dropna()
        return {
            "min"   : round(float(col.min()),    4),
            "max"   : round(float(col.max()),    4),
            "mean"  : round(float(col.mean()),   4),
            "median": round(float(col.median()), 4),
        }
    return {
        "min"   : None,
        "max"   : None,
        "mean"  : None,
        "median": None,
    }


def detect_outliers(column: pd.Series) -> dict:
    """
    Detect outliers using the IQR method.
    Kept as a separate nested key "outliers" — not part of the required
    flat structure but stored alongside it for deeper analysis.
    Returns None values for non-numeric columns.
    """
    if pd.api.types.is_numeric_dtype(column):
        col      = column.dropna()
        q1       = col.quantile(0.25)
        q3       = col.quantile(0.75)
        iqr      = q3 - q1
        lower    = q1 - (1.5 * iqr)
        upper    = q3 + (1.5 * iqr)
        outliers = col[(col < lower) | (col > upper)].tolist()
        return {
            "outlier_count": len(outliers),
            "lower_bound"  : round(float(lower), 4),
            "upper_bound"  : round(float(upper), 4),
            "samples"      : outliers[:5],
        }
    return {
        "outlier_count": None,
        "lower_bound"  : None,
        "upper_bound"  : None,
        "samples"      : None,
    }
