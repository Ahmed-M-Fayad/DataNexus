"""
column_stats.py
---------------
Computes statistical profile for a single pandas Series (one column).
Called by ProfilerEngine._profile_column() for every column in the dataset.

Public interface
----------------
    compute(column: pd.Series) -> dict

Output dict shape
-----------------
{
    "null_pct":       float,   # percentage of null rows (0.0–100.0)
    "distinct_count": int,     # number of unique non-null values
    "min":            float | None,   # None for non-numeric columns
    "max":            float | None,
    "mean":           float | None,
    "median":         float | None,
}
"""

import pandas as pd


def compute(column: pd.Series) -> dict:
    """
    Compute all column-level statistics in one call.

    Returns a flat dict that ProfilerEngine merges directly into the column
    profile. Numeric stats are None for non-numeric columns.

    Parameters
    ----------
    column : one column from the dataset DataFrame

    Returns
    -------
    dict matching the shape above — all keys always present.
    """
    total      = len(column)
    null_count = int(column.isnull().sum())

    # ── Numeric stats (None for text / boolean / datetime columns) ────────────
    if pd.api.types.is_numeric_dtype(column):
        col = column.dropna()
        if len(col) > 0:
            min_val    = round(float(col.min()),    4)
            max_val    = round(float(col.max()),    4)
            mean_val   = round(float(col.mean()),   4)
            median_val = round(float(col.median()), 4)
        else:
            # all values were null
            min_val = max_val = mean_val = median_val = None
    else:
        min_val = max_val = mean_val = median_val = None

    return {
        "null_pct"      : round((null_count / total * 100), 2) if total else 0.0,
        "distinct_count": int(column.nunique()),
        "min"           : min_val,
        "max"           : max_val,
        "mean"          : mean_val,
        "median"        : median_val,
    }