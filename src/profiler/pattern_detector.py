"""
pattern_detector.py
--------------------
Uses regex to detect common data patterns inside string columns.
Called by profiler_engine.py for every column in the dataset.

Returns a dict with exactly two keys: "pattern" and "rate".
This dict is nested under the "pattern" key in the column profile.
"""

import re
import pandas as pd


PATTERNS = {
    "national_id": r"^\d{14}$",
    "email"      : r"^[A-Za-z0-9._-]+@[A-Za-z0-9]+\.[A-Za-z]+$",
    "phone"      : r"^(?:\+2)?01[0125]\s?\d{8}$",
    "date"       : r"^(?:[012]?\d|3?[01])(?:-|/)(1?[120]|0?[1-9])(?:-|/)(?:\d{4})$",
}

# Required output shape — always returned, even when no pattern found
_EMPTY = {"pattern": None, "rate": None}


def pattern_detector(column: pd.Series) -> dict:

    if not (column.dtype == "object" or pd.api.types.is_string_dtype(column)):
        return _EMPTY.copy()

    col = column.dropna()
    if col.empty:
        return _EMPTY.copy()

    for pattern_name, pattern_regex in PATTERNS.items():

        sample    = col.iloc[:100]
        s_matches = sum(1 for s in sample if re.fullmatch(pattern_regex, str(s).strip()))

        if s_matches / len(sample) < 0.5:
            continue

        full_matches   = sum(1 for r in col if re.fullmatch(pattern_regex, str(r).strip()))
        full_match_prc = full_matches / len(col)

        if full_match_prc > 0.75:
            return {
                "pattern": pattern_name,
                "rate"   : round(full_match_prc, 4),
            }

    return _EMPTY.copy()
