"""
pattern_detector.py
--------------------
Detects common data patterns inside string columns using regex.
Called by ProfilerEngine._profile_column() for every column.

Public interface
----------------
    detect(column: pd.Series) -> str | None

Returns a pattern name string ("email", "phone", "date", "national_id")
if a dominant pattern is found, or None if no pattern is detected.
This value is stored as "detected_pattern" in the column profile JSON.
"""

import re
import pandas as pd


PATTERNS = {
    "national_id": r"^\d{14}$",
    "email"      : r"^[A-Za-z0-9._-]+@[A-Za-z0-9]+\.[A-Za-z]+$",
    "phone"      : r"^(?:\+2)?01[0125]\s?\d{8}$",
    # Month group fixed: 1?[120] was wrong (matched "0","20","120", missed "11")
    # Correct: 0?[1-9] covers months 1–9, 1[0-2] covers months 10–12
    "date"       : r"^(?:[012]?\d|3?[01])(?:-|/)(?:0?[1-9]|1[0-2])(?:-|/)\d{4}$",
}


def detect(column: pd.Series) -> str | None:
    """
    Detect whether a string column follows a known pattern.

    Strategy: first sample the first 100 rows for speed. If ≥50% of the
    sample matches, run the full column. If >75% of all rows match, the
    pattern is declared dominant and its name is returned.

    Parameters
    ----------
    column : one column from the dataset DataFrame

    Returns
    -------
    Pattern name string, or None if no dominant pattern found.
    Non-string columns always return None.
    """
    if not (column.dtype == "object" or pd.api.types.is_string_dtype(column)):
        return None

    col = column.dropna()
    if col.empty:
        return None

    for pattern_name, pattern_regex in PATTERNS.items():

        sample    = col.iloc[:100]
        s_matches = sum(1 for s in sample if re.fullmatch(pattern_regex, str(s).strip()))

        if s_matches / len(sample) < 0.5:
            continue

        full_matches   = sum(1 for r in col if re.fullmatch(pattern_regex, str(r).strip()))
        full_match_prc = full_matches / len(col)

        if full_match_prc > 0.75:
            return pattern_name

    return None