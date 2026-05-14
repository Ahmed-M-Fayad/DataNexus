import re
import pandas as pd


PATTERNS = {
    "national_id": r"^\d{14}$",
    "email"      : r"^[A-Za-z0-9._-]+@[A-Za-z0-9]+\.[A-Za-z]+$",
    "phone"      : r"^(?:\+2)?01[0125]\s?\d{8}$",
    "date"       : r"^(?:[012]?\d|3?[01])(?:-|/)(?:0?[1-9]|1[0-2])(?:-|/)\d{4}$",
}


def detect(column: pd.Series) -> str | None:
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