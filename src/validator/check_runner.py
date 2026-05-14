import logging

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_CHECK_TYPES: frozenset[str] = frozenset({
    "not_null",
    "completeness",          # alias of not_null
    "not_empty",
    "unique",
    "range",
    "regex",
    "in_set",
    "foreign_key",           # alias of in_set
    "referential_integrity", # alias of in_set
    "freshness",
})


class CheckRunner:
    """Executes a single validation check against a pandas DataFrame. Stateless."""

    def __init__(self, use_ge: bool = False) -> None:
        self.use_ge = use_ge
        self._ge_adapter = None  # lazy-loaded on first GE call

    def run(self, df: pd.DataFrame, check: dict) -> dict:
        if self.use_ge:
            return self._run_via_ge(df, check)
        return self._run_natively(df, check)

    # ── GE path ───────────────────────────────────────────────────────────────

    def _run_via_ge(self, df: pd.DataFrame, check: dict) -> dict:
        if self._ge_adapter is None:
            from src.ge_adapter import GEAdapter  # noqa: PLC0415
            self._ge_adapter = GEAdapter()
        raw = self._ge_adapter.run_expectation(df, check)
        raw["status"]        = "pass" if raw["success"] else "fail"
        raw["error_message"] = None
        return raw

    # ── Native path ───────────────────────────────────────────────────────────

    def _run_natively(self, df: pd.DataFrame, check: dict) -> dict:
        check_name: str   = check.get("name", "unnamed")
        check_type: str   = check.get("check_type", "")
        column:     str   = check.get("column", "")
        threshold:  float = float(check.get("threshold", 1.0))

        required = {"name", "check_type", "column", "severity"}
        missing  = required - check.keys()
        if missing:
            raise ValueError(
                f"Check '{check_name}' is missing required keys: {sorted(missing)}."
            )

        if check_type not in SUPPORTED_CHECK_TYPES:
            raise ValueError(
                f"Check '{check_name}': unknown check_type '{check_type}'. "
                f"Supported: {sorted(SUPPORTED_CHECK_TYPES)}"
            )

        if column not in df.columns:
            raise ValueError(
                f"Check '{check_name}': column '{column}' not found. "
                f"Available: {sorted(df.columns.tolist())}"
            )

        col   = df[column]
        total = len(df)

        _dispatch = {
            "not_null":              self._check_not_null,
            "completeness":          self._check_not_null,
            "not_empty":             self._check_not_empty,
            "unique":                self._check_unique,
            "range":                 self._check_range,
            "regex":                 self._check_regex,
            "in_set":                self._check_in_set,
            "foreign_key":           self._check_in_set,
            "referential_integrity": self._check_in_set,
            "freshness":             self._check_freshness,
        }

        failing_mask:  pd.Series = _dispatch[check_type](col, check)
        failing_count: int       = int(failing_mask.sum())
        pass_rate:     float     = ((total - failing_count) / total) if total > 0 else 1.0
        success:       bool      = pass_rate >= threshold
        samples:       list      = col[failing_mask].dropna().head(5).tolist()

        logger.info(
            f"Check '{check_name}' [{check_type}] on '{column}': "
            f"{'PASSED' if success else 'FAILED'} | "
            f"failing={failing_count}/{total} ({100*(1-pass_rate):.1f}% bad) | "
            f"threshold={threshold}"
        )
        return {
            "status":             "pass" if success else "fail",
            "success":            success,
            "failing_count":      failing_count,
            "total_count":        total,
            "unexpected_samples": samples,
            "error_message":      None,
        }

    # ── Check implementations (True = row FAILED) ─────────────────────────────

    def _check_not_null(self, col: pd.Series, check: dict) -> pd.Series:
        # Covers not_null and completeness.
        return col.isna()

    def _check_not_empty(self, col: pd.Series, check: dict) -> pd.Series:
        null_mask  = col.isna()
        empty_mask = col.astype(str).str.strip() == ""
        return null_mask | empty_mask

    def _check_unique(self, col: pd.Series, check: dict) -> pd.Series:
        # keep=False marks every row that participates in a duplicate, not just extras.
        return col.duplicated(keep=False)

    def _check_range(self, col: pd.Series, check: dict) -> pd.Series:
        min_val = check.get("min")
        max_val = check.get("max")
        if min_val is None and max_val is None:
            raise ValueError(
                f"Check '{check.get('name')}' uses 'range' but neither 'min' nor 'max' provided."
            )
        null_mask    = col.isna()
        numeric      = pd.to_numeric(col, errors="coerce")
        out_of_range = pd.Series(False, index=col.index)
        if min_val is not None: out_of_range = out_of_range | (numeric < min_val)
        if max_val is not None: out_of_range = out_of_range | (numeric > max_val)
        return null_mask | out_of_range

    def _check_regex(self, col: pd.Series, check: dict) -> pd.Series:
        pattern = check.get("regex")
        if not pattern:
            raise ValueError(f"Check '{check.get('name')}' uses 'regex' but 'regex' key is missing.")
        null_mask     = col.isna()
        no_match_mask = ~col.astype(str).str.match(pattern, na=False)
        return null_mask | no_match_mask

    def _check_in_set(self, col: pd.Series, check: dict) -> pd.Series:
        values = check.get("values")
        if values is None:
            raise ValueError(
                f"Check '{check.get('name')}' uses 'in_set'/'foreign_key' "
                f"but 'values' key is missing."
            )
        return ~col.isin(values)

    def _check_freshness(self, col: pd.Series, check: dict) -> pd.Series:
        min_val = check.get("min")
        max_val = check.get("max")
        if min_val is None and max_val is None:
            raise ValueError(
                f"Check '{check.get('name')}' uses 'freshness' but neither 'min' nor 'max' provided."
            )
        null_mask    = col.isna()
        dt_col       = pd.to_datetime(col, errors="coerce")
        out_of_range = pd.Series(False, index=col.index)
        if min_val is not None: out_of_range = out_of_range | (dt_col < pd.Timestamp(min_val))
        if max_val is not None: out_of_range = out_of_range | (dt_col > pd.Timestamp(max_val))
        return null_mask | out_of_range
