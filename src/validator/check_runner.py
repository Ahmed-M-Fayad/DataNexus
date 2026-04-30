"""
DataNexus — Check Runner
Executes a single validation check against a pandas DataFrame.

This module is pure computation — no database access, no I/O.
The Validation Engine calls this once per check in a config.

Two execution paths
-------------------
    Native (default, use_ge=False):
        Runs each check using plain pandas operations.
        Fast, zero external dependencies, works offline.

    GE-backed (use_ge=True):
        Delegates every call to GEAdapter → great_expectations.
        Useful when you want GE's richer result metadata.
        Requires great-expectations==0.18.8 to be installed.

Return contract
---------------
    Every call to run() returns EXACTLY this dict shape —
    the Validation Engine and score_calculator depend on it:

        {
            "status":             "pass" | "fail" | "error",
            "success":            bool,
            "failing_count":      int,
            "total_count":        int,
            "unexpected_samples": list,   # up to 5 example failing values
            "error_message":      str | None,
        }

Usage
-----
    runner = CheckRunner()                   # native mode
    runner = CheckRunner(use_ge=True)        # GE-backed mode

    result = runner.run(df=my_dataframe, check={
        "name":       "email_not_null",
        "check_type": "not_null",
        "column":     "email",
        "threshold":  0.95,
        "severity":   "high",
    })
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Supported native check types ─────────────────────────────────────────────
# This set mirrors the 6 built-in test_definitions seeded in the migration
# PLUS the aliases defined in the GE Adapter.
# Any check_type NOT in this set raises ValueError immediately.
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
    """
    Executes a single validation check against a pandas DataFrame.

    Stateless — one instance can be reused for every check in a run.
    The Validation Engine creates one CheckRunner and calls run() in a loop.
    """

    def __init__(self, use_ge: bool = False) -> None:
        """
        Args:
            use_ge: if True, delegate checks to GEAdapter instead of native pandas.
                    GE adapter must be built (Step 6) and great-expectations==0.18.8
                    must be installed.  Defaults to False.
        """
        self.use_ge = use_ge
        # _ge_adapter is lazy-loaded on first use to avoid ImportError at startup
        # if great_expectations is not yet installed.
        self._ge_adapter = None

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, check: dict) -> dict:
        """
        Executes a single check against the DataFrame.

        Args:
            df:    pandas DataFrame of the dataset being validated.
            check: dict with the check definition. Required keys:
                     "name"       (str)   — human-readable identifier
                     "check_type" (str)   — e.g. "not_null", "range"
                     "column"     (str)   — column to inspect
                     "threshold"  (float) — fraction of rows that must PASS (0.0–1.0)
                     "severity"   (str)   — "critical"|"high"|"medium"|"low"
                   Optional keys (check-type-specific):
                     "min"    (number)   — lower bound for "range" / "freshness"
                     "max"    (number)   — upper bound for "range" / "freshness"
                     "regex"  (str)      — pattern for "regex" check
                     "values" (list)     — allowed set for "in_set" / "foreign_key"

        Returns:
            dict with keys:
                "status"             → "pass" | "fail" | "error"
                "success"            → bool (True when status == "pass")
                "failing_count"      → int
                "total_count"        → int
                "unexpected_samples" → list (up to 5 failing values)
                "error_message"      → str | None

        Raises:
            ValueError: unknown check_type, missing required key, bad config.
            (RuntimeError from GEAdapter is re-raised as-is when use_ge=True.)
        """
        if self.use_ge:
            return self._run_via_ge(df, check)
        return self._run_natively(df, check)

    # ── GE-backed path ────────────────────────────────────────────────────────

    def _run_via_ge(self, df: pd.DataFrame, check: dict) -> dict:
        """
        Delegates the check to GEAdapter (Step 6).
        Lazy-loads the adapter on first call so the module can be imported
        even when great_expectations is not yet installed.
        """
        # Lazy-load: import only when actually needed, not at module load time.
        if self._ge_adapter is None:
            from src.ge_adapter import GEAdapter  # Step 6 dependency
            self._ge_adapter = GEAdapter()

        # GEAdapter.run_expectation() already returns the agreed result shape.
        raw = self._ge_adapter.run_expectation(df, check)

        # GEAdapter returns "success" bool but not "status" string.
        # We add "status" here so the result is consistent with native output.
        raw["status"]        = "pass" if raw["success"] else "fail"
        raw["error_message"] = None
        return raw

    # ── Native path ───────────────────────────────────────────────────────────

    def _run_natively(self, df: pd.DataFrame, check: dict) -> dict:
        """
        Runs the check using plain pandas — no external library required.

        Steps:
          1. Validate the check dict has all required keys.
          2. Confirm the target column exists in the DataFrame.
          3. Dispatch to the right _check_* method.
          4. Compute pass/fail based on threshold.
          5. Return the agreed result dict.
        """
        check_name: str  = check.get("name", "unnamed")
        check_type: str  = check.get("check_type", "")
        column:     str  = check.get("column", "")
        threshold:  float = float(check.get("threshold", 1.0))

        # ── guard 1: required keys ────────────────────────────────────────────
        required = {"name", "check_type", "column", "severity"}
        missing  = required - check.keys()
        if missing:
            raise ValueError(
                f"Check '{check_name}' is missing required keys: {sorted(missing)}. "
                f"Every check dict must have: {sorted(required)}"
            )

        # ── guard 2: check type is supported ──────────────────────────────────
        if check_type not in SUPPORTED_CHECK_TYPES:
            raise ValueError(
                f"Check '{check_name}': unknown check_type '{check_type}'. "
                f"Supported types: {sorted(SUPPORTED_CHECK_TYPES)}"
            )

        # ── guard 3: column exists in the DataFrame ───────────────────────────
        if column not in df.columns:
            raise ValueError(
                f"Check '{check_name}': column '{column}' not found in DataFrame. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

        col   = df[column]     # the pandas Series we are checking
        total = len(df)        # total number of rows

        # ── dispatch table: check_type → handler method ────────────────────
        # Using a dict dispatch instead of if/elif chains so adding a new
        # check type only requires adding one entry here + one _check_* method.
        _dispatch = {
            "not_null":              self._check_not_null,
            "completeness":          self._check_not_null,   # alias
            "not_empty":             self._check_not_empty,
            "unique":                self._check_unique,
            "range":                 self._check_range,
            "regex":                 self._check_regex,
            "in_set":                self._check_in_set,
            "foreign_key":           self._check_in_set,     # alias
            "referential_integrity": self._check_in_set,     # alias
            "freshness":             self._check_freshness,
        }

        # Each handler returns a boolean pandas Series:
        #   True  → this row FAILED the check
        #   False → this row PASSED the check
        failing_mask: pd.Series = _dispatch[check_type](col, check)

        failing_count: int = int(failing_mask.sum())

        # pass_rate = fraction of rows that PASSED (not failed)
        pass_rate: float = ((total - failing_count) / total) if total > 0 else 1.0

        # The check passes when enough rows satisfy the rule.
        # threshold=0.95 means 95 % of rows must pass.
        success: bool = pass_rate >= threshold

        # Collect up to 5 example failing values for the dashboard.
        # We drop NaN from samples because they are hard to display.
        samples: list = col[failing_mask].dropna().head(5).tolist()

        result = {
            "status":             "pass" if success else "fail",
            "success":            success,
            "failing_count":      failing_count,
            "total_count":        total,
            "unexpected_samples": samples,
            "error_message":      None,
        }

        logger.info(
            f"Check '{check_name}' [{check_type}] on '{column}': "
            f"{'PASSED' if success else 'FAILED'} | "
            f"failing={failing_count}/{total} ({100*(1-pass_rate):.1f}% bad) | "
            f"threshold={threshold}"
        )
        return result

    # ── Individual check implementations ─────────────────────────────────────
    # Each method receives the column Series and the full check dict.
    # Each method returns a boolean Series where True = this row FAILED.

    def _check_not_null(self, col: pd.Series, check: dict) -> pd.Series:
        """
        Fails rows where the value is NULL / None / NaN.
        Covers check_types: "not_null", "completeness".
        """
        return col.isna()
        # col.isna() returns True for every cell that is NULL/None/NaN.
        # Pandas treats Python None, float NaN, and pd.NaT all as "NA".

    def _check_not_empty(self, col: pd.Series, check: dict) -> pd.Series:
        """
        Fails rows where the value is:
          - NULL / None / NaN  (isna)
          - An empty string ""
          - A whitespace-only string "   " or "\t"

        This catches data that passes not_null but is still useless.
        """
        null_mask  = col.isna()
        # .astype(str) converts NaN → "nan", but null_mask already covers those.
        # .str.strip() removes leading/trailing whitespace before comparing to "".
        empty_mask = col.astype(str).str.strip() == ""
        return null_mask | empty_mask   # True if EITHER condition holds

    def _check_unique(self, col: pd.Series, check: dict) -> pd.Series:
        """
        Fails ALL rows that participate in a duplicate value (keeps no copy).
        Example: if "A" appears 3 times, all 3 rows are marked as failing.
        keep=False means "mark every duplicate, not just extras".
        """
        return col.duplicated(keep=False)

    def _check_range(self, col: pd.Series, check: dict) -> pd.Series:
        """
        Fails rows where the numeric value is outside [min, max].
        NULL values are counted as failures (missing data fails range checks).
        At least one of "min" or "max" must be provided.
        """
        min_val = check.get("min")  # None if not provided
        max_val = check.get("max")  # None if not provided

        if min_val is None and max_val is None:
            raise ValueError(
                f"Check '{check.get('name')}' uses 'range' but neither "
                f"'min' nor 'max' was provided. At least one is required."
            )

        null_mask = col.isna()

        # pd.to_numeric(errors="coerce") converts non-numeric strings to NaN
        # so they are treated as failures rather than raising a TypeError.
        numeric = pd.to_numeric(col, errors="coerce")

        out_of_range = pd.Series(False, index=col.index)
        if min_val is not None:
            out_of_range = out_of_range | (numeric < min_val)   # below minimum
        if max_val is not None:
            out_of_range = out_of_range | (numeric > max_val)   # above maximum

        return null_mask | out_of_range

    def _check_regex(self, col: pd.Series, check: dict) -> pd.Series:
        """
        Fails rows where the string value does NOT match the regex pattern.
        NULL values are counted as failures.
        """
        pattern = check.get("regex")
        if not pattern:
            raise ValueError(
                f"Check '{check.get('name')}' uses 'regex' but "
                f"the 'regex' key is missing from the check definition."
            )

        null_mask = col.isna()

        # .str.match() returns True if the BEGINNING of the string matches.
        # na=False means NaN cells return False (not a match = failing row),
        # but we cover NaN separately with null_mask for clarity.
        no_match_mask = ~col.astype(str).str.match(pattern, na=False)

        return null_mask | no_match_mask

    def _check_in_set(self, col: pd.Series, check: dict) -> pd.Series:
        """
        Fails rows whose value is NOT in the provided allowed set.
        NULL values are treated as failures (None is not in any normal set).
        Covers check_types: "in_set", "foreign_key", "referential_integrity".
        """
        values = check.get("values")
        if values is None:
            raise ValueError(
                f"Check '{check.get('name')}' uses 'in_set'/'foreign_key' but "
                f"the 'values' key is missing. Provide a list of allowed values."
            )

        # col.isin(values) returns True for matching rows; ~ inverts to get failures.
        # isin() treats NaN as not-in-set automatically, so nulls always fail.
        return ~col.isin(values)

    def _check_freshness(self, col: pd.Series, check: dict) -> pd.Series:
        """
        Fails rows where the datetime value is outside [min, max].
        Accepts ISO 8601 strings or datetime objects for min/max.
        NULL values are counted as failures.
        """
        min_val = check.get("min")
        max_val = check.get("max")

        if min_val is None and max_val is None:
            raise ValueError(
                f"Check '{check.get('name')}' uses 'freshness' but neither "
                f"'min' nor 'max' was provided. At least one is required."
            )

        null_mask = col.isna()

        # pd.to_datetime(errors="coerce") turns unparseable values into NaT.
        dt_col = pd.to_datetime(col, errors="coerce")

        out_of_range = pd.Series(False, index=col.index)
        if min_val is not None:
            out_of_range = out_of_range | (dt_col < pd.Timestamp(min_val))
        if max_val is not None:
            out_of_range = out_of_range | (dt_col > pd.Timestamp(max_val))

        return null_mask | out_of_range
