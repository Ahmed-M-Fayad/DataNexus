# ─────────────────────────────────────────────────────
# Built against great-expectations==0.18.8 (0.x API)
# ge.from_pandas() is NOT available in GE 1.x
# If requirements.txt is upgraded to GE 1.x, this file
# needs a full rewrite using DataContext + Validator approach
# ─────────────────────────────────────────────────────

import logging
import great_expectations as ge
import pandas as pd
from .expectation_mapper import map_check_type, get_auto_kwargs

logger = logging.getLogger(__name__)


class GEAdapter:
    """
    Wraps Great Expectations behind a single interface.
    The rest of the codebase never imports great_expectations directly —
    all GE calls go through this class only.

    Usage:
        adapter = GEAdapter()
        result = adapter.run_expectation(df, check)

    Input:
        df    — pandas DataFrame containing the dataset to check
        check — dict with keys: name, check_type, column,
                and optional: min, max, regex, values, threshold

    Output:
        {
            "success":            bool,
            "failing_count":      int,
            "total_count":        int,
            "unexpected_samples": list
        }
    """

    def run_expectation(self, df: pd.DataFrame, check: dict) -> dict:
        """
        Runs a single GE expectation against a pandas DataFrame.

        Args:
            df:    pandas DataFrame containing the data to validate
            check: dict confirmed by Validation Engine team with keys:
                   - name        (str)   check identifier
                   - check_type  (str)   e.g. 'not_null', 'range'
                   - column      (str)   column name to check
                   - threshold   (float) pass threshold 0.0 - 1.0
                   - severity    (str)   Low / Medium / High / Critical
                   - min         (opt)   used by 'range' and 'freshness'
                   - max         (opt)   used by 'range' and 'freshness'
                   - regex       (opt)   used by 'regex'
                   - values      (opt)   used by 'in_set' and 'foreign_key'

        Returns:
            dict with keys:
                success            (bool)  did the check pass?
                failing_count      (int)   number of rows that failed
                total_count        (int)   total rows checked
                unexpected_samples (list)  up to 5 examples of failing values

        Raises:
            ValueError: if check_type is unknown
            RuntimeError: if GE fails to run the expectation
        """
        check_name = check.get("name", "unnamed")
        check_type = check.get("check_type")
        column     = check.get("column")

        # ── guard: validate required keys before doing anything ──
        # If the Validation Engine accidentally passes an incomplete check
        # dict, we catch it here immediately with a clear error message
        # instead of getting a confusing crash deep inside GE.
        required_keys = ["name", "check_type", "column", "threshold", "severity"]
        missing = [k for k in required_keys if k not in check]
        if missing:
            raise ValueError(
                f"Check dict '{check_name}' is missing required keys: {missing}. "
                f"Every check must have: {required_keys}"
            )

        logger.info(f"Running check '{check_name}' | type: '{check_type}' | column: '{column}'")

        # ── step 1: convert DataFrame to GE dataset ──────────
        try:
            ge_dataset = ge.from_pandas(df)
        except Exception as e:
            logger.error(f"Failed to create GE dataset: {e}")
            raise RuntimeError(f"GE dataset creation failed: {e}") from e

        # ── step 2: get the GE method name from the mapper ───
        expectation_name = map_check_type(check_type)

        # ── step 3: build kwargs from the check dict ─────────
        kwargs = self._build_kwargs(check, column)

        # ── step 4: get the GE method and run it ─────────────
        try:
            method = getattr(ge_dataset, expectation_name)
            raw_result = method(**kwargs)
        except Exception as e:
            logger.error(f"GE expectation '{expectation_name}' failed: {e}")
            raise RuntimeError(
                f"Failed to run expectation '{expectation_name}' "
                f"on column '{column}': {e}"
            ) from e

        # ── step 5: extract and return clean result ───────────
        result = self._extract_result(raw_result, check_name)
        logger.info(
            f"Check '{check_name}' → "
            f"{'PASSED' if result['success'] else 'FAILED'} | "
            f"failing: {result['failing_count']}/{result['total_count']}"
        )
        return result

    def _build_kwargs(self, check: dict, column: str) -> dict:
        """
        Builds the keyword arguments dict to pass to the GE method.

        FIX: 'threshold' from the check dict is now mapped to GE's 'mostly'
        parameter. 'mostly' tells GE what fraction of rows must pass for the
        check to be considered successful (e.g. mostly=0.95 means 95% of
        rows must pass). Without this, GE defaults to requiring 100% of rows
        to pass, which ignores the threshold the user configured.

        Args:
            check:  the full check definition dict
            column: the column name to check

        Returns:
            dict of kwargs ready to unpack into the GE method
        """
        kwargs = {"column": column}

        # ── inject auto kwargs for special check types ────────
        # Some check types (e.g. not_empty) always need a specific
        # internal parameter the user never supplies themselves.
        # We pull it from the mapper so this file never hardcodes details.
        check_type = check.get("check_type", "")
        auto = get_auto_kwargs(check_type)
        kwargs.update(auto)

        # Guard: range/freshness checks must have at least one bound
        if check_type in ("range", "freshness"):
            if "min" not in check and "max" not in check:
                raise ValueError(
                    f"Check '{check.get('name')}' uses check_type '{check_type}' "
                    f"but provides neither 'min' nor 'max'. "
                    f"At least one bound is required."
                )
            
        # ── FIX: map threshold → mostly ──────────────────────
        # 'threshold' in our config means "at least X fraction of rows must
        # pass this check". GE calls this same concept 'mostly'.
        # Example: threshold=0.95 → mostly=0.95 → GE passes if 95%+ rows pass
        if "threshold" in check:
            kwargs["mostly"] = check["threshold"]

        if "min" in check:
            kwargs["min_value"] = check["min"]
        if "max" in check:
            kwargs["max_value"] = check["max"]
        if "regex" in check:
            if check_type == "not_empty":
                logger.warning(
                    f"Check '{check.get('name')}' supplied a 'regex' key for "
                    f"'not_empty' — this is ignored. The adapter always uses "
                    f"the correct regex for not_empty automatically."
                )
            else:
                kwargs["regex"] = check["regex"]
        if "values" in check:
            kwargs["value_set"] = check["values"]

        logger.debug(f"Built kwargs for check '{check.get('name')}': {kwargs}")
        return kwargs

    def _extract_result(self, raw_result, check_name: str) -> dict:
        """
        Extracts the clean result dict from GE's raw result object.

        Args:
            raw_result: the result object returned by GE
            check_name: used for logging only

        Returns:
            clean dict matching the agreed contract:
            {success, failing_count, total_count, unexpected_samples}
        """
        try:
            success       = raw_result["success"]
            result_data   = raw_result.get("result", {})
            total_count   = result_data.get("element_count", 0)
            failing_count = result_data.get("unexpected_count", 0)
            samples       = result_data.get("partial_unexpected_list", [])[:5]

            return {
                "success":            success,
                "failing_count":      failing_count,
                "total_count":        total_count,
                "unexpected_samples": samples,
            }

        except Exception as e:
            logger.error(f"Failed to extract result for '{check_name}': {e}")
            raise RuntimeError(
                f"Could not parse GE result for check '{check_name}': {e}"
            ) from e