import logging
import great_expectations as ge
import pandas as pd
from .expectation_mapper import map_check_type, get_auto_kwargs

logger = logging.getLogger(__name__)


class GEAdapter:
    """Wraps Great Expectations. Nothing outside this file imports great_expectations."""

    def run_expectation(self, df: pd.DataFrame, check: dict) -> dict:
        check_name = check.get("name", "unnamed")
        check_type = check.get("check_type")
        column     = check.get("column")

        required_keys = ["name", "check_type", "column", "threshold", "severity"]
        missing = [k for k in required_keys if k not in check]
        if missing:
            raise ValueError(
                f"Check '{check_name}' is missing required keys: {missing}."
            )

        logger.info(f"Running check '{check_name}' | type: '{check_type}' | column: '{column}'")

        try:
            ge_dataset = ge.from_pandas(df)
        except Exception as e:
            raise RuntimeError(f"GE dataset creation failed: {e}") from e

        expectation_name = map_check_type(check_type)
        kwargs           = self._build_kwargs(check, column)

        try:
            raw_result = getattr(ge_dataset, expectation_name)(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to run expectation '{expectation_name}' on column '{column}': {e}"
            ) from e

        result = self._extract_result(raw_result, check_name)
        logger.info(
            f"Check '{check_name}' → "
            f"{'PASSED' if result['success'] else 'FAILED'} | "
            f"failing: {result['failing_count']}/{result['total_count']}"
        )
        return result

    def _build_kwargs(self, check: dict, column: str) -> dict:
        kwargs     = {"column": column}
        check_type = check.get("check_type", "")

        # Inject any kwargs that are always required for this check type.
        kwargs.update(get_auto_kwargs(check_type))

        if check_type in ("range", "freshness"):
            if "min" not in check and "max" not in check:
                raise ValueError(
                    f"Check '{check.get('name')}' uses '{check_type}' "
                    f"but provides neither 'min' nor 'max'."
                )

        # 'threshold' in our schema == GE's 'mostly' (fraction of rows that must pass).
        if "threshold" in check:
            kwargs["mostly"] = check["threshold"]

        if "min" in check: kwargs["min_value"] = check["min"]
        if "max" in check: kwargs["max_value"] = check["max"]

        if "regex" in check:
            if check_type == "not_empty":
                logger.warning(
                    f"Check '{check.get('name')}': 'regex' key ignored for 'not_empty' — "
                    f"the adapter injects the correct regex automatically."
                )
            else:
                kwargs["regex"] = check["regex"]

        if "values" in check:
            kwargs["value_set"] = check["values"]

        logger.debug(f"Built kwargs for check '{check.get('name')}': {kwargs}")
        return kwargs

    def _extract_result(self, raw_result, check_name: str) -> dict:
        try:
            result_data = raw_result.get("result", {})
            return {
                "success":            raw_result["success"],
                "failing_count":      result_data.get("unexpected_count", 0),
                "total_count":        result_data.get("element_count", 0),
                "unexpected_samples": result_data.get("partial_unexpected_list", [])[:5],
            }
        except Exception as e:
            raise RuntimeError(f"Could not parse GE result for check '{check_name}': {e}") from e
