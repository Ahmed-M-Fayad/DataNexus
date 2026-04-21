"""
validator.py — Validates a parsed ParsedConfig object.

After parser.py turns raw YAML into Python objects, this module checks that
every field makes sense (required fields present, values in allowed ranges,
check types recognised, etc.).

Usage:
    from src.config_parser.validator import ConfigValidator

    issues = ConfigValidator.validate(config)   # returns list[str]
    if issues:
        raise ValueError("\\n".join(issues))
"""

import re
from typing import List

from .schemas import (
    ParsedConfig,
    CheckConfig,
    VALID_CHECK_TYPES,
    VALID_SEVERITIES,
    VALID_ALERT_CHANNELS,
)


# ---------------------------------------------------------------------------
# Cron expression pattern — simple sanity check (5 fields, no deep parsing)
# ---------------------------------------------------------------------------
# Each cron field can be: *, a number, a range (1-5), a list (1,3),
# a step (*/6 or 1-5/2), or combinations thereof.
# We use [0-9,\-*/]+ as a simple sanity check — it validates the character
# set and that there are exactly 5 space-separated fields without trying
# to parse every possible cron variation.
_CRON_RE = re.compile(
    r"^[0-9,\-*/]+\s+"   # minute
    r"[0-9,\-*/]+\s+"    # hour
    r"[0-9,\-*/]+\s+"    # day-of-month
    r"[0-9,\-*/]+\s+"    # month
    r"[0-9,\-*/]+$"      # day-of-week
)


class ConfigValidator:
    """
    Validates a ParsedConfig object and returns a list of human-readable
    error strings.  An empty list means the config is valid.

    All methods are static — no need to instantiate this class.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def validate(config: ParsedConfig) -> List[str]:
        """
        Run all validation rules against *config*.

        Returns:
            A list of error message strings.  Empty list → config is valid.
        """
        errors: List[str] = []

        errors.extend(ConfigValidator._validate_top_level(config))
        errors.extend(ConfigValidator._validate_checks(config.checks))

        return errors

    # ------------------------------------------------------------------
    # Top-level field checks
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_top_level(config: ParsedConfig) -> List[str]:
        errors: List[str] = []

        # dataset is required and must be a non-empty string
        if not config.dataset or not config.dataset.strip():
            errors.append("'dataset' is required and must not be empty.")

        # checks list must not be empty
        if not config.checks:
            errors.append("'checks' must contain at least one check definition.")

        # quality_threshold must be between 0 and 100
        if not (0.0 <= config.quality_threshold <= 100.0):
            errors.append(
                f"'quality_threshold' must be between 0 and 100, "
                f"got {config.quality_threshold}."
            )

        # alert_channels must only contain known values
        for channel in config.alert_channels:
            if channel not in VALID_ALERT_CHANNELS:
                errors.append(
                    f"Unknown alert channel '{channel}'. "
                    f"Allowed: {sorted(VALID_ALERT_CHANNELS)}."
                )

        # schedule_cron — if provided, must look like a valid cron expression
        if config.schedule_cron is not None:
            cron = config.schedule_cron.strip()
            if not _CRON_RE.match(cron):
                errors.append(
                    f"'schedule_cron' does not look like a valid 5-field cron "
                    f"expression: '{cron}'."
                )

        return errors

    # ------------------------------------------------------------------
    # Per-check validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_checks(checks: List[CheckConfig]) -> List[str]:
        errors: List[str] = []
        seen_names: set = set()

        for idx, check in enumerate(checks):
            prefix = f"Check [{idx}] ('{check.name}')"

            # name must be non-empty
            if not check.name or not check.name.strip():
                errors.append(f"Check [{idx}]: 'name' is required and must not be empty.")

            # duplicate names
            if check.name in seen_names:
                errors.append(f"{prefix}: duplicate check name — each check must have a unique name.")
            seen_names.add(check.name)

            # check_type must be one of the known types
            if check.check_type not in VALID_CHECK_TYPES:
                errors.append(
                    f"{prefix}: unknown check_type '{check.check_type}'. "
                    f"Allowed: {sorted(VALID_CHECK_TYPES)}."
                )

            # severity must be one of the known values (all lowercase after parser normalisation)
            if check.severity not in VALID_SEVERITIES:
                errors.append(
                    f"{prefix}: unknown severity '{check.severity}'. "
                    f"Allowed: {sorted(VALID_SEVERITIES)}."
                )

            # threshold must be 0.0–1.0 when provided
            if check.threshold is not None:
                if not (0.0 <= check.threshold <= 1.0):
                    errors.append(
                        f"{prefix}: 'threshold' must be between 0.0 and 1.0, "
                        f"got {check.threshold}."
                    )

            # check_type-specific rules
            errors.extend(ConfigValidator._validate_check_type_fields(check, prefix))

        return errors

    # ------------------------------------------------------------------
    # Check-type-specific field rules
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_check_type_fields(check: CheckConfig, prefix: str) -> List[str]:
        """
        Validates that the fields required by each check_type are present
        and that no obviously wrong combinations exist.
        """
        errors: List[str] = []
        ct = check.check_type

        # FIX: Added 'completeness' — it measures the fill-rate of a specific
        # column, so a column name is required just like not_null.
        column_required_types = {
            "not_null", "unique", "range", "regex",
            "accepted_values", "completeness",
        }
        if ct in column_required_types and not check.column:
            errors.append(f"{prefix}: check_type '{ct}' requires a 'column' field.")

        # Range check: must provide at least one of min_value / max_value
        if ct == "range":
            if check.min_value is None and check.max_value is None:
                errors.append(
                    f"{prefix}: check_type 'range' requires at least one of "
                    f"'min_value' or 'max_value'."
                )
            # If both are provided, min must be ≤ max
            if (
                check.min_value is not None
                and check.max_value is not None
                and check.min_value > check.max_value
            ):
                errors.append(
                    f"{prefix}: 'min_value' ({check.min_value}) must be ≤ "
                    f"'max_value' ({check.max_value})."
                )

        # Regex check: must provide a pattern
        if ct == "regex":
            if not check.pattern:
                errors.append(f"{prefix}: check_type 'regex' requires a 'pattern' field.")
            else:
                # Try to compile it — catch obviously broken patterns
                try:
                    re.compile(check.pattern)
                except re.error as exc:
                    errors.append(
                        f"{prefix}: 'pattern' is not a valid regular expression: {exc}."
                    )

        # accepted_values check: must provide a non-empty list
        if ct == "accepted_values":
            if not check.accepted_values:
                errors.append(
                    f"{prefix}: check_type 'accepted_values' requires a non-empty "
                    f"'accepted_values' list."
                )

        # row_count check: must supply min_value as the minimum acceptable row count.
        # Note: 'threshold' (0.0–1.0 pass-rate) is a separate concept and is NOT
        # used for row_count checks.
        if ct == "row_count" and check.min_value is None:
            errors.append(
                f"{prefix}: check_type 'row_count' requires a 'min_value' "
                f"(the minimum acceptable number of rows)."
            )

        # FIX: schema check: must supply expected_schema so the engine knows
        # which columns and types to compare against.
        if ct == "schema" and not check.expected_schema:
            errors.append(
                f"{prefix}: check_type 'schema' requires an 'expected_schema' field "
                f"(a dict mapping column names to expected types)."
            )

        return errors