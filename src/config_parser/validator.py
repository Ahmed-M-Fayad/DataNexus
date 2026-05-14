import re
from typing import List

from .schemas import (
    ParsedConfig,
    CheckConfig,
    VALID_CHECK_TYPES,
    VALID_SEVERITIES,
    VALID_ALERT_CHANNELS,
)

# Simple 5-field cron sanity check (character-set only, not semantic).
_CRON_RE = re.compile(
    r"^[0-9,\-*/]+\s+"
    r"[0-9,\-*/]+\s+"
    r"[0-9,\-*/]+\s+"
    r"[0-9,\-*/]+\s+"
    r"[0-9,\-*/]+$"
)


def validate(config: ParsedConfig) -> List[str]:
    errors: List[str] = []
    errors.extend(_validate_top_level(config))
    errors.extend(_validate_checks(config.checks))
    return errors


def _validate_top_level(config: ParsedConfig) -> List[str]:
    errors: List[str] = []

    if not config.dataset or not config.dataset.strip():
        errors.append("'dataset' is required and must not be empty.")

    if not config.checks:
        errors.append("'checks' must contain at least one check definition.")

    if not (0.0 <= config.quality_threshold <= 100.0):
        errors.append(
            f"'quality_threshold' must be between 0 and 100, got {config.quality_threshold}."
        )

    for channel in config.alert_channels:
        if channel not in VALID_ALERT_CHANNELS:
            errors.append(
                f"Unknown alert channel '{channel}'. Allowed: {sorted(VALID_ALERT_CHANNELS)}."
            )

    if config.schedule_cron is not None:
        cron = config.schedule_cron.strip()
        if not _CRON_RE.match(cron):
            errors.append(
                f"'schedule_cron' does not look like a valid 5-field cron expression: '{cron}'."
            )

    return errors


def _validate_checks(checks: List[CheckConfig]) -> List[str]:
    errors: List[str] = []
    seen_names: set = set()

    for idx, check in enumerate(checks):
        prefix = f"Check [{idx}] ('{check.name}')"

        if not check.name or not check.name.strip():
            errors.append(f"Check [{idx}]: 'name' is required and must not be empty.")

        if check.name in seen_names:
            errors.append(f"{prefix}: duplicate check name — each check must have a unique name.")
        seen_names.add(check.name)

        if check.check_type not in VALID_CHECK_TYPES:
            errors.append(
                f"{prefix}: unknown check_type '{check.check_type}'. "
                f"Allowed: {sorted(VALID_CHECK_TYPES)}."
            )

        if check.severity not in VALID_SEVERITIES:
            errors.append(
                f"{prefix}: unknown severity '{check.severity}'. "
                f"Allowed: {sorted(VALID_SEVERITIES)}."
            )

        if check.threshold is not None and not (0.0 <= check.threshold <= 1.0):
            errors.append(
                f"{prefix}: 'threshold' must be between 0.0 and 1.0, got {check.threshold}."
            )

        errors.extend(_validate_check_type_fields(check, prefix))

    return errors


def _validate_check_type_fields(check: CheckConfig, prefix: str) -> List[str]:
    errors: List[str] = []
    ct = check.check_type

    column_required = {"not_null", "unique", "range", "regex", "accepted_values", "completeness"}
    if ct in column_required and not check.column:
        errors.append(f"{prefix}: check_type '{ct}' requires a 'column' field.")

    if ct == "range":
        if check.min_value is None and check.max_value is None:
            errors.append(
                f"{prefix}: check_type 'range' requires at least one of 'min_value' or 'max_value'."
            )
        if (
            check.min_value is not None
            and check.max_value is not None
            and check.min_value > check.max_value
        ):
            errors.append(
                f"{prefix}: 'min_value' ({check.min_value}) must be ≤ 'max_value' ({check.max_value})."
            )

    if ct == "regex":
        if not check.pattern:
            errors.append(f"{prefix}: check_type 'regex' requires a 'pattern' field.")
        else:
            try:
                re.compile(check.pattern)
            except re.error as exc:
                errors.append(f"{prefix}: 'pattern' is not a valid regular expression: {exc}.")

    if ct == "accepted_values" and not check.accepted_values:
        errors.append(
            f"{prefix}: check_type 'accepted_values' requires a non-empty 'accepted_values' list."
        )

    if ct == "row_count" and check.min_value is None:
        errors.append(
            f"{prefix}: check_type 'row_count' requires a 'min_value' "
            f"(the minimum acceptable number of rows)."
        )

    if ct == "schema" and not check.expected_schema:
        errors.append(
            f"{prefix}: check_type 'schema' requires an 'expected_schema' field "
            f"(a dict mapping column names to expected types)."
        )

    return errors


# Backwards-compatibility shim.
class ConfigValidator:
    validate = staticmethod(validate)
