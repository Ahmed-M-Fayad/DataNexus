import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Fallback map used if the YAML config file is missing
# or cannot be loaded. Keeps the system running safely.
# ─────────────────────────────────────────────────────
# IMPORTANT: When adding a new check type, update BOTH:
#   1. config/expectation_mappings.yaml  (primary source)
#   2. This fallback dict                (safety net)
# ─────────────────────────────────────────────────────

_FALLBACK_MAP = {
    "not_null":              "expect_column_values_to_not_be_null",
    "completeness":          "expect_column_values_to_not_be_null",
    "regex":                 "expect_column_values_to_match_regex",
    "range":                 "expect_column_values_to_be_between",
    "in_set":                "expect_column_values_to_be_in_set",
    "unique":                "expect_column_values_to_be_unique",
    "foreign_key":           "expect_column_values_to_be_in_set",
    "freshness":             "expect_column_values_to_be_between",
    # not_empty — catches "" and "   " (whitespace-only) that not_null misses.
    # GE's not_match_regex rejects any row where the regex matches.
    # regex ^\s*$ matches empty strings AND strings that are only spaces/tabs.
    "not_empty":             "expect_column_values_to_not_match_regex",
    # referential_integrity — same behaviour as foreign_key (values must exist
    # in a reference set) but named as the SRS and implementation plan define it.
    # Both names are kept so either works in a validation config YAML.
    "referential_integrity": "expect_column_values_to_be_in_set",
}


def _load_expectation_map() -> dict:
    """
    Loads expectation mappings from config/expectation_mappings.yaml.
    Falls back to hardcoded defaults if the file is missing or unreadable.

    Returns:
        dict mapping check_type strings to GE expectation method names
    """
    config_path = (
        Path(__file__).parent  # src/ge_adapter/
        .parent                # src/
        .parent                # datanexus/ (project root)
        / "config"
        / "expectation_mappings.yaml"
    )

    # ── defensive check 1: file exists ──────────────────
    if not config_path.exists():
        logger.warning(
            f"Expectation mappings config not found at '{config_path}'. "
            f"Falling back to hardcoded defaults."
        )
        return _FALLBACK_MAP.copy()

    # ── defensive check 2: file is readable and valid YAML ──
    try:
        with open(config_path, "r") as f:
            mapping = yaml.safe_load(f)

        if not isinstance(mapping, dict):
            raise ValueError("Mappings file did not parse to a dictionary.")

        if not mapping:
            raise ValueError("Mappings file is empty.")

        logger.info(
            f"Loaded {len(mapping)} expectation mappings "
            f"from '{config_path}'"
        )
        return mapping

    except Exception as e:
        logger.warning(
            f"Failed to load mappings config: {e}. "
            f"Falling back to hardcoded defaults."
        )
        return _FALLBACK_MAP.copy()


# Load once at module import time — not on every function call
EXPECTATION_MAP = _load_expectation_map()


# ─────────────────────────────────────────────────────
# FIX: check types that require auto-injected kwargs.
# When the Validation Engine calls not_empty, the user
# does not supply a regex — the adapter injects it here
# automatically because ^$ is always the correct regex
# for "empty string detection" and should never change.
# ─────────────────────────────────────────────────────
AUTO_KWARGS = {
    # ^\s*$ matches empty strings ("") AND whitespace-only strings ("  ", "\t")
    # ^$ would only catch truly empty strings and miss "   " which is equally bad data
    "not_empty": {"regex": r"^\s*$"},
}


def map_check_type(check_type: str) -> str:
    """
    Maps a DataNexus check type to its GE expectation name.

    Args:
        check_type: string from the validation config
                    e.g. 'not_null', 'range', 'regex'

    Returns:
        GE expectation method name as a string
        e.g. 'expect_column_values_to_be_between'

    Raises:
        ValueError: if check_type is not found in the mappings
    """
    if check_type not in EXPECTATION_MAP:
        logger.error(f"Unknown check type: '{check_type}'")
        raise ValueError(
            f"Unknown check type: '{check_type}'. "
            f"Supported types: {list(EXPECTATION_MAP.keys())}"
        )

    expectation = EXPECTATION_MAP[check_type]
    logger.info(f"Mapped '{check_type}' -> '{expectation}'")
    return expectation


def get_auto_kwargs(check_type: str) -> dict:
    """
    Returns any kwargs that must be auto-injected for a given check type,
    regardless of what the user provided in the check config.

    This exists because some check types (like not_empty) always need
    a specific internal parameter (regex='^$') that is never user-supplied.

    Args:
        check_type: the check type string e.g. 'not_empty'

    Returns:
        dict of extra kwargs to inject, or empty dict if none needed
    """
    return AUTO_KWARGS.get(check_type, {})