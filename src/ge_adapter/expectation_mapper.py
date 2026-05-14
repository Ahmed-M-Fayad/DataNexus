import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_FALLBACK_MAP = {
    "not_null":              "expect_column_values_to_not_be_null",
    "completeness":          "expect_column_values_to_not_be_null",
    "regex":                 "expect_column_values_to_match_regex",
    "range":                 "expect_column_values_to_be_between",
    "in_set":                "expect_column_values_to_be_in_set",
    "unique":                "expect_column_values_to_be_unique",
    "foreign_key":           "expect_column_values_to_be_in_set",
    "freshness":             "expect_column_values_to_be_between",
    "not_empty":             "expect_column_values_to_not_match_regex",
    "referential_integrity": "expect_column_values_to_be_in_set",
}


def _load_expectation_map() -> dict:
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

AUTO_KWARGS = {
    "not_empty": {"regex": r"^\s*$"},
}


def map_check_type(check_type: str) -> str:
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
    return AUTO_KWARGS.get(check_type, {})