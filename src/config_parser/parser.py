import yaml
from typing import Union
from pathlib import Path

from .schemas import ParsedConfig, CheckConfig
from .validator import validate


class ConfigParseError(Exception):
    """Raised when a config file cannot be parsed or fails validation."""
    pass


def from_file(path: Union[str, Path]) -> ParsedConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigParseError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigParseError(f"Path is not a file: {path}")
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigParseError(f"Could not read config file '{path}': {exc}") from exc
    return from_string(raw_text, source=str(path))


def from_string(yaml_text: str, source: str = "<string>") -> ParsedConfig:
    try:
        raw_dict = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ConfigParseError(f"Failed to parse YAML from '{source}': {exc}") from exc

    if not isinstance(raw_dict, dict):
        raise ConfigParseError(
            f"Config from '{source}' must be a YAML mapping (got {type(raw_dict).__name__})."
        )
    return from_dict(raw_dict, source=source)


def from_dict(raw: dict, source: str = "<dict>") -> ParsedConfig:
    try:
        config = _build_config(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigParseError(f"Could not construct config from '{source}': {exc}") from exc

    errors = validate(config)
    if errors:
        bullet_list = "\n  - ".join(errors)
        raise ConfigParseError(
            f"Config from '{source}' has {len(errors)} validation error(s):\n  - {bullet_list}"
        )
    return config


# ── Internal builders ─────────────────────────────────────────────────────────

def _build_config(raw: dict) -> ParsedConfig:
    dataset           = raw.get("dataset", "")
    name              = raw.get("name")
    description       = raw.get("description")
    schedule_cron     = raw.get("schedule_cron")
    quality_threshold = float(raw.get("quality_threshold", 80.0))

    raw_channels = raw.get("alert_channels", [])
    if isinstance(raw_channels, str):
        raw_channels = [raw_channels]
    alert_channels = [str(c).strip() for c in raw_channels]

    raw_checks = raw.get("checks", [])
    if not isinstance(raw_checks, list):
        raise TypeError("'checks' must be a YAML list.")

    checks = [_build_check(c, idx) for idx, c in enumerate(raw_checks)]

    return ParsedConfig(
        dataset=dataset,
        name=name,
        description=description,
        schedule_cron=schedule_cron,
        alert_channels=alert_channels,
        quality_threshold=quality_threshold,
        checks=checks,
    )


def _build_check(raw: dict, idx: int) -> CheckConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"Check [{idx}] must be a YAML mapping, got {type(raw).__name__}.")

    threshold_raw = raw.get("threshold")
    threshold = float(threshold_raw) if threshold_raw is not None else None

    accepted_values_raw = raw.get("accepted_values")
    if accepted_values_raw is not None and not isinstance(accepted_values_raw, list):
        accepted_values_raw = [accepted_values_raw]

    # Normalise to lowercase — the DB Severity enum is all-lowercase.
    severity = str(raw.get("severity", "medium")).strip().lower()

    return CheckConfig(
        name=raw.get("name", ""),
        check_type=raw.get("check_type", ""),
        severity=severity,
        column=raw.get("column"),
        threshold=threshold,
        min_value=raw.get("min_value"),
        max_value=raw.get("max_value"),
        pattern=raw.get("pattern"),
        accepted_values=accepted_values_raw,
        expected_schema=raw.get("expected_schema"),
    )


# Backwards-compatibility shim so existing code that does
#   from src.config_parser.parser import ConfigParser
# still works during the transition to CLI.
class ConfigParser:
    from_file   = staticmethod(from_file)
    from_string = staticmethod(from_string)
    from_dict   = staticmethod(from_dict)
