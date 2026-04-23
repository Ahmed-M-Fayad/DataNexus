"""
parser.py — Reads a YAML validation config file and returns a ParsedConfig.

This is the only file in the project that reads YAML from disk.
Everything else works with ParsedConfig / CheckConfig objects.

Usage:
    from src.config_parser.parser import ConfigParser

    # From a file path
    config = ConfigParser.from_file("config/examples/customers.yaml")

    # From a YAML string (useful when the config is stored in the DB)
    config = ConfigParser.from_string(yaml_text)

    # From a plain dict (useful for testing)
    config = ConfigParser.from_dict(raw_dict)
"""

import yaml
from typing import Union
from pathlib import Path

from .schemas import ParsedConfig, CheckConfig
from .validator import ConfigValidator


class ConfigParseError(Exception):
    """Raised when a config file cannot be parsed or fails validation."""
    pass


class ConfigParser:
    """
    Turns YAML config files (or strings / dicts) into ParsedConfig objects.

    All three entry points (from_file, from_string, from_dict) do the same
    thing internally:
        1. Produce a raw Python dict from the input.
        2. Call _build_config(raw_dict) to create the dataclass objects.
        3. Call ConfigValidator.validate() to check for errors.
        4. Return the validated ParsedConfig, or raise ConfigParseError.

    All methods are static — no need to instantiate this class.
    """

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    @staticmethod
    def from_file(path: Union[str, Path]) -> ParsedConfig:
        """
        Load and parse a YAML config from a file path.

        Args:
            path: Absolute or relative path to the .yaml file.

        Returns:
            A validated ParsedConfig object.

        Raises:
            ConfigParseError: If the file does not exist, cannot be parsed,
                              or fails validation.
        """
        path = Path(path)

        if not path.exists():
            raise ConfigParseError(f"Config file not found: {path}")

        if not path.is_file():
            raise ConfigParseError(f"Path is not a file: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw_text = fh.read()
        except OSError as exc:
            raise ConfigParseError(f"Could not read config file '{path}': {exc}") from exc

        return ConfigParser.from_string(raw_text, source=str(path))

    @staticmethod
    def from_string(yaml_text: str, source: str = "<string>") -> ParsedConfig:
        """
        Parse a YAML string into a ParsedConfig.

        Args:
            yaml_text: Raw YAML content.
            source:    Optional label used in error messages (e.g. a filename).

        Returns:
            A validated ParsedConfig object.

        Raises:
            ConfigParseError: If the YAML is malformed or fails validation.
        """
        try:
            raw_dict = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            raise ConfigParseError(
                f"Failed to parse YAML from '{source}': {exc}"
            ) from exc

        if not isinstance(raw_dict, dict):
            raise ConfigParseError(
                f"Config from '{source}' must be a YAML mapping (got {type(raw_dict).__name__})."
            )

        return ConfigParser.from_dict(raw_dict, source=source)

    @staticmethod
    def from_dict(raw: dict, source: str = "<dict>") -> ParsedConfig:
        """
        Build and validate a ParsedConfig from a plain Python dict.

        This is the lowest-level entry point — useful in tests and when the
        config is already in memory.

        Args:
            raw:    Dictionary matching the expected config schema.
            source: Optional label used in error messages.

        Returns:
            A validated ParsedConfig object.

        Raises:
            ConfigParseError: If required fields are missing or validation fails.
        """
        try:
            config = ConfigParser._build_config(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigParseError(
                f"Could not construct config from '{source}': {exc}"
            ) from exc

        errors = ConfigValidator.validate(config)
        if errors:
            bullet_list = "\n  - ".join(errors)
            raise ConfigParseError(
                f"Config from '{source}' has {len(errors)} validation error(s):\n  - {bullet_list}"
            )

        return config

    # ------------------------------------------------------------------
    # Internal builder — raw dict → dataclass objects
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(raw: dict) -> ParsedConfig:
        """
        Convert a raw dictionary into a ParsedConfig dataclass.

        No validation is done here — ConfigValidator handles that.
        This method only does type coercion and structural mapping.
        """
        # ---- top-level fields ----------------------------------------
        dataset = raw.get("dataset", "")
        name = raw.get("name")
        description = raw.get("description")
        schedule_cron = raw.get("schedule_cron")
        quality_threshold = float(raw.get("quality_threshold", 80.0))

        # alert_channels can be a list or a single string
        raw_channels = raw.get("alert_channels", [])
        if isinstance(raw_channels, str):
            raw_channels = [raw_channels]
        alert_channels = [str(c).strip() for c in raw_channels]

        # ---- checks list -----------------------------------------------
        raw_checks = raw.get("checks", [])
        if not isinstance(raw_checks, list):
            raise TypeError("'checks' must be a YAML list.")

        checks = [ConfigParser._build_check(c, idx) for idx, c in enumerate(raw_checks)]

        return ParsedConfig(
            dataset=dataset,
            name=name,
            description=description,
            schedule_cron=schedule_cron,
            alert_channels=alert_channels,
            quality_threshold=quality_threshold,
            checks=checks,
        )

    @staticmethod
    def _build_check(raw: dict, idx: int) -> CheckConfig:
        """
        Convert one raw check dict into a CheckConfig dataclass.
        """
        if not isinstance(raw, dict):
            raise TypeError(f"Check [{idx}] must be a YAML mapping, got {type(raw).__name__}.")

        # threshold: stored as 0.0–1.0 in configs
        threshold_raw = raw.get("threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else None

        # accepted_values: normalise to a list
        accepted_values_raw = raw.get("accepted_values")
        if accepted_values_raw is not None and not isinstance(accepted_values_raw, list):
            accepted_values_raw = [accepted_values_raw]

        # FIX: Normalise severity to lowercase to match the database Severity
        # enum (critical / high / medium / low).  This means the YAML can be
        # written as "High", "HIGH", or "high" and they all work correctly.
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