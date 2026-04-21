"""
schemas.py — Dataclass definitions for validation config objects.

These are the Python objects that parser.py produces and validator.py checks.
The rest of the system (ValidationEngine, CLI, API) works with these objects,
never with raw dicts or YAML strings.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Any


# ---------------------------------------------------------------------------
# Allowed literal values (used by validator.py for enum-style checks)
# ---------------------------------------------------------------------------

VALID_CHECK_TYPES = {
    "not_null",
    "unique",
    "range",
    "regex",
    "completeness",
    "accepted_values",
    "row_count",
    "schema",
}

# FIX: Use lowercase to match the database Severity enum exactly.
# The DB model defines: critical / high / medium / low (all lowercase).
# The parser normalises whatever the user writes to lowercase before validation.
VALID_SEVERITIES = {"low", "medium", "high", "critical"}

VALID_ALERT_CHANNELS = {"slack", "email"}


# ---------------------------------------------------------------------------
# Per-check dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckConfig:
    """
    Represents a single validation check inside a config file.

    Example YAML block that maps to this object:
        - name: email_not_null
          column: email
          check_type: not_null
          threshold: 0.95
          severity: high
    """

    name: str
    check_type: str
    severity: str = "medium"          # FIX: lowercase default to match DB enum

    # Optional fields — not every check type needs all of these
    column: Optional[str] = None          # column to run the check on
    threshold: Optional[float] = None     # 0.0–1.0 pass-rate threshold
    min_value: Optional[Any] = None       # for range / row_count checks
    max_value: Optional[Any] = None       # for range checks
    pattern: Optional[str] = None         # for regex checks
    accepted_values: Optional[List[Any]] = None   # for accepted_values checks
    expected_schema: Optional[dict] = None        # for schema checks

    def to_dict(self) -> dict:
        """Serialize back to a plain dictionary (useful for testing / logging)."""
        return {
            "name": self.name,
            "check_type": self.check_type,
            "severity": self.severity,
            "column": self.column,
            "threshold": self.threshold,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "pattern": self.pattern,
            "accepted_values": self.accepted_values,
            "expected_schema": self.expected_schema,
        }


# ---------------------------------------------------------------------------
# Top-level config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedConfig:
    """
    Represents a complete, parsed validation config file.

    NOTE ON NAMING: This class is intentionally called ParsedConfig (not
    ValidationConfig) to avoid a name clash with the SQLAlchemy ORM class
    database.models.ValidationConfig.  Both classes can exist in the same
    Python module without confusion:

        from src.config_parser import ParsedConfig          # this class
        from src.database.models import ValidationConfig    # the DB row

    Example YAML that maps to this object:
        dataset: public.customers
        schedule_cron: '0 */6 * * *'
        alert_channels: [slack, email]
        checks:
          - name: email_not_null
            ...
    """

    dataset: str                             # "schema.table" or a CSV file path
    checks: List[CheckConfig]                # one or more check definitions

    # Optional top-level fields
    name: Optional[str] = None               # human-readable name for this config
    description: Optional[str] = None
    schedule_cron: Optional[str] = None      # cron expression, e.g. "0 */6 * * *"
    alert_channels: List[str] = field(default_factory=list)  # ["slack", "email"]

    # quality_threshold is stored on a 0–100 scale in the YAML (e.g. 85.0)
    # because that is more intuitive for the person writing the config.
    # Use db_quality_threshold below when writing this value to the DB column,
    # which stores it on a 0.0–1.0 scale.
    quality_threshold: float = 80.0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def db_quality_threshold(self) -> float:
        """
        Returns quality_threshold converted to the 0.0–1.0 scale used by the
        database ValidationConfig.quality_threshold column.

        Example:
            parsed.quality_threshold        → 85.0
            parsed.db_quality_threshold     → 0.85
        """
        return self.quality_threshold / 100.0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "dataset": self.dataset,
            "schedule_cron": self.schedule_cron,
            "alert_channels": self.alert_channels,
            "quality_threshold": self.quality_threshold,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_yaml(self) -> str:
        """
        Serialize to a YAML string, ready to be stored in the database
        ValidationConfig.config_yaml column (which is a Text field).

        Example:
            yaml_text = parsed_config.to_yaml()
            db_row.config_yaml = yaml_text
            session.commit()
        """
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )