import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Any


VALID_CHECK_TYPES = {
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
}

# Severity values must be lowercase — matches the DB Severity enum.
VALID_SEVERITIES = {"low", "medium", "high", "critical"}

VALID_ALERT_CHANNELS = {"slack", "email"}


@dataclass
class CheckConfig:
    name: str
    check_type: str
    severity: str = "medium"

    column: Optional[str] = None
    threshold: Optional[float] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    pattern: Optional[str] = None
    accepted_values: Optional[List[Any]] = None
    expected_schema: Optional[dict] = None

    def to_dict(self) -> dict:
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


@dataclass
class ParsedConfig:
    dataset: str
    checks: List[CheckConfig]

    name: Optional[str] = None
    description: Optional[str] = None
    schedule_cron: Optional[str] = None
    alert_channels: List[str] = field(default_factory=list)
    quality_threshold: float = 80.0

    @property
    def db_quality_threshold(self) -> float:
        # DB column stores threshold as 0.0–1.0; YAML uses 0–100.
        return self.quality_threshold / 100.0

    def to_dict(self) -> dict:
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
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
