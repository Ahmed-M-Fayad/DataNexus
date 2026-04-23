"""
src/config_parser/__init__.py

Public API for the Config Parser module.

Typical usage from anywhere else in the project:

    from src.config_parser import ConfigParser, ConfigParseError, ParsedConfig

    config = ConfigParser.from_file("config/examples/customers.yaml")
    print(config.dataset)                   # "public.customers"
    print(len(config.checks))               # 3
    print(config.checks[0].name)            # "email_not_null"
    print(config.checks[0].severity)        # "high"  (always lowercase)
    print(config.db_quality_threshold)      # 0.85  (converted from 85.0)

NOTE: The parsed config class is called ParsedConfig (not ValidationConfig)
to avoid a name clash with database.models.ValidationConfig (the DB row class).
If you need both in the same file, import them like this:

    from src.config_parser import ParsedConfig
    from src.database.models import ValidationConfig as ValidationConfigModel
"""

from .parser import ConfigParser, ConfigParseError
from .schemas import ParsedConfig, CheckConfig, VALID_CHECK_TYPES, VALID_SEVERITIES
from .validator import ConfigValidator

__all__ = [
    "ConfigParser",
    "ConfigParseError",
    "ConfigValidator",
    "ParsedConfig",
    "CheckConfig",
    "VALID_CHECK_TYPES",
    "VALID_SEVERITIES",
]