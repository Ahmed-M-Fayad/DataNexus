# Public API.
# Preferred style:  from src.config_parser import from_file, from_string, from_dict
# Compat style:     from src.config_parser import ConfigParser   (still works)

from .parser import (
    from_file,
    from_string,
    from_dict,
    ConfigParser,       # backwards-compat shim
    ConfigParseError,
)
from .validator import validate, ConfigValidator  # ConfigValidator is a compat shim
from .schemas import (
    ParsedConfig,
    CheckConfig,
    VALID_CHECK_TYPES,
    VALID_SEVERITIES,
)

__all__ = [
    "from_file",
    "from_string",
    "from_dict",
    "ConfigParser",
    "ConfigParseError",
    "validate",
    "ConfigValidator",
    "ParsedConfig",
    "CheckConfig",
    "VALID_CHECK_TYPES",
    "VALID_SEVERITIES",
]
