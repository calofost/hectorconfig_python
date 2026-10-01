"""Python tools for configuring HECTOR probe fields.

The public entry point is :func:`configure_hector`, which is deliberately
named after the R package function so that the two workflows are easy to
compare while the Python version is being validated.
"""

from .constants import HectorConstants, hector_engine_parameters, load_constants
from .check import CheckResult, check_configuration
from .configure import ConfigurationResult, configure_hector, refresh_result_tables
from .io import hector_read_tables, hector_write_tables

__all__ = [
    "ConfigurationResult",
    "CheckResult",
    "HectorConstants",
    "configure_hector",
    "check_configuration",
    "refresh_result_tables",
    "hector_engine_parameters",
    "hector_read_tables",
    "hector_write_tables",
    "load_constants",
]

__version__ = "0.1.10"
