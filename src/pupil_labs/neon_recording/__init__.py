"""Top-level entry-point for the pl-neon-recording package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.neon_recording")
except PackageNotFoundError:
    # package is not installed
    pass

import os
import structlog
import logging

log = structlog.get_logger(__name__)

# obtained from: https://betterstack.com/community/guides/logging/structlog/
def filter_function(_, __, event_dict):
    if event_dict.get("func_name") == "delete_files":
        raise structlog.DropEvent
    return event_dict


level = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, level)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL),
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FUNC_NAME]
        ),
        filter_function,
        structlog.dev.ConsoleRenderer(),
    ]
)

log.info("NeonRecording: package loaded.")

from .neon_recording import load

__all__ = [
    "__version__",
    "load"
    ]
